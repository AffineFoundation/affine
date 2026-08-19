#!/usr/bin/env python3
"""Merge LoRA adapter into kevin base → full safetensors dir for vllm serve."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def _file_sha256(path: Path, nbytes: int = 0) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        if nbytes:
            h.update(f.read(nbytes))
        else:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def _first_shard(model_dir: Path) -> Path:
    idx = model_dir / "model.safetensors.index.json"
    if idx.is_file():
        weight_map = json.loads(idx.read_text()).get("weight_map") or {}
        shard = weight_map.get("model.embed_tokens.weight") or next(
            iter(weight_map.values()), None
        )
        if shard:
            return model_dir / shard
    single = model_dir / "model.safetensors"
    if single.is_file():
        return single
    raise FileNotFoundError(f"no safetensors shard under {model_dir}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--adapter", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--device-map",
        default="cpu",
        help="transformers device_map; use 'auto' on free GPUs 6,7 after train "
        "(much faster than cpu for 35B merge under TTL pressure)",
    )
    ap.add_argument(
        "--max-shard-size",
        default="5GB",
        help="save_pretrained max_shard_size (5GB avoids gocryptfs hang on "
        "huge first shard; use /tmp out dir too)",
    )
    ap.add_argument(
        "--save-original-format",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="transformers save_pretrained save_original_format; set "
        "--no-save-original-format when device_map=auto offlading breaks "
        "shard conversion (RuntimeError: offlading / shard files)",
    )
    args = ap.parse_args()

    t0 = time.time()
    if args.out.exists():
        shutil.rmtree(args.out)
    args.out.mkdir(parents=True)

    print(
        f"[merge] load base {args.base} device_map={args.device_map} "
        f"max_shard_size={args.max_shard_size} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
        flush=True,
    )
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.base,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map,
        trust_remote_code=False,
    )
    print(f"[merge] load adapter {args.adapter}", flush=True)
    model = PeftModel.from_pretrained(model, str(args.adapter))
    print("[merge] merge_and_unload", flush=True)
    model = model.merge_and_unload()
    # Contig-clone before safetensors serialize — peft/mmap tensors on
    # overlay|/tmp hit EFAULT "Bad address" (os error 14) otherwise
    # (H109/H110 recover; R15 p2203).
    print(
        "[merge] contiguous-clone state_dict (EFAULT workaround)",
        flush=True,
    )
    sd = {
        k: v.detach().to("cpu").contiguous().clone()
        for k, v in model.state_dict().items()
    }
    print(
        f"[merge] save {args.out} shard={args.max_shard_size} "
        f"n_tensors={len(sd)} save_original_format={args.save_original_format}",
        flush=True,
    )
    model.save_pretrained(
        str(args.out),
        state_dict=sd,
        safe_serialization=True,
        max_shard_size=args.max_shard_size,
        save_original_format=args.save_original_format,
    )
    del sd
    tok.save_pretrained(str(args.out))

    # AutoModelForCausalLM.save_pretrained writes the *text* config
    # (qwen3_5_moe_text / Qwen3_5MoeForCausalLM). Stock vLLM on the eval
    # stack expects the multimodal wrapper (qwen3_5_moe /
    # Qwen3_5MoeForConditionalGeneration) that the king ships. Observed
    # 2026-08-07: chall serve crashed with TypeError expecting
    # Qwen3_5MoeConfig, got Qwen3_5MoeTextConfig. Restore wrapper config
    # + vision preprocessor sidecars from the base snapshot (weights
    # already use model.language_model.* keys; sizes match).
    base_path = Path(args.base)
    for name in (
        "config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "video_preprocessor_config.json",
    ):
        src = base_path / name
        if src.is_file():
            shutil.copy2(src, args.out / name)
            print(f"[merge] restored {name} from base", flush=True)
    # Tok331102 ships processor_config.json (nested image_processor) but not
    # preprocessor_config.json. Stock vLLM/transformers ImageProcessingMixin
    # still requires preprocessor_config.json — without it chall dies at
    # MultiModalBudget (H79 pass307). Derive it from processor_config when
    # the flat file is absent.
    pre_out = args.out / "preprocessor_config.json"
    proc_src = base_path / "processor_config.json"
    if not pre_out.is_file() and proc_src.is_file():
        proc = json.loads(proc_src.read_text())
        img = proc.get("image_processor", proc)
        pre_out.write_text(json.dumps(img, indent=2) + "\n")
        print(
            "[merge] derived preprocessor_config.json from "
            "processor_config.json image_processor",
            flush=True,
        )

    # CausalLM save also drops the vision tower (model.visual.*).
    # With the wrapper config restored, vLLM requires those weights.
    #
    # Two layouts observed:
    #   kevin: separate model-visual*.safetensors — copy whole files.
    #   TalentPigs: visual packed into model-00016-of-00016.safetensors
    #   alongside a few language tensors — cannot copy the whole shard
    #   (would clobber merged LoRA language weights). Extract only missing
    #   keys into model-visual-restored.safetensors.
    out_idx_path = args.out / "model.safetensors.index.json"
    base_idx_path = base_path / "model.safetensors.index.json"
    out_idx = (
        json.loads(out_idx_path.read_text())
        if out_idx_path.is_file()
        else {"metadata": {}, "weight_map": {}}
    )
    base_idx = (
        json.loads(base_idx_path.read_text())
        if base_idx_path.is_file()
        else {"weight_map": {}}
    )
    added = 0
    visual_shards = sorted(base_path.glob("model-visual*.safetensors"))
    for src in visual_shards:
        shutil.copy2(src, args.out / src.name)
        print(f"[merge] restored visual shard {src.name}", flush=True)
    for key, shard in (base_idx.get("weight_map") or {}).items():
        if key not in out_idx["weight_map"] and (
            args.out / shard
        ).is_file():
            # Key lives in a shard we already copied (kevin-style).
            out_idx["weight_map"][key] = shard
            added += 1

    # CausalLM can leave visual keys in the index pointing at language
    # shards that do not contain them (Tok331102 2-shard layout). Treat
    # those phantom entries as missing so we extract a real visual shard.
    from safetensors import safe_open

    def _shard_keyset(path: Path) -> set[str]:
        if not path.is_file():
            return set()
        with safe_open(str(path), framework="pt") as f:
            return set(f.keys())

    out_shard_keys: dict[str, set[str]] = {}
    phantom = {}
    for key, shard in list(out_idx["weight_map"].items()):
        if "visual" not in key:
            continue
        if shard not in out_shard_keys:
            out_shard_keys[shard] = _shard_keyset(args.out / shard)
        if key not in out_shard_keys[shard]:
            phantom[key] = shard
    if phantom:
        print(
            f"[merge] phantom visual index entries (not in shard files): "
            f"{len(phantom)} — will extract from base",
            flush=True,
        )
        for key in phantom:
            out_idx["weight_map"].pop(key, None)

    missing = {
        k: sh
        for k, sh in (base_idx.get("weight_map") or {}).items()
        if k not in out_idx["weight_map"]
    }
    if missing:
        from collections import defaultdict

        from safetensors.torch import save_file

        by_shard: dict[str, list[str]] = defaultdict(list)
        for key, shard in missing.items():
            by_shard[shard].append(key)
        restored: dict[str, torch.Tensor] = {}
        for shard, keys in by_shard.items():
            src = base_path / shard
            if not src.is_file():
                raise FileNotFoundError(
                    f"missing base shard for visual restore: {src}"
                )
            # Clone off safetensors mmap before the handle closes — otherwise
            # save_file can hit EFAULT / "Bad address" (os error 14) (R9 p2187).
            with safe_open(str(src), framework="pt", device="cpu") as f:
                for key in keys:
                    t = f.get_tensor(key)
                    restored[key] = t.detach().contiguous().clone()
            print(
                f"[merge] extracted {len(keys)} missing keys from {shard}",
                flush=True,
            )
        out_vis = args.out / "model-visual-restored.safetensors"
        # Overlay /tmp save_file can still EFAULT after clone (R3 p2189;
        # LESSONS p2123). Serialize on /root then copy into merge out.
        tmp_vis = Path("/root") / f".merge_vis_{os.getpid()}.safetensors"
        try:
            if tmp_vis.exists():
                tmp_vis.unlink()
            save_file(restored, str(tmp_vis))
            shutil.copyfile(tmp_vis, out_vis)
        finally:
            try:
                tmp_vis.unlink(missing_ok=True)
            except TypeError:
                if tmp_vis.exists():
                    tmp_vis.unlink()
        for key in restored:
            out_idx["weight_map"][key] = out_vis.name
            added += 1
        print(
            f"[merge] wrote {out_vis.name} with {len(restored)} tensors "
            f"(TalentPigs-style packed visual restore; via /root cipher)",
            flush=True,
        )

    if added or visual_shards or missing or phantom:
        total = sum(p.stat().st_size for p in args.out.glob("*.safetensors"))
        out_idx.setdefault("metadata", {})
        out_idx["metadata"]["total_size"] = total
        if "total_parameters" in (base_idx.get("metadata") or {}):
            out_idx["metadata"]["total_parameters"] = base_idx["metadata"][
                "total_parameters"
            ]
        out_idx_path.write_text(json.dumps(out_idx, indent=2) + "\n")
        n_vis = sum(1 for k in out_idx["weight_map"] if "visual" in k)
        # Refuse on index-only visual (phantom) — verify tensors exist.
        vis_resolved = 0
        for key, shard in out_idx["weight_map"].items():
            if "visual" not in key:
                continue
            if shard not in out_shard_keys:
                out_shard_keys[shard] = _shard_keyset(args.out / shard)
            if key in out_shard_keys[shard] or (
                (args.out / shard).is_file()
                and key in _shard_keyset(args.out / shard)
            ):
                vis_resolved += 1
        # Refresh keyset for newly written visual shard.
        if (args.out / "model-visual-restored.safetensors").is_file():
            out_shard_keys["model-visual-restored.safetensors"] = _shard_keyset(
                args.out / "model-visual-restored.safetensors"
            )
            vis_resolved = sum(
                1
                for k, sh in out_idx["weight_map"].items()
                if "visual" in k and k in out_shard_keys.get(sh, set())
            )
        print(
            f"[merge] index now has {added} restored keys; "
            f"visual_keys={n_vis} visual_resolved={vis_resolved}",
            flush=True,
        )
        base_has_vis = any(
            "visual" in k for k in (base_idx.get("weight_map") or {})
        )
        if base_has_vis and vis_resolved == 0:
            raise SystemExit(
                "REFUSE: base has model.visual.* but merged output has none "
                "— chall serve would crash under wrapper config"
            )
        if base_has_vis and vis_resolved < n_vis:
            raise SystemExit(
                f"REFUSE: phantom visual index — resolved {vis_resolved}/{n_vis}"
            )

    # Hygiene: no *.py, strip auto_map if present.
    cfg_path = args.out / "config.json"
    cfg = json.loads(cfg_path.read_text())
    if "auto_map" in cfg:
        del cfg["auto_map"]
        cfg_path.write_text(json.dumps(cfg, indent=2) + "\n")
    for p in args.out.rglob("*.py"):
        p.unlink()

    # Hard rule: weight-identical to king/base is rejected at submit. Refuse
    # here so we never burn ~66 min of n40+n80 sim on a no-op merge.
    #
    # Do NOT use only the first 1MiB of shard-1: that window is almost always
    # embed/lm_head (untouched by LoRA), so a real LoRA merge false-positives
    # as identical (observed 2026-08-07 H1: head equal, mid equal, tail ≠,
    # and q/k/v/o_proj tensors differ). Sample head/mid/tail on every
    # safetensors shard; refuse only if ALL sampled windows match.
    base_shard = _first_shard(base_path)
    out_shard = _first_shard(args.out)
    base_fp = _file_sha256(base_shard, nbytes=1 << 20)
    out_fp = _file_sha256(out_shard, nbytes=1 << 20)

    def _window_sha(path: Path, offset: int, nbytes: int = 1 << 20) -> str:
        h = hashlib.sha256()
        size = path.stat().st_size
        off = max(0, min(offset, max(0, size - nbytes)))
        with open(path, "rb") as f:
            f.seek(off)
            h.update(f.read(nbytes))
        return h.hexdigest()

    shard_windows: dict[str, dict] = {}
    any_diff = False
    # Only numbered language-model shards. Base may ship unused extras
    # (e.g. model-visual-extra.safetensors) that save_pretrained omits.
    shard_paths = sorted(base_path.glob("model-*-of-*.safetensors"))
    if not shard_paths:
        shard_paths = sorted(
            p for p in base_path.glob("*.safetensors") if p.name.startswith("model")
        )
    for shard_path in shard_paths:
        name = shard_path.name
        out_p = args.out / name
        if not out_p.is_file():
            any_diff = True
            shard_windows[name] = {"out_missing": True}
            continue
        size = shard_path.stat().st_size
        windows = {
            "head": (_window_sha(shard_path, 0), _window_sha(out_p, 0)),
            "mid": (
                _window_sha(shard_path, size // 2),
                _window_sha(out_p, size // 2),
            ),
            "tail": (
                _window_sha(shard_path, max(0, size - (1 << 20))),
                _window_sha(out_p, max(0, out_p.stat().st_size - (1 << 20))),
            ),
        }
        eqs = {k: a == b for k, (a, b) in windows.items()}
        if not all(eqs.values()):
            any_diff = True
        shard_windows[name] = {
            "size_base": size,
            "size_out": out_p.stat().st_size,
            "equal": eqs,
        }

    # Legacy field kept for harvest parsers; true identity uses window probe.
    first_1mib_identical = base_fp == out_fp
    identical = (not any_diff) and first_1mib_identical

    meta = {
        "base": args.base,
        "adapter": str(args.adapter),
        "out": str(args.out),
        "device_map": args.device_map,
        "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "base_first_shard": str(base_shard),
        "out_first_shard": str(out_shard),
        "base_first_1MiB_sha256": base_fp,
        "out_first_1MiB_sha256": out_fp,
        "first_1MiB_identical": first_1mib_identical,
        "shard_windows": shard_windows,
        "weight_identical": identical,
        "elapsed_s": time.time() - t0,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    (args.out / "merge_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    # Stage where host harvest looks. Name by out path so H1v2 merge does not
    # clobber H1's meta if harvest hasn't SCP'd it yet (pass 58).
    Path("/root/affine_data").mkdir(parents=True, exist_ok=True)
    out_s = str(args.out)
    if "/h5b/" in out_s or out_s.rstrip("/").endswith("h5b/merged"):
        stage_name = "h5b_merge_meta.json"
    elif "/h1v2/" in out_s or out_s.rstrip("/").endswith("h1v2/merged"):
        stage_name = "h1v2_merge_meta.json"
    else:
        stage_name = "h1_merge_meta.json"
    Path(f"/root/affine_data/{stage_name}").write_text(
        json.dumps(meta, indent=2) + "\n"
    )
    print(json.dumps(meta, indent=2), flush=True)
    if identical:
        raise SystemExit(
            "merge looks weight-identical to base (all shard head/mid/tail "
            "windows match) — refuse sim"
        )
    if first_1mib_identical and any_diff:
        print(
            "[merge] NOTE: first_1MiB of first shard matches base (expected "
            "for embed-leading shards); later windows differ — merge OK",
            flush=True,
        )
    print("[merge] DONE", flush=True)


if __name__ == "__main__":
    main()
