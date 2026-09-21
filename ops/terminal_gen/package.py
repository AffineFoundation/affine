"""Stage 6: package the clean tasks for the `affine_terminal_gen_v1` taskset.

Takes the tasks that passed validate.py AND decontam.py and writes

    rollouts/envs/affine_terminal_gen_v1/affine_terminal_gen_v1/data/e<epoch>/
        tasks.tar.gz          the Harbor task dirs (solution/ included: the pod
                              never mounts it; Harbor's parser ignores it)
        manifest.json         counts per stage with drop reasons, model, prompt
                              version, seed, licence, sha256 of the tarball
        decontam_report.md    the report from decontam.py (ships with the set)

The taskset extracts the tarball on first load (`~/.cache/affine/terminal_gen/e<epoch>`)
and the catalog enumerates the same dirs. Set `--env.taskset.data-epoch <epoch>`
in the `[source.affine_terminal_gen]` stanza and redeploy the pods to serve it.

    python package.py --epoch 63 --out ~/terminal_gen/out
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import tarfile
import time
from pathlib import Path

from common import GENERATOR_VERSION, LICENSE, LICENSE_URL, out_dir, read_jsonl

log = logging.getLogger("terminal_gen.package")

REPO = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO / "rollouts" / "envs" / "affine_terminal_gen_v1" / "affine_terminal_gen_v1" / "data"


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--out", default="~/terminal_gen/out")
    ap.add_argument("--dest", default=str(DATA_ROOT), help="env package data dir")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = out_dir(args.out, args.epoch)

    validated = {r["task_id"] for r in read_jsonl(out / "validation.jsonl") if r.get("ok")}
    clean = {r["task_id"] for r in read_jsonl(out / "decontam.jsonl") if r.get("ok")}
    if not (out / "decontam.json").exists():
        raise SystemExit("decontam.json missing: run decontam.py before packaging")
    final = sorted(validated & clean)
    if not final:
        raise SystemExit("no task survived validation + decontamination")

    dest = Path(args.dest).expanduser() / f"e{args.epoch}"
    dest.mkdir(parents=True, exist_ok=True)
    tar_path = dest / "tasks.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        for tid in final:
            tar.add(out / "tasks" / tid, arcname=tid, recursive=True)
    sha = hashlib.sha256(tar_path.read_bytes()).hexdigest()

    rendered = {r["task_id"]: r for r in read_jsonl(out / "rendered.jsonl")}
    by_site: dict[str, int] = {}
    by_domain: dict[str, int] = {}
    by_difficulty: dict[str, int] = {}
    for tid in final:
        r = rendered.get(tid, {})
        by_site[r.get("site", "?")] = by_site.get(r.get("site", "?"), 0) + 1
        by_domain[r.get("domain", "?")] = by_domain.get(r.get("domain", "?"), 0) + 1
        by_difficulty[r.get("difficulty", "?")] = by_difficulty.get(r.get("difficulty", "?"), 0) + 1

    synth = _load_json(out / "synth_summary.json")
    validation = _load_json(out / "validation_summary.json")
    decontam = _load_json(out / "decontam.json")
    manifest = {
        "generator": GENERATOR_VERSION,
        "data_epoch": args.epoch,
        "seed": args.epoch,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "tasks": len(final),
        "task_ids": final,
        "tarball_sha256": sha,
        "stages": {
            "posts": synth.get("posts"),
            "specs": synth.get("specs"),
            "synth_rejects": synth.get("rejects"),
            "rendered": len(rendered),
            "built": validation.get("built"),
            "build_success_rate": validation.get("build_success_rate"),
            "validated": validation.get("ok"),
            "validated_yield": validation.get("validated_yield"),
            "validation_drops": validation.get("drops"),
            "decontam_checked": decontam.get("checked"),
            "decontam_dropped": decontam.get("dropped"),
            "decontam_benchmarks": decontam.get("benchmarks"),
        },
        "synth": {"model": synth.get("model"), "prompt_version": synth.get("prompt_version"),
                  "usd": round(synth.get("usd", 0.0), 2)},
        "by_site": by_site, "by_domain": by_domain, "by_difficulty": by_difficulty,
        "license": LICENSE, "license_url": LICENSE_URL,
        "license_note": "Tasks are derived from Stack Exchange posts (CC BY-SA 4.0); each task.toml "
                        "[metadata] carries se_site / se_qid / se_url attribution.",
    }
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=1))
    report = out / "decontam_report.md"
    if report.exists():
        (dest / "decontam_report.md").write_text(report.read_text())
    log.info("packaged %d tasks -> %s (%.1f MB, sha256 %s)", len(final), tar_path,
             tar_path.stat().st_size / 1e6, sha[:12])
    log.info("next: commit %s, set extra_flags = [\"--env.taskset.data-epoch\", \"%d\"] and share = 1.0 on "
             "[source.affine_terminal_gen], then the datagen worker deploys (ops/king-datagen/deploy_pods.sh "
             "--restart --all) after the handshake in internal/benchsuite/requests.md", dest, args.epoch)


if __name__ == "__main__":
    main()
