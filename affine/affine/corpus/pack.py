"""Pack per-turn JSONL into trajectory chunks + a Parquet turn index."""

from __future__ import annotations

import gzip
import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from .materialize import materialize_turn, stratum_key

log = logging.getLogger("affine.corpus.pack")

DEFAULT_CHUNK_TRAJS = 1000
ROUNDTRIP_SAMPLE = 64

INDEX_COLUMNS = [
    "turn_id", "traj_id", "turn_idx", "stratum", "phase", "source",
    "language", "chunk_key", "traj_line", "msg_pos", "n_prefix_chars",
]


@dataclass
class PackResult:
    chunk_paths: list[Path]          # local .jsonl (uncompressed) paths
    chunk_meta: list[dict]           # {key, sha256, n_trajectories, format, active}
    index_path: Path                 # local .parquet
    index_sha256: str
    n_turns: int
    n_trajectories: int
    compat_path: Path | None = None  # optional flat turn JSONL for dual-publish


def _load_turns(paths: list[Path]) -> list[dict]:
    turns: list[dict] = []
    for path in paths:
        opener = gzip.open if str(path).endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                turns.append(json.loads(line))
    return turns


def rebuild_trajectory(turns: list[dict]) -> dict:
    """Rebuild one traj record from ordered per-turn records of the same traj_id."""
    turns = sorted(turns, key=lambda t: int(t["turn_idx"]))
    tid = turns[0]["traj_id"]
    messages: list[dict] = []
    turn_metas: list[dict] = []
    for t in turns:
        prefix = t["prefix"]
        ref = t.get("reference_turn")
        if not isinstance(ref, str) or not ref:
            raise ValueError(f"{tid}:{t['turn_idx']} missing reference_turn")
        if not messages:
            messages = [{"role": m["role"], "content": m["content"]} for m in prefix]
        else:
            if len(prefix) < len(messages):
                raise ValueError(
                    f"{tid}:{t['turn_idx']} prefix shorter than rebuilt messages")
            for i, m in enumerate(messages):
                if (prefix[i].get("role") != m["role"]
                        or prefix[i].get("content") != m["content"]):
                    raise ValueError(
                        f"{tid}:{t['turn_idx']} prefix diverges at message {i}")
            messages.extend(
                {"role": m["role"], "content": m["content"]}
                for m in prefix[len(messages):])
        msg_pos = len(messages)
        messages.append({"role": "assistant", "content": ref})
        turn_metas.append({
            "turn_idx": int(t["turn_idx"]),
            "msg_pos": msg_pos,
            "phase": t.get("phase", ""),
            "suffix_len": int(t.get("suffix_len", len(ref))),
            "n_prefix_chars": int(t.get(
                "n_prefix_chars", sum(len(m["content"]) for m in prefix))),
        })
    head = turns[0]
    return {
        "traj_id": tid,
        "messages": messages,
        "turns": turn_metas,
        "instance_id": head.get("instance_id", ""),
        "repo": head.get("repo", ""),
        "model": head.get("model", ""),
        "action_kind": head.get("action_kind", "bash"),
        "generated_at": head.get("generated_at", ""),
        "source": head.get("source", "swe"),
        "language": head.get("language", "python"),
    }


def _check_roundtrip(trajs: list[dict], originals: dict[str, dict],
                     n_sample: int = ROUNDTRIP_SAMPLE) -> None:
    """Fail if materialize(prefix/ref) differs from the source turn records."""
    checked = 0
    for traj in trajs:
        for meta in traj["turns"]:
            tid = f"{traj['traj_id']}:{meta['turn_idx']}"
            orig = originals.get(tid)
            if orig is None:
                continue
            got = materialize_turn(traj, meta)
            if got["prefix"] != orig["prefix"]:
                raise ValueError(f"roundtrip prefix mismatch for {tid}")
            if got["reference_turn"] != orig["reference_turn"]:
                raise ValueError(f"roundtrip reference_turn mismatch for {tid}")
            checked += 1
            if checked >= n_sample:
                log.info("roundtrip ok: checked %d turns", checked)
                return
    if checked == 0:
        raise ValueError("roundtrip check found no overlapping turns")
    log.info("roundtrip ok: checked %d turns", checked)


def pack_turns_to_chunks(
    turn_paths: list[Path],
    out_dir: Path,
    *,
    epoch: int,
    chunk_trajs: int = DEFAULT_CHUNK_TRAJS,
    write_compat: bool = False,
    chunks_prefix: str = "turns/chunks",
) -> PackResult:
    """Convert turn-flat JSONL into traj chunks + Parquet index.

    Chunk keys are ``{chunks_prefix}/chunk_{epoch:04d}_{i:04d}.jsonl.gz``
    (gzip written at upload time; local files are uncompressed jsonl so
    sha256 matches the corpus_push uncompressed-sha convention).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    turns = _load_turns(turn_paths)
    if not turns:
        raise ValueError("no turns to pack")

    by_traj: dict[str, list[dict]] = defaultdict(list)
    originals: dict[str, dict] = {}
    traj_order: list[str] = []
    for t in turns:
        tid = t["traj_id"]
        if tid not in by_traj:
            traj_order.append(tid)
        by_traj[tid].append(t)
        originals[f"{tid}:{t['turn_idx']}"] = t

    # Preserve first-seen traj order so index row order matches the source
    # JSONL within each stratum — required for sample_slice seed parity.
    trajs = [rebuild_trajectory(by_traj[tid]) for tid in traj_order]
    traj_by_id = {tr["traj_id"]: tr for tr in trajs}
    _check_roundtrip(trajs, originals)

    chunk_paths: list[Path] = []
    chunk_meta: list[dict] = []
    # traj_id → (chunk_key, line_in_chunk)
    traj_locus: dict[str, tuple[str, int]] = {}

    for ci in range(0, len(trajs), chunk_trajs):
        batch = trajs[ci: ci + chunk_trajs]
        local = out_dir / f"chunk_{epoch:04d}_{ci // chunk_trajs:04d}.jsonl"
        lines = [json.dumps(tr, separators=(",", ":"), ensure_ascii=False)
                 for tr in batch]
        raw = ("\n".join(lines) + "\n").encode("utf-8")
        local.write_bytes(raw)
        sha = hashlib.sha256(raw).hexdigest()
        key = (f"{chunks_prefix}/chunk_{epoch:04d}_"
               f"{ci // chunk_trajs:04d}.jsonl.gz")
        chunk_paths.append(local)
        chunk_meta.append({
            "key": key,
            "sha256": sha,
            "n_trajectories": len(batch),
            "n_turns": sum(len(tr["turns"]) for tr in batch),
            "format": "traj_v1",
            "active": True,
        })
        for line_i, tr in enumerate(batch):
            traj_locus[tr["traj_id"]] = (key, line_i)

    # Index rows follow the original turn JSONL order (not traj-grouped).
    index_rows: dict[str, list] = {c: [] for c in INDEX_COLUMNS}
    for t in turns:
        tid = t["traj_id"]
        tr = traj_by_id[tid]
        meta = next(m for m in tr["turns"]
                    if int(m["turn_idx"]) == int(t["turn_idx"]))
        chunk_key, line_i = traj_locus[tid]
        index_rows["turn_id"].append(f"{tid}:{t['turn_idx']}")
        index_rows["traj_id"].append(tid)
        index_rows["turn_idx"].append(int(t["turn_idx"]))
        index_rows["stratum"].append(stratum_key({"traj_id": tid}))
        index_rows["phase"].append(str(meta.get("phase") or t.get("phase")
                                       or ""))
        index_rows["source"].append(str(tr.get("source") or ""))
        index_rows["language"].append(str(tr.get("language") or ""))
        index_rows["chunk_key"].append(chunk_key)
        index_rows["traj_line"].append(line_i)
        index_rows["msg_pos"].append(int(meta["msg_pos"]))
        index_rows["n_prefix_chars"].append(
            int(meta.get("n_prefix_chars") or 0))

    index_path = out_dir / f"turns_{epoch:04d}.parquet"
    table = pa.table({
        "turn_id": pa.array(index_rows["turn_id"], type=pa.string()),
        "traj_id": pa.array(index_rows["traj_id"], type=pa.string()),
        "turn_idx": pa.array(index_rows["turn_idx"], type=pa.int32()),
        "stratum": pa.array(index_rows["stratum"], type=pa.string()),
        "phase": pa.array(index_rows["phase"], type=pa.string()),
        "source": pa.array(index_rows["source"], type=pa.string()),
        "language": pa.array(index_rows["language"], type=pa.string()),
        "chunk_key": pa.array(index_rows["chunk_key"], type=pa.string()),
        "traj_line": pa.array(index_rows["traj_line"], type=pa.int32()),
        "msg_pos": pa.array(index_rows["msg_pos"], type=pa.int32()),
        "n_prefix_chars": pa.array(index_rows["n_prefix_chars"],
                                   type=pa.int32()),
    })
    pq.write_table(table, index_path, compression="zstd")
    index_sha = hashlib.sha256(index_path.read_bytes()).hexdigest()

    compat_path = None
    if write_compat:
        compat_path = out_dir / f"turns_epoch_{epoch:04d}_compat.jsonl"
        with open(compat_path, "w", encoding="utf-8") as f:
            for t in turns:
                f.write(json.dumps(t, separators=(",", ":"),
                                   ensure_ascii=False))
                f.write("\n")

    n_turns = len(index_rows["turn_id"])
    log.info("packed %d turns / %d trajs → %d chunks + index (%s)",
             n_turns, len(trajs), len(chunk_paths), index_sha[:12])
    return PackResult(
        chunk_paths=chunk_paths,
        chunk_meta=chunk_meta,
        index_path=index_path,
        index_sha256=index_sha,
        n_turns=n_turns,
        n_trajectories=len(trajs),
        compat_path=compat_path,
    )


def expand_all_turns(chunk_paths: list[Path]) -> list[dict]:
    """Materialize every turn from local uncompressed chunk jsonl files."""
    out: list[dict] = []
    for path in chunk_paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                traj = json.loads(line)
                for meta in traj["turns"]:
                    out.append(materialize_turn(traj, meta))
    return out
