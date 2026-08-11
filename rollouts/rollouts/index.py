"""Parquet rollout index: one row per rollout, one file per stored chunk.

The queryable face of the trace store — serves the scheduler (kept-turn
shares), the fold (what exists per source/model), analytics (cost/yield per
endpoint), and future training queries ("all rollouts of task X across
models" via env_id + task_uid).

    <root>/index/<chunk stem>.parquet

Index files are derived data: deleting <root>/index and re-running
`python -m rollouts.index --rebuild` regenerates them from the chunks.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from rollouts.schema import (
    trace_error_type,
    trace_reward_score,
    trace_stats,
)

log = logging.getLogger("rollouts.index")

SCHEMA = pa.schema([
    ("rollout_id", pa.string()),
    ("source", pa.string()),
    ("env_id", pa.string()),
    ("task_uid", pa.string()),
    ("sid", pa.string()),
    ("repo", pa.string()),
    ("language", pa.string()),
    ("policy_id", pa.string()),
    ("model", pa.string()),        # provider-qualified label
    ("harness", pa.string()),
    ("endpoint", pa.string()),
    ("ok", pa.bool_()),
    ("error", pa.string()),
    ("stop_condition", pa.string()),
    ("resolved", pa.float64()),    # telemetry scalar; null if unscored
    ("rewards_json", pa.string()),
    ("n_nodes", pa.int32()),
    ("n_calls", pa.int32()),
    ("prompt_tokens", pa.int64()),
    ("completion_tokens", pa.int64()),
    ("cached_tokens", pa.int64()),
    ("cost_usd", pa.float64()),
    ("agent_wall_s", pa.float64()),
    ("kept_turns", pa.int32()),    # duel_turns view records that survived
    ("chunk", pa.string()),
    ("stored_at", pa.string()),
])


def _row(env: dict, chunk_key: str, kept_turns: int) -> dict:
    trace = env["trace"]
    task = env["task"]
    policy = env["policy"]
    stats = trace_stats(trace)
    score = trace_reward_score(trace)
    return {
        "rollout_id": env["rollout_id"],
        "source": env["source"],
        "env_id": env["env_id"],
        "task_uid": task["uid"],
        "sid": task.get("sid") or "",
        "repo": task.get("repo") or "",
        "language": task.get("language") or "",
        "policy_id": policy.get("id") or "",
        "model": policy.get("model") or "",
        "harness": policy.get("harness") or "",
        "endpoint": policy.get("endpoint") or "",
        "ok": bool(trace.get("ok", False)),
        "error": trace_error_type(trace),
        "stop_condition": trace.get("stop_condition"),
        "resolved": None if score is None else float(score),
        "rewards_json": json.dumps(trace.get("rewards") or {},
                                   sort_keys=True),
        "n_nodes": len(trace.get("nodes") or []),
        "n_calls": stats["n_calls"],
        "prompt_tokens": stats["prompt_tokens"],
        "completion_tokens": stats["completion_tokens"],
        "cached_tokens": stats["cached_tokens"],
        "cost_usd": stats["cost_usd"],
        "agent_wall_s": stats["agent_wall_s"],
        "kept_turns": kept_turns,
        "chunk": chunk_key,
        "stored_at": env["stored_at"],
    }


class RolloutIndex:
    def __init__(self, root: Path):
        self.dir = root / "index"

    def append(self, envelopes: list[dict], chunk_key: str,
               kept_turns: dict[str, int]) -> None:
        """One parquet file per chunk; kept_turns maps rollout_id -> count
        of duel_turns records that survived validation."""
        if not envelopes:
            return
        rows = [_row(e, chunk_key, kept_turns.get(e["rollout_id"], 0))
                for e in envelopes]
        table = pa.Table.from_pylist(rows, schema=SCHEMA)
        self.dir.mkdir(parents=True, exist_ok=True)
        stem = Path(chunk_key).name.removesuffix(".jsonl.gz")
        path = self.dir / f"{stem}.parquet"
        tmp = path.with_suffix(".tmp")
        pq.write_table(table, tmp)
        tmp.replace(path)

    def _dataset(self) -> ds.Dataset | None:
        if not self.dir.is_dir() or not any(self.dir.glob("*.parquet")):
            return None
        return ds.dataset(str(self.dir), format="parquet")

    def kept_turns_by(self, *keys: str) -> dict[tuple, int]:
        """Sum of kept turns grouped by the given columns, e.g.
        kept_turns_by("source") or kept_turns_by("source", "policy_id")."""
        data = self._dataset()
        if data is None:
            return {}
        table = data.to_table(columns=[*keys, "kept_turns"])
        out: dict[tuple, int] = {}
        cols = [table.column(k).to_pylist() for k in keys]
        kept = table.column("kept_turns").to_pylist()
        for i, n in enumerate(kept):
            group = tuple(c[i] for c in cols)
            out[group] = out.get(group, 0) + int(n or 0)
        return out

    def summary(self) -> dict:
        data = self._dataset()
        if data is None:
            return {"rollouts": 0}
        table = data.to_table(columns=["source", "kept_turns", "cost_usd",
                                       "resolved"])
        per_source: dict[str, dict] = {}
        for src, kept, cost, res in zip(
                table.column("source").to_pylist(),
                table.column("kept_turns").to_pylist(),
                table.column("cost_usd").to_pylist(),
                table.column("resolved").to_pylist()):
            row = per_source.setdefault(
                src, {"rollouts": 0, "kept_turns": 0, "cost_usd": 0.0,
                      "resolved": 0})
            row["rollouts"] += 1
            row["kept_turns"] += int(kept or 0)
            row["cost_usd"] = round(row["cost_usd"] + (cost or 0.0), 4)
            row["resolved"] += 1 if res == 1.0 else 0
        return {"rollouts": table.num_rows, "per_source": per_source}


def rebuild(root: Path) -> None:
    """Regenerate every index file from the chunk store. kept_turns is
    recomputed by re-deriving the duel_turns view (validation included)."""
    from rollouts.store import TraceStore
    from rollouts.views.duel_turns import derive_and_validate

    store = TraceStore(root)
    index = RolloutIndex(root)
    by_chunk: dict[str, list[dict]] = {}
    for chunk in store._load_manifest()["chunks"]:
        by_chunk[chunk["key"]] = list(store.iter_envelopes(chunk["key"]))
    for key, envelopes in by_chunk.items():
        kept: dict[str, int] = {}
        for env in envelopes:
            records, _ = derive_and_validate([env])
            kept[env["rollout_id"]] = len(records)
        index.append(envelopes, key, kept)
        log.info("rebuilt index for %s (%d rollouts)", key, len(envelopes))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.rebuild:
        rebuild(args.root)
    print(json.dumps(RolloutIndex(args.root).summary(), indent=1))


if __name__ == "__main__":
    main()
