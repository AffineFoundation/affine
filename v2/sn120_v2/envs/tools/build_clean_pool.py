"""Build math pool v2.1 from the pinned third-party audit lists.

Run from any directory with the grader extras installed:

    python -m sn120_v2.envs.tools.build_clean_pool

The old pool is read only. The two output files are deterministic. The
KEY_FORMAT case is the one computed part: it grades the proposed clean key
against the unchanged task and its original answer key with the live grader.
No task is re-keyed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from ...canonical import canonical_json, file_sha256
from ..math_grader import check_runtime, grade_answer
from ..math_pool import POOL_V2_0_FILE, POOL_V2_0_SHA256, POOL_V2_1_FILE, POOL_V2_1_SHA256, load_pool

SOURCE = "catoneone/sn120-k1-fixes@427295a"
AUDIT_DIR = Path(__file__).resolve().parents[1] / "audit"
AUDIT_SHA256 = {
    "pool_audit_v2new_lists.json": "3faec92a01cb656b0f87474f0a845ee607d44dd0857a88f1ca6210db47ef61c7",
    "pool_audit_lists.json": "3b0bd39c8a75c2f09df2e631599ec442fba5ec15fef2cbcec9b4bcd82c6bbba4",
}
DROPPED_FILE = POOL_V2_1_FILE.with_suffix(".dropped.json")
RULE = (
    "D starts empty. For every item in v2new.remove and 1108.remove, add its id to D. For every item in "
    "v2new.fix_key and 1108.fix_key: if verdict is BAD_KEY, add its id to D; if verdict is KEY_FORMAT, grade "
    "that item's own new_key against the pool answer with grade_answer(gold, problem, "
    "'<think>x</think> \\\\boxed{' + new_key + '}', False) and add its id to D if code is not P. "
    "An id is dropped if ANY of its items says drop. kept_key_format is the ids of KEY_FORMAT items whose grade "
    "was P, minus D. There is no precedence between audits. No task is re-keyed."
)


def _read_audit(name: str, sha256: str) -> dict[str, Any]:
    path = AUDIT_DIR / name
    found = file_sha256(path)
    if found != sha256:
        raise ValueError(f"{path} has sha256 {found}, expected {sha256}")
    data = json.loads(path.read_bytes())
    if set(data) < {"remove", "fix_key"}:
        raise ValueError(f"{path} is not an audit list")
    return data


def _order(task_id: str) -> tuple[str, int]:
    prefix, number = task_id.split(":")
    return prefix, int(number)


def build() -> tuple[bytes, bytes]:
    """Return the exact bytes of the new pool and its public dropped list."""
    check_runtime()
    audits = {name: _read_audit(name, sha256) for name, sha256 in AUDIT_SHA256.items()}
    old = load_pool(POOL_V2_0_FILE, POOL_V2_0_SHA256)
    verdicts: dict[str, dict[str, str]] = {}
    for name, prefix in (("pool_audit_v2new_lists.json", "v2new"), ("pool_audit_lists.json", "1108")):
        for row in [*audits[name]["remove"], *audits[name]["fix_key"]]:
            task = old.task(row["id"])
            if row["answer_key"] != task.answer:
                raise ValueError(f"{row['id']}: audit answer_key differs from the old pool")
            verdicts.setdefault(task.task_id, {})[prefix] = row["verdict"]

    dropped: set[str] = set()
    passed_key_format: set[str] = set()
    for audit in audits.values():
        dropped.update(row["id"] for row in audit["remove"])
        for row in audit["fix_key"]:
            task = old.task(row["id"])
            if row["verdict"] == "BAD_KEY":
                dropped.add(row["id"])
            elif row["verdict"] == "KEY_FORMAT":
                grade = grade_answer(
                    gold=task.answer,
                    problem=task.problem,
                    output="<think>x</think> \\boxed{" + row["new_key"] + "}",
                    capped=False,
                )
                if grade.code == "P":
                    passed_key_format.add(row["id"])
                else:
                    dropped.add(row["id"])
    kept_key_format = passed_key_format - dropped

    raw_lines = POOL_V2_0_FILE.read_bytes().splitlines(keepends=True)
    if len(raw_lines) != len(old) or any(not line.endswith(b"\n") for line in raw_lines):
        raise ValueError("the old pool must have one newline-terminated task per line")
    lines = []
    for line in raw_lines:
        task_id = json.loads(line)["id"]
        if task_id not in dropped:
            lines.append(line)
    new_pool = b"".join(lines)
    if len(dropped) != 1454 or len(kept_key_format) != 156 or len(lines) != 20120:
        raise ValueError(f"unexpected counts: dropped={len(dropped)}, kept_key_format={len(kept_key_format)}, kept={len(lines)}")
    if old.task_ids[:4] != tuple(f"math:{i}" for i in range(4)) or set(old.task_ids[:4]) & dropped:
        raise ValueError("the first four pool tasks must remain for the admission probe")
    if hashlib.sha256(new_pool).hexdigest() != POOL_V2_1_SHA256:
        raise ValueError("the new pool does not have its expected sha256")

    rows = [{"id": task_id, "verdicts": [f"{prefix}:{verdict}" for prefix, verdict in verdicts[task_id].items()]} for task_id in dropped]
    rows.sort(key=lambda row: _order(row["id"]))
    document = {
        "from_sha256": POOL_V2_0_SHA256,
        "to_sha256": POOL_V2_1_SHA256,
        "source": SOURCE,
        "rule": RULE,
        "dropped": rows,
        "kept_key_format": sorted(kept_key_format, key=_order),
    }
    return new_pool, canonical_json(document) + b"\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args(argv)
    pool, dropped = build()
    for path, content in ((POOL_V2_1_FILE, pool), (DROPPED_FILE, dropped)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    print(json.dumps({"pool": str(POOL_V2_1_FILE), "sha256": file_sha256(POOL_V2_1_FILE), "dropped": str(DROPPED_FILE), "tasks": len(pool.splitlines())}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
