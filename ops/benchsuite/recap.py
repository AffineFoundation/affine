#!/usr/bin/env python
"""Keep cells that ran at an OLD completion cap under a budget tag, so the plain
column of a card always means the current cap.

2026-09-19: the caps of MMLU-Pro / GPQA / MATH-500 / LiveCodeBench rose (8k/16k ->
32k/64k; suite.toml `cap_history`). A cell whose summary.json records an earlier
`max_tokens` is renamed <model>/<env>@<cap>k__t<T> and its summary gets
env = "<env>@<cap>k", base_env, budget.tag — the same shape harbor_cell.py uses
for swebench-verified@4h250, so publish.py / the kingboard show it as its own
column next to the new-cap cell. Idempotent: an already-tagged cell is left alone.

  recap.py --run-dir ~/benchsuite/runs/<run_id> [--teacher-from <run_id>] [--publish state|full]

--teacher-from copies the reference run's NEW-cap teacher cells (plain names) into
this run's teacher/ (light copy, like fast_pass.sh does at pass start) once they
exist, so every card compares king and teacher at the same cap.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
SUITE = tomllib.loads((HERE / "suite.toml").read_text())
CAPPED = {e["id"]: (int(e["max_tokens"]), [int(c) for c in e.get("cap_history") or []]) for e in SUITE["envs"] if e.get("cap_history")}
LIGHT_EXCLUDE = ("traces.jsonl", "traces.jsonl.gz", "logs", "eval.log", "harbor", "are")


def tag_of(cap: int) -> str:
    return f"{cap // 1024}k"


def recap_run(run_dir: Path) -> list[str]:
    changed = []
    for summ in sorted(run_dir.glob("*/*/summary.json")):
        cell = summ.parent
        try:
            s = json.loads(summ.read_text())
        except ValueError:
            continue
        env = s.get("env") or ""
        if "@" in env or env not in CAPPED:
            continue
        cur, history = CAPPED[env]
        cap = s.get("max_tokens")
        if cap is None or int(cap) == cur:
            continue
        if int(cap) not in history:
            print(f"  ? {cell.relative_to(run_dir)}: max_tokens {cap} is neither the current cap nor in cap_history; left alone")
            continue
        tag = tag_of(int(cap))
        new_env = f"{env}@{tag}"
        new_dir = cell.parent / cell.name.replace(f"{env}__", f"{new_env}__", 1)
        if new_dir.exists():
            print(f"  ! {new_dir.relative_to(run_dir)} exists; {cell.name} left alone")
            continue
        s["env"] = new_env
        s["base_env"] = env
        s["budget"] = {**(s.get("budget") or {}), "tag": tag, "max_tokens": int(cap),
                       "note": f"completion cap {cap} (the default before 2026-09-19; now {cur})"}
        cell.rename(new_dir)
        (new_dir / "summary.json").write_text(json.dumps(s, indent=1))
        changed.append(f"{cell.relative_to(run_dir)} -> {new_dir.relative_to(run_dir)}")
    return changed


def copy_teacher(run_dir: Path, teacher_from: Path) -> list[str]:
    copied = []
    for env, (cur, _) in CAPPED.items():
        for src in sorted((teacher_from / "teacher").glob(f"{env}__t*")):
            if not (src / "summary.json").exists():
                continue
            s = json.loads((src / "summary.json").read_text())
            if int(s.get("max_tokens") or 0) != cur:
                continue
            dst = run_dir / "teacher" / src.name
            if dst.exists():
                continue
            shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*LIGHT_EXCLUDE))
            copied.append(src.name)
    return copied


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True, nargs="+")
    ap.add_argument("--teacher-from", default="", help="reference run id / path whose new-cap teacher cells are copied in")
    ap.add_argument("--publish", choices=["", "state", "full"], default="", help="state: publish.py --only-state; full: upload too")
    ap.add_argument("--python", default=sys.executable)
    a = ap.parse_args()
    rc = 0
    for rd in a.run_dir:
        run_dir = Path(rd).expanduser()
        if not run_dir.is_dir():
            print(f"{run_dir}: not a directory"); rc = 1; continue
        changed = recap_run(run_dir)
        copied = []
        if a.teacher_from and (run_dir / "teacher").is_dir():
            tf = Path(a.teacher_from).expanduser()
            if not tf.is_dir():
                tf = run_dir.parent / a.teacher_from
            if tf.resolve() != run_dir.resolve():
                copied = copy_teacher(run_dir, tf)
        print(f"{run_dir.name}: {len(changed)} cell(s) re-tagged, {len(copied)} teacher cell(s) copied in")
        for c in changed:
            print("  " + c)
        if a.publish and (changed or copied):
            cmd = [a.python, str(HERE / "publish.py"), "--run-dir", str(run_dir)]
            if a.publish == "state":
                cmd.append("--only-state")
            r = subprocess.run(cmd)
            rc = rc or r.returncode
    return rc


if __name__ == "__main__":
    sys.exit(main())
