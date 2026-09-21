# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///
"""Rebuild the container state of a verifiers `bash` harness rollout.

Replays the prefix's `bash` / `edit` tool calls with the same semantics as
harnesses/bash/program.py (bash -c in the cwd; edit = single-occurrence
string replacement), in order, and compares each result with the recorded
tool message. Unknown or malformed calls (the original loop answered them
with an error string) are skipped. Writes a JSON report.

argv: --state-file --report --timeout
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path


def run_bash(command: str, timeout: int) -> str:
    try:
        result = subprocess.run(["bash", "-c", command], capture_output=True,
                                text=True, timeout=timeout, check=False)
        return result.stdout + result.stderr
    except Exception as e:  # noqa: BLE001 - mirrors the harness program
        return f"error: {e}"


def run_edit(path, old_str, new_str) -> str:
    if not isinstance(path, str) or not path:
        return "error: 'path' is required"
    if not isinstance(old_str, str) or not isinstance(new_str, str):
        return "error: 'old_str' and 'new_str' must be strings"
    if not old_str:
        return "error: 'old_str' must be a non-empty string"
    filepath = Path(path)
    if not filepath.is_absolute():
        filepath = Path.cwd() / filepath
    if not filepath.exists():
        return f"error: {path} not found"
    try:
        content = filepath.read_text()
    except Exception as e:  # noqa: BLE001
        return f"error: could not read {path}: {e}"
    count = content.count(old_str)
    if count != 1:
        return f"error: old_str must appear exactly once in {path} (found {count})"
    try:
        filepath.write_text(content.replace(old_str, new_str, 1))
    except Exception as e:  # noqa: BLE001
        return f"error: could not write {path}: {e}"
    return f"Edited {path}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--state-file", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()
    state = json.loads(Path(args.state_file).read_text(encoding="utf-8"))
    messages = state["messages"]
    recorded = {m.get("tool_call_id"): m.get("content") or ""
                for m in messages if m.get("role") == "tool"}
    report = {"replay": [], "replay_n": 0, "replay_match": 0, "replay_seconds": 0.0}
    t0 = time.time()
    for m in messages:
        if m.get("role") != "assistant" or not m.get("tool_calls"):
            continue
        for call in m["tool_calls"]:
            fn = call.get("function") or {}
            name = fn.get("name")
            try:
                targs = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                report["replay"].append({"id": call.get("id"), "skipped": "bad_json"})
                continue
            if not isinstance(targs, dict):
                report["replay"].append({"id": call.get("id"), "skipped": "non_object"})
                continue
            if name == "bash":
                got = run_bash(targs.get("command", ""), args.timeout)
            elif name == "edit":
                got = run_edit(targs.get("path"), targs.get("old_str"), targs.get("new_str"))
            else:
                report["replay"].append({"id": call.get("id"), "skipped": f"tool:{name}"})
                continue
            want = recorded.get(call.get("id"))
            match = None if want is None else (got.strip() == want.strip())
            report["replay"].append({"id": call.get("id"), "tool": name, "output_match": match})
            report["replay_n"] += 1
            report["replay_match"] += 1 if match else 0
    report["replay_seconds"] = round(time.time() - t0, 1)
    Path(args.report).write_text(json.dumps(report))


if __name__ == "__main__":
    main()
