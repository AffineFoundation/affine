#!/usr/bin/env python
"""bench_fail: the benchsuite's FAILED king trials as trace envelopes for D.

Jacob 2026-09-24 17:53/17:56 UTC: "add the failure runs from the benchmarks
into the dataset ... upsampling the runs where we are doing badly"; mode (a)
DIRECT: king-failed / teacher-passed benchmark trials enter D as themselves
("a fair backward pass at this point in the training"). Data event, no wvk.

Input: the benchsuite's stored cells under ~/benchsuite/runs/<run>/king/<cell>/
  * Harbor cells (`<cell>/harbor/<trial>/`): SWE-bench Verified (mini-swe-agent
    trajectory `agent/mini-swe-agent.trajectory.json`, our mini_swe adapter)
    and Terminal-Bench 2 (Terminus-2 ATIF `agent/trajectory.json`, rebuilt into
    the terminus_json dialect: assistant = {"analysis","plan","commands"},
    observation = user turn). Verdict = `result.json` verifier reward.
  * verifiers cells (`<cell>/traces.jsonl[.gz]`): chat / tau2 / gaia2 suites,
    verifiers trace v1 episodes (our verifiers adapter path). Verdict = the
    env's primary reward (affine.corpus.view.PRIMARY_REWARD_KEYS).
Selection: king reward 0 on a live trial (infra / cancelled trials skipped);
where any teacher cell of the same suite has the task, the teacher must have
passed it (the king_fail band); trials whose transcript already holds a
successful upstream fetch (affine.corpus.upstream) are dropped as leaked.

Output envelopes (rollouts.schema.make_envelope): source `bench_<suite>`
(swebench_verified, terminal_bench_2, aime25, ...), policy `bench_<harness>`
with model `king/king-<digest12>`, task {uid: "<suite>/<task>", sid:
"bench_<suite>__<task>", repo: "bench/<suite>", language, bench: <suite>,
king_digest, reign, cell, trial, temp, teacher_passed}, trace.rewards.solved
= 0 (outcome stamped), trace.info.upstream_fetch. Published through the normal
TraceStore + R2TraceMirror into the SEPARATE prefix `traces-bench/` (own
manifest at data.affine.io/traces-bench/manifest.json; chunk keys start with
`bench_<suite>`), never into `traces/`: the fold's derive path has no
unknown-source gate (an unlisted source lands in the default coding group as
a teacher-side row), so the fold reads this prefix only when
`[bench_fail].enabled` / `[decontamination].allow_bench_groups` name it. One
prefix = auditable and removable in one fold.

    .venv/bin/python ops/bench_fail/ingest.py --dry-run          # counts only
    .venv/bin/python ops/bench_fail/ingest.py --publish          # store + R2
Options: --runs DIR, --min-reign N, --suites a,b, --out DIR (store dir).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / "rollouts"), str(REPO / "affine")]
from affine.corpus.upstream import rollout_has_upstream_fetch  # noqa: E402
from affine.corpus.view import PRIMARY_REWARD_KEYS  # noqa: E402
from rollouts.adapters.mini_swe import traj_to_trace  # noqa: E402
from rollouts.schema import PolicyStamp, make_envelope  # noqa: E402

RUNS = Path(os.environ.get("BENCHSUITE_RUNS", str(Path.home() / "benchsuite" / "runs")))
MATRIX_URL = "https://kings.affine.io/api/matrix.json"
RUN_RE = re.compile(r"^\d{8}T\d{4,6}Z-([0-9a-f]{12})(?:-([a-z0-9]+))?$")
CELL_RE = re.compile(r"^(?P<suite>[^_]+(?:@[^_]+)?)__t(?P<temp>[0-9.]+)(?:\.(?P<tag>.+))?$")
SUITE_LANGUAGE = {"swebench-verified": "python", "swebench-pro": "python", "terminal-bench-2": "shell"}
# Harbor's ATIF stores the Terminus JSON reply parsed; rebuild the dialect.
TERMINUS_KEYS = ("analysis", "plan", "commands")


def suite_slug(suite: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", suite.split("@")[0].lower()).strip("_")


def reigns_by_digest() -> dict[str, dict]:
    try:
        m = httpx.get(MATRIX_URL, headers={"User-Agent": "affine-bench-fail/0.1"}, timeout=30).json()
    except Exception as e:  # noqa: BLE001
        print(f"kingboard matrix unreachable ({e!r}); reigns unknown", file=sys.stderr)
        return {}
    out = {}
    for r in m.get("rows", []):
        if r.get("digest12"):
            out[r["digest12"]] = {"reign": r.get("reign"), "kind": r.get("kind"), "label": r.get("label"), "current": r.get("current")}
    return out


def primary_score(rewards: dict) -> float | None:
    """The env's primary grade: the fold's keys first, else the cell's single
    grade (benchmark envs name it tau2_reward / passed / followed_instructions /
    proved / ...; every benchsuite cell carries exactly one)."""
    for k in PRIMARY_REWARD_KEYS:
        v = rewards.get(k)
        if isinstance(v, dict) and isinstance(v.get("score"), (int, float)) and not isinstance(v.get("score"), bool):
            return float(v["score"])
    nums = [float(v["score"]) for v in rewards.values()
            if isinstance(v, dict) and isinstance(v.get("score"), (int, float)) and not isinstance(v.get("score"), bool)]
    return nums[0] if len(nums) == 1 else (None if not nums else max(nums))


# -- Harbor -------------------------------------------------------------------

def harbor_trials(cell: Path):
    for rp in sorted((cell / "harbor").glob("*/result.json")):
        try:
            r = json.loads(rp.read_text())
        except ValueError:
            continue
        if "trial_name" not in r:
            continue
        exc = r.get("exception_info") or {}
        etype = exc.get("exception_type") or ""
        reward = ((r.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        agent_started = bool((r.get("agent_execution") or {}).get("started_at"))
        # infra: an exception before the agent ran, or no verdict at all
        infra = bool(etype) and not ("Timeout" in etype and "Agent" in etype) and reward is None
        yield rp.parent, r, (None if reward is None else float(reward)), infra or not agent_started


def terminus_trace(traj: dict, task_name: str, model_label: str) -> dict | None:
    steps = traj.get("steps") or []
    nodes = []
    for s in steps:
        src = s.get("source")
        if src == "system":
            nodes.append({"message": {"role": "system", "content": s.get("message") or ""}, "sampled": False, "timestamp": s.get("timestamp")})
        elif src == "user":
            nodes.append({"message": {"role": "user", "content": s.get("message") or ""}, "sampled": False, "timestamp": s.get("timestamp")})
        elif src == "agent":
            msg = s.get("message") or ""
            analysis, plan = msg, ""
            m = re.match(r"\s*Analysis:\s*(.*?)\s*Plan:\s*(.*)\Z", msg, re.S)
            if m:
                analysis, plan = m.group(1).strip(), m.group(2).strip()
            cmds = []
            for tc in s.get("tool_calls") or []:
                a = tc.get("arguments") or {}
                cmds.append({"keystrokes": a.get("keystrokes", ""), "duration": a.get("duration", 1.0)})
            body = {"analysis": analysis, "plan": plan, "commands": cmds}
            if not cmds:
                body["task_complete"] = True
            content = json.dumps(body, ensure_ascii=False, indent=2)
            node = {"message": {"role": "assistant", "content": content}, "sampled": True, "timestamp": s.get("timestamp")}
            if s.get("reasoning_content"):
                node["message"]["reasoning_content"] = s["reasoning_content"]
            nodes.append(node)
            obs = ((s.get("observation") or {}).get("results") or [])
            text = "\n".join(str(o.get("content") or "") for o in obs if isinstance(o, dict))
            if text:
                nodes.append({"message": {"role": "user", "content": text}, "sampled": False, "timestamp": s.get("timestamp")})
    if not any(n["sampled"] for n in nodes):
        return None
    raw = json.dumps(traj, sort_keys=True).encode()
    return {
        "version": 1, "id": hashlib.sha256(raw).hexdigest()[:32],
        "run": {"type": "harbor", "id": traj.get("session_id") or ""},
        "task": {"type": "terminal_bench_2", "data": {"name": task_name}},
        "agent": {"config": {"model": model_label, "harness": {"id": "terminus"}, "sampling": {"temperature": ((traj.get("agent") or {}).get("extra") or {}).get("temperature")}}, "name": "terminus-2"},
        "nodes": nodes, "calls": [], "rewards": {}, "metrics": {},
        "info": {"bench_agent": traj.get("agent"), "final_metrics": traj.get("final_metrics")},
        "is_completed": True, "ok": True, "stop_condition": "agent_completed", "errors": [], "timing": {},
    }


# -- main ---------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", default=str(RUNS))
    ap.add_argument("--min-reign", type=int, default=12, help="kings on the board with reign >= this (default 12)")
    ap.add_argument("--suites", default="", help="comma list of suite names (cell prefix before __t); default all")
    ap.add_argument("--out", default="/tmp/bench_fail_store")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--publish", action="store_true", help="write the TraceStore and mirror to R2 (DATA_R2_* creds)")
    ap.add_argument("--prefix", default="traces-bench/", help="R2 prefix (own manifest); never traces/ -- see the docstring")
    ap.add_argument("--stats", default=str(REPO / "affine" / "state" / "bench_fail" / "ingest_stats.json"))
    a = ap.parse_args()
    runs = Path(a.runs)
    want_suites = {s.strip() for s in a.suites.split(",") if s.strip()}
    reigns = reigns_by_digest()

    # teacher pass sets per suite: task -> True if any teacher trial passed
    teacher_pass: dict[str, dict[str, bool]] = defaultdict(dict)
    for cell in sorted(runs.glob("*/teacher/*")):
        m = CELL_RE.match(cell.name)
        if not m:
            continue
        suite = m.group("suite").split("@")[0]
        if (cell / "harbor").is_dir():
            for _, r, reward, infra in harbor_trials(cell):
                if infra or reward is None:
                    continue
                t = r.get("task_name")
                teacher_pass[suite][t] = teacher_pass[suite].get(t, False) or reward >= 1.0
        else:
            for tr, name, score in verifiers_episodes(cell):
                if score is None:
                    continue
                teacher_pass[suite][name] = teacher_pass[suite].get(name, False) or score >= 1.0

    stats: dict[str, Counter] = defaultdict(Counter)
    envelopes: list[dict] = []
    seen_trials: set[str] = set()
    for run in sorted(runs.iterdir()):
        m = RUN_RE.match(run.name)
        if not m:
            continue
        digest = m.group(1)
        info = reigns.get(digest) or {}
        reign = info.get("reign")
        if info.get("kind") != "king" or reign is None or int(reign) < a.min_reign:
            continue                      # kings on the board only (challenger / reference / genesis runs are not the seat)
        for cell in sorted((run / "king").glob("*")) if (run / "king").is_dir() else []:
            cm = CELL_RE.match(cell.name)
            if not cm:
                continue
            suite_full, temp = cm.group("suite"), float(cm.group("temp"))
            suite = suite_full.split("@")[0]
            if want_suites and suite not in want_suites:
                continue
            key = f"{suite}|reign {reign if reign is not None else digest}"
            slug = suite_slug(suite)
            source = f"bench_{slug}"
            if (cell / "harbor").is_dir():
                for tdir, r, reward, infra in harbor_trials(cell):
                    stats[key]["trials"] += 1
                    if infra:
                        stats[key]["infra_skipped"] += 1; continue
                    if reward is None or reward >= 1.0:
                        stats[key]["king_passed" if reward else "unscored"] += 1; continue
                    stats[key]["king_failed"] += 1
                    task = r.get("task_name") or tdir.name.rsplit("__", 1)[0]
                    tp = teacher_pass.get(suite, {}).get(task)
                    if tp is False:
                        stats[key]["teacher_failed_too"] += 1; continue
                    trial_key = f"{digest}|{cell.name}|{tdir.name}"
                    if trial_key in seen_trials:
                        continue
                    seen_trials.add(trial_key)
                    model_label = f"king/king-{digest}"
                    if (tdir / "agent" / "mini-swe-agent.trajectory.json").exists():
                        harness, action_kind = "mini_swe_textbased", "bash"
                        try:
                            trace = traj_to_trace((tdir / "agent" / "mini-swe-agent.trajectory.json").read_bytes(),
                                                  instance_id=task, model_label=model_label, endpoint="bench", resolved=0.0)
                        except (ValueError, OSError):
                            stats[key]["no_trajectory"] += 1; continue   # empty / truncated file (trial cut mid-write)
                        # mini-swe's exit_status ("Submitted", ...) is not one of the fold's clean stop
                        # conditions; the trial ran to its own end, so stamp agent_completed and keep the
                        # exit status in info (rollout_outcome would otherwise call it errored).
                        if trace.get("is_completed"):
                            trace["stop_condition"] = "agent_completed"; trace["errors"] = []
                    elif (tdir / "agent" / "trajectory.json").exists():
                        harness, action_kind = "terminus", "terminus_json"
                        try:
                            trace = terminus_trace(json.loads((tdir / "agent" / "trajectory.json").read_text()), task, model_label)
                        except ValueError:
                            trace = None
                        if trace is None:
                            stats[key]["no_trajectory"] += 1; continue
                        trace["rewards"] = {"solved": {"score": 0.0, "weight": 1.0}}
                    else:
                        stats[key]["no_trajectory"] += 1; continue
                    trace.setdefault("info", {})["upstream_fetch"] = rollout_has_upstream_fetch(trace.get("nodes") or [])
                    if trace["info"]["upstream_fetch"]:
                        stats[key]["leaked_dropped"] += 1; continue
                    trace["info"].update({"bench": suite_full, "cell": cell.name, "trial": tdir.name, "reign": reign, "harbor_result": {"exception": (r.get("exception_info") or {}).get("exception_type"), "started_at": r.get("started_at"), "finished_at": r.get("finished_at")}})
                    meta = {"uid": f"{suite}/{task}", "sid": f"bench_{slug}__{task}", "repo": f"bench/{suite}", "language": SUITE_LANGUAGE.get(suite, "shell"),
                            "bench": suite_full, "king_digest": digest, "reign": reign, "cell": cell.name, "trial": tdir.name, "temp": temp, "teacher_passed": tp}
                    stamp = PolicyStamp(policy_id=f"bench_{harness}", model=model_label, harness=harness, endpoint="bench", action_kind=action_kind, temperature=temp)
                    envelopes.append(make_envelope(source=source, env_id=f"bench-{slug}", task=meta, policy=stamp, trace=trace))
                    stats[key]["eligible"] += 1
            else:
                for tr, name, score in verifiers_episodes(cell):
                    stats[key]["trials"] += 1
                    if score is None:
                        stats[key]["unscored"] += 1; continue
                    if score >= 1.0:
                        stats[key]["king_passed"] += 1; continue
                    stats[key]["king_failed"] += 1
                    tp = teacher_pass.get(suite, {}).get(name)
                    if tp is False:
                        stats[key]["teacher_failed_too"] += 1; continue
                    trial_key = f"{digest}|{cell.name}|{name}"
                    if trial_key in seen_trials:
                        continue
                    seen_trials.add(trial_key)
                    harness = ((tr.get("agent") or {}).get("config") or {}).get("harness", {}).get("id") or "null"
                    action_kind = "boxed" if "boxed" in json.dumps((tr.get("nodes") or [{}])[0])[:2000] else ("tool_call" if tr.get("tools") else "text")
                    tr.setdefault("info", {})["upstream_fetch"] = rollout_has_upstream_fetch(tr.get("nodes") or [])
                    if tr["info"]["upstream_fetch"]:
                        stats[key]["leaked_dropped"] += 1; continue
                    tr["info"].update({"bench": suite_full, "cell": cell.name, "reign": reign})
                    tr["rewards"] = {**(tr.get("rewards") or {}), "solved": {"score": 0.0, "weight": 1.0}}
                    meta = {"uid": f"{suite}/{name}", "sid": f"bench_{slug}__{name}", "repo": f"bench/{suite}", "language": "chat",
                            "bench": suite_full, "king_digest": digest, "reign": reign, "cell": cell.name, "trial": name, "temp": temp, "teacher_passed": tp}
                    stamp = PolicyStamp(policy_id=f"bench_{harness}", model=f"king/king-{digest}", harness=harness, endpoint="bench", action_kind=action_kind, temperature=temp)
                    envelopes.append(make_envelope(source=source, env_id=f"bench-{slug}", task=meta, policy=stamp, trace=tr))
                    stats[key]["eligible"] += 1

    table = {k: dict(v) for k, v in sorted(stats.items())}
    by_suite = defaultdict(Counter)
    for k, v in stats.items():
        by_suite[k.split("|")[0]].update(v)
    summary = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "min_reign": a.min_reign,
               "by_suite_reign": table, "by_suite": {k: dict(v) for k, v in sorted(by_suite.items())},
               "eligible_total": len(envelopes), "teacher_pass_sets": {s: len(v) for s, v in teacher_pass.items()}}
    Path(a.stats).parent.mkdir(parents=True, exist_ok=True)
    Path(a.stats).write_text(json.dumps(summary, indent=1))
    print(f"{'suite | reign':34} {'trials':>6} {'k_fail':>6} {'t_fail':>6} {'leak':>5} {'elig':>5}")
    for k, v in table.items():
        print(f"{k:34} {v.get('trials',0):6d} {v.get('king_failed',0):6d} {v.get('teacher_failed_too',0):6d} {v.get('leaked_dropped',0):5d} {v.get('eligible',0):5d}")
    print(f"eligible envelopes: {len(envelopes)}; stats -> {a.stats}")
    if a.dry_run or not a.publish:
        return 0
    from rollouts.store import TraceStore
    from rollouts.r2mirror import R2TraceMirror
    store = TraceStore(Path(a.out))
    by_source = defaultdict(list)
    for e in envelopes:
        by_source[e["source"]].append(e)
    stamp_t = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    for src, evs in by_source.items():
        for i in range(0, len(evs), 25):
            store.append_batch(evs[i:i + 25], tag=f"{src}-{stamp_t}-{i // 25:04d}")
    def env_value(k):
        if os.environ.get(k):
            return os.environ[k]
        for p in (REPO / ".env", Path.home() / ".affine-validator.env"):
            try:
                for line in p.read_text().splitlines():
                    line = line.strip().removeprefix("export ").strip()
                    if line.startswith(k + "="):
                        return line.split("=", 1)[1].strip().strip('"').strip("'")
            except OSError:
                pass
        return ""
    r2 = R2TraceMirror(bucket=env_value("DATA_R2_BUCKET") or "affine-data", endpoint=env_value("DATA_R2_ENDPOINT"),
                       access_key_id=env_value("DATA_R2_ACCESS_KEY_ID"), secret_access_key=env_value("DATA_R2_SECRET_ACCESS_KEY"),
                       prefix=a.prefix)
    if a.prefix == "traces/":
        sys.exit("refusing to publish bench envelopes into traces/ (the fold would take them as coding teacher rows); use traces-bench/")
    n = r2.mirror(store)
    print(f"published {n} chunk(s) to {a.prefix} (keys bench_*)")
    return 0


def verifiers_episodes(cell: Path):
    """(trace, task_name, primary score) per episode of a verifiers cell."""
    p = cell / "traces.jsonl.gz"
    op = gzip.open
    if not p.exists():
        p = cell / "traces.jsonl"; op = open
    if not p.exists():
        return
    with op(p, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                ep = json.loads(line)
            except ValueError:
                continue
            traces = ep.get("traces") or [ep]
            for tr in traces:
                task = tr.get("task") or ep.get("task") or {}
                data = task.get("data") or {}
                name = str(data.get("name") or data.get("idx") if data.get("name") or data.get("idx") is not None else task.get("hash") or task.get("key") or tr.get("id"))
                yield tr, name, primary_score(tr.get("rewards") or {})


if __name__ == "__main__":
    sys.exit(main())
