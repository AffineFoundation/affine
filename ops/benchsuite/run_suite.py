#!/usr/bin/env python
"""Run the Affine benchmark suite (suite.toml) against OpenAI-compatible
endpoints with Prime Intellect's `verifiers` (the `eval` CLI of the pinned
checkout, called as <verifiers>/.venv/bin/eval — never `uv run`, which
re-syncs the venv and drops the editable tasksets), one run
directory per (model, env, temperature), every rollout kept.

  python run_suite.py run --run-id 20260912T1500Z-0ce59769300c \
      --verifiers-dir ~/benchsuite/verifiers --out ~/benchsuite/runs \
      --king-url http://127.0.0.1:8001/v1 --king-model king-0ce59769300c \
      --teacher-url http://127.0.0.1:8002/v1 --teacher-model teacher \
      --key-env BENCH_API_KEY --runtime prime --envs aime25,math500 \
      --models king,teacher --temps primary,secondary --push

Layout under <out>/<run_id>/:
  manifest.json                        who / what / where / how much (no secrets)
  <model>/<env>__t<temp>/traces.jsonl  every rollout (verifiers Trace: messages,
                                       calls with usage + timing, rewards, errors)
  <model>/<env>__t<temp>/summary.json  n, score, 95% CI, tokens, wall time, cost
  <model>/<env>__t<temp>/eval.log      the eval's own log

Resumable: a cell whose summary.json exists is skipped. The king and the
teacher run the same env concurrently (two eval processes). `publish.py`
copies the run to R2 and writes the scorecard JSON the kingboard reads.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
SUITE = tomllib.loads((HERE / "suite.toml").read_text())
PRIME_API = "https://api.primeintellect.ai/api/v1"


def log(msg: str) -> None:
    print(f"[benchsuite] {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} {msg}",
          flush=True)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial proportion k/n."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def mean_ci(xs: list[float], z: float = 1.96) -> tuple[float, float, float]:
    """(mean, lo, hi) normal-approximation interval for a bounded score."""
    n = len(xs)
    if n == 0:
        return (0.0, 0.0, 0.0)
    m = sum(xs) / n
    if n == 1:
        return (m, m, m)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    half = z * sd / math.sqrt(n)
    return (m, max(0.0, m - half), min(1.0, m + half))


def prime_wallet(api_key: str | None) -> float | None:
    if not api_key:
        return None
    try:
        r = httpx.get(f"{PRIME_API}/billing/wallet",
                      headers={"Authorization": f"Bearer {api_key}"}, timeout=20)
        r.raise_for_status()
        d = r.json()
        for k in ("balance", "balance_usd", "amount"):
            if k in d:
                return float(d[k])
        return float(d.get("wallet", {}).get("balance"))
    except (httpx.HTTPError, ValueError, TypeError, AttributeError):
        return None


# ------------------------------------------------------------- summaries
def summarize_traces(path: Path, reward_name: str) -> dict:
    """Per-rollout records + aggregate from a verifiers traces.jsonl."""
    rows = []
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            # traces.jsonl holds one EPISODE per line: {id, env, task, run, ok,
            # errors, traces: [Trace per agent]}. Single-agent envs -> traces[0].
            inner = e.get("traces") or []
            t = inner[0] if inner else {}
            episode_errors = [x.get("type") for x in (e.get("errors") or [])]
            rw = t.get("rewards") or {}
            r = rw.get(reward_name)
            score = None if r is None else float(r.get("score") if isinstance(r, dict) else r)
            if score is None and rw:
                # fall back to the weighted sum verifiers itself reports as the reward
                tot = 0.0
                for v in rw.values():
                    if isinstance(v, dict) and v.get("score") is not None:
                        tot += float(v["score"]) * float(v.get("weight", 1.0))
                score = tot
            usage_p = usage_c = usage_r = 0
            for c in t.get("calls") or []:
                u = c.get("usage") or {}
                usage_p += int(u.get("prompt_tokens") or u.get("input_tokens") or 0)
                usage_c += int(u.get("completion_tokens") or u.get("output_tokens") or 0)
                det = u.get("completion_tokens_details") or {}
                usage_r += int(u.get("reasoning_tokens") or det.get("reasoning_tokens") or 0)
            tm = t.get("timing") or {}
            ag = tm.get("agent") or {}
            task = t.get("task") or e.get("task") or {}
            data = task.get("data") or {}
            rows.append({
                "episode_id": e.get("id"), "trace_id": t.get("id"),
                "task_idx": data.get("idx"),
                "task_key": task.get("key") or data.get("name"),
                "score": score,
                "rewards": {k: (v.get("score") if isinstance(v, dict) else v)
                            for k, v in rw.items()},
                "metrics": t.get("metrics") or {},
                "prompt_tokens": usage_p,
                "completion_tokens": usage_c,
                "reasoning_tokens": usage_r,
                "n_calls": len(t.get("calls") or []),
                "finish_reasons": [c.get("finish_reason") for c in (t.get("calls") or [])][-3:],
                "agent_seconds": round(float(ag.get("end", 0) or 0) - float(ag.get("start", 0) or 0), 2),
                "stop_condition": t.get("stop_condition"),
                "ok": bool(e.get("ok")) and bool(t.get("ok")),
                "errors": episode_errors + [x.get("type") for x in (t.get("errors") or [])],
                "error_messages": [(x.get("message") or "")[:120] for x in (e.get("errors") or []) + (t.get("errors") or [])],
                "n_nodes": len(t.get("nodes") or []),
            })
    # Error classes. A rollout that ran out of its time budget or overflowed
    # the model's context is the MODEL failing the task (score 0, like any
    # benchmark harness treats budget exhaustion); an infrastructure error
    # (sandbox died, proxy cut, tool server bug) is excluded from n.
    def err_class(r):
        msgs = " ".join(r["error_messages"])
        if "agent timeout" in msgs:
            return "timeout"
        if "maximum context length" in msgs:
            return "context_overflow"
        return "infra" if r["errors"] else None
    for r in rows:
        r["error_class"] = err_class(r)
        if r["error_class"] in ("timeout", "context_overflow") and r["score"] is None:
            r["score"] = 0.0
    scored = [r["score"] for r in rows if r["score"] is not None]
    binary = all(s in (0.0, 1.0) for s in scored)
    if binary:
        k = int(sum(scored))
        lo, hi = wilson(k, len(scored))
        mean = k / len(scored) if scored else 0.0
    else:
        mean, lo, hi = mean_ci(scored)
    return {
        "n": len(rows), "n_scored": len(scored),
        "n_errored": sum(1 for r in rows if r["error_class"] == "infra"),
        "n_timeout": sum(1 for r in rows if r["error_class"] == "timeout"),
        "n_context_overflow": sum(1 for r in rows if r["error_class"] == "context_overflow"),
        "score": round(mean, 4), "ci95": [round(lo, 4), round(hi, 4)],
        "binary": binary,
        "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
        "completion_tokens": sum(r["completion_tokens"] for r in rows),
        "reasoning_tokens": sum(r["reasoning_tokens"] for r in rows),
        "finish_length_frac": round(sum(1 for r in rows if "length" in (r["finish_reasons"] or [])) / max(1, len(rows)), 4),
        "rollouts": rows,
    }


# ------------------------------------------------------------- eval cells
def cell_dir(out: Path, model: str, env_id: str, temp: float) -> Path:
    return out / model / f"{env_id}__t{temp:g}"


def build_cmd(env: dict, model: str, url: str, key_env: str, temp: float,
              rollouts: int, out: Path, dirname: str, runtime: str,
              concurrency: int, push: bool, run_name: str,
              rollout_timeout: int, verifiers_dir: Path) -> list[str]:
    cmd = [
        str(verifiers_dir / ".venv" / "bin" / "eval"), env["taskset"],
        "-m", model,
        "--client.base-url", url,
        "--client.api-key-var", key_env,
        "--env.agent.harness.id", env["harness"],
        "--sampling.temperature", str(temp),
        "--sampling.max-tokens", str(env["max_tokens"]),
        "-r", str(rollouts),
        "-c", str(concurrency),
        "--no-rich",
        "--push" if push else "--no-push",
        "-o", str(out),
        "--run.dir", dirname,
        "--run.name", run_name,
        "--env.agent.timeout.setup", "1800",
        "--env.agent.timeout.rollout", str(rollout_timeout),
        "--env.agent.timeout.scoring", "1800",
    ]
    if env.get("n", -1) and int(env["n"]) > 0:
        cmd += ["-n", str(int(env["n"])), "--shuffle"]
    if env.get("max_turns"):
        cmd += ["--env.agent.max-turns", str(int(env["max_turns"]))]
    # Even single-turn chat tasks need a container runtime: the tasksets set
    # `network_allow = []` and verifiers refuses to run a network policy under
    # the subprocess runtime. runtime="none" cells therefore also use the
    # container runtime given on the command line (docker: python:3.11-slim,
    # ~5 s boot per task).
    cmd += ["--env.agent.runtime.type", runtime]
    if runtime == "prime":
        # Tasks that restrict egress (`network_allow = []`) need a micro-VM
        # sandbox; container sandboxes cannot enforce a policy. The first use
        # of an image as a VM builds it once on Prime's side (~10 min).
        cmd += ["--env.agent.runtime.vm", "true" if env.get("sandbox_vm") else "false"]
    cmd += list(env.get("args") or [])
    if env.get("taskset_config"):
        # Typed taskset knobs (Literal ints etc.) do not parse from CLI strings;
        # pass them as a TOML overlay, which keeps native types.
        overlay = out / dirname / "taskset.toml"
        overlay.parent.mkdir(parents=True, exist_ok=True)
        lines = ["[env.taskset]"]
        for k, v in env["taskset_config"].items():
            lines.append(f"{k} = {json.dumps(v)}")
        overlay.write_text("\n".join(lines) + "\n")
        # `@ file` must not follow a bare boolean flag (the parser would take
        # `@` as that flag's value), so it goes right after the taskset id.
        cmd[2:2] = ["@", str(overlay)]
    return cmd


def run_cell(env: dict, model_label: str, model: str, url: str, key_env: str,
             temp: float, rollouts: int, out: Path, verifiers_dir: Path,
             runtime: str, concurrency: int, push: bool, run_id: str,
             rollout_timeout: int, results: dict) -> None:
    d = cell_dir(out, model_label, env["id"], temp)
    if (d / "summary.json").exists():
        log(f"skip {model_label}/{env['id']} t={temp:g}: done")
        results[(model_label, env["id"], temp)] = json.loads((d / "summary.json").read_text())
        return
    d.mkdir(parents=True, exist_ok=True)
    dirname = str(d.relative_to(out))
    run_name = f"affine-bench-{run_id}-{model_label}-{env['id']}-t{temp:g}"
    resolved = d / "configs" / "resolved" / "eval.json"
    if resolved.exists() and (d / "traces.jsonl").exists():
        # An interrupted cell: re-run only its missing/errored rollouts in place.
        cmd = [str(verifiers_dir / ".venv" / "bin" / "eval"), "@", str(resolved), "--resume"]
        log(f"resume {model_label}/{env['id']} t={temp:g}")
    else:
        cmd = build_cmd(env, model, url, key_env, temp, rollouts, out, dirname, runtime,
                        concurrency, push, run_name, rollout_timeout, verifiers_dir)
    with (d / "cmd.txt").open("a") as fh:
        fh.write(" ".join(cmd) + "\n")
    log(f"start {model_label}/{env['id']} t={temp:g}: {' '.join(cmd[:6])} ...")
    t0 = time.time()
    with (d / "eval.log").open("a") as fh:
        p = subprocess.run(cmd, cwd=str(verifiers_dir), stdout=fh, stderr=subprocess.STDOUT,
                           env=os.environ.copy())
    wall = time.time() - t0
    traces = d / "traces.jsonl"
    if not traces.exists():
        log(f"FAIL {model_label}/{env['id']} t={temp:g}: exit={p.returncode}, no traces "
            f"(see {d / 'eval.log'})")
        results[(model_label, env["id"], temp)] = {"error": f"exit {p.returncode}", "wall_seconds": wall}
        return
    summ = summarize_traces(traces, env["reward"])
    summ.update({
        "env": env["id"], "taskset": env["taskset"], "model": model_label,
        "temperature": temp, "rollouts_per_task": rollouts,
        "harness": env["harness"], "runtime": runtime,
        "max_tokens": env["max_tokens"], "reward": env["reward"],
        "wall_seconds": round(wall, 1), "exit_code": p.returncode,
        "task_subset": ({"n": int(env["n"]), "shuffle_seed": 0} if int(env.get("n", -1)) > 0
                        else {"n": "all"}),
    })
    (d / "summary.json").write_text(json.dumps(summ, indent=1))
    log(f"done {model_label}/{env['id']} t={temp:g}: n={summ['n']} score={summ['score']} "
        f"ci={summ['ci95']} tokens_out={summ['completion_tokens']} wall={wall/60:.1f}min "
        f"exit={p.returncode}")
    results[(model_label, env["id"], temp)] = summ


def temps_of(a: argparse.Namespace) -> list[float]:
    out = []
    if "primary" in a.temps.split(","):
        out.append(float(SUITE["sampling"]["primary_temperature"]))
    if "secondary" in a.temps.split(","):
        out.append(float(SUITE["sampling"]["secondary_temperature"]))
    return out


def copy_teacher_cells(src_run: Path, dst_run: Path, envs: list[dict], temps: list[float]) -> int:
    """Copy the teacher's summary.json of every requested cell from an earlier
    run (rollout rows dropped, `reused_from` stamped). The traces stay in the
    source run's R2 prefix, which the stamp points at."""
    n = 0
    for env in envs:
        for temp in temps:
            if temp != float(SUITE["sampling"]["primary_temperature"]) and not env.get("secondary"):
                continue
            src = cell_dir(src_run, "teacher", env["id"], temp) / "summary.json"
            dst = cell_dir(dst_run, "teacher", env["id"], temp) / "summary.json"
            if not src.exists() or dst.exists():
                continue
            summ = json.loads(src.read_text())
            summ.pop("rollouts", None)
            summ["reused_from"] = src_run.name
            summ["traces"] = f"{SUITE['suite']['r2_prefix']}{src_run.name}/teacher/{src.parent.name}/traces.jsonl.gz"
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_text(json.dumps(summ, indent=1))
            n += 1
    return n


def cmd_compare(a: argparse.Namespace) -> int:
    """Did the king move? For every finished king cell of --run-id, compare its
    score with the same cell of --against (a run id or a published scorecard
    JSON): "moved" when the new score lies outside the previous run's 95%
    interval or the previous interval's centre lies outside the new one.
    Prints the moved envs; exit 0 = moved (or no baseline), 1 = all inside."""
    out = Path(a.out).expanduser()
    new_run = out / a.run_id
    prev_cells = {}
    against = Path(a.against).expanduser()
    if against.suffix == ".json" and against.exists():
        card = json.loads(against.read_text())
        for r in card.get("rows", []):
            if r.get("king"):
                prev_cells[(r["env"], float(r["temperature"]))] = r["king"]
    else:
        for summ in (out / a.against).glob("king/*/summary.json"):
            s = json.loads(summ.read_text())
            prev_cells[(s["env"], float(s["temperature"]))] = s
    moved = []
    for summ in sorted(new_run.glob("king/*/summary.json")):
        s = json.loads(summ.read_text())
        key = (s["env"], float(s["temperature"]))
        prev = prev_cells.get(key)
        if prev is None:
            moved.append((s["env"], s["temperature"], s["score"], None, "no baseline"))
            continue
        lo, hi = prev["ci95"]
        nlo, nhi = s["ci95"]
        prev_mid = prev["score"]
        if not (lo <= s["score"] <= hi) or not (nlo <= prev_mid <= nhi):
            moved.append((s["env"], s["temperature"], s["score"], prev_mid, f"outside [{lo:.3f}, {hi:.3f}]"))
    for env, temp, new, old, why in moved:
        print(f"MOVED {env} t={temp:g}: {new} vs {old} ({why})")
    if not moved:
        print("no king cell moved beyond the previous run's 95% interval")
    return 0 if moved else 1


def cmd_run(a: argparse.Namespace) -> int:
    out = Path(a.out).expanduser() / a.run_id
    out.mkdir(parents=True, exist_ok=True)
    verifiers_dir = Path(a.verifiers_dir).expanduser()
    env_ids = [e.strip() for e in a.envs.split(",")] if a.envs else [e["id"] for e in SUITE["envs"]]
    by_id = {e["id"]: e for e in SUITE["envs"]}
    envs = [by_id[i] for i in env_ids if i in by_id]      # in --envs order
    missing = set(env_ids) - {e["id"] for e in envs}
    if missing:
        raise SystemExit(f"unknown envs: {sorted(missing)}")
    models = {}
    if "king" in a.models.split(","):
        models["king"] = (a.king_model, a.king_url)
    if "teacher" in a.models.split(","):
        models["teacher"] = (a.teacher_model, a.teacher_url)
    if a.teacher_from:
        # Cheap mode: the teacher is frozen and the tasks are fixed, so its
        # baseline is copied forward from an earlier run instead of re-served.
        n_copied = copy_teacher_cells(Path(a.out).expanduser() / a.teacher_from, out, envs, temps_of(a))
        log(f"teacher baseline: {n_copied} cells reused from run {a.teacher_from}")
    temps = []
    if "primary" in a.temps.split(","):
        temps.append(("primary", float(SUITE["sampling"]["primary_temperature"])))
    if "secondary" in a.temps.split(","):
        temps.append(("secondary", float(SUITE["sampling"]["secondary_temperature"])))
    prime_key = os.environ.get("PRIME_API_KEY")
    manifest_path = out / a.manifest      # a second concurrent runner uses its own file
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.setdefault("run_id", a.run_id)
    manifest.setdefault("created_at", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    if a.meta:
        meta_path = Path(a.meta).expanduser()
        manifest.update(json.loads(meta_path.read_text() if meta_path.exists() else a.meta))
    manifest.setdefault("suite", {k: SUITE["suite"][k] for k in
                                  ("name", "version", "verifiers_commit", "research_envs_commit")})
    manifest.setdefault("serving", SUITE["serving"])
    manifest.setdefault("sampling", SUITE["sampling"])
    manifest.setdefault("envs", {e["id"]: {k: v for k, v in e.items() if k != "note"}
                                 for e in SUITE["envs"]})
    manifest.setdefault("cells", {})
    manifest.setdefault("prime_wallet_log", [])

    cells = []
    for env in envs:
        for kind, temp in temps:
            if kind == "secondary" and not env.get("secondary"):
                continue
            rollouts = int(env.get("secondary_rollouts", 1)) if kind == "secondary" else 1
            cells.append((env, temp, rollouts))
    max_total = float(SUITE["prime"]["max_total_usd"])
    results: dict = {}
    lock = threading.Lock()
    share = 1.0 / max(1, len(models))     # each model owns half the pod

    def record(env, temp, rollouts, label, wall):
        with lock:
            cell_key = f"{env['id']}__t{temp:g}"
            cell = manifest["cells"].setdefault(cell_key, {
                "env": env["id"], "temperature": temp, "rollouts_per_task": rollouts,
                "models": {}, "pod_cost_usd": 0.0})
            r = results.get((label, env["id"], temp), {})
            cell["models"][label] = {k: v for k, v in r.items() if k != "rollouts"}
            cell["models"][label]["wall_seconds"] = round(wall, 1)
            cell["pod_cost_usd"] = round(cell.get("pod_cost_usd", 0.0)
                                         + wall / 3600 * float(a.pod_usd_per_hour) * share, 2)
            cell["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            bal = prime_wallet(prime_key)
            manifest["prime_wallet_log"].append({"at": cell["finished_at"], "cell": cell_key,
                                                 "model": label, "balance": bal})
            manifest["prime_spent_usd"] = round(
                sum(float(c.get("pod_cost_usd") or 0) for c in manifest["cells"].values()), 2)
            manifest_path.write_text(json.dumps(manifest, indent=1))

    def chain(label, model, url):
        """One model works through every cell in order; `--parallel-envs` cells
        at a time so a long tail (a few slow tasks) does not idle its GPUs."""
        sem = threading.Semaphore(max(1, a.parallel_envs))
        threads = []

        def one(env, temp, rollouts):
            with sem:
                if float(manifest.get("prime_spent_usd", 0.0)) >= max_total:
                    log(f"STOP {label}: spend >= cap ${max_total:.2f}")
                    return
                t0 = time.time()
                run_cell(env, label, model, url, a.key_env, temp, rollouts, out,
                         verifiers_dir, a.runtime, a.concurrency, a.push, a.run_id,
                         int(env.get("rollout_timeout", a.rollout_timeout)), results)
                record(env, temp, rollouts, label, time.time() - t0)
        for env, temp, rollouts in cells:
            th = threading.Thread(target=one, args=(env, temp, rollouts), daemon=True)
            th.start()
            threads.append(th)
            time.sleep(0.5)
        for th in threads:
            th.join()

    chains = [threading.Thread(target=chain, args=(label, model, url), daemon=True)
              for label, (model, url) in models.items()]
    for th in chains:
        th.start()
    for th in chains:
        th.join()
    log("suite pass complete")
    return 0


def cmd_summarize(a: argparse.Namespace) -> int:
    """Re-derive summary.json for every cell that has traces (after edits or a crash)."""
    out = Path(a.out).expanduser() / a.run_id
    by_id = {e["id"]: e for e in SUITE["envs"]}
    seen = set()
    for traces in sorted(list(out.glob("*/*/traces.jsonl")) + list(out.glob("*/*/traces.jsonl.gz"))):
        d = traces.parent
        if d in seen:
            continue          # prefer the plain file when both exist
        seen.add(d)
        env_id, _, temp = d.name.rpartition("__t")
        env = by_id.get(env_id)
        if env is None:
            continue
        summ = summarize_traces(traces, env["reward"])
        prev = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {}
        summ.update({k: prev.get(k) for k in ("wall_seconds", "exit_code") if k in prev})
        summ.update({"env": env_id, "taskset": env["taskset"], "model": d.parent.name,
                     "temperature": float(temp), "reward": env["reward"],
                     "harness": env["harness"], "max_tokens": env["max_tokens"]})
        (d / "summary.json").write_text(json.dumps(summ, indent=1))
        log(f"{d.parent.name}/{d.name}: n={summ['n']} score={summ['score']} ci={summ['ci95']}")
    return 0


def cmd_retry(a: argparse.Namespace) -> int:
    """Re-run the errored rollouts of finished cells in place (`eval @ <resolved>
    --resume` re-runs only missing/errored rollouts), then re-summarize.
    Infra hiccups (a proxy 502, a sandbox that never came up) otherwise count
    as score 0 for that task."""
    out = Path(a.out).expanduser() / a.run_id
    verifiers_dir = Path(a.verifiers_dir).expanduser()
    by_id = {e["id"]: e for e in SUITE["envs"]}
    todo = []
    for summ_path in sorted(out.glob("*/*/summary.json")):
        s = json.loads(summ_path.read_text())
        if int(s.get("n_errored") or 0) >= a.min_errors and (summ_path.parent / "configs/resolved/eval.json").exists():
            todo.append((summ_path.parent, s))
    log(f"{len(todo)} cells with errored rollouts")
    threads = []
    sem = threading.Semaphore(a.parallel)

    def one(d: Path, s: dict) -> None:
        with sem:
            env = by_id[s["env"]]
            cmd = [str(verifiers_dir / ".venv/bin/eval"), "@", str(d / "configs/resolved/eval.json"), "--resume"]
            log(f"retry {d.parent.name}/{d.name}: {s['n_errored']} errored")
            t0 = time.time()
            with (d / "eval.log").open("a") as fh:
                subprocess.run(cmd, cwd=str(verifiers_dir), stdout=fh, stderr=subprocess.STDOUT)
            new = summarize_traces(d / "traces.jsonl", env["reward"])
            s.update({k: new[k] for k in new})
            s["wall_seconds"] = round(float(s.get("wall_seconds") or 0) + time.time() - t0, 1)
            s["retried"] = int(s.get("retried") or 0) + 1
            (d / "summary.json").write_text(json.dumps(s, indent=1))
            log(f"retry done {d.parent.name}/{d.name}: n={s['n']} errored={s['n_errored']} score={s['score']}")

    for d, s in todo:
        th = threading.Thread(target=one, args=(d, s), daemon=True)
        th.start()
        threads.append(th)
    for th in threads:
        th.join()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    cp = sub.add_parser("compare")
    cp.add_argument("--run-id", required=True)
    cp.add_argument("--out", required=True)
    cp.add_argument("--against", required=True, help="previous run id under --out, or a scorecard JSON")
    rt = sub.add_parser("retry")
    rt.add_argument("--run-id", required=True)
    rt.add_argument("--out", required=True)
    rt.add_argument("--verifiers-dir", required=True)
    rt.add_argument("--min-errors", type=int, default=1)
    rt.add_argument("--parallel", type=int, default=4)
    r = sub.add_parser("run")
    r.add_argument("--run-id", required=True)
    r.add_argument("--verifiers-dir", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--king-url")
    r.add_argument("--king-model")
    r.add_argument("--teacher-url")
    r.add_argument("--teacher-model", default="teacher")
    r.add_argument("--key-env", default="BENCH_API_KEY")
    r.add_argument("--runtime", default="prime", choices=["prime", "docker", "subprocess"])
    r.add_argument("--envs", default="")
    r.add_argument("--models", default="king,teacher")
    r.add_argument("--temps", default="primary,secondary")
    r.add_argument("--concurrency", type=int, default=64)
    r.add_argument("--parallel-envs", type=int, default=1,
                   help="cells run at once per model (keeps GPUs busy through slow tails)")
    r.add_argument("--rollout-timeout", type=int, default=7200)
    r.add_argument("--pod-usd-per-hour", type=float, default=0.0)
    r.add_argument("--push", action="store_true")
    r.add_argument("--teacher-from", default="",
                   help="cheap mode: reuse the teacher cells of this earlier run id instead of serving the teacher")
    r.add_argument("--manifest", default="manifest.json",
                   help="manifest file name under the run dir (a concurrent runner on other "
                        "envs must use a different name; publish.py merges manifest*.json)")
    r.add_argument("--meta", default="", help="JSON (string or file) merged into manifest.json: "
                   "king digest/reign/hotkey, teacher, where, pod, code commit")
    s = sub.add_parser("summarize")
    s.add_argument("--run-id", required=True)
    s.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd in ("run", "retry") and not (Path(a.verifiers_dir).expanduser() / ".venv/bin/eval").exists():
        raise SystemExit("verifiers venv missing: run install_eval_env.sh first")
    return {"run": cmd_run, "summarize": cmd_summarize, "retry": cmd_retry,
            "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
