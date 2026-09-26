"""Harvest N = 8 controller (store doc §7/§8, 2026-09-22) — the box-side loop
around the adaptive pod drivers (run_states.py --adaptive 4/8, launched by
run_pods.sh launch harvest).

    python harvest_ctl.py collect     pull every pod's results → harvest_n8/continuations.jsonl
    python harvest_ctl.py arms        plan + sample the Phase B / C arms, ship them to the pods
    python harvest_ctl.py king        3 reign-20 king samples at every kept state (free)
    python harvest_ctl.py tables      harvest_n8/tables.jsonl (per state {class -> (solved, total)}, B)
    python harvest_ctl.py report      VAV re-sim + F1-vs-TX + N7 → harvest_n8/report.txt
    python harvest_ctl.py cycle       all of the above once
    python harvest_ctl.py loop [--every 7200]

State classes (arm T, teacher continuations @T0.8, s solved of N graded):
  phase_a    N + dead < 4 — Phase A still running
  dropped    s = N at N >= 4 — the teacher always solves, nothing to rank
  ceiling    s = 0 — kept; Phase B tops up to 8, F1/TX proposals forced at it
  split      0 < s < N — kept; Phase B tops up to 8; Z (thought test) candidates
  abandoned  a permanent error (image / grader) or no graded continuation
Arms shipped from here (rows with continuations_needed = 1, picked up by the
drivers' --watch):
  F1 / F1b   glm-5.3 greedy / @T0.8 first action forced, teacher finishes
  TX / TXb   qwen3.8-27b greedy / @T0.8 EXTRA first action forced the same way
             (control: forcing mechanism + extra teacher draws at the same state)
  ZG<k>/ZB<k> k = 1..4: the teacher continues from a stored SOLVED / FAILED
             first thought (thought closed, action chosen by the teacher via
             raw /completions here), that first reply forced, teacher finishes
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import json
import math
import os
import re
import statistics as st
import subprocess
import sys
import tarfile
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import common  # noqa: E402
from common import (Engy, TEACHER_ENGY, THINK_CLOSE, THINK_OPEN, jaccard, norm_action,  # noqa: E402
                    read_jsonl, write_jsonl)
from sample import FRONTIER, SAMPLE_MAX_TOKENS, action_of  # noqa: E402
from select_states import META_DROP, forced_reply_of  # noqa: E402
from split_yield import first_action  # noqa: E402
import vav_sim  # noqa: E402
from vav_sim import (K_KING, _f, _mean, baseline, canon, classes_of, generic_action, match,  # noqa: E402
                     paired, reduced, repeat_last, sign_test, wire_to_template)

OUT = common.REPO / "research" / "results" / "frontier_arbiter" / "harvest_n8"
ARMS = Path("/tmp/fa_harvest/arms")
COLLECT = Path("/tmp/fa_harvest/collect")
RUN_PODS = HERE / "run_pods.sh"
PODS_SPEC = os.environ.get("PODS_SPEC") or \
    "dg5 64.247.196.242 40299 0;dg6 87.203.233.23 20100 1;dg2 86.125.172.23 57598 2;bf4 73.139.34.205 20008 3"
PODS = {e.split()[0]: (e.split()[1], e.split()[2]) for e in PODS_SPEC.split(";")}
WD = "/root/recoverable/frontier"
BOX = ["ssh", "-i", str(Path.home() / ".ssh" / "arbos_box"), "-o", "LogLevel=ERROR", "const@204.12.171.6"]
POD_OPTS = ("-i $HOME/.ssh/id_ed25519 -o UserKnownHostsFile=ops/king-datagen/state/known_hosts "
            "-o StrictHostKeyChecking=accept-new -o LogLevel=ERROR")

PHASE_A, PHASE_B = 4, 8
MAX_RETRIES = 2
PERMANENT_ERROR_MARKS = ("docker_build_failed", "pull access denied", "manifest unknown",
                         "checkout HEAD~1 failed", "no such image")
Z_STATES = 20
Z_CONT = 4
Z_MIN_THOUGHT_CHARS = 80
Z_MAX_TOKENS = 4096
LOOSE = 0.5
TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.S)
KEPT = ("split", "ceiling")
F_ARMS = ("F1", "F1b")
X_ARMS = ("TX", "TXb")
GLM_TOTAL_BUDGET_USD = 40.0
PARALLEL_STATES = 4

CONT = OUT / "continuations.jsonl"
PROPOSALS = OUT / "proposals.jsonl"
Z_PLAN = OUT / "z_plan.jsonl"
TABLES = OUT / "tables.jsonl"
REPORT = OUT / "report.txt"
NOTES = OUT / "run_notes.txt"
HISTORY = OUT / "history.jsonl"
SHIPPED = OUT / "shipped.json"
KING_SAMPLES = OUT / "king_samples.jsonl"
COST_LOG = OUT / "engy_cost.jsonl"


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())


def note(msg: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    line = f"{now()}  {msg}"
    print(line, flush=True)
    with open(NOTES, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def log_cost(stage: str, engy: Engy) -> None:
    common.append_jsonl(COST_LOG, {"at": now(), "stage": stage, "cost_usd": round(engy.cost_usd, 5),
                                   "usage": engy.usage})


def engy_spent() -> dict[str, float]:
    by: dict[str, float] = collections.defaultdict(float)
    for r in read_jsonl(COST_LOG):
        by[r["stage"]] += float(r.get("cost_usd") or 0)
    return dict(by)


def base_of(state_id: str) -> str:
    return state_id.rsplit(":", 1)[0]


def stem(sid: str) -> str:
    return sid.replace(":", "_")


# ------------------------------------------------------------------ pods
def box(cmd: str, timeout: int = 900) -> str:
    return subprocess.run([*BOX, cmd], capture_output=True, text=True, timeout=timeout).stdout


def pod_sh(pod: str, cmd: str, timeout: int = 600) -> str:
    host, port = PODS[pod]
    q = cmd.replace("'", "'\\''")
    return box(f"cd ~/subnet120 && ssh {POD_OPTS} -p {port} root@{host} '{q}' 2>&1 | grep -v setlocale", timeout)


def pod_put(pod: str, local: Path, remote: str) -> None:
    host, port = PODS[pod]
    subprocess.run(["scp", "-q", "-i", str(Path.home() / ".ssh" / "arbos_box"), str(local),
                    f"const@204.12.171.6:/tmp/{local.name}"], check=True, timeout=600)
    box(f"cd ~/subnet120 && scp -q {POD_OPTS} -P {port} /tmp/{local.name} root@{host}:{remote}", 600)


# ------------------------------------------------------------------ collect
def cmd_collect(args) -> None:
    env = dict(os.environ, PODS_SPEC=PODS_SPEC)
    t0 = time.time()
    p = subprocess.run(["bash", str(RUN_PODS), "collect", "harvest", str(COLLECT)], env=env,
                       capture_output=True, text=True, timeout=1800)
    tail = (p.stdout + p.stderr).strip().splitlines()[-3:]
    src = COLLECT / "continuations.jsonl"
    if not src.exists():
        note(f"collect FAILED ({time.time() - t0:.0f}s): {tail}")
        return
    rows = read_jsonl(src)
    OUT.mkdir(parents=True, exist_ok=True)
    # a row's arm is the state id suffix (collect_results); `pod` is stamped from the archive dir
    write_jsonl(CONT, rows)
    per_pod = collections.Counter(r["pod"] for r in rows)
    ok = sum(r.get("status") == "ok" for r in rows)
    note(f"collect: {len(rows)} rows ({ok} ok) from {dict(per_pod)} in {time.time() - t0:.0f}s; {tail[-1] if tail else ''}")


# ------------------------------------------------------------------ state summaries
def is_permanent(r: dict) -> bool:
    err = str(r.get("error") or "")
    return any(m in err for m in PERMANENT_ERROR_MARKS)


def is_dead(r: dict) -> bool:
    return r.get("status") != "ok" and (is_permanent(r) or int(r.get("attempts") or 1) > MAX_RETRIES)


def graded(r: dict) -> bool:
    return r.get("status") == "ok" and r.get("outcome") in ("solved", "failed")


def load_states() -> dict[str, dict]:
    """base id -> state JSON (arm files share the base's messages; the T file wins)."""
    out: dict[str, dict] = {}
    for f in sorted(ARMS.glob("*/T/states/*.json")):
        try:
            d = json.load(open(f))
        except ValueError:
            continue
        b = base_of(d["state_id"])
        if b not in out or d["state_id"].endswith(":T"):
            out[b] = d
    return out


def selected_meta() -> dict[str, dict]:
    metas = {}
    for f in ARMS.glob("*/T/states.jsonl"):
        pod = f.parts[-3]
        for m in read_jsonl(f):
            m = dict(m, pod=pod)
            metas[m["state_id"]] = m
    return metas


def summarize(rows: list[dict]) -> dict[str, dict]:
    """base id -> {phase, s, n, dead, n8, pod, harness, kind, arms: {arm: rows}}."""
    by: dict[str, dict[str, list[dict]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        by[base_of(r["state_id"])][r["arm"]].append(r)
    metas = selected_meta()
    out = {}
    for b, arms in by.items():
        T = sorted(arms.get("T", []), key=lambda r: int(r.get("continuation") or 0))
        ok = [r for r in T if graded(r)]
        dead = [r for r in T if is_dead(r)]
        s, n = sum(r["outcome"] == "solved" for r in ok), len(ok)
        if any(is_permanent(r) for r in T):
            phase = "abandoned"
        elif n + len(dead) < PHASE_A:
            phase = "phase_a"
        elif n == 0:
            phase = "abandoned"
        elif s == n:
            phase = "dropped"
        elif s == 0:
            phase = "ceiling"
        else:
            phase = "split"
        meta = metas.get(b + ":T") or {}
        r0 = T[0] if T else [r for rs in arms.values() for r in rs][0]
        out[b] = {"sid": b, "phase": phase, "kept": phase in KEPT, "s": s, "n": n, "dead": len(dead),
                  "n8": phase in KEPT and n + len(dead) >= PHASE_B,
                  "pod": r0.get("pod"), "harness": r0.get("harness"), "resume_kind": r0.get("resume_kind"),
                  "kind": meta.get("action_kind"), "depth": meta.get("depth"), "source": meta.get("source"),
                  "T": T, "ok": ok, "arms": arms}
    return out


# ------------------------------------------------------------------ arms (Phase B proposals, Phase C thoughts)
def parse_tool_calls(text: str) -> tuple[str, list[dict]]:
    """Qwen3.8 raw completion → (visible content without the XML, OpenAI-style tool_calls)."""
    calls = []
    for i, m in enumerate(TOOL_CALL_RE.finditer(text)):
        body = m.group(1)
        name, args = None, None
        try:
            d = json.loads(body)
            if isinstance(d, dict) and d.get("name"):
                name, args = d["name"], d.get("arguments") or {}
        except ValueError:
            # Qwen3.8 template form: <function=edit><parameter=path>\n…\n</parameter></function>
            fm = vav_sim.XML_FUNC_RE.search(body)
            if fm:
                name = fm.group(1)
                args = {k: v for k, v in vav_sim.XML_PARAM_RE.findall(fm.group(2))}
        if name:
            calls.append({"id": f"call_z{i}", "type": "function",
                          "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)}})
    return TOOL_CALL_RE.sub("", text).strip(), calls


def relation(y: str, teacher_ys: list[str], kind: str) -> tuple[str, float]:
    if not teacher_ys:
        return "no_teacher", 0.0
    if any(norm_action(y, kind) == norm_action(t, kind) for t in teacher_ys):
        return "exact", 1.0
    bj = max(vav_sim.body_jaccard(y, t, kind) for t in teacher_ys)
    return ("jaccard" if bj >= LOOSE else "new"), bj


def usable(p: dict | None) -> bool:
    return bool(p and p.get("y") and p.get("harness_valid", True) and not p.get("finish_reply"))


def arm_row(stt: dict, sid: str, arm: str, forced: dict, y: str, extra: dict) -> tuple[dict, Path]:
    s = {k: v for k, v in stt.items() if k not in ("forced_reply", "forced_action", "arm", "probe")}
    s.update(state_id=sid, arm=arm, probe="harvest_n8", forced_reply=forced, forced_action=y, **extra)
    s["continuations_needed"] = 1
    pod = extra["pod"]
    path = ARMS / pod / "T" / "states" / (stem(sid) + ".json")
    path.write_text(json.dumps(s, ensure_ascii=False))
    meta = {k: v for k, v in s.items() if k not in META_DROP}
    meta["path"] = str(path)
    return meta, path


def append_meta(pod: str, meta: dict) -> None:
    with open(ARMS / pod / "T" / "states.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")


async def sample_proposals(engy: Engy, stt: dict, model: str, budget_ok) -> list[dict]:
    """greedy + T0.8 first actions of `model` at the state (T0.8 re-drawn up to 2x when it
    repeats the greedy action); each entry: source, y, reply (for forced_reply_of)."""
    harness, kind = stt["harness"], stt["action_kind"]
    extra = {"tools": stt["tools"]} if harness == "bash" and stt.get("tools") else {}
    out: list[dict] = []
    for source, temp, tries in (("greedy", 0.0, 2), ("t08", 0.8, 3)):
        got = dup = None
        for _ in range(tries):
            if not budget_ok():
                return out
            try:
                rep = await engy.chat(model, stt["messages"], temperature=temp, max_tokens=SAMPLE_MAX_TOKENS, **extra)
            except Exception as e:  # noqa: BLE001 - a dead Engy skips the state this cycle
                note(f"  {model} sample failed at {stt['state_id'][:12]}: {repr(e)[:160]}")
                return out
            p = action_of(rep, harness, kind)
            if not usable(p):
                continue
            if source == "t08" and out and norm_action(p["y"], kind) == norm_action(out[0]["y"], kind):
                dup = (p, rep)
                continue
            got = (p, rep, False)
            break
        if got is None and dup is not None:
            # every T0.8 draw repeated the greedy action: keep one as a second verified row of that class
            got = (dup[0], dup[1], True)
        if got is not None:
            p, rep, is_dup = got
            out.append({"source": source, "y": p["y"], "p": p, "finish": rep.get("finish"), "dup": is_dup})
    return out


async def plan_ceiling(engy: Engy, summ: dict[str, dict], states: dict[str, dict]) -> int:
    done = {p["state_id"] for p in read_jsonl(PROPOSALS)}
    targets = [x for x in summ.values() if x["phase"] == "ceiling" and x["sid"] + ":frontier" not in done
               and x["pod"] in PODS and x["sid"] in states]
    glm_spent = engy_spent().get("glm_proposals", 0.0)

    def budget_ok():
        return glm_spent + engy.usage.get(FRONTIER, {}).get("cost_usd", 0.0) < GLM_TOTAL_BUDGET_USD
    sem = asyncio.Semaphore(PARALLEL_STATES)

    async def one(x: dict) -> int:
        stt = states[x["sid"]]
        harness, kind = stt["harness"], stt["action_kind"]
        teacher_ys = [first_action(r, kind) for r in x["ok"]]
        teacher_ys = [y for y in teacher_ys if y]
        async with sem:
            fr, tx = await asyncio.gather(sample_proposals(engy, stt, FRONTIER, budget_ok),
                                          sample_proposals(engy, stt, TEACHER_ENGY, lambda: True))
        if not fr and not tx:
            return 0
        n_rows = 0
        props = []
        for arms, lst, model in ((F_ARMS, fr, FRONTIER), (X_ARMS, tx, TEACHER_ENGY)):
            for arm, q in zip(arms, lst):
                sid = f"{x['sid']}:{arm}"
                forced = forced_reply_of({"frontier_greedy": q["p"], "harness": harness})
                rel, bj = relation(q["y"], teacher_ys, kind)
                meta, _ = arm_row(stt, sid, arm, forced, q["y"],
                                  {"pod": x["pod"], "proposal_model": model, "proposal_source": q["source"],
                                   "teacher_desc": f"{x['s']}/{x['n']}"})
                append_meta(x["pod"], meta)
                props.append({"arm": arm, "model": model, "source": q["source"], "y": q["y"], "dup_of_greedy": q["dup"],
                              "relation_to_teacher": rel, "best_body_jaccard": round(bj, 3), "finish": q["finish"]})
                n_rows += 1
        common.append_jsonl(PROPOSALS, {"state_id": x["sid"] + ":frontier", "harness": harness, "action_kind": kind,
                                        "source": x["source"], "depth": x["depth"], "pod": x["pod"],
                                        "teacher_desc": f"{x['s']}/{x['n']}", "teacher_first_actions": teacher_ys,
                                        "proposals": props, "planned_at": now()})
        print(f"  ceiling {x['sid'][:12]} {harness:18s} T {x['s']}/{x['n']}  "
              f"{[(q['arm'], q['relation_to_teacher']) for q in props]}", flush=True)
        return n_rows
    return sum(await asyncio.gather(*[one(x) for x in targets]))


async def teacher_from_thought(engy: Engy, stt: dict, thought: str, temperature: float = 0.8) -> dict:
    """Raw /completions: the prefix rendered by the teacher's own chat template, the
    assistant turn opened with the stored thought CLOSED (`<think>\\n…\\n</think>\\n\\n`),
    the teacher writes the visible reply = its own action. Engy's chat endpoint does
    not honour continue_final_message (checked 2026-09-21), hence the raw path."""
    tok = common.teacher_tokenizer()
    kw = {"tools": stt["tools"]} if stt.get("tools") else {}
    prompt = tok.apply_chat_template(wire_to_template(stt["messages"]), tokenize=False, add_generation_prompt=True, **kw)
    if prompt.rstrip().endswith(THINK_OPEN):
        prompt = prompt.rstrip() + "\n"
    elif not prompt.endswith(THINK_OPEN + "\n"):
        prompt = prompt + THINK_OPEN + "\n"
    prompt += thought.strip() + "\n" + THINK_CLOSE + "\n\n"
    payload = {"model": TEACHER_ENGY, "prompt": prompt, "max_tokens": Z_MAX_TOKENS, "temperature": temperature}
    d = await engy._post("/completions", payload, TEACHER_ENGY)
    cost = engy._tally(TEACHER_ENGY, d)
    ch = d["choices"][0]
    text = ch.get("text") or ""
    content, calls = parse_tool_calls(text) if stt["harness"] == "bash" else (text, [])
    return {"reasoning": "", "content": content, "tool_calls": calls, "finish": ch.get("finish_reason"),
            "usage": d.get("usage"), "cost_usd": cost, "prompt_chars": len(prompt)}


def pick_thoughts(x: dict, kind: str) -> tuple[dict, dict] | None:
    """(solved row, failed row) among the T continuations whose first reply carries the
    latent thought (harvest rows only; yesterday's seeds have none)."""
    def thought(r):
        return ((r.get("first_reply") or {}).get("reasoning") or "").strip()
    good = [r for r in x["ok"] if r["outcome"] == "solved" and len(thought(r)) >= Z_MIN_THOUGHT_CHARS and first_action(r, kind)]
    bad = [r for r in x["ok"] if r["outcome"] == "failed" and len(thought(r)) >= Z_MIN_THOUGHT_CHARS and first_action(r, kind)]
    if not good or not bad:
        return None
    # the thought whose action was NOT the majority class is the interesting one; simplest: first of each
    return good[0], bad[0]


async def plan_z(engy: Engy, summ: dict[str, dict], states: dict[str, dict]) -> int:
    planned = read_jsonl(Z_PLAN)
    done = {p["state_id"] for p in planned}
    if len(done) >= Z_STATES:
        return 0
    # split states first at N = 8 (both labels are then better supported), then N >= 4
    cands = sorted([x for x in summ.values() if x["phase"] == "split" and x["sid"] not in done
                    and x["pod"] in PODS and x["sid"] in states], key=lambda x: (-x["n"], x["sid"]))
    picked = []
    for x in cands:
        if len(done) + len(picked) >= Z_STATES:
            break
        if pick_thoughts(x, states[x["sid"]]["action_kind"]) is not None:
            picked.append(x)
    sem = asyncio.Semaphore(PARALLEL_STATES)

    async def one(x: dict) -> int:
        stt = states[x["sid"]]
        kind, harness = stt["action_kind"], stt["harness"]
        good, bad = pick_thoughts(x, kind)
        forced_rows, cost = [], 0.0
        async with sem:
            for label, src_row in (("ZG", good), ("ZB", bad)):
                thought = src_row["first_reply"]["reasoning"].strip()
                k = tries = 0
                while k < Z_CONT and tries < Z_CONT * 3:
                    tries += 1
                    try:
                        rep = await teacher_from_thought(engy, stt, thought)
                    except Exception as e:  # noqa: BLE001
                        note(f"  Z sample failed at {x['sid'][:12]}: {repr(e)[:160]}")
                        break
                    cost += float(rep.get("cost_usd") or 0)
                    p = action_of(rep, harness, kind)
                    if not usable(p):
                        continue
                    k += 1
                    arm = f"{label}{k}"
                    sid = f"{x['sid']}:{arm}"
                    forced = forced_reply_of({"frontier_greedy": p, "harness": harness})
                    meta, _ = arm_row(stt, sid, arm, forced, p["y"],
                                      {"pod": x["pod"], "z_label": label, "z_source_outcome": src_row["outcome"],
                                       "z_source_continuation": src_row.get("continuation"), "z_thought_chars": len(thought),
                                       "teacher_desc": f"{x['s']}/{x['n']}"})
                    append_meta(x["pod"], meta)
                    forced_rows.append({"arm": arm, "label": label, "y": p["y"], "finish": rep.get("finish"),
                                        "relation_to_source_action": relation(p["y"], [first_action(src_row, kind)], kind)[0]})
        if not forced_rows:
            return 0
        common.append_jsonl(Z_PLAN, {"state_id": x["sid"], "harness": harness, "action_kind": kind, "pod": x["pod"],
                                     "teacher_desc": f"{x['s']}/{x['n']}",
                                     "good": {"continuation": good.get("continuation"), "thought_chars": len(good["first_reply"]["reasoning"]),
                                              "action": first_action(good, kind)},
                                     "bad": {"continuation": bad.get("continuation"), "thought_chars": len(bad["first_reply"]["reasoning"]),
                                             "action": first_action(bad, kind)},
                                     "forced": forced_rows, "cost_usd": round(cost, 4), "planned_at": now()})
        print(f"  Z {x['sid'][:12]} {harness:18s} T {x['s']}/{x['n']}  "
              f"ZG {sum(f['label'] == 'ZG' for f in forced_rows)} ZB {sum(f['label'] == 'ZB' for f in forced_rows)} "
              f"same-as-source {sum(f['relation_to_source_action'] == 'exact' for f in forced_rows)}", flush=True)
        return len(forced_rows)
    return sum(await asyncio.gather(*[one(x) for x in picked]))


def ship_arms() -> None:
    """New arm state files + the pod's states.jsonl (atomically replaced) → the pod."""
    shipped = json.loads(SHIPPED.read_text()) if SHIPPED.exists() else {}
    for pod in PODS:
        d = ARMS / pod / "T"
        if not d.exists():
            continue
        have = set(shipped.get(pod) or [])
        new = [p for p in sorted((d / "states").glob("*.json")) if p.name not in have and not p.name.endswith("_T.json")]
        if not new:
            continue
        tgz = Path(f"/tmp/fa_arms_new_{pod}.tgz")
        with tarfile.open(tgz, "w:gz") as tf:
            for p in new:
                tf.add(p, arcname=f"states/{p.name}")
            tf.add(d / "states.jsonl", arcname="states.jsonl.new")
        try:
            pod_put(pod, tgz, f"{WD}/arms_new_{pod}.tgz")
            out = pod_sh(pod, f"cd {WD}/arms/{pod}/T && tar xzf {WD}/arms_new_{pod}.tgz && mv states.jsonl.new states.jsonl "
                              f"&& wc -l states.jsonl && ls states | wc -l")
        except Exception as e:  # noqa: BLE001 - an unreachable pod is retried next cycle
            note(f"ship to {pod} FAILED: {repr(e)[:200]}")
            continue
        if not out.strip():
            note(f"ship to {pod}: no confirmation, retry next cycle")
            continue
        shipped[pod] = sorted(have | {p.name for p in new})
        SHIPPED.write_text(json.dumps(shipped, indent=1))
        note(f"shipped {len(new)} arm state(s) to {pod}: {' '.join(out.split())}")


def cmd_arms(args) -> None:
    rows = read_jsonl(CONT)
    summ = summarize(rows)
    states = load_states()
    engy = Engy(concurrency=6, timeout=1500)

    async def go():
        n_c = await plan_ceiling(engy, summ, states)
        n_z = await plan_z(engy, summ, states)
        return n_c, n_z
    n_c, n_z = asyncio.run(go())
    # cost split by model: glm rows under glm_proposals, teacher rows (TX + Z) under teacher_proposals
    glm = engy.usage.get(FRONTIER, {}).get("cost_usd", 0.0)
    common.append_jsonl(COST_LOG, {"at": now(), "stage": "glm_proposals", "cost_usd": round(glm, 5),
                                   "usage": {FRONTIER: engy.usage.get(FRONTIER)}})
    common.append_jsonl(COST_LOG, {"at": now(), "stage": "teacher_proposals", "cost_usd": round(engy.cost_usd - glm, 5),
                                   "usage": {TEACHER_ENGY: engy.usage.get(TEACHER_ENGY)}})
    note(f"arms: {n_c} F1/TX rows, {n_z} Z rows planned; Engy ${engy.cost_usd:.3f} (glm ${glm:.3f})")
    ship_arms()


# ------------------------------------------------------------------ king
def cmd_king(args) -> None:
    summ = summarize(read_jsonl(CONT))
    vav_sim.KING_SAMPLES = KING_SAMPLES
    vav_sim.STATES_GLOB = str(ARMS / "*" / "T" / "states" / "*_T.json")
    tables = [{"sid": x["sid"], "kind": x["kind"]} for x in summ.values() if x["kept"] and x["kind"]]
    if not tables:
        return
    try:
        asyncio.run(vav_sim.sample_king(tables, args.concurrency))
    except SystemExit as e:
        note(f"king: {e}")
        return
    n = len([r for r in read_jsonl(KING_SAMPLES) if "error" not in r])
    note(f"king: {n} samples stored for {len(tables)} kept states")


# ------------------------------------------------------------------ tables
def loose_classes(rows: list[dict], kind: str) -> list[list[int]]:
    """Single-linkage groups: norm-exact OR body Jaccard >= 0.5."""
    n = len(rows)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if rows[i]["norm"] == rows[j]["norm"] or vav_sim.body_jaccard(rows[i]["y"], rows[j]["y"], kind) >= LOOSE:
                parent[find(i)] = find(j)
    groups: dict[int, list[int]] = collections.defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    return sorted(groups.values(), key=lambda g: g[0])


def forced_y_of(sid: str, metas: dict[str, dict]) -> str | None:
    m = metas.get(sid)
    return (m or {}).get("forced_action")


def build_tables(summ: dict[str, dict]) -> list[dict]:
    metas = selected_meta()
    out = []
    for x in summ.values():
        if not x["kept"] or not x["kind"]:
            continue
        kind = x["kind"]
        table = []
        for i, r in enumerate(x["ok"]):
            y = canon(first_action(r, kind), kind)
            if y:
                table.append({"src": "T", "arm": "T", "idx": i, "cont": r.get("continuation"), "y": y,
                              "norm": norm_action(y, kind), "outcome": r["outcome"]})
        for src, arms in (("F", F_ARMS), ("X", X_ARMS)):
            j = 0
            for arm in arms:
                for r in x["arms"].get(arm, []):
                    y = forced_y_of(f"{x['sid']}:{arm}", metas)
                    if graded(r) and y:
                        table.append({"src": src, "arm": arm, "idx": j, "y": canon(y, kind), "norm": norm_action(y, kind),
                                      "outcome": r["outcome"]})
                        j += 1
        z_rows = []
        for arm, rs in x["arms"].items():
            if arm[:2] in ("ZG", "ZB"):
                for r in rs:
                    z_rows.append({"arm": arm, "label": arm[:2], "y": forced_y_of(f"{x['sid']}:{arm}", metas),
                                   "outcome": r.get("outcome") if graded(r) else None, "status": r.get("status")})
        cl = classes_of(table)
        B = baseline(table, cl)
        groups = loose_classes(table, kind)
        loose = []
        for g in groups:
            mem = [table[i] for i in g]
            loose.append({"y": mem[0]["y"][:300], "solved": sum(m["outcome"] == "solved" for m in mem), "total": len(mem),
                          "members": [f"{m['src']}{m['idx']}" for m in mem]})
        loose_v = {i: (c["solved"] / c["total"]) for c, g in zip(loose, groups) for i in g}
        B_loose = _mean([loose_v[i] for i, t in enumerate(table) if t["src"] == "T"])
        pending = sum(1 for arm in (*F_ARMS, *X_ARMS) for r in x["arms"].get(arm, []) if not graded(r) and not is_dead(r))
        out.append({"sid": x["sid"], "tag": x["phase"], "kind": kind, "harness": x["harness"], "depth": x["depth"],
                    "source": x["source"], "pod": x["pod"], "s": x["s"], "n": x["n"], "dead": x["dead"], "n8": x["n8"],
                    "table": table,
                    "n_F": sum(t["src"] == "F" for t in table), "s_F": sum(t["src"] == "F" and t["outcome"] == "solved" for t in table),
                    "n_X": sum(t["src"] == "X" for t in table), "s_X": sum(t["src"] == "X" and t["outcome"] == "solved" for t in table),
                    "arms_pending": pending,
                    "classes_strict": [{"y": c["y"][:300], "solved": c["solved"], "total": c["total"],
                                        "members": [f"{s}{i}" for s, i in c["members"]]} for c in cl.values()],
                    "classes_loose": loose, "B": B, "B_loose": B_loose, "z_rows": z_rows})
    return out


def cmd_tables(args) -> list[dict]:
    summ = summarize(read_jsonl(CONT))
    tables = build_tables(summ)
    write_jsonl(TABLES, tables)
    note(f"tables: {len(tables)} kept states ({sum(t['n8'] for t in tables)} at N>=8) -> tables.jsonl")
    return tables


# ------------------------------------------------------------------ VAV re-sim (surface cascade, no LLM judge)
MINERS = ("teacher_heldout", "king", "frontier_loo", "teacher_extra_loo", "generic", "repeat")
SRC_MINER = {"T": "teacher_heldout", "F": "frontier_loo", "X": "teacher_extra_loo"}
CONTRASTS = (("teacher_heldout", "king"), ("teacher_heldout", "generic"), ("teacher_heldout", "repeat"),
             ("king", "generic"), ("king", "repeat"), ("frontier_loo", "teacher_extra_loo"),
             ("frontier_loo", "king"), ("teacher_extra_loo", "king"))


def king_rows(sid: str, ks: list[dict]) -> list[dict]:
    return sorted([r for r in ks if r["sid"] == sid and "error" not in r], key=lambda r: r["i"])


def score_state(t: dict, stt: dict | None, ks: list[dict]) -> dict:
    kind = t["kind"]
    cands = []
    for r in t["table"]:
        cands.append({"miner": SRC_MINER[r["src"]], "label": f"{r['src']}{r['idx']}", "y": r["y"], "exclude": (r["src"], r["idx"]),
                      "outcome": r["outcome"]})
    for r in king_rows(t["sid"], ks):
        cands.append({"miner": "king", "label": f"K{r['i']}", "y": r["y"] if r["parsed"] else "", "exclude": None,
                      "forfeit": not r["parsed"]})
    cands.append({"miner": "generic", "label": "gen", "y": generic_action(kind), "exclude": None})
    if stt:
        rl = repeat_last(stt["messages"], kind)
        if rl:
            cands.append({"miner": "repeat", "label": "rep", "y": rl, "exclude": None})
    rec = {"sid": t["sid"], "tag": t["tag"], "n8": t["n8"], "s": t["s"], "n": t["n"], "B": t["B"], "cands": []}
    for c in cands:
        tab = reduced(t["table"], c["exclude"])
        if not c["y"]:
            m = {"how": "forfeit", "V": None, "B": baseline(tab, classes_of(tab)), "term": 0.0}
        else:
            m = match(c["y"], kind, tab, t["sid"], None)
        rec["cands"].append({"miner": c["miner"], "label": c["label"], "how": m["how"], "V": m["V"], "B": m["B"], "term": m["term"],
                             "outcome": c.get("outcome"), "y": c["y"][:200]})
    per = {}
    for miner in MINERS:
        cs = [c for c in rec["cands"] if c["miner"] == miner]
        per[miner] = None if not cs else {
            "n": len(cs), "term": st.mean(c["term"] for c in cs),
            "credited": st.mean(1.0 if c["how"] in ("exact", "jaccard") else 0.0 for c in cs),
            "how": dict(collections.Counter(c["how"] for c in cs)),
            "forfeit": st.mean(1.0 if c["how"] == "forfeit" else 0.0 for c in cs) if miner == "king" else None}
    rec["per_miner"] = per
    return rec


def aggregate(rows: list[dict]) -> dict:
    out: dict = {"n_states": len(rows)}
    for miner in MINERS:
        rs = [r for r in rows if r["per_miner"].get(miner)]
        if not rs:
            out[miner] = None
            continue
        terms = [r["per_miner"][miner]["term"] for r in rs]
        out[miner] = {"n_states": len(rs), "n_actions": sum(r["per_miner"][miner]["n"] for r in rs), "term": st.mean(terms),
                      "se": st.stdev(terms) / math.sqrt(len(terms)) if len(terms) > 1 else None,
                      "credited_actions": _mean([r["per_miner"][miner]["credited"] for r in rs]),
                      "credited_states": st.mean(1.0 if r["per_miner"][miner]["credited"] > 0 else 0.0 for r in rs),
                      "positive_states": st.mean(1.0 if r["per_miner"][miner]["term"] > 0 else 0.0 for r in rs),
                      "how": dict(sum((collections.Counter(r["per_miner"][miner]["how"]) for r in rs), collections.Counter())),
                      "forfeit": _mean([r["per_miner"][miner]["forfeit"] for r in rs]) if miner == "king" else None}
    out["paired"] = {}
    for a, b in CONTRASTS:
        d = [r["per_miner"][a]["term"] - r["per_miner"][b]["term"] for r in rows if r["per_miner"].get(a) and r["per_miner"].get(b)]
        out["paired"][f"{a}-{b}"] = paired(d)
    return out


# ------------------------------------------------------------------ side experiments
def f1_vs_tx(tables: list[dict]) -> dict:
    per = []
    for t in tables:
        F = [r for r in t["table"] if r["src"] == "F"]
        X = [r for r in t["table"] if r["src"] == "X"]
        if not F and not X:
            continue
        per.append({"sid": t["sid"], "tag": t["tag"], "harness": t["harness"], "s": t["s"], "n": t["n"],
                    "F": (sum(r["outcome"] == "solved" for r in F), len(F)), "X": (sum(r["outcome"] == "solved" for r in X), len(X)),
                    "F_new_class": sum(1 for r in F if all(r["norm"] != q["norm"] for q in t["table"] if q["src"] == "T")),
                    "X_new_class": sum(1 for r in X if all(r["norm"] != q["norm"] for q in t["table"] if q["src"] == "T"))})
    both = [p for p in per if p["F"][1] and p["X"][1]]
    diffs = [p["F"][0] / p["F"][1] - p["X"][0] / p["X"][1] for p in both]
    sF = sum(p["F"][0] for p in per); nF = sum(p["F"][1] for p in per)
    sX = sum(p["X"][0] for p in per); nX = sum(p["X"][1] for p in per)
    return {"states": len(per), "states_both": len(both), "F_pooled": (sF, nF), "X_pooled": (sX, nX),
            "F_rate": sF / nF if nF else None, "X_rate": sX / nX if nX else None,
            "F_states_with_solve": sum(p["F"][0] > 0 for p in per), "X_states_with_solve": sum(p["X"][0] > 0 for p in per),
            "F_new_class_rows": sum(p["F_new_class"] for p in per), "X_new_class_rows": sum(p["X_new_class"] for p in per),
            "paired": paired(diffs), "per": per}


def n7(tables: list[dict]) -> dict:
    per = []
    for t in tables:
        zs = [z for z in t["z_rows"] if z["outcome"] in ("solved", "failed")]
        if not zs:
            continue
        G = [z for z in zs if z["label"] == "ZG"]
        Bd = [z for z in zs if z["label"] == "ZB"]
        per.append({"sid": t["sid"], "harness": t["harness"], "s": t["s"], "n": t["n"],
                    "ZG": (sum(z["outcome"] == "solved" for z in G), len(G)), "ZB": (sum(z["outcome"] == "solved" for z in Bd), len(Bd)),
                    "pending": sum(1 for z in t["z_rows"] if z["outcome"] is None and z["status"] != "errored")})
    both = [p for p in per if p["ZG"][1] and p["ZB"][1]]
    diffs = [p["ZG"][0] / p["ZG"][1] - p["ZB"][0] / p["ZB"][1] for p in both]
    sG = sum(p["ZG"][0] for p in per); nG = sum(p["ZG"][1] for p in per)
    sB = sum(p["ZB"][0] for p in per); nB = sum(p["ZB"][1] for p in per)
    t_rate = _mean([p["s"] / p["n"] for p in both])
    return {"states": len(per), "states_both": len(both), "ZG_pooled": (sG, nG), "ZB_pooled": (sB, nB),
            "ZG_rate": sG / nG if nG else None, "ZB_rate": sB / nB if nB else None, "teacher_rate_same_states": t_rate,
            "gap_paired": paired(diffs), "per": per}


# ------------------------------------------------------------------ report
def progress(rows: list[dict], summ: dict[str, dict]) -> dict:
    metas = selected_meta()
    selected = {base_of(m["state_id"]) for m in metas.values() if m["state_id"].endswith(":T")}
    phases = collections.Counter(x["phase"] for x in summ.values())
    phases["not_started"] = len(selected - set(summ))
    by_h = collections.defaultdict(collections.Counter)
    for x in summ.values():
        by_h[x["harness"]][x["phase"]] += 1
    T_rows = [r for r in rows if r["arm"] == "T"]
    arm_rows = [r for r in rows if r["arm"] != "T"]
    cost_T = sum(float(r.get("cost_usd") or 0) for r in T_rows)
    cost_arms = sum(float(r.get("cost_usd") or 0) for r in arm_rows)
    walls = collections.defaultdict(float)
    for r in rows:
        walls[r["pod"]] += float(r.get("wall_s") or 0) / 3600
    ok_T = [r for r in T_rows if graded(r)]
    return {"selected": len(selected), "phases": dict(phases), "by_harness": {k: dict(v) for k, v in by_h.items()},
            "kept": sum(x["kept"] for x in summ.values()), "n8": sum(x["n8"] for x in summ.values()),
            "T_rows": len(T_rows), "T_ok": len(ok_T), "T_errored": sum(r.get("status") != "ok" for r in T_rows),
            "T_solved": sum(r["outcome"] == "solved" for r in ok_T),
            "arm_rows": len(arm_rows), "arm_ok": sum(graded(r) for r in arm_rows),
            "arm_by": dict(collections.Counter(r["arm"] for r in arm_rows)),
            "cost_T": cost_T, "cost_arms": cost_arms, "engy_side": engy_spent(),
            "container_hours": dict(walls), "wall_median_s": st.median([float(r["wall_s"]) for r in ok_T]) if ok_T else None,
            "turn_cap": sum(1 for r in ok_T if r.get("turn_cap_error")),
            "per_pod_rows": dict(collections.Counter(r["pod"] for r in rows))}


def render(prog: dict, tables: list[dict], scored: list[dict], fx: dict, z: dict, hist: list[dict]) -> str:
    L: list[str] = []
    P = L.append
    P(f"Harvest N = 8 — verified-action tables for the VAV rule (store doc §7/§8) — {now()}")
    P("Terms: state = rollout_id:turn_idx prefix of a FAILED teacher trajectory; T = teacher continuation @T0.8 from the state;")
    P("       s/N = solved/graded T continuations; split = 0<s<N, ceiling = s=0, dropped = s=N (at N>=4); kept = split|ceiling; N=8 = 8 slots settled;")
    P("       table = {class -> (solved,total)} over graded continuations whose FIRST action is in the class (T rows + forced F1/TX rows);")
    P("       class strict = norm_action-exact; loose = strict OR body-Jaccard>=0.5 single-linkage; V = solved/total; B = mean V over the teacher's own first actions;")
    P("       term = V(class(y)) − B on the table WITHOUT the scored action's own row (LOO), unmatched → 0; credited = joined a class (exact / Jaccard);")
    P("       miners: teacher_heldout = the N teacher first actions (LOO mean); king = 3 reign-20 samples at the prefix (forfeit = no </think>/action, term 0);")
    P("       frontier_loo = forced glm-5.3 proposals (F1 greedy, F1b T0.8); teacher_extra_loo = forced EXTRA teacher draws (TX greedy, TXb T0.8);")
    P("       generic = `ls -la` in the dialect; repeat = previous assistant action re-issued; sign = exact two-sided binomial on per-state sign (ties dropped);")
    P("       Z = teacher continues from a stored SOLVED (ZG) / FAILED (ZB) first thought, its own action forced, teacher finishes (N7 thought-value test).")
    P("       No LLM equivalence judge in this re-sim (surface cascade only) — the split-states VAV sim reports the judge's effect.")
    P("")
    P("== 0. PROGRESS ==")
    ph = prog["phases"]
    P(f"selected {prog['selected']} states: phase_a {ph.get('phase_a', 0)}  dropped(all solved) {ph.get('dropped', 0)}  "
      f"split {ph.get('split', 0)}  ceiling {ph.get('ceiling', 0)}  abandoned {ph.get('abandoned', 0)}  no result yet {ph.get('not_started', 0)}")
    P(f"kept {prog['kept']}, at N=8: {prog['n8']}   per harness: {prog['by_harness']}")
    P(f"T continuations {prog['T_rows']} rows ({prog['T_ok']} graded, {prog['T_solved']} solved, {prog['T_errored']} errored/pending-retry, "
      f"{prog['turn_cap']} turn-cap=failed); median wall {_f(prog['wall_median_s'], 6, 0)} s")
    P(f"arm continuations {prog['arm_rows']} rows ({prog['arm_ok']} graded) by arm {prog['arm_by']}")
    side = prog["engy_side"]
    total = prog["cost_T"] + prog["cost_arms"] + sum(side.values())
    P(f"$ Engy: T continuations {prog['cost_T']:.2f} + arm continuations {prog['cost_arms']:.2f} + proposals/thought sampling "
      f"{ {k: round(v, 2) for k, v in side.items()} } = ${total:.2f} (list prices; king sampling free)")
    P(f"container-hours by pod (sum of eval walls): { {k: round(v, 1) for k, v in prog['container_hours'].items()} }; rows per pod {prog['per_pod_rows']}")
    P("")
    P("== 1. TABLES per kept state (strict classes; loose classes in tables.jsonl) ==")
    P(f"{'tag':<7} {'n8':<2} {'state':<8} {'harness':<19} {'d':>2} {'T s/N':>5} {'F s/n':>5} {'X s/n':>5} {'cls':>3} {'clsL':>4} {'B':>6} {'B_L':>6}  Z rows")
    for t in sorted(tables, key=lambda t: (not t["n8"], t["tag"], t["sid"])):
        zg = [zr for zr in t["z_rows"] if zr["label"] == "ZG" and zr["outcome"]]
        zb = [zr for zr in t["z_rows"] if zr["label"] == "ZB" and zr["outcome"]]
        zs = (f"ZG {sum(zr['outcome'] == 'solved' for zr in zg)}/{len(zg)} ZB {sum(zr['outcome'] == 'solved' for zr in zb)}/{len(zb)}"
              if t["z_rows"] else "")
        P(f"{t['tag']:<7} {'*' if t['n8'] else ' ':<2} {t['sid'][:8]:<8} {t['harness']:<19} {t['depth']:>2} {t['s']}/{t['n']:<3} "
          f"{t['s_F']}/{t['n_F']:<3} {t['s_X']}/{t['n_X']:<3} {len(t['classes_strict']):>3} {len(t['classes_loose']):>4} "
          f"{_f(t['B'], 6, 3)} {_f(t['B_loose'], 6, 3)}  {zs}")
        for c in sorted(t["classes_strict"], key=lambda c: (-(c["solved"] / c["total"]), -c["total"])):
            P(f"{'':<62} {c['solved']}/{c['total']} [{','.join(c['members']):<14}] {c['y'].replace(chr(10), ' ')[:100]}")
    P("")
    P("== 2. VAV re-sim — per-state term (V − B) by miner ==")
    P(f"{'tag':<7} {'n8':<2} {'state':<8} {'B':>6} | {'teacherLOO':>10} {'king':>7} {'frontLOO':>8} {'xtraLOO':>8} {'generic':>7} {'repeat':>7} | king how")
    for r in sorted(scored, key=lambda r: (not r["n8"], r["tag"], r["sid"])):
        pm = r["per_miner"]

        def cell(m, w=7):
            return _f(pm[m]["term"], w, 3) if pm.get(m) else " " * (w - 3) + "n/a"
        P(f"{r['tag']:<7} {'*' if r['n8'] else ' ':<2} {r['sid'][:8]:<8} {_f(r['B'], 6, 3)} | {cell('teacher_heldout', 10)} {cell('king')} "
          f"{cell('frontier_loo', 8)} {cell('teacher_extra_loo', 8)} {cell('generic')} {cell('repeat')} | {pm['king']['how'] if pm.get('king') else '-'}")
    P("")
    subsets = (("N=8 all kept", [r for r in scored if r["n8"]]), ("N=8 split", [r for r in scored if r["n8"] and r["tag"] == "split"]),
               ("N=8 ceiling", [r for r in scored if r["n8"] and r["tag"] == "ceiling"]),
               ("N>=4 all kept (interim)", scored), ("N>=4 split (interim)", [r for r in scored if r["tag"] == "split"]))
    for label, rs in subsets:
        a = aggregate(rs)
        P(f"-- {label} (n states {a['n_states']}) -- mean term per miner, credited share, positive-state share")
        P(f"{'miner':<18} {'states':>6} {'acts':>5} {'term':>8} {'se':>7} {'credited/act':>12} {'credited/state':>14} {'term>0':>7}  how | king forfeit")
        for m in MINERS:
            g = a.get(m)
            if not g:
                continue
            P(f"{m:<18} {g['n_states']:>6} {g['n_actions']:>5} {_f(g['term'], 8, 4)} {_f(g['se'], 7, 4)} {_f(g['credited_actions'], 12, 2)} "
              f"{_f(g['credited_states'], 14, 2)} {_f(g['positive_states'], 7, 2)}  {g['how']} | {_f(g['forfeit'], 4, 2) if g['forfeit'] is not None else ''}")
        P("   paired contrasts (a − b): mean, z, sign test +/−/ties, p")
        for k, p in a["paired"].items():
            P(f"     {k:<36} n {p['n']:>3} mean {_f(p['mean'], 8, 4)} z {_f(p['z'], 6, 2)} sign +{p['pos']}/−{p['neg']}/={p['ties']} p {_f(p['p'], 6, 3)}")
        P("")
    P("== 3. Frontier proposals vs extra teacher draws at ceiling states (forced first action, teacher finishes) ==")
    P(f"states with arms {fx['states']} (both F and X graded {fx['states_both']}); pooled solve F1/F1b {fx['F_pooled'][0]}/{fx['F_pooled'][1]} "
      f"= {_f(fx['F_rate'], 5, 3)} vs TX/TXb {fx['X_pooled'][0]}/{fx['X_pooled'][1]} = {_f(fx['X_rate'], 5, 3)}; "
      f"states with >=1 solve: F {fx['F_states_with_solve']} X {fx['X_states_with_solve']}")
    pp = fx["paired"]
    P(f"paired per-state rate F − X: n {pp['n']} mean {_f(pp['mean'], 7, 3)} z {_f(pp['z'], 5, 2)} sign +{pp['pos']}/−{pp['neg']}/={pp['ties']} p {_f(pp['p'], 6, 3)}; "
      f"rows in a class NO teacher continuation used: F {fx['F_new_class_rows']}/{fx['F_pooled'][1]} X {fx['X_new_class_rows']}/{fx['X_pooled'][1]}")
    for p in fx["per"]:
        P(f"   {p['tag']:<7} {p['sid'][:8]} {p['harness']:<19} T {p['s']}/{p['n']}  F {p['F'][0]}/{p['F'][1]}  X {p['X'][0]}/{p['X'][1]}")
    P("")
    P("== 4. N7 thought-value test (Z arms at split states) ==")
    P(f"states {z['states']} (both labels graded {z['states_both']}); pooled solve good-thought ZG {z['ZG_pooled'][0]}/{z['ZG_pooled'][1]} = {_f(z['ZG_rate'], 5, 3)} "
      f"vs bad-thought ZB {z['ZB_pooled'][0]}/{z['ZB_pooled'][1]} = {_f(z['ZB_rate'], 5, 3)}; teacher's own s/N at the same states {_f(z['teacher_rate_same_states'], 5, 3)}")
    gp = z["gap_paired"]
    P(f"paired gap ZG − ZB: n {gp['n']} mean {_f(gp['mean'], 7, 3)} z {_f(gp['z'], 5, 2)} sign +{gp['pos']}/−{gp['neg']}/={gp['ties']} p {_f(gp['p'], 6, 3)}")
    for p in z["per"]:
        P(f"   {p['sid'][:8]} {p['harness']:<19} T {p['s']}/{p['n']}  ZG {p['ZG'][0]}/{p['ZG'][1]}  ZB {p['ZB'][0]}/{p['ZB'][1]}  pending {p['pending']}")
    P("")
    P("== 5. HISTORY (one line per collect cycle) ==")
    for hrow in hist:
        P(f"   {hrow['at']}  kept {hrow['kept']:>3} n8 {hrow['n8']:>3} split {hrow['split']:>3} ceiling {hrow['ceiling']:>3} T rows {hrow['T_rows']:>4} "
          f"${hrow['cost']:>6.2f} | N=8 teacher−king mean {_f(hrow.get('tk_mean'), 7, 4)} sign +{hrow.get('tk_pos', 0)}/−{hrow.get('tk_neg', 0)}/={hrow.get('tk_ties', 0)} "
          f"p {_f(hrow.get('tk_p'), 5, 3)} | interim(N>=4) mean {_f(hrow.get('tk4_mean'), 7, 4)} p {_f(hrow.get('tk4_p'), 5, 3)}")
    return "\n".join(L) + "\n"


def cmd_report(args) -> None:
    rows = read_jsonl(CONT)
    summ = summarize(rows)
    tables = build_tables(summ)
    write_jsonl(TABLES, tables)
    states = load_states()
    ks = read_jsonl(KING_SAMPLES) if KING_SAMPLES.exists() else []
    scored = [score_state(t, states.get(t["sid"]), ks) for t in tables]
    fx = f1_vs_tx(tables)
    z = n7(tables)
    prog = progress(rows, summ)
    a8 = aggregate([r for r in scored if r["n8"]])["paired"]["teacher_heldout-king"]
    a4 = aggregate(scored)["paired"]["teacher_heldout-king"]
    hrow = {"at": now(), "kept": prog["kept"], "n8": prog["n8"], "split": prog["phases"].get("split", 0),
            "ceiling": prog["phases"].get("ceiling", 0), "T_rows": prog["T_rows"],
            "cost": prog["cost_T"] + prog["cost_arms"] + sum(prog["engy_side"].values()),
            "tk_mean": a8["mean"], "tk_pos": a8["pos"], "tk_neg": a8["neg"], "tk_ties": a8["ties"], "tk_p": a8["p"],
            "tk4_mean": a4["mean"], "tk4_p": a4["p"]}
    common.append_jsonl(HISTORY, hrow)
    txt = render(prog, tables, scored, fx, z, read_jsonl(HISTORY))
    REPORT.write_text(txt)
    (OUT / "report.json").write_text(json.dumps({"progress": prog, "f1_vs_tx": fx, "n7": z,
                                                 "agg_n8": aggregate([r for r in scored if r["n8"]]),
                                                 "agg_all": aggregate(scored), "scored": scored}, indent=1, default=str))
    note(f"report: kept {prog['kept']} (n8 {prog['n8']}, split {prog['phases'].get('split', 0)}, ceiling {prog['phases'].get('ceiling', 0)}); "
         f"N=8 teacher−king mean {_f(a8['mean'], 7, 4)} sign +{a8['pos']}/−{a8['neg']}/={a8['ties']} p {_f(a8['p'], 5, 3)}; ${hrow['cost']:.2f}")


# ------------------------------------------------------------------ cycle / loop
def cmd_cycle(args) -> None:
    for step in (cmd_collect, cmd_arms, cmd_king, cmd_tables, cmd_report):
        try:
            step(args)
        except Exception:  # noqa: BLE001 - one failing step must not stop the harvest
            note(f"{step.__name__} CRASHED: {traceback.format_exc()[-800:]}")


def cmd_loop(args) -> None:
    while True:
        t0 = time.time()
        cmd_cycle(args)
        wait = max(60.0, args.every - (time.time() - t0))
        print(f"next cycle in {wait / 60:.0f} min", flush=True)
        time.sleep(wait)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("collect", "arms", "king", "tables", "report", "cycle", "loop"))
    ap.add_argument("--every", type=float, default=7200)
    ap.add_argument("--concurrency", type=int, default=6)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"collect": cmd_collect, "arms": cmd_arms, "king": cmd_king, "tables": cmd_tables, "report": cmd_report,
     "cycle": cmd_cycle, "loop": cmd_loop}[args.cmd](args)


if __name__ == "__main__":
    main()
