#!/usr/bin/env python
"""Analyse the first-divergence probe output and write the king_divergence side-table."""
import json, re, collections, statistics as st, math, sys
from pathlib import Path
REPO = Path.home() / "subnet120"
import os
KING12 = os.environ.get("DIV_KING", "6d0ee567e33e")
_D = Path(os.environ.get("DIV_OUT", str(REPO / "affine/state/king_divergence")))
IN = _D / f"{KING12}.turns.jsonl"
SIDE = _D / f"{KING12}.jsonl"
WS = re.compile(r"\s+")
def norm(s): return WS.sub(" ", s or "").strip().lower()
def coarse(a: str) -> str:
    """tool name + first path-like token, or bash command head (first 2 words) + first path."""
    a = a or ""
    m = re.search(r'"name"\s*:\s*"([^"]+)"', a) or re.search(r"<function=([^>\s]+)", a) or re.search(r"<(\w+)>", a)
    body = a
    if m:
        name = m.group(1)
        path = re.search(r"(/[\w./-]+|[\w-]+\.(?:py|go|ts|js|rs|java|md|txt|json|toml|sqlite))", a[m.end():])
        return f"{name} {path.group(1) if path else ''}".strip().lower()
    body = re.sub(r"^```[^\n]*\n", "", body.strip()).rstrip("`").strip()
    toks = body.split()
    head = " ".join(toks[:2]).lower() if toks else ""
    path = re.search(r"(/[\w./-]+|[\w-]+\.(?:py|go|ts|js|rs|java|md|txt|json|toml|sqlite))", body)
    return f"{head} {path.group(1) if path else ''}".strip()

rows = {}
for pth in sorted(IN.parent.glob(f"{KING12}.turns*.jsonl")):
    for l in open(pth):
        r = json.loads(l); rows.setdefault(r["rollout_id"], r)
rows = list(rows.values())
print("rollouts", len(rows), collections.Counter((r["cls"], r["outcome"]) for r in rows))
turns_all = 0; stop_turns = 0
per = []
for r in rows:
    T = r["turns"]
    turns_all += len(T); stop_turns += sum(t["stop_eligible"] for t in T)
    fd = {"strict": None, "coarse": None, "confident": None, "stopdiv": None}
    for t in T:
        refs = [x["action"] for x in t["refs"] if x["action"]]
        rn = {norm(a) for a in refs}; rc = {coarse(a) for a in refs}
        ka = t["king_action"]
        agree_strict = t["agree"]
        agree_coarse = (bool(ka) and coarse(ka) in rc) or (t["king_stop"] and t["ref_stop"] >= 1)
        unanimous_c = (len(refs) == 3 and len(rc) == 1) or (t["ref_stop"] == 3)
        t["_coarse_agree"] = agree_coarse
        t["_stopdiv"] = t["ref_stop"] >= 2 and not t["king_stop"]
        if fd["strict"] is None and not agree_strict: fd["strict"] = t["turn_idx"]
        if fd["coarse"] is None and not agree_coarse: fd["coarse"] = t["turn_idx"]
        if fd["confident"] is None and unanimous_c and not agree_coarse: fd["confident"] = t["turn_idx"]
        if fd["stopdiv"] is None and t["_stopdiv"]: fd["stopdiv"] = t["turn_idx"]
    r["_fd"] = fd
    per.append(r)
base_stop = stop_turns / turns_all
print(f"turns probed {turns_all}; stop-eligible turns {stop_turns} ({100*base_stop:.1f} %)")
print(f"turn-level: exact a_match {100*st.mean(t['a_match'] for r in rows for t in r['turns']):.1f} %, agree(strict) {100*st.mean(t['agree'] for r in rows for t in r['turns']):.1f} %, agree(coarse) {100*st.mean(t['_coarse_agree'] for r in rows for t in r['turns']):.1f} %")
print("\n== Prediction (a): share of first divergences on stop-eligible turns vs base rate")
out_lines = []
for variant in ("strict", "coarse", "confident", "stopdiv"):
    n_div = 0; n_stop = 0; n_none = 0; depth = []
    for r in per:
        k = r["_fd"][variant]
        if k is None: n_none += 1; continue
        t = next(x for x in r["turns"] if x["turn_idx"] == k)
        n_div += 1; n_stop += t["stop_eligible"]; depth.append(k)
    line = f"{variant:10} first divergence found in {n_div}/{len(per)} rollouts (none within probed turns: {n_none}); on stop-eligible turns {n_stop}/{n_div} = {100*n_stop/max(n_div,1):.1f} % (base rate of stop-eligible turns {100*base_stop:.1f} %); depth p50 {st.median(depth) if depth else float('nan')}"
    print(line); out_lines.append(line)
print("\n   by class (coarse variant): first-div on stop-eligible % / base stop % / median depth / n")
for cls in ("agent", "tool", "notool"):
    R = [r for r in per if r["cls"] == cls]
    T = [t for r in R for t in r["turns"]]
    d = [(r["_fd"]["coarse"], next(x for x in r["turns"] if x["turn_idx"] == r["_fd"]["coarse"])) for r in R if r["_fd"]["coarse"] is not None]
    if not d: continue
    print(f"   {cls:8} {100*st.mean(t['stop_eligible'] for _,t in d):.1f} / {100*st.mean(t['stop_eligible'] for t in T):.1f} / {st.median(k for k,_ in d)} / {len(d)}")
print("\n== Prediction (b): failure vs depth of first CONFIDENT divergence (refs unanimous incl. unanimous stop, king differs) and of first STOP divergence (>=2 refs stop, king acts), multi-turn classes")
bins = [(0,1,"turn 0"),(1,3,"1-2"),(3,6,"3-5"),(6,12,"6-11"),(12,31,"12+"),(None,None,"none within 30")]
for variant in ("confident", "stopdiv"):
    print(f"   -- {variant}")
    for lo,hi,l in bins:
        R = [r for r in per if r["cls"] != "notool" and ((r["_fd"][variant] is None) if lo is None else (r["_fd"][variant] is not None and lo <= r["_fd"][variant] < hi))]
        if R: print(f"   first {variant} {l:16} n={len(R):3d} failed {100*st.mean(r['outcome']=='failed' for r in R):5.1f} %  loop-guard/max_turns stop {100*st.mean(r['stop_condition'] in ('loop_guard','max_turns') for r in R):5.1f} %  rollout turns p50 {st.median(r['n_turns_total'] for r in R)}")
print("\n   stop-divergence turns per rollout (multi-turn classes): mean", st.mean(sum(t['_stopdiv'] for t in r['turns']) for r in per if r['cls']!='notool'), " share of rollouts with >=1:", st.mean(any(t['_stopdiv'] for t in r['turns']) for r in per if r['cls']!='notool'))
print("\n== H3 by class: turns where ALL 3 refs stopped -> king acted %")
for cls in ("agent","tool","notool"):
    T=[t for r in per if r["cls"]==cls for t in r["turns"] if t["ref_stop"]==3]
    Tany=[t for r in per if r["cls"]==cls for t in r["turns"]]
    print(f"   {cls:8} all-3-stop turns {len(T)}/{len(Tany)} ({100*len(T)/max(1,len(Tany)):.1f} % of probed turns); king acted there {100*st.mean(not t['king_stop'] for t in T) if T else float('nan'):.1f} %; king_loop label at those turns: {collections.Counter(t['king_loop'] for t in T).most_common(3)}")
print("   (selection was balanced 50/50 solved/failed per class, so 50 % = no relation)")
# king acted where refs stopped
acted_at_stop = [t for r in per for t in r["turns"] if t["stop_eligible"] and not t["king_stop"]]
stopped_at_stop = [t for r in per for t in r["turns"] if t["stop_eligible"] and t["king_stop"]]
print(f"\n== H3 direct: at stop-eligible turns the king acted on {len(acted_at_stop)}/{len(acted_at_stop)+len(stopped_at_stop)} = {100*len(acted_at_stop)/max(1,len(acted_at_stop)+len(stopped_at_stop)):.1f} %; where ALL 3 refs stopped: ", end="")
all3 = [t for r in per for t in r["turns"] if t["ref_stop"] == 3]
print(f"king acted {100*st.mean(not t['king_stop'] for t in all3) if all3 else float('nan'):.1f} % (n={len(all3)})")
print("\n== Prediction (c): G and a_match at first-divergence turns vs control turns")
def gstats(ts, key):
    vals = [t["echo"][key] for t in ts if t.get("echo") and t["echo"].get(key) is not None]
    return (len(vals), 100*st.mean(v >= 0 for v in vals) if vals else float('nan'), st.mean(vals) if vals else float('nan'))
div_t = [next(x for x in r["turns"] if x["turn_idx"] == r["_fd"]["strict"]) for r in per if r["_fd"]["strict"] is not None]
div_t = [t for t in div_t if t.get("echo")]
ctrl_t = [t for r in per for t in r["turns"] if t.get("echo") and not t.get("first_divergence")]
for label, ts in (("first-divergence turns", div_t), ("control (agreeing) turns", ctrl_t)):
    for key in ("G_c2", "G_c4"):
        n, pos, mean = gstats(ts, key)
        print(f"   {label:26} {key}: n={n} G>=0 on {pos:.1f} %  mean {mean:+.4f}   a_match {100*st.mean(t['a_match'] for t in ts) if ts else float('nan'):.1f} %  stop-eligible {100*st.mean(t['stop_eligible'] for t in ts) if ts else float('nan'):.0f} %")
    ms = [t["echo"]["m"] for t in ts if t.get("echo") and t["echo"].get("m") is not None]
    tt = [x for t in ts if t.get("echo") for x in t["echo"].get("t", [])]
    print(f"   {'':26} king thought per-byte lp mean {st.mean(ms) if ms else float('nan'):+.4f} vs teacher refs {st.mean(tt) if tt else float('nan'):+.4f}; king thought chars p50 {st.median(t['king_thought_len'] for t in ts) if ts else float('nan')}")
# side table
n_side = 0
with open(SIDE, "w") as f:
    for r in per:
        if r["outcome"] != "failed": continue
        k = r["_fd"]["coarse"] if r["_fd"]["coarse"] is not None else r["_fd"]["strict"]
        if k is None: continue
        t = next(x for x in r["turns"] if x["turn_idx"] == k)
        kind = "confident" if r["_fd"]["confident"] == k else ("coarse" if r["_fd"]["coarse"] == k else "strict")
        ka = t["king_action"] or ""
        if t["ref_stop"] >= 2 and not t["king_stop"]:
            label = "acts_where_teacher_stops"
            sysblob = r.get("_sys", "")
            ids = re.findall(r"[A-Za-z][A-Za-z0-9_]*_[0-9]{2,}|[A-Z]{2,}[0-9]{2,}|\b\d{3}-\d{3}-\d{4}\b", ka)
            if any(i in sysblob for i in ids): label = "schema_example_where_teacher_asked"
        elif t["king_loop"] in ("in_loop", "loop_onset"):
            label = "repeats_where_teacher_moves_on"
        elif t["king_stop"] and t["ref_stop"] == 0:
            label = "stops_where_teacher_acts"
        else:
            label = "different_action_same_kind"
        row = {"turn_id": f"{r['traj_id']}:{k}", "rollout_id": r["rollout_id"], "traj_id": r["traj_id"], "node_id": t["node_id"], "turn_idx": k,
               "king": f"king-{KING12}", "source": r["source"], "harness": r["harness"], "kind": t["kind"], "outcome": r["outcome"],
               "stop_condition": r["stop_condition"], "n_turns_total": r["n_turns_total"], "divergence_kind": kind,
               "stop_eligible": t["stop_eligible"], "ref_stop": t["ref_stop"], "king_stop": t["king_stop"], "king_action": t["king_action"],
               "refs": t["refs"], "ref_n_valid": t["ref_n_valid"], "ref_unanimous": t["ref_unanimous"], "a_match": t["a_match"], "agree": t["agree"],
               "echo": t.get("echo"), "failure_label": label, "axis": {"agent": "coding", "tool": "tool_use", "notool": "tool_use"}.get(r["cls"], "other"), "prefix_chars": t["prefix_chars"], "king_thought_len": t["king_thought_len"], "king_finish": t["king_finish"],
               "king_loop": t["king_loop"], "obs_kind": t["obs_kind"], "probe_model": "engy/qwen3.8-27b T0.8 4096", "probed_at": "2026-09-16"}
        f.write(json.dumps(row) + "\n"); n_side += 1
print(f"\nside-table {SIDE}: {n_side} first-divergence states of failed rollouts")
