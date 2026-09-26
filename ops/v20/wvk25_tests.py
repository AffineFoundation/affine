"""wvk-25 staged code: (1) parity of the new sdmeter against the live one at default knobs,
(2) sequential_run unit test with fakes, (3) bundle counterfactual on the last N verdicts (Qwen refs)."""
import asyncio, gzip, json, math, statistics as st, sys, importlib.util
import numpy as np
sys.path.insert(0, "/home/const/wvk25_stage/affine")
from evalsrv import sdmeter as new_sd, dueling as new_du
from affine import dialects
from affine.config import load_config
spec = importlib.util.spec_from_file_location("old_sd", "/home/const/subnet120/affine/evalsrv/sdmeter.py")
old_sd = importlib.util.module_from_spec(spec); sys.modules["old_sd"] = old_sd; spec.loader.exec_module(old_sd)

c = load_config("/home/const/wvk25_stage/affine/affine.toml"); d = c.duel
print("config:", c.weight_version_key, "seq", d.seq_enabled, d.seq_look_every, d.seq_k, d.seq_consecutive, "| sd", d.sd_meter.get("miner_empty_rule"), d.sd_meter.get("empty_gate_ratio"), d.sd_meter.get("r_cap_teacher"))

rows = [json.loads(l) for l in open("/home/const/subnet120/affine/state/history.jsonl") if l.strip()]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
vs = [r for r in rows if isinstance(r.get("verdict"), dict) and (r["verdict"].get("duel_params") or {}).get("score_mode") == "sd_min_rga"][-N:]
def kind_of(refs, tid):
    ys = [r["y"] for r in refs.get(tid, [])]
    for k in ("tool_call", "boxed", "terminus_json", "bash"):
        if ys and all(dialects.count_actions(y, k) > 0 for y in ys): return k
    return "text"

# (1) parity + (3) counterfactual
BUNDLE = {"miner_empty_rule": "drop_typ", "empty_gate_ratio": 2.0, "r_cap_teacher": True}
par_ok = 0; table = []
for r in vs:
    cid = r["challenge_id"]; v = r["verdict"]
    art = json.load(gzip.open(f"/home/const/subnet120/affine/state/evals/{cid}.json.gz")); refs = art["teacher_refs"]
    kinds = {t: kind_of(refs, t) for t in refs}; tau = v["duel_params"]["tau"]
    base = {**v["duel_params"]["sd_meter"], "shadow": True}
    o = old_sd.shadow_verdict(art["challenger_rows"], art["king_rows"], refs, kinds, tau, old_sd.settings({"sd_meter": base}), live_gates_pass=True)["by_anchor"]["loo"]
    n = new_sd.shadow_verdict(art["challenger_rows"], art["king_rows"], refs, kinds, tau, new_sd.settings({"sd_meter": base}), live_gates_pass=True)["by_anchor"]["loo"]
    par_ok += (abs(o["margin"] - n["margin"]) < 1e-12 and o["n_paired_turns"] == n["n_paired_turns"] and o["would_crown_rule_only"] == n["would_crown_rule_only"])
    res = {"live": n}
    for name, over in (("empty", {"miner_empty_rule": "drop_typ"}), ("empty+gate", {"miner_empty_rule": "drop_typ", "empty_gate_ratio": 2.0}), ("rcap", {"r_cap_teacher": True}), ("bundle", BUNDLE)):
        sh = new_sd.shadow_verdict(art["challenger_rows"], art["king_rows"], refs, kinds, tau, new_sd.settings({"sd_meter": {**base, **over}}), live_gates_pass=True)
        res[name] = sh["by_anchor"]["loo"]; res[name + "_gate"] = sh["empty_gate"]
    table.append((cid, r.get("uid"), r["event"], v["margin"], v["z"], res))
print(f"parity old vs new sdmeter at default knobs: {par_ok}/{len(vs)} identical")

print("\n| duel | uid | live (stamp) | replay live | empty rule | + gate 2× | R cap | bundle | empty share c/k/teacher | gate over c/k |")
print("|---|---|---|---|---|---|---|---|---|---|")
flips = {k: 0 for k in ("empty", "empty+gate", "rcap", "bundle")}; ctrl = {k: [] for k in ("live", "bundle")}
for cid, uid, ev, lm, lz, res in table:
    cells = []
    for k in ("live", "empty", "empty+gate", "rcap", "bundle"):
        b = res[k]; cells.append(f"{b['margin']:+.3f} z {b['z']:+.1f}{' **crown**' if b['would_crown_rule_only'] else ''}")
        if k != "live": flips[k] += b["would_crown_rule_only"] != res["live"]["would_crown_rule_only"]
    g = res["bundle_gate"]
    for k in ("live", "bundle"):
        cm = res[k].get("control_matched", {}).get("all", {})
        if cm.get("z") is not None: ctrl[k].append(cm["margin"])
    print(f"| {cid} | {uid} | {lm:+.3f} z {lz:+.1f}{' **crown**' if ev=='crowned' else ''} | " + " | ".join(cells) + f" | {g['challenger']['empty_share']:.3f}/{g['king']['empty_share']:.3f}/{g['challenger']['teacher_share']:.3f} | {g['challenger']['over_gate']}/{g['king']['over_gate']} |")
print(f"\nflips vs live: {flips}; control_matched(all) median live {st.median(ctrl['live']):+.3f} bundle {st.median(ctrl['bundle']):+.3f}")
sdd = {k: [] for k in ("live", "bundle")}
for *_, res in table:
    for k in sdd: sdd[k].append(res[k]["sd_diff"])
print(f"paired sd_diff median live {st.median(sdd['live']):.3f} bundle {st.median(sdd['bundle']):.3f}")

# (2) sequential_run unit test
class R:  # minimal DuelResult stand-in
    def __init__(self, d):
        n = len(d); self.margin = st.mean(d); self.se = st.stdev(d) / math.sqrt(n) if n > 1 else float("inf"); self.z = self.margin / self.se if self.se else 0; self.min_margin = 0.2
def run_case(diffs, seq):
    turns = [{"i": i} for i in range(len(diffs))]
    async def score_slice(batch, done_before):
        idx = [t["i"] for t in batch]
        return [{"turn_id": i, "d": 0.0} for i in idx], [{"turn_id": i, "d": diffs[i]} for i in idx]
    def decide(c_rows, k_rows):
        return R([r["d"] for r in c_rows])
    stamp = {"looks": [], "stopped_at": None, "reason": None}
    k, c, n = asyncio.run(new_du.sequential_run(turns, score_slice, decide, seq, stamp))
    return n, stamp["reason"], len(stamp["looks"])
rng = np.random.default_rng(1); seq = {"enabled": True, "look_every": 100, "k": 2.6, "consecutive": 2}
print("seq: strong win  ", run_case(list(rng.normal(0.6, 1.5, 1000)), seq))
print("seq: clear loss  ", run_case(list(rng.normal(-0.3, 1.5, 1000)), seq))
print("seq: null        ", run_case(list(rng.normal(0.0, 1.5, 1000)), seq))
print("seq: near delta  ", run_case(list(rng.normal(0.2, 1.5, 1000)), seq))
