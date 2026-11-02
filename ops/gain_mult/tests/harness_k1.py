"""Dry harness: original vs patched k1_service.scores_doc on the live inputs (no writes outside tests/). + synthetic m test."""
import importlib.util, json, sys, tempfile, os
from pathlib import Path
T = Path(__file__).parent
def load(name, p):
    s = importlib.util.spec_from_file_location(name, p); m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
orig = load("k1o", "/home/const/subnet120-rewrite/state/root-audits/20261027-k1-per-uid/code/k1_service.py")
pat = load("k1p", str(T / "k1_service.patched.py"))
chain = orig.rj(orig.ST / "K1-CHAIN.json")
recs = orig.verdicts(chain)
a = orig.scores_doc(recs, chain, None); b = pat.scores_doc(recs, chain, None)
live = json.load(open("/home/const/subnet120/affine/website/scores.json"))
va = [(m["uid"], m["miner_hex"], m["share_now"]) for m in a["miners"]]; vb = [(m["uid"], m["miner_hex"], m["share_now"]) for m in b["miners"]]
vl = [(m["uid"], m["miner_hex"], m["share_now"]) for m in live["miners"]]
res = dict(identical_orig_vs_patched=va == vb, sha_orig=a["score_sha256"], sha_patched=b["score_sha256"], sha_live=live["score_sha256"],
           identical_to_live=vb == vl, shares_patched=[(u, s) for u, _, s in vb], gain_m=[(m["uid"], m.get("gain_m"), m.get("gain_settled")) for m in b["miners"]])
# synthetic: activation 560, king 567 settled m=2, an unsettled 999, an out-of-range one -> 479 gets 2/11
d = Path(tempfile.mkdtemp(dir=str(T))); (d / "code").mkdir(); (d / "state").mkdir()
cfg = json.load(open(T.parent / "code/config.json")); cfg["activation_epoch"] = 560
json.dump(cfg, open(d / "code/config.json", "w"))
json.dump(dict(kings=[dict(epoch=567, m=2.0, settled=True), dict(epoch=557, m=2.0, settled=True), dict(epoch=999, m=1.7, settled=False)]), open(d / "state/GAIN-MULT.private.json", "w"))
pat.GAINM = d
c = pat.scores_doc(recs, chain, None)
sh = {m["uid"]: m["share_now"] for m in c["miners"]}
res["synthetic"] = dict(gm=pat.gain_mults(), shares=sh, total=sum(sh.values()), uid479=sh.get(479), expect479=2/11, others=sorted(set(round(v, 12) for u, v in sh.items() if u != 479)))
json.dump(dict(kings=[dict(epoch=567, m=5.0, settled=True)]), open(d / "state/GAIN-MULT.private.json", "w"))
res["synthetic_out_of_range_gm"] = pat.gain_mults()
(d / "state/GAIN-MULT.private.json").write_text("{broken")
res["synthetic_broken_gm"] = pat.gain_mults()
import shutil; shutil.rmtree(d)
print(json.dumps(res, indent=1))
