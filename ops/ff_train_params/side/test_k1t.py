import json, os, sys, shutil, tempfile
from pathlib import Path
T = Path(tempfile.mkdtemp(prefix="tpk1t-")); os.environ["K1T_WEB"] = str(T / "web")
sys.path.insert(0, ".")
import k1_transparency as K
L = Path("/home/const/subnet120-rewrite/state/mimo-mainnet-v1/epochs")
e = max(int(p.name) for p in L.iterdir() if p.name.isdigit() and (p / "TRAIN-DISPATCH-FF.json").exists() and (p / "TRAINING-RESULT.json").exists())
src = L / str(e); ed = T / "ed"; ed.mkdir()
for n in ("TRAIN-DISPATCH-FF.json", "TRAINING-RESULT.json", "K1-TURN.json"): shutil.copy(src / n, ed / n)
shutil.copytree(src / "ff", ed / "ff")
env = json.loads((ed / "TRAIN-DISPATCH-FF.json").read_text()); (env.get("payload") or env)["train"] = dict(lr=8e-7, kl=0.01, given=["kl", "lr"], defaults=dict(lr=5e-7, kl=0.02))
(ed / "TRAIN-DISPATCH-FF.json").write_text(json.dumps(env))
res = json.loads((ed / "TRAINING-RESULT.json").read_text()); r = res.get("payload", res)
r["report"]["training"]["train_params"] = dict(version="v833j-turn-train-params-v1", lr=8e-7, kl=0.01)
(ed / "TRAINING-RESULT.json").write_text(json.dumps(res))
turn = json.loads((ed / "K1-TURN.json").read_text()); turn["train"] = dict(lr=8e-7, kl=0.01); turn["train_given"] = ["kl", "lr"]; turn.setdefault("format", "affine-batch-v2")
for verdict in ("green", "red"):
    dec = dict(decision=verdict, candidate=None)
    out = T / ("turn-%s.json" % verdict)
    ok = K.publish_ff_turn(e, dec, turn, ed, out)
    s = out.read_text(); d = json.loads(s)
    print(verdict, ok, "train_params" in d["training"], d["turn"].get("train"), d["dispatch"].get("train"), "8e-07" in s, "0.01," in s or "0.01}" in s)
shutil.rmtree(T)
