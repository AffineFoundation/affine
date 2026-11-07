import os, sys, tempfile, time, json
from pathlib import Path
td = tempfile.mkdtemp(); os.environ["PE_STATE"] = td + "/state"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from safetensors.torch import save_file, load_file
import pe_apply as A

def ck(name, tensors):
    d = Path(td) / name; d.mkdir()
    save_file({k: v[:2].contiguous() for k, v in tensors.items()} if False else tensors, str(d / "model-00001-of-00001.safetensors"), metadata={"format": "pt"})
    (d / "config.json").write_text('{"a":1}')
    return d
g = torch.Generator().manual_seed(0)
def rnd(): return {"w1": torch.randn(64, 32, generator=g).to(torch.bfloat16), "w2": torch.randn(128, generator=g).to(torch.bfloat16)}
H0 = rnd()
C1 = {k: (v.float() + 1e-2 * torch.randn(v.shape, generator=g)).to(torch.bfloat16) for k, v in H0.items()}
C2 = {k: (v.float() + 1e-2 * torch.randn(v.shape, generator=g)).to(torch.bfloat16) for k, v in H0.items()}
dH0, dC1, dC2 = ck("H0", H0), ck("C1", C1), ck("C2", C2)
# (a) fresh: H0 + (C1 - H0) == C1 bitwise
r = A.merge_dirs(dH0, dC1, dH0, Path(td) / "M_fresh")
o = load_file(str(Path(td) / "M_fresh/model-00001-of-00001.safetensors"))
assert all(torch.equal(o[k], C1[k]) for k in C1), "fresh merge not bitwise == candidate"
print("a fresh bitwise ok", r["stats"])
# (b) stale: head moved H0->C1; C2 trained from H0 -> M = C1 + (C2 - H0) in fp32
r = A.merge_dirs(dC1, dC2, dH0, Path(td) / "M12", fp32_out_dir=Path(td) / "M12_fp32")
o = load_file(str(Path(td) / "M12/model-00001-of-00001.safetensors")); o32 = load_file(str(Path(td) / "M12_fp32/model-00001-of-00001.safetensors"))
for k in H0:
    ref = C1[k].float() + (C2[k].float() - H0[k].float())
    assert torch.equal(o32[k], ref) and o32[k].dtype == torch.float32
    assert torch.equal(o[k], ref.to(torch.bfloat16)) and o[k].dtype == torch.bfloat16
assert (Path(td) / "M12/config.json").exists() and not (Path(td) / "M12.tmp").exists()
print("b stale fp32 merge ok", r["stats"])
# (c) staleness chain + green-time order + overflow
A.set_genesis("H0")
assert A.staleness("H0") == 0
A.add_green(10, "C2", "H0", green_at=200)   # later green
A.add_green(9, "C1", "H0", green_at=100)    # earlier green
assert A.next_green()["candidate"] == "C1"
try:
    A.record_applied(dict(candidate="C2", epoch=10, training_head="H0", green_at=200), "fresh", 0, "C2", "H0"); raise SystemExit("order not enforced")
except RuntimeError as ex: print("c order enforced:", ex)
it = A.next_green(); kind, s = A.decide_apply(it); assert (kind, s) == ("fresh", 0)
A.record_applied(it, kind, s, "C1", "H0")
it = A.next_green(); assert it["candidate"] == "C2"; kind, s = A.decide_apply(it); assert (kind, s) == ("merge", 1), (kind, s)
A.record_applied(it, kind, s, "M12", "C1")
for i, nh in enumerate(["X3", "X4"]):
    A.add_green(20 + i, "c%d" % i, A.current_head_of(A.chain()), green_at=300 + i)
    it = A.next_green(); kind, s = A.decide_apply(it); assert kind == "fresh"; A.record_applied(it, kind, s, nh, A.current_head_of(A.chain()))
assert A.staleness("H0") == 4 and A.staleness("C1") == 3 and A.staleness("M12") == 2 and A.staleness("X4") == 0
A.add_green(30, "late", "C1", green_at=400); it = A.next_green(); assert A.decide_apply(it) == ("stale", 3); A.record_stale(it, 3)
A.add_green(31, "ok2", "M12", green_at=401); it = A.next_green(); assert A.decide_apply(it) == ("merge", 2)
A.add_green(32, "unk", "never-head", green_at=402)
assert A.decide_apply(dict(training_head="never-head")) == ("stale", None)
# (d) base-head-moved guard
try:
    A.record_applied(it, "merge", 2, "Y", "WRONG"); raise SystemExit("base guard missing")
except RuntimeError as ex: print("d base guard:", ex)
print("ALL PE_APPLY TESTS PASS")
