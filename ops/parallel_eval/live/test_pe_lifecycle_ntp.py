"""pe_lifecycle.install_pe composed over the RA/20261107-timeout-promote ntp wrapper (outer=ntp -> v829i -> base)."""
import json, os, sys, tempfile
from pathlib import Path
td = Path(tempfile.mkdtemp()); os.environ["PE_LIFE_STATE"] = str(td / "life"); os.environ["PE_FLAG"] = str(td / "flag.json")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe_lifecycle as PL
import pe_apply as PA
def setflag(on): (td / "flag.json").write_text(json.dumps(dict(parallel_eval=dict(enabled=on))))
seen = []
class Life:
    def _training(self, directory, opening, dispatch, current, now):
        seen.append("base"); (Path(directory) / "TRAINING-JOB.json").write_text("{}")
        return dict(status="updated", checkpoint=dict(id="C"), optimizer_step=2, checkpoint_authorization=dict(input="H"))
def v829i(L):
    prior = L._training
    def _training(self, *a):
        seen.append("v829i"); r = prior(self, *a); return None if r and r["status"] == "updated" else r
    L._training = _training
def ntp(L):
    real_flag = lambda: {}
    inner = L._training
    def _training(self, *a):
        seen.append("ntp"); real_flag(); return inner(self, *a)
    L._training = _training
v829i(Life); ntp(Life); meta = PL.install_pe(Life)
assert meta["pe"] is True
cur = dict(checkpoint=dict(id="H"), optimizer_step=1)
d = td / "e" / "5"; d.mkdir(parents=True)
setflag(False); seen.clear(); assert Life()._training(d, None, None, cur, 0) is None and seen == ["ntp", "v829i", "base"], seen
setflag(True); PL._wj(td / "life/TRAINER-AT.json", dict(head="H")); (d / "TRAINING-JOB.json").unlink()
seen.clear(); r = Life()._training(d, None, None, cur, 0)
assert seen == ["base"] and r["status"] == "no-update" and r["reason"].startswith("pe-pending-e5-C"), (seen, r)
# fresh-only: staleness 0 fresh, 1-2 merge (blocked), >2 stale
os.environ["PE_STATE"] = str(td / "pa")
c = dict(genesis="H0", applied=[dict(new_head="H1"), dict(new_head="H2"), dict(new_head="H3")])
assert PA.APPLY_STALENESS_MAX == 0 and PA.STALENESS_MAX == 2
assert PA.decide_apply(dict(training_head="H3"), c) == ("fresh", 0)
assert PA.decide_apply(dict(training_head="H2"), c) == ("merge", 1)
assert PA.decide_apply(dict(training_head="H1"), c) == ("merge", 2)
assert PA.decide_apply(dict(training_head="H0"), c) == ("stale", 3)
assert PA.decide_apply(dict(training_head="X"), c) == ("stale", None)
print("ALL PE_LIFECYCLE NTP-COMPOSE + FRESH-ONLY TESTS PASS")
# from_epoch handover: an epoch before from_epoch stays on the serial outer path even with PE enabled
(td / "flag.json").write_text(json.dumps(dict(parallel_eval=dict(enabled=True, from_epoch=7))))
d6 = td / "e" / "6"; d6.mkdir(parents=True); seen.clear()
assert Life()._training(d6, None, None, cur, 0) is None and seen == ["ntp", "v829i", "base"], seen
d7 = td / "e" / "7"; d7.mkdir(parents=True); seen.clear(); PL._wj(td / "life/TRAINER-AT.json", dict(head="H"))
r = Life()._training(d7, None, None, cur, 0); assert seen == ["base"] and r["status"] == "no-update", (seen, r)
print("FROM_EPOCH HANDOVER TESTS PASS")
