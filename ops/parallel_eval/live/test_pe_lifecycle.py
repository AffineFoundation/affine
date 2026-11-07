import json, os, sys, tempfile, time
from pathlib import Path
td = Path(tempfile.mkdtemp()); os.environ["PE_LIFE_STATE"] = str(td / "life"); os.environ["PE_FLAG"] = str(td / "flag.json")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pe_lifecycle as PL
def setflag(on): (td / "flag.json").write_text(json.dumps(dict(parallel_eval=dict(enabled=on))))
calls = []
class Life:
    def _training(self, directory, opening, dispatch, current, now):   # base: trains C from current head
        calls.append(("base", Path(directory).name))
        (Path(directory) / "TRAINING-JOB.json").write_text("{}")
        if self.next is None: return None
        return dict(status="updated", checkpoint=dict(id=self.next), optimizer_step=current["optimizer_step"] + 1,
                    checkpoint_authorization=dict(input=current["checkpoint"]["id"], id=self.next))
def v829i(Lifecycle):   # stand-in for the real v829i: holds every 'updated' for a gate decision
    prior = Lifecycle._training
    def _training(self, *a): r = prior(self, *a); return None if (r and r["status"] == "updated") else r
    Lifecycle._training = _training
v829i(Life); PL.install_pe(Life)
L = Life(); cur = dict(checkpoint=dict(id="H"), optimizer_step=10, checkpoint_authorization=None)
def ep(e): d = td / "epochs" / str(e); d.mkdir(parents=True, exist_ok=True); return d
# PE off: v829i hold semantics unchanged
setflag(False); L.next = "C1"; assert L._training(ep(1), None, None, cur, 0) is None
setflag(True)
# restore guard: trainer not known on H -> wait
assert L._training(ep(2), None, None, cur, 0) is None and (td / "life/RESTORE-REQUEST-e2.json").exists()
PL._wj(td / "life/TRAINER-AT.json", dict(head="H"))
L.next = "C2"
# trained C2 from H -> pending, head kept (no-update), no hold
r = L._training(ep(2), None, None, cur, 0); assert r["status"] == "no-update" and r["reason"].startswith("pe-pending-e2-C2"), r
p = json.loads((td / "life/pending/e2.json").read_text()); assert p["training_head"] == "H" and p["candidate"] == "C2"
# next epoch: trainer must be restored to H first
(td / "epochs/3").mkdir(parents=True, exist_ok=True); assert L._training(ep(3), None, None, cur, 0) is None
PL._wj(td / "life/TRAINER-AT.json", dict(head="H"))
# GREEN for C2 ready on base H (fresh) -> epoch 3 result = C2 with its own authorization; C3 becomes pending
PL._wj(td / "life/apply/C2.json", dict(state="ready", base_head="H", green_at=5, checkpoint=p["checkpoint"], optimizer_step=p["optimizer_step"], checkpoint_authorization=p["checkpoint_authorization"]))
L.next = "C3"; r = L._training(ep(3), None, None, cur, 0)
assert r["status"] == "updated" and r["checkpoint"]["id"] == "C2" and r["checkpoint_authorization"]["input"] == "H", r
assert json.loads((td / "life/apply/C2.json").read_text())["state"] == "offered"
# Crash/retry before FINALIZED/CURRENT: exact same GREEN remains offered, not lost.
r_again = L._training(ep(3), None, None, cur, 0)
assert r_again == r
assert PL.confirm_applied(td, dict(cur, epoch_index=3)) == []
# Even with FINALIZED present, CURRENT still on base H is NOT an acknowledgment.
PL._wj(td / "epochs/3/FINALIZED.json", dict(payload=dict(input_checkpoint="H", result=r)))
assert PL.confirm_applied(td, dict(cur, epoch_index=3)) == []
# Only durable CURRENT advanced to the exact finalized checkpoint consumes it.
newcur = dict(cur, checkpoint=r["checkpoint"], epoch_index=4)
assert len(PL.confirm_applied(td, newcur)) == 1
assert PL.confirm_applied(td, newcur) == []
assert json.loads((td / "life/apply/C2.json").read_text())["state"] == "consumed"
assert json.loads((td / "life/pending/e3.json").read_text())["training_head"] == "H"
# apply record for another base head is ignored
PL._wj(td / "life/apply/X.json", dict(state="ready", base_head="OTHER", green_at=1, checkpoint=dict(id="X"), optimizer_step=1, checkpoint_authorization={}))
cur2 = dict(cur, checkpoint=dict(id="C2")); PL._wj(td / "life/TRAINER-AT.json", dict(head="C2")); L.next = None
assert L._training(ep(4), None, None, cur2, 0) is None
assert PL.install_pe(Life).get("already") is True
print("ALL PE_LIFECYCLE (staged, durable apply crash/retry) TESTS PASS")
