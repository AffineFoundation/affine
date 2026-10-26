import importlib.util, json, os, sys, tempfile, time, types
from pathlib import Path
spec = importlib.util.spec_from_file_location("L", sys.argv[1]); L = importlib.util.module_from_spec(spec); spec.loader.exec_module(L)
tmp = Path(tempfile.mkdtemp()); L.PG_FLAG = tmp / "PROMOTION-GATE.json"; L.PG_STATE = tmp / "gate"; L.CONTROL_PATH = tmp / "TRAIN-CONTROL.json"
L.join_audit_population = lambda self, d: None
sys.modules['subnet'] = types.ModuleType('subnet'); bj = types.ModuleType('subnet.backend_jobs'); bj.signed = lambda o, i: dict(epoch='e'); sys.modules['subnet.backend_jobs'] = bj
H, C = 'a' * 64, 'c' * 64
calls = []
class Lc:
    identity = types.SimpleNamespace(id='x')
    def _training(self, directory, opening, dispatch, current, now):
        calls.append(directory); return dict(status='updated', checkpoint=dict(id=C), optimizer_step=8, checkpoint_authorization='authC')
L.install_v829i_promotion_gate(Lc)
cur = dict(checkpoint=dict(id=H), optimizer_step=7, checkpoint_authorization='authH')
ep = tmp / 'epochs' / '10'; ep.mkdir(parents=True); (ep / 'TRAINING-RESULT.json').write_text('{}')
lc = Lc(); ok = []
# flag absent -> pass-through
r = lc._training(ep, {}, {}, cur, 0); ok.append(r['status'] == 'updated')
L.PG_FLAG.write_text(json.dumps(dict(enabled=True)))
r = lc._training(ep, {}, {}, cur, 0); ok.append(r is None)                # hold
n = len(calls); lc._training(ep, {}, {}, cur, 0); ok.append(len(calls) == n)  # cached, no inner re-call
(L.PG_STATE / '10').mkdir(parents=True); (L.PG_STATE / '10' / 'DECISION.json').write_text(json.dumps(dict(head=H, candidate=C, decision='green', evaluated={})))
r = lc._training(ep, {}, {}, cur, 0); ok.append(r['status'] == 'updated' and r['checkpoint']['id'] == C)
(L.PG_STATE / '10' / 'DECISION.json').write_text(json.dumps(dict(head=H, candidate=C, decision='red', evaluated={})))
r = lc._training(ep, {}, {}, cur, 0); ok.append(r['status'] == 'no-update' and r['checkpoint']['id'] == H and r['optimizer_step'] == 7 and r['checkpoint_authorization'] == 'authH' and r['reason'].startswith('promotion-gate-red'))
# decision for another candidate -> hold; timeout -> promote
(L.PG_STATE / '10' / 'DECISION.json').write_text(json.dumps(dict(head=H, candidate='d' * 64, decision='red')))
ok.append(lc._training(ep, {}, {}, cur, 0) is None)
os.utime(ep / 'TRAINING-RESULT.json', (time.time() - 4000, time.time() - 4000))
r = lc._training(ep, {}, {}, cur, 0); ok.append(r['status'] == 'updated' and json.loads((ep / 'PROMOTION-GATE.json').read_text())['decision'] == 'timeout-promote')
# restore guard for e11 after red e10
(L.PG_STATE / '10' / 'DECISION.json').write_text(json.dumps(dict(head=H, candidate=C, decision='red')))
(ep / 'FINALIZED.json').write_text(json.dumps(dict(payload=dict(result=dict(reason='promotion-gate-red-e10-cccc-kept-aaaa')))))
e11 = tmp / 'epochs' / '11'; e11.mkdir(); n = len(calls)
ok.append(lc._training(e11, {}, {}, cur, 0) is None and len(calls) == n)   # waits, no job
lc._v829i_restore_wait[11] = time.time() - 5000
r = lc._training(e11, {}, {}, cur, 0); ok.append(r['status'] == 'no-update' and r['reason'] == 'promotion-gate-restore-pending')
(L.PG_STATE / '10' / 'TRAINER-RESTORED.json').write_text('{}'); e12 = tmp / 'epochs' / '12'; e12.mkdir()
# restored -> e11-like epoch trains (use fresh dir 11b semantics via new epoch 11 dir w/o marker)
import shutil; shutil.rmtree(e11); e11.mkdir(); lc._v829i_results.clear()
r = lc._training(e11, {}, {}, cur, 0); ok.append(r is None and len(calls) > n)  # inner called, now holding for its own gate
# TRAIN-CONTROL pause bypasses the restore guard
(L.PG_STATE / '10' / 'TRAINER-RESTORED.json').unlink(); L.CONTROL_PATH.write_text(json.dumps(dict(mode='pause'))); shutil.rmtree(e11); e11.mkdir(); lc._v829i_results.clear(); n = len(calls)
lc._training(e11, {}, {}, cur, 0); ok.append(len(calls) == n + 1)
print('RESULTS', ok); print('ALL PASS' if all(ok) else 'FAIL')
