import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, sys.argv[1]); import flip_scorer as F
H, C1, C2, C3 = 'h'*64, 'a'*64, 'b'*64, 'c'*64
# e10 head H probed at e10. e10 RED (candidate C1 discarded), e11 RED (C2 discarded), e12 GREEN (H->C3). C3 probed at e13.
lineage = {10: dict(input=H, output=H, auth_input='p', trained=False), 11: dict(input=H, output=H, auth_input='p', trained=False),
           12: dict(input=H, output=C3, auth_input=H, trained=True)}
ok = []
e, why = F.credit_epoch_for_pair(10, 13, H, lineage); ok.append(e == 12); print('pair 10->13 credit', e, why)
# rollback composition: probes e5 (X) -> e13 (C3) where H was a rollback target probed at e3 and e5..e9 differ
lin2 = dict(lineage); lin2.update({5: dict(input='x'*64, output=H, auth_input='q', trained=False)})
ckpt_of = {3: H, 5: 'x'*64, 13: C3}
e1, why1 = F.credit_epoch_for_pair(5, 13, 'x'*64, lin2); ok.append(e1 is None)
e2, par = F.rollback_parent_for_pair(5, 13, ckpt_of, lin2, {3, 5, 13}); ok.append((e2, par) == (12, 3)); print('rollback compose', e2, par)
# explicit red guard
d = Path(tempfile.mkdtemp()); (d / '12').mkdir(); (d / '12' / 'PROMOTION-GATE.json').write_text(json.dumps(dict(decision='red')))
ok.append(F.gate_decision(d, 12) == 'red' and F.gate_decision(d, 11) is None)
print(ok, 'ALL PASS' if all(ok) else 'FAIL')
