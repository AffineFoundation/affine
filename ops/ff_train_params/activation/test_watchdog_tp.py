import json, tempfile, os
from pathlib import Path
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'watchdog_tp.py')).read().split("log(f'armed")[0]
T = Path(tempfile.mkdtemp()); os.environ['FROM'] = '1'
g = {'__name__': 'wd'}; exec(src, g); g['AV'] = T; g['D'] = T
def mk(train, tp, raw=0.0224, um=0.0224, dir_='normgrad', status='updated'):
    d = Path(tempfile.mkdtemp(dir=T))
    (d/'TRAIN-DISPATCH-FF.json').write_text(json.dumps(dict(payload=dict(train=train) if train else {})))
    tr = dict(status=status, objective_loss=0.1, preclip_gradient_norm=3.0, optimizer=dict(sgd=dict(raw_update_norm=raw, sgd_direction=dir_, fp32_master=dict(update_norm_master=um))))
    if tp: tr['train_params'] = tp
    (d/'TRAINING-RESULT.json').write_text(json.dumps(dict(payload=dict(report=dict(training=tr)))))
    return g['check_result'](1, d)
dflt = dict(lr=5e-7, kl=0.02, given=[], defaults=dict(lr=5e-7, kl=0.02))
ok_tp = dict(lr=5e-7, kl=0.02, lr_multiplier=1.0, kl_applied=0.02, defaults_used=True, step=dict(path='v833i-exact', raw_update_norm=0.0224, raw_update_norm_default=0.0224))
cases = [
 ('flag off / legacy, ok', mk(None, None), None),
 ('v1 default ok', mk(dflt, ok_tp), None),
 ('default but scaled path', mk(dflt, dict(ok_tp, step=dict(ok_tp['step'], path='scaled'))), 'x'),
 ('default but norm off band', mk(dflt, dict(ok_tp, step=dict(ok_tp['step'], raw_update_norm=0.05)), raw=0.05, um=0.05), 'x'),
 ('default master off raw', mk(dflt, ok_tp, um=0.0230), 'x'),
 ('NaN', mk(dflt, ok_tp, um=float('nan')), 'x'),
 ('sign dir', mk(dflt, ok_tp, dir_='sign'), 'x'),
 ('train but no report', mk(dflt, None), 'x'),
 ('m=0.5 ok', mk(dict(lr=2.5e-7, kl=0.05), dict(lr=2.5e-7, kl=0.05, lr_multiplier=0.5, kl_applied=0.05, step=dict(path='scaled', raw_update_norm=0.0112, raw_update_norm_default=0.0224)), um=0.0112), None),
 ('m=0.5 wrong raw', mk(dict(lr=2.5e-7, kl=0.05), dict(lr=2.5e-7, kl=0.05, lr_multiplier=0.5, kl_applied=0.05, step=dict(path='scaled', raw_update_norm=0.02, raw_update_norm_default=0.0224)), um=0.02), 'x'),
 ('m=2 ok', mk(dict(lr=1e-6, kl=0.02), dict(lr=1e-6, kl=0.02, lr_multiplier=2.0, kl_applied=0.02, step=dict(path='scaled', raw_update_norm=0.0240, raw_update_norm_default=0.0224)), um=0.0240), None),
 ('m=2 above m*raw', mk(dict(lr=1e-6, kl=0.02), dict(lr=1e-6, kl=0.02, lr_multiplier=2.0, kl_applied=0.02, step=dict(path='scaled', raw_update_norm=0.05, raw_update_norm_default=0.0224)), um=0.05), 'x'),
 ('kl mismatch', mk(dict(lr=1e-6, kl=0.05), dict(lr=1e-6, kl=0.05, lr_multiplier=2.0, kl_applied=0.02, step=dict(path='scaled', raw_update_norm=0.024, raw_update_norm_default=0.0224)), um=0.024), 'x'),
 ('kl only ok', mk(dict(lr=5e-7, kl=0.1), dict(lr=5e-7, kl=0.1, lr_multiplier=1.0, kl_applied=0.1, defaults_used=False, step=dict(path='v833i-exact', raw_update_norm=0.0224, raw_update_norm_default=0.0224))), None),
 ('not updated', mk(dflt, None, status='rejected'), None),
]
bad = 0
for name, got, want in cases:
    ok = (got is None) == (want is None); bad += not ok
    print('PASS' if ok else 'FAIL', name, '->', got)
print('ALL PASS' if not bad else '%d FAIL' % bad)
