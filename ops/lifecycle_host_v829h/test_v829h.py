"""Offline test of the v829h boundary re-stamp (tmp dir; no prod state touched)."""
import importlib.util, json, sys, tempfile
from pathlib import Path
SRC = "/home/const/subnet120-rewrite/state/root-audits/20261015-mimo-v8.2-release-succession/release-v822-signed/deployment-final-source-v822"
sys.path.insert(0, SRC)
from subnet.controller import save_manifest
ST = Path(sys.argv[1])
def load(seconds):
    spec = importlib.util.spec_from_file_location("l%d" % seconds, ST / ("s%d" % seconds) / "mimo_v829_train_gate_launcher.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
class Base:
    def __init__(self, root, cs): self.root, self.collection_seconds = Path(root), cs
    def _current(self):  # mirrors mimo_lifecycle.py:100-104 equality check
        cur = json.loads((self.root / 'CURRENT.json').read_bytes())
        assert cur['collection_seconds'] == self.collection_seconds, 'same original lifecycle release/configuration'
        return cur
res = []
for target, start in ((1200, 1800), (1800, 1200)):
    L = type("L%d" % target, (Base,), {}); load(target).install_v829h_collection(L)
    with tempfile.TemporaryDirectory() as d:
        d = Path(d); (d/'epochs/151').mkdir(parents=True)
        save_manifest(d/'CURRENT.json', dict(epoch_index=151, collection_seconds=start, max_tasks=256))
        (d/'epochs/151/PLAN.json').write_text('{}')
        lc = L(d, 1800)  # config value is always 1800 (signed lifecycle config)
        c = lc._current(); assert c['collection_seconds'] == start and lc.collection_seconds == start, 'in-flight keeps window'
        save_manifest(d/'CURRENT.json', dict(c, epoch_index=152))
        c = lc._current(); assert c['collection_seconds'] == target and lc.collection_seconds == target, 'boundary re-stamp'
        assert json.loads((d/'CURRENT.json').read_bytes())['max_tasks'] == 256
        (d/'epochs/152').mkdir(); (d/'epochs/152/PLAN.json').write_text('{}')
        c = lc._current(); assert lc.collection_seconds == target
        save_manifest(d/'CURRENT.json', dict(c, collection_seconds=999))
        try: lc._current(); raise SystemExit('should fail closed')
        except AssertionError: pass
    res.append((start, target, 'PASS'))
# fresh process restart while planned (config 1800, CURRENT 1200 stamped, PLAN exists) -> keeps 1200
L = type("Lr", (Base,), {}); load(1200).install_v829h_collection(L)
with tempfile.TemporaryDirectory() as d:
    d = Path(d); (d/'epochs/152').mkdir(parents=True); (d/'epochs/152/PLAN.json').write_text('{}')
    save_manifest(d/'CURRENT.json', dict(epoch_index=152, collection_seconds=1200, max_tasks=256))
    lc = L(d, 1800); lc._current(); assert lc.collection_seconds == 1200
res.append(('restart-planned-1200', 'PASS'))
print(json.dumps(res))
