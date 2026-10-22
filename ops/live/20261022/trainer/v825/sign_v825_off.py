"""Add the pre-signed rollback variant TRAINER-HOST.v8.2.5-off (identical, engine.drop_unresolved=false)."""
import json, os, subprocess, sys
from pathlib import Path
ROOT_ID = '3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd'
RA = Path('/home/const/subnet120-rewrite/state/root-audits')
SRC = RA/'20261015-mimo-v8.2-release-succession/release-v822-signed/deployment-final-source-v822'
OUT = RA/'20261021-v825-drop-unresolved/release-v825-signed'
sys.path.insert(0, str(SRC))
from subnet.storage import Identity, canonical, sha
from subnet.backend_jobs import signed
from subnet.mimo_release_policy import sign
on = json.loads((OUT/'trainer-host-v825/TRAINER-HOST.v8.2.5.ROOT-ADMITTED.private.json').read_text())
signed(on['admission'], ROOT_ID)
prop = {k: v for k, v in on.items() if k != 'admission'}; prop['engine'] = dict(prop['engine'], drop_unresolved=False)
life = json.loads((RA/'20261017-v824/release-v824-signed/lifecycle.v8.2.4.ACTIVE.private.json').read_text())
actor = Identity(bytes.fromhex(Path(life['authority_seed_file']).read_text().strip())); assert actor.id == ROOT_ID
cfg = dict(prop, admission=sign(actor, dict(version=prop['version'], configuration_sha256=sha(canonical(prop)), source_unchanged=True, scope=prop['scope'])))
actor = None
assert signed(cfg['admission'], ROOT_ID)['configuration_sha256'] == sha(canonical(prop))
p = OUT/'trainer-host-v825/TRAINER-HOST.v8.2.5-off.ROOT-ADMITTED.private.json'
fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600); os.write(fd, canonical(cfg)); os.close(fd)
os.replace(OUT/'SIGNED-MANIFEST.sha256', RA/'20261021-v825-drop-unresolved/SIGNED-MANIFEST.sha256.pre-off-variant')
subprocess.run('find . -type f ! -name SIGNED-MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > SIGNED-MANIFEST.sha256 && chmod 600 SIGNED-MANIFEST.sha256', shell=True, cwd=OUT, check=True)
print(json.dumps(dict(off_variant=str(p), configuration_sha256=sha(canonical(prop)))))
