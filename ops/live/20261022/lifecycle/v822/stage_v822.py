"""Stage the v8.2.2 successor release candidate (UNSIGNED). Reads v8.2.1 staged inputs + v822-tree; writes only NEW
files into staged-v822-successor/. Nothing signed, uploaded or activated."""
import copy, hashlib, json, os, shutil, sys, tarfile, tempfile
from pathlib import Path
D = Path('/home/const/subnet120-rewrite/state/root-audits/20261015-mimo-v8.2-release-succession')
S1 = D/'staged-v821-successor'; O = D/'staged-v822-successor'; TREE = D/'v822-tree'
MAX_TASKS = int(sys.argv[1]) if len(sys.argv) > 1 else 128
sys.path.insert(0, str(TREE))
from subnet.mimo_release import build, verify
from subnet.mimo_worker import source_inventory
from subnet.storage import canonical, sha
from subnet.mimo_grpo import TrainerConfig
digest = lambda v: sha(canonical(v)); h = lambda b: hashlib.sha256(b).hexdigest()
O.mkdir(mode=0o700)
def write(name, value=None, raw=None, private=True):
    p = O/name; assert not p.exists(), p
    raw = raw if raw is not None else (json.dumps(value, indent=1, sort_keys=True) + '\n').encode()
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600 if private else 0o644)
    with os.fdopen(fd, 'wb') as f: f.write(raw); f.flush(); os.fsync(f.fileno())
    return h(raw)
b1 = json.loads((S1/'SUCCESSOR-BINDING.json').read_bytes()); A2 = b1['archive_sha256']; assert A2.startswith('e6bba768')
rel1 = json.loads((S1/'source-release-v821/RELEASE.json').read_bytes()); names = set(rel1['source'])
REL = O/'source-release-v822'
probe = Path(tempfile.mkdtemp(prefix='v822-rel-'))/'rel'
rec = build(TREE, probe, settings=rel1['settings'])
missing = sorted(names - set(rec['source']))
if missing:
    shutil.rmtree(probe.parent); probe = Path(tempfile.mkdtemp(prefix='v822-rel-'))/'rel'
    rec = build(TREE, probe, settings=rel1['settings'], extra=missing)
assert set(rec['source']) == names, ('file set', sorted(names ^ set(rec['source']))[:10])
changed = sorted(n for n in names if rec['source'][n]['sha256'] != rel1['source'][n]['sha256'])
assert changed == ['ops/mimo_lifecycle.py'], changed
assert verify(probe) == rec
shutil.copytree(probe, REL); os.chmod(REL, 0o700); shutil.rmtree(probe.parent)
release = json.loads((REL/'RELEASE.json').read_bytes()); ARCHIVE = release['archive_sha256']
assert h((REL/'source.tar.gz').read_bytes()) == ARCHIVE and ARCHIVE != A2
SRC = O/'deployment-final-source-v822'; SRC.mkdir(mode=0o700)
with tarfile.open(REL/'source.tar.gz') as t: t.extractall(SRC, filter='data')
assert {str(p.relative_to(SRC)) for p in SRC.rglob('*') if p.is_file()} == names
for n, row in release['source'].items(): assert h((SRC/n).read_bytes()) == row['sha256'], n
inv = source_inventory(SRC.resolve()); assert digest(inv) == b1['source_inventory_sha256'] and digest(inv).startswith('d0b0475d'), digest(inv)
pol1 = json.loads((S1/'PUBLIC-POLICY.v8.2.1.UNSIGNED.private.json').read_bytes()); pol = copy.deepcopy(pol1)
old_scope = pol['scope_sha256']; assert old_scope == b1['scope_sha256']
pol['scope']['immutable_manifest']['source_bundle'] = dict(sha256=ARCHIVE, size=release['archive_bytes'])
pol['scope_sha256'] = digest(pol['scope']); pol['qualifications'] = []
diffs = [k for k in pol['scope']['immutable_manifest'] if k != 'source_bundle' and pol['scope']['immutable_manifest'][k] != pol1['scope']['immutable_manifest'][k]]
assert not diffs and [k for k in pol['scope'] if k != 'immutable_manifest' and pol['scope'][k] != pol1['scope'][k]] == [], diffs
psha = write('PUBLIC-POLICY.v8.2.2.UNSIGNED.private.json', pol)
q = json.loads((S1/'QUALIFICATIONS.v8.2.1.UNSIGNED.private.json').read_bytes())
assert q['scope_sha256'] == old_scope and q['ready'] == 7
q2 = copy.deepcopy(q); q2['scope_sha256'] = pol['scope_sha256']; q2['candidate_archive_sha256'] = ARCHIVE
for p in q2['payloads']: assert p['scope_sha256'] == old_scope; p['scope_sha256'] = pol['scope_sha256']
qsha = write('QUALIFICATIONS.v8.2.2.UNSIGNED.private.json', q2)
life1 = json.loads((S1/'lifecycle.v8.2.1.UNSIGNED.private.json').read_bytes()); life = copy.deepcopy(life1)
assert life['adapters']['job_seconds'] == 7200 and life['lifecycle']['max_tasks'] == 80 == life['trainer_config']['max_tasks']
assert life['audit_scheduler']['max_execution_retries'] == 6 and len(life['archive_execution']) == 3 and A2 in life['archive_execution']
old_key = life['adapters']['source_bundle_key']; assert A2 in old_key
life['adapters']['source_bundle_key'] = old_key.replace(A2, ARCHIVE)
assert life['adapters']['source_files'] == inv
life['lifecycle']['max_tasks'] = MAX_TASKS; life['trainer_config']['max_tasks'] = MAX_TASKS
life['archive_execution'][ARCHIVE] = dict(runtime_versions=copy.deepcopy(life['archive_execution'][A2]['runtime_versions']), source_files=inv)
life['audit_scheduler']['max_execution_retries'] = 3
TrainerConfig(**life['trainer_config']).validate()
def flat(v, p=''):
    return {p: v} if not isinstance(v, dict) else {k2: v2 for k, x in v.items() for k2, v2 in flat(x, p + '/' + k).items()}
f0, f1 = flat(life1), flat(life)
delta = sorted(k for k in set(f0) | set(f1) if f0.get(k) != f1.get(k) and not k.startswith('/archive_execution/' + ARCHIVE))
assert delta == ['/adapters/source_bundle_key', '/audit_scheduler/max_execution_retries', '/lifecycle/max_tasks', '/trainer_config/max_tasks'], delta
lsha = write('lifecycle.v8.2.2.UNSIGNED.private.json', life)
shutil.copytree(S1/'writer-accel-v821', O/'writer-accel-v822')
binding = dict(version='mimo-v822-successor-binding-v1', archive_sha256=ARCHIVE, archive_bytes=release['archive_bytes'],
    source_inventory_sha256=digest(inv), changed_files=changed, scope_sha256=pol['scope_sha256'], previous_scope_sha256=old_scope,
    previous_archive_sha256=A2, policy_file_sha256=psha, qualifications_file_sha256=qsha, lifecycle_file_sha256=lsha,
    max_tasks=MAX_TASKS, job_seconds=7200, max_execution_retries=3, lifecycle_delta=delta,
    source_bundle_key=life['adapters']['source_bundle_key'], git='release/mimo-v8.2.1-trainer-restore @ a7bc7f77',
    archive_execution=sorted(life['archive_execution']), auditable_archives=sorted([A2, ARCHIVE]))
write('SUCCESSOR-BINDING.json', binding, private=False)
print(json.dumps(binding))
