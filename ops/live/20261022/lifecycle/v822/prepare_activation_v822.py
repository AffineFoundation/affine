#!/usr/bin/env python3
"""Build NOT-INSTALLED v8.2.2 successor activation material. On worker hosts it only adds NEW inert files
(source-final-v822 extracted from the signed A3 archive + source-registry.PREPARED-v822s.json); never edits live
configs, units, CURRENT or queue. Local outputs under activation-v822/ (0600/0700)."""
import hashlib, json, os, re, shlex, subprocess, sys
from pathlib import Path
os.umask(0o077)
D = Path('/home/const/subnet120-rewrite/state/root-audits/20261015-mimo-v8.2-release-succession')
SG = D/'release-v822-signed'; ST = D/'staged-v822-successor'; A = D/'activation-v822'; A.mkdir(mode=0o700, exist_ok=True)
UNITS = Path('/home/const/.config/systemd/user')
B = json.loads((ST/'SUCCESSOR-BINDING.json').read_bytes()); NEW = B['archive_sha256']; A821 = B['previous_archive_sha256']
TS = __import__('time').strftime('%Y%m%dT%H%M%SZ', __import__('time').gmtime())
P1 = json.loads((D/'activation-v821/ACTIVATION-PLAN.json').read_bytes())['hosts']
def h(b): return hashlib.sha256(b).hexdigest()
def sshargs(role):
    return ['-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ConnectTimeout=10', '-o', 'UserKnownHostsFile=/home/const/.lium/known_hosts/' + P1[role]['known_hosts']]
def ssh(role, cmd, stdin=None):
    return subprocess.run(['ssh', '-T', *sshargs(role), '-p', str(P1[role]['port']), 'root@' + P1[role]['host'], cmd], input=stdin, capture_output=True, timeout=300, check=True).stdout
def scp(role, local, remote):
    subprocess.run(['scp', '-q', *sshargs(role), '-P', str(P1[role]['port']), str(local), f"root@{P1[role]['host']}:{remote}"], check=True, timeout=300)
def put(p, raw, mode=0o600):
    p = Path(p); p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if p.exists(): assert p.read_bytes() == raw, 'existing differs: ' + str(p); return p
    tmp = p.with_suffix(p.suffix + '.tmp'); tmp.write_bytes(raw); os.chmod(tmp, mode); os.replace(tmp, p); return p
life = json.loads((SG/'lifecycle.v8.2.2.ACTIVE.private.json').read_bytes()); inv = life['archive_execution'][NEW]['source_files']
sys.path.insert(0, str(D/'v822-tree')); from subnet.storage import canonical, sha
assert sha(canonical(inv)) == B['source_inventory_sha256'] and inv == life['archive_execution'][A821]['source_files']
rel = ST/'source-release-v822/RELEASE.json'; tgz = ST/'source-release-v822/source.tar.gz'
assert h(tgz.read_bytes()) == NEW
plan = dict(version='mimo-v822-activation-plan-v1', archive_sha256=NEW, previous_archive_sha256=A821, hosts={})
for role, p1 in P1.items():
    base = p1['base']; hd = A/'hosts'/role; hd.mkdir(parents=True, exist_ok=True, mode=0o700)
    cur_drop = UNITS/f'affine-mimo-{role}-supervised.service.d'/'zzzz~~~v821s-owned-worker.conf'
    text = cur_drop.read_text()
    m = re.search(r'([^\s"]+/python) -I -B (\S+/owned-worker-v7\.py) --config (\S+\.json) --sha256 ([0-9a-f]{64})', text); assert m, role
    hostpy, owscript, cfg_now, sha_now = m.groups(); owpy = hostpy + ' -I -B ' + owscript
    assert cfg_now == p1['config_v821s'] and sha_now == p1['config_v821s_sha256'], (role, cfg_now)
    raw_cfg = ssh(role, 'cat ' + shlex.quote(cfg_now)); assert h(raw_cfg) == sha_now, role + ' live config digest'
    reg_now = p1['registry_v821s']
    regb = ssh(role, 'cat ' + shlex.quote(reg_now)); assert h(regb) == p1['registry_v821s_sha256'], role + ' live registry digest'
    reg = json.loads(regb); assert A821 in reg and NEW not in reg and reg[A821]['source_files'] == inv, role
    src = base + '/source-final-v822b'
    exists = ssh(role, f'test -d {shlex.quote(src)} && echo yes || echo no').strip() == b'yes'
    scp(role, Path('/tmp/verify_tree.py'), '/root/verify_tree_v822.py'); scp(role, rel, '/root/RELEASE-v822.json')
    if not exists:
        scp(role, tgz, '/root/v822-source.tar.gz')
        ssh(role, f'umask 077; mkdir {shlex.quote(src)} && tar -C {shlex.quote(src)} --no-same-owner -xzf /root/v822-source.tar.gz && rm -f /root/v822-source.tar.gz')
    out = ssh(role, f'{hostpy} -I -B /root/verify_tree_v822.py {shlex.quote(src)} /root/RELEASE-v822.json && rm -f /root/verify_tree_v822.py /root/RELEASE-v822.json').decode().strip()
    assert out.startswith('tree-ok'), (role, out)
    reg[NEW] = dict(path=src, source_files=inv)
    regraw = json.dumps(reg, sort_keys=True).encode()
    put(hd/'source-registry.qualified-v822s.json', regraw)
    P = f'{base}/source-registry.PREPARED-v822s.json'
    ssh(role, f'umask 077; cat > {P}.new && if cmp -s {P}.new {P}; then rm -f {P}.new; else (test ! -e {P} || mv {P} {P}.superseded-{TS}) && mv {P}.new {P}; fi', stdin=regraw)
    assert h(ssh(role, f'cat {P}')) == h(regraw), role
    txt = raw_cfg.decode(); oldsrc = base + '/source-final-v821b"'
    assert txt.count(oldsrc) >= 1, (role, 'source path in config')
    txt = txt.replace(oldsrc, base + '/source-final-v822b"')
    assert txt.count('source-registry.qualified-v821s.json') == 1, role
    txt = txt.replace('source-registry.qualified-v821s.json', 'source-registry.qualified-v822s.json')
    cfgraw = txt.encode(); put(hd/'owned-worker-v822s.json', cfgraw); sha_new = h(cfgraw)
    drop = text.replace(f'{cfg_now} --sha256 {sha_now}', f'{base}/owned-worker-v822s.json --sha256 {sha_new}')
    drop = re.sub(r'^# MiMo .*$', '# MiMo v8.2.2 successor: owned worker on source-final-v822b + registry qualified-v822s (v7 + v8.2 + v8.2.1 + v8.2.2).', drop, count=1, flags=re.M)
    assert drop != text and 'owned-worker-v822s.json' in drop
    put(hd/'zzzz~~~~v822s-owned-worker.conf', drop.encode())
    plan['hosts'][role] = dict(host=p1['host'], port=p1['port'], known_hosts=p1['known_hosts'], base=base, owpy=owpy, config_now=cfg_now, config_now_sha256=sha_now,
        config_v822s=base + '/owned-worker-v822s.json', config_v822s_sha256=sha_new, registry_now=reg_now, registry_now_sha256=p1['registry_v821s_sha256'],
        registry_v822s=base + '/source-registry.qualified-v822s.json', registry_v822s_sha256=h(regraw), source_v822=src, source_v821=p1['source_v821'], tree=out)
    print(json.dumps(dict(role=role, tree=out[:60], staged_source_new=not exists, config_v822s_sha256=sha_new[:12], registry_v822s_sha256=h(regraw)[:12])), flush=True)
PY = '/home/const/subnet120-rewrite/.venv/bin/python'
uidop = SG/'uid-operator.v8.2.2.ROOT-ADMITTED.private.json'
put(A/'arbos/affine-mimo-lifecycle.service.d/zzzz~~~~~~~v822s-release.conf', (
    '# MiMo v8.2.2 successor: uid launcher -> capture operator -> lifecycle.v8.2.2.ACTIVE (chain v7 -> v8.2 -> v8.2.1 -> v8.2.2).\n'
    '# Installed by activate_v822.sh; removed by rollback_v822.sh.\n[Service]\nExecStart=\n'
    f'ExecStart={PY} -B /home/const/subnet120-rewrite/state/root-audits/20261013-null-pow-v1/uid1000-host-prospective/mimo_uid_capacity_launcher.py --operator-config {uidop}\n').encode())
wd = (UNITS/'affine-live-reward-writer.service.d/zzzz~~~~mimo-v821-population-index.conf').read_text()
old = str(D/'release-v821-signed/HOST-COMPOSITION.v8.2.1-accel.ROOT-SIGNED.private.json'); assert wd.count(old) == 1, 'writer drop-in'
wd2 = re.sub(r'^# MiMo v8\.2\.1 writer', '# MiMo v8.2.2 writer', wd.replace(old, str(SG/'HOST-COMPOSITION.v8.2.2-accel.ROOT-SIGNED.private.json')), count=1, flags=re.M)
put(A/'arbos/affine-live-reward-writer.service.d/zzzz~~~~~mimo-v822-population-index.conf', wd2.encode())
put(A/'ACTIVATION-PLAN.json', (json.dumps(plan, indent=1, sort_keys=True) + '\n').encode())
files = sorted(p for p in A.rglob('*') if p.is_file() and p.name not in ('MANIFEST.sha256', 'PREFLIGHT-PASS', 'LAST-ACTIVATION'))
(A/'MANIFEST.sha256').write_text(''.join(f'{h(p.read_bytes())}  {p}\n' for p in files)); os.chmod(A/'MANIFEST.sha256', 0o600)
print(json.dumps(dict(done=True, files=len(files))))
