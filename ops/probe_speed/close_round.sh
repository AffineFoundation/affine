set -u
RA=/home/const/subnet120-rewrite/state/root-audits; D=$RA/20261030-probe-speed
/home/const/subnet120-rewrite/.venv/bin/python - "$1" "$2" <<'PYX'
import sys, json, os, time
sys.path.insert(0, '/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/code')
from pathlib import Path
import heldout as HO
def cr(ts=None):
    ts = time.time() if ts is None else ts; return time.strftime('%Y-%m-%d %H:%M:%S CR (UTC-6)', time.gmtime(ts - 6 * 3600))
def wj(path, obj, mode=0o600):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    t = path.with_name('.' + path.name + '.tmp'); fd = os.open(t, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
    with os.fdopen(fd, 'w') as f: f.write(json.dumps(obj, indent=1, sort_keys=True) + '\n')
    os.chmod(t, mode); os.replace(t, path)
def rj(path, default=None):
    try: return json.loads(Path(path).read_text())
    except Exception: return default
def log(**k):
    k['at_cr'] = cr(); k['by'] = 'probe-speed'
    open('/home/const/subnet120-rewrite/state/root-audits/20261030-probe-speed/rounds.log', 'a').write(json.dumps(k) + '\n')
R = HO.Rounds(Path('/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/state/heldout'), wj, rj, log, cr)
c = R.current()
if not c: print('no-open-round'); sys.exit(0)

print("closed" if R.close(c["id"], sys.argv[1], int(sys.argv[2]) or None) else "already-closed", c["id"])
PYX
