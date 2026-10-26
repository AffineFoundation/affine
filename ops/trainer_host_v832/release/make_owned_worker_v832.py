"""Runs ON the prod node: owned-worker-v832{,-off}.json = owned-worker-v831.json with only the host launcher/config swapped to host-v832. O_EXCL."""
import json, os, sys
POD = '/workspace/affine-mimo-trainer-prod'
s = json.load(open(POD + '/owned-worker-v831.json'))
for out, cfgname in (('owned-worker-v832.json', 'TRAINER-HOST.v8.3.2.ROOT-ADMITTED.private.json'),
                     ('owned-worker-v832-off.json', 'TRAINER-HOST.v8.3.2-off.ROOT-ADMITTED.private.json')):
    cmd = [c.replace(POD + '/host-v831/mimo_host_trainer_v83.py', POD + '/host-v832/mimo_host_trainer_v83.py')
            .replace(POD + '/host-v831/TRAINER-HOST.v8.3.1.ROOT-ADMITTED.private.json', POD + '/host-v832/' + cfgname)
           for c in s['command']]
    assert cmd != s['command'] and sum('host-v832/' in c for c in cmd) == 2 and len(cmd) == len(s['command']) and not any('host-v831' in c for c in cmd)
    p = POD + '/' + out
    if os.path.exists(p):
        print('exists', p); continue
    fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600); os.write(fd, json.dumps(dict(s, command=cmd), sort_keys=True).encode()); os.close(fd)
    print(p)
