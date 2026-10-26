#!/usr/bin/env python3
"""Build stage launcher v829h from the signed v829g launcher (exact-site edits only).
usage: v829h_patch.py SRC_LAUNCHER OUT_LAUNCHER SECONDS"""
import sys
from pathlib import Path
src, out, seconds = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
assert seconds in (1200, 1800)
s = src.read_text()

def rep(old, new):
    global s
    assert s.count(old) == 1, old[:80]
    s = s.replace(old, new)

rep('Trainer job lifetime (14400), gate, train control, bans, TOPLOC, nonces, roster: unchanged.\n"""',
    'Trainer job lifetime (14400), gate, train control, bans, TOPLOC, nonces, roster: unchanged.\n\n'
    'v8.2.9h (Jacob 2026-10-26 13:38 CR "Cut gather to 20 minutes"): miner collection window\n'
    '(lifecycle collection_seconds, signed manifest deadline = start + collection_seconds) 1800 s ->\n'
    'COLLECTION_SECONDS. Same epoch-BOUNDARY pattern as the v8.2.4 max_tasks 128<->256 switch: CURRENT.json\n'
    'is re-stamped to COLLECTION_SECONDS only while the current epoch has no PLAN yet; an in-flight epoch\n'
    'keeps its planned window. Only values in V829H_COLLECTION_SECONDS are accepted (fail closed otherwise).\n'
    'A sibling release with COLLECTION_SECONDS = 1800 is the one-step revert. max_tasks (256), trainer,\n'
    'gate, sidecar, bans, TOPLOC, nonces, roster: unchanged.\n"""')
rep('AUDIT_JOB_SECONDS = 7200\n',
    'AUDIT_JOB_SECONDS = 7200\n'
    '# v8.2.9h (Jacob 2026-10-26): miner collection window, switched only at an epoch boundary.\n'
    'COLLECTION_SECONDS = %d\n'
    'V829H_COLLECTION_SECONDS = (1200, 1800)\n' % seconds)
rep('\ndef main():\n', r'''
def install_v829h_collection(Lifecycle):
    """v8.2.9h: boundary re-stamp of CURRENT.collection_seconds (pattern of v8.2.4 max_tasks)."""
    need(COLLECTION_SECONDS in V829H_COLLECTION_SECONDS, "v829h collection window")
    from subnet.controller import save_manifest
    prior_current = Lifecycle._current

    def _current(self):
        target = COLLECTION_SECONDS
        path = self.root / 'CURRENT.json'
        if path.is_file() and not path.is_symlink():
            current = json.loads(path.read_bytes())
            stamped = current.get('collection_seconds')
            if stamped in V829H_COLLECTION_SECONDS and stamped != target:
                planned = (self.root / 'epochs' / str(current['epoch_index']) / 'PLAN.json').exists()
                if planned:
                    self.collection_seconds = stamped              # in-flight epoch keeps its planned window
                else:
                    save_manifest(path, dict(current, collection_seconds=target))  # epoch-boundary re-stamp
                    print(json.dumps(dict(component="collection-seconds-v829h", epoch_index=current['epoch_index'],
                                          previous=stamped, collection_seconds=target)), flush=True)
                    self.collection_seconds = target
            elif stamped == target:
                self.collection_seconds = target
        return prior_current(self)

    _current.__qualname__, _current.__module__ = prior_current.__qualname__, prior_current.__module__
    Lifecycle._current = _current
    return dict(v829h=True, collection_seconds=COLLECTION_SECONDS, accepted=list(V829H_COLLECTION_SECONDS))


def main():
''')
rep("    v829_meta = install_v829(module, Lifecycle)\n",
    "    v829_meta = install_v829(module, Lifecycle)\n"
    "    v829_meta['v829h'] = install_v829h_collection(Lifecycle)\n")
out.write_text(s)
print("ok", out, seconds)
