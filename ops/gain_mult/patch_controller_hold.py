#!/usr/bin/env python3
"""Patch priors/controller.py: optional bounded parent hold (RA/20261102-gain-mult). When the gate head changes and the head the pod is
serving (the new king's parent) has fewer than parent_hold.min_tasks distinct gate-pool tasks sampled at the current cap
(cells since parent_hold.cap_from_t), keep sampling it for up to parent_hold.max_s before switching. Off unless controller.json has
"parent_hold": {"enabled": true, ...}. Idempotent (marker). usage: patch_controller_hold.py FILE"""
import sys
P = sys.argv[1]
s = open(P).read()
MARK = '# gain-mult parent hold 20261102'
if MARK in s:
    print('already patched'); sys.exit(0)

def rep(old, new):
    global s
    assert s.count(old) == 1, ('anchor count', s.count(old), old[:80])
    s = s.replace(old, new)

rep("def main():\n",
'''HOLD = {}   ''' + MARK + '''


def parent_hold(table, new, old):
    """True = keep sampling the old head (the new king's parent) this poll. Bounded by max_s per head change; any error -> no hold."""
    h = CFG.get("parent_hold") or {}
    if h.get("enabled") is not True or not old or not new or new == old: return False
    st = HOLD.setdefault(new, dict(since=time.time(), logged=0))
    if st.get("done"): return False
    pull(table)                       # PodUnreachable propagates to the normal handler
    try:
        n = table.db.execute("SELECT count(DISTINCT task_id) FROM cells WHERE head=? AND t>=?", (old, float(h["cap_from_t"]))).fetchone()[0]
    except Exception as x:
        log(event="parent_hold_error", error="%s: %s" % (type(x).__name__, x)); st["done"] = True; return False
    waited = time.time() - st["since"]
    if n >= int(h["min_tasks"]) or waited >= float(h["max_s"]):
        if st["logged"]: log(event="parent_hold_done", parent=old[:12], new_head=new[:12], parent_tasks=n, held_s=round(waited),
                             why="enough" if n >= int(h["min_tasks"]) else "max_s")
        st["done"] = True; return False
    if time.time() - st["logged"] >= 300:
        log(event="parent_hold", parent=old[:12], new_head=new[:12], parent_tasks=n, need=int(h["min_tasks"]), held_s=round(waited),
            max_s=float(h["max_s"])); st["logged"] = time.time()
    return True


def main():
''')
rep('''            if g and s is not None and g != (s or {}).get("ckpt"):''',
    '''            if g and s is not None and g != (s or {}).get("ckpt") and not parent_hold(table, g, (s or {}).get("ckpt")):''')
open(P, 'w').write(s)
print('patched')
