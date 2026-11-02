import json, sys, tempfile, time, sqlite3, importlib.util, os
from pathlib import Path
d = Path(tempfile.mkdtemp()); cfgp = d / "c.json"
json.dump(dict(state_dir=str(d / "st"), pod_ssh="/bin/false", parent_hold=dict(enabled=True, min_tasks=3, max_s=2, cap_from_t=100)), open(cfgp, "w"))
sys.argv = ["controller.py", str(cfgp)]
sys.path.insert(0, "/home/const/subnet120-rewrite")
s = importlib.util.spec_from_file_location("ctl", sys.argv_path if hasattr(sys, "argv_path") else str(Path(__file__).parent / "controller.patched.py"))
C = importlib.util.module_from_spec(s); s.loader.exec_module(C)
class T:  # fake table with the cells schema
    def __init__(s): s.db = sqlite3.connect(":memory:"); s.db.execute("CREATE TABLE cells(task_id, env, head, n, c, b, t, source)")
t = T(); C.pull = lambda table: 0; C.log = lambda **kw: print("LOG", kw)
for i in range(2): t.db.execute("INSERT INTO cells VALUES(?,?,?,?,?,?,?,?)", ("x%d" % i, "dapo", "OLD", 4, 1, 0, 200, "sampler"))
t.db.execute("INSERT INTO cells VALUES(?,?,?,?,?,?,?,?)", ("y", "dapo", "OLD", 4, 1, 0, 50, "sampler"))  # before cap_from: not counted
r = [C.parent_hold(t, "NEW", "OLD")]                      # 2 < 3 -> hold
t.db.execute("INSERT INTO cells VALUES(?,?,?,?,?,?,?,?)", ("x9", "dapo", "OLD", 4, 1, 0, 300, "sampler"))
r.append(C.parent_hold(t, "NEW", "OLD"))                  # 3 >= 3 -> release
r.append(C.parent_hold(t, "NEW", "OLD"))                  # done -> stays released
r.append(C.parent_hold(t, "NEW2", "OLD2"))                # 0 -> hold
time.sleep(2.1); r.append(C.parent_hold(t, "NEW2", "OLD2"))  # max_s -> release
C.CFG.pop("parent_hold"); r.append(C.parent_hold(t, "NEW3", "OLD3"))   # disabled -> no hold
print("RESULT", r, "expect", [True, False, False, True, False, False], r == [True, False, False, True, False, False])
