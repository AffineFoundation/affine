"""Per-task prior table (PRIVATE: never publish or commit; it reveals which tasks are informative for the head).
SQLite, one row per task:
  task_id   env-prefixed id (math:5442, dapo:123)        env      affine_math | dapo
  s, n, b   effective successes / trials / cap hits (floats: old-head counts are discounted)
  head      head hash the fresh counts come from (the head the table is currently tracking)
  s_head, n_head, b_head   raw (undiscounted) counts from `head` only
  disc_from last head whose counts were discounted into s/n/b (NULL if never)    discounts  how many head changes discounted this row
  last_seen unix time of the last ingested trial
Head change (a GREEN promotes a new head): every row keeps its counts but s, n, b are multiplied by `discount` (default 0.5) and the
raw per-head counts restart at 0, so old-head evidence still informs selection but with wider uncertainty and is quickly
overwritten by new-head trials. The raw cells ledger (`cells`) keeps every ingested record per head for audit / recompute."""
import json, sqlite3, time

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks(task_id TEXT PRIMARY KEY, env TEXT NOT NULL, s REAL NOT NULL DEFAULT 0, n REAL NOT NULL DEFAULT 0,
  b REAL NOT NULL DEFAULT 0, head TEXT, s_head INTEGER NOT NULL DEFAULT 0, n_head INTEGER NOT NULL DEFAULT 0, b_head INTEGER NOT NULL DEFAULT 0,
  disc_from TEXT, discounts INTEGER NOT NULL DEFAULT 0, last_seen REAL);
CREATE TABLE IF NOT EXISTS cells(id INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT, env TEXT, head TEXT, n INTEGER, c INTEGER,
  b INTEGER, t REAL, source TEXT);
CREATE INDEX IF NOT EXISTS cells_head ON cells(head);
CREATE TABLE IF NOT EXISTS heads(head TEXT PRIMARY KEY, since REAL, discount REAL, note TEXT);
CREATE TABLE IF NOT EXISTS meta(k TEXT PRIMARY KEY, v TEXT);
"""


class PriorTable:
    def __init__(self, path, discount=0.5):
        self.db = sqlite3.connect(path, timeout=60); self.db.executescript(SCHEMA); self.discount = float(discount)

    def meta(self, k, v=None):
        if v is None:
            r = self.db.execute("SELECT v FROM meta WHERE k=?", (k,)).fetchone(); return json.loads(r[0]) if r else None
        self.db.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (k, json.dumps(v))); self.db.commit()

    def head(self):
        return self.meta("head")

    def set_head(self, head, note="", discount=None):
        """Track a new head: discount every row's effective counts, restart the raw per-head counts. Idempotent."""
        cur = self.head()
        if cur == head: return False
        d = self.discount if discount is None else float(discount)
        with self.db:
            if cur is not None:
                self.db.execute("UPDATE tasks SET s=s*?, n=n*?, b=b*?, disc_from=head, discounts=discounts+1, head=?, "
                                "s_head=0, n_head=0, b_head=0", (d, d, d, head))
            self.db.execute("INSERT OR REPLACE INTO heads VALUES(?,?,?,?)", (head, time.time(), d, note))
            self.db.execute("INSERT OR REPLACE INTO meta VALUES('head',?)", (json.dumps(head),))
        return True

    def ensure_tasks(self, task_ids_envs):
        with self.db:
            self.db.executemany("INSERT OR IGNORE INTO tasks(task_id, env, head) VALUES(?,?,?)",
                                [(t, e, self.head()) for t, e in task_ids_envs])

    def ingest(self, recs, source="sampler"):
        """recs: dicts {task_id, env, head, n, c, b, t}. A record from a head other than the tracked one counts with the discount
        (it is stale evidence); records from the tracked head add fully to s/n/b and to the raw per-head counts."""
        H = self.head(); k = 0
        with self.db:
            for r in recs:
                n = int(r.get("n") or 0)
                if n <= 0: continue
                c = int(r.get("c") or 0); b = int(r.get("b") or 0); t = float(r.get("t") or time.time())
                self.db.execute("INSERT INTO cells(task_id, env, head, n, c, b, t, source) VALUES(?,?,?,?,?,?,?,?)",
                                (r["task_id"], r["env"], r["head"], n, c, b, t, source))
                self.db.execute("INSERT OR IGNORE INTO tasks(task_id, env, head) VALUES(?,?,?)", (r["task_id"], r["env"], H))
                if r["head"] == H:
                    self.db.execute("UPDATE tasks SET s=s+?, n=n+?, b=b+?, s_head=s_head+?, n_head=n_head+?, b_head=b_head+?, "
                                    "last_seen=max(coalesce(last_seen,0),?) WHERE task_id=?", (c, n, b, c, n, b, t, r["task_id"]))
                else:
                    d = self.discount
                    self.db.execute("UPDATE tasks SET s=s+?, n=n+?, b=b+?, last_seen=max(coalesce(last_seen,0),?) WHERE task_id=?",
                                    (d * c, d * n, d * b, t, r["task_id"]))
                k += 1
        return k

    def rows(self, envs=None):
        q = "SELECT task_id, env, s, n, b, head, s_head, n_head, b_head, disc_from, discounts, last_seen FROM tasks"
        cols = ["task_id", "env", "s", "n", "b", "head", "s_head", "n_head", "b_head", "disc_from", "discounts", "last_seen"]
        out = [dict(zip(cols, r)) for r in self.db.execute(q)]
        return [r for r in out if envs is None or r["env"] in envs]

    def digest(self, envs=None):
        """sha256 of the (task_id, s, n) snapshot a selection was drawn from (recorded privately with the draw)."""
        import hashlib
        h = hashlib.sha256()
        for r in sorted(self.rows(envs), key=lambda r: r["task_id"]):
            h.update(("%s|%.6f|%.6f\n" % (r["task_id"], r["s"], r["n"])).encode())
        return h.hexdigest()
