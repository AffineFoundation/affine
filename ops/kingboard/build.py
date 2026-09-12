"""Kingboard stats builder: king vs teacher solve rates per datagen env.

Reads the public trace store on data.affine.io (R2 bucket `affine-data`):

    traces/manifest.json                -> list of immutable chunk objects
    traces/chunks/<src>-<utc>-<sha12>.jsonl.gz   one envelope per rollout

Every envelope carries the source / env id, the policy that played the
agent seat (`king_*` vs `teacher_*`), the served model label
(`king/king-<digest12>`), the env grade under `trace.rewards`, the stop
condition, the error list and the model-call timing. This script keeps one
summary row per rollout in a local sqlite file, processes only chunks it
has not seen (keyed by the manifest's chunk sha256), and writes
`state/stats.json` for the web page.

Outcome labelling mirrors `affine/affine/corpus/view.py::rollout_outcome`
(the fold's king_fail rule) so the board and D agree on what "failed"
means:

    errored   real error recorded, or stop_condition not in
              {agent_completed, max_turns}
    solved    primary grade >= 1.0   (keys tried in order:
              rewards.solved.score, rewards.correct.score,
              rewards.passed_fraction.score)
    failed    primary grade < 1.0, or no grade at the turn cap
    unscored  no numeric grade and not at the turn cap

Solve rate = solved / (solved + failed). Errored and unscored rollouts are
counted but excluded from the rate.

Run once: `python build.py` (env: DATA_R2_ACCESS_KEY_ID /
DATA_R2_SECRET_ACCESS_KEY / DATA_R2_ENDPOINT for S3 reads; without them the
public HTTPS mirror is used).
"""

from __future__ import annotations

import concurrent.futures as cf
import gzip
import hashlib
import io
import json
import logging
import math
import os
import sqlite3
import statistics
import sys
import time
import tomllib
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

try:
    import boto3
    from botocore.config import Config as BotoConfig
except ImportError:  # public HTTPS reads still work
    boto3 = None
try:
    import orjson

    def loads(b: bytes) -> dict:
        return orjson.loads(b)
except ImportError:
    def loads(b: bytes) -> dict:
        return json.loads(b)

log = logging.getLogger("kingboard.build")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STATE_DIR = Path(os.environ.get("KINGBOARD_STATE_DIR", HERE / "state"))
DB_PATH = STATE_DIR / "rollouts.sqlite"
STATS_PATH = STATE_DIR / "stats.json"
SOURCES_TOML = Path(os.environ.get(
    "KINGBOARD_SOURCES_TOML", REPO / "rollouts" / "rollouts" / "sources.toml"))
VALIDATOR_STATE = Path(os.environ.get(
    "KINGBOARD_VALIDATOR_STATE", REPO / "affine" / "state" / "state.json"))
DATA_URL = os.environ.get("KINGBOARD_DATA_URL", "https://data.affine.io").rstrip("/")
R2_BUCKET = os.environ.get("DATA_R2_BUCKET", "affine-data")
MANIFEST_KEY = "traces/manifest.json"
USER_AGENT = "affine-kingboard/0.1"
DOWNLOAD_WORKERS = int(os.environ.get("KINGBOARD_WORKERS", "8"))
TREND_BUCKETS = 14           # 24 h buckets shown in the trend
ROLLOUT_TIMEOUT_S = 3600.0   # datagen rollout wall cap (rollouts/run.py)
TIMEOUT_MARGIN_S = 100.0     # a rollout this close to the cap is a timeout
TIMEOUT_MARKER = "agent timeout"

CLEAN_STOP_CONDITIONS = {"agent_completed", "max_turns"}
TURN_CAP_STOP = "max_turns"
TURN_CAP_ARTIFACT = "rollout stopped: max_turns"
PRIMARY_REWARD_KEYS = ("solved", "correct", "passed_fraction")

# policy.harness (+ action_kind for the null harness) -> board label
HARNESS_LABELS = {
    "mini_swe_textbased": "textbased",
    "bash": "bashtool",
    "pi": "pi",
    "claude_code": "claude_code",
    "kimi_code": "kimi_code",
    "hermes_agent": "hermes_agent",
    "terminus_2": "terminus_2",
    "codex": "codex",
}

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    key TEXT PRIMARY KEY, sha256 TEXT NOT NULL, n_rollouts INTEGER,
    created_at TEXT, processed_at REAL
);
CREATE TABLE IF NOT EXISTS rollouts (
    rollout_id TEXT PRIMARY KEY, chunk TEXT NOT NULL,
    source TEXT, env_id TEXT, grp TEXT,
    policy_id TEXT, harness TEXT, seat TEXT, model_label TEXT, digest12 TEXT,
    task_uid TEXT, task_sid TEXT, repo TEXT, language TEXT,
    ts REAL, stored_at TEXT,
    outcome TEXT, score REAL, stop TEXT, n_calls INTEGER,
    error_type TEXT, wall_s REAL, timeout INTEGER
);
CREATE INDEX IF NOT EXISTS rollouts_seat_ts ON rollouts (seat, ts);
CREATE INDEX IF NOT EXISTS rollouts_digest ON rollouts (digest12);
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
"""


# -- fetching ------------------------------------------------------------------
class Fetcher:
    """Bytes for a bucket key: S3 with the DATA_R2_* key when present, else
    the public HTTPS mirror (Cloudflare rejects Python's default UA)."""

    def __init__(self) -> None:
        ak = os.environ.get("DATA_R2_ACCESS_KEY_ID")
        sk = os.environ.get("DATA_R2_SECRET_ACCESS_KEY")
        ep = os.environ.get("DATA_R2_ENDPOINT")
        self.s3 = None
        if boto3 is not None and ak and sk and ep:
            self.s3 = boto3.client(
                "s3", endpoint_url=ep, region_name="auto",
                aws_access_key_id=ak, aws_secret_access_key=sk,
                config=BotoConfig(signature_version="s3v4",
                                  retries={"max_attempts": 5, "mode": "standard"},
                                  max_pool_connections=DOWNLOAD_WORKERS + 2))
        self.mode = "s3" if self.s3 else "https"

    def get(self, key: str) -> bytes:
        if self.s3 is not None:
            return self.s3.get_object(Bucket=R2_BUCKET, Key=key)["Body"].read()
        req = urllib.request.Request(f"{DATA_URL}/{key}",
                                     headers={"User-Agent": USER_AGENT})
        last: Exception | None = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=300) as r:
                    return r.read()
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                last = e
                time.sleep(2 ** attempt)
        raise RuntimeError(f"fetch failed for {key}: {last}")


# -- envelope -> row -------------------------------------------------------------
def rollout_outcome(trace: dict) -> tuple[str, float | None]:
    """(outcome, primary score) — same rule as affine.corpus.view."""
    stop = trace.get("stop_condition")
    errors = [e for e in (trace.get("errors") or [])
              if not (stop == TURN_CAP_STOP
                      and TURN_CAP_ARTIFACT in str(e.get("message") or ""))]
    rewards = trace.get("rewards") or {}
    score = next(((rewards.get(k) or {}).get("score")
                  for k in PRIMARY_REWARD_KEYS if rewards.get(k)), None)
    value: float | None = None
    if not isinstance(score, bool) and isinstance(score, (int, float, str)):
        try:
            value = float(score)
        except (TypeError, ValueError):
            value = None
    if errors or stop not in CLEAN_STOP_CONDITIONS:
        return "errored", value
    if value is None:
        return ("failed" if stop == TURN_CAP_STOP else "unscored"), None
    return ("solved" if value >= 1.0 else "failed"), value


def error_type(trace: dict) -> str | None:
    stop = trace.get("stop_condition")
    for e in trace.get("errors") or []:
        if stop == TURN_CAP_STOP and TURN_CAP_ARTIFACT in str(e.get("message") or ""):
            continue
        return e.get("type") or e.get("error") or "unknown"
    return None


def is_timeout(trace: dict, wall: float | None) -> bool:
    """The datagen supervisor kills a rollout at ROLLOUT_TIMEOUT_S; the
    harness records it as a HarnessError "agent timeout: rollout exceeded
    its Ns budget" with stop_condition "error". The wall check catches the
    same event when the message is missing."""
    if trace.get("stop_condition") == "timeout":
        return True
    for e in trace.get("errors") or []:
        if TIMEOUT_MARKER in str(e.get("message") or ""):
            return True
    return wall is not None and wall >= ROLLOUT_TIMEOUT_S - TIMEOUT_MARGIN_S


def seat_of(policy_id: str) -> str:
    if policy_id.startswith("king_"):
        return "king"
    if policy_id.startswith("teacher_"):
        return "teacher"
    return "other"


def harness_label(policy: dict) -> str:
    h = policy.get("harness") or ""
    if h == "null":
        # The null harness plays three dialects: math-style \boxed{} answers,
        # native tool calling, and (env wave 1) the whole visible reply.
        kind = policy.get("action_kind")
        if kind == "boxed":
            return "boxed"
        if kind == "text":
            return "text"
        return "toolcall"
    return HARNESS_LABELS.get(h, h or "unknown")


def digest12_of(policy: dict) -> str | None:
    model = policy.get("model") or ""
    tail = model.rsplit("/", 1)[-1]
    if tail.startswith("king-"):
        return tail[len("king-"):]
    return None


def parse_iso(s: str | None) -> float | None:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def envelope_row(env: dict, chunk_key: str, groups: dict[str, str]) -> tuple:
    trace = env.get("trace") or {}
    task = env.get("task") or {}
    policy = env.get("policy") or {}
    timing = trace.get("timing") or {}
    agent = timing.get("agent") or {}
    # stored_at = when the finished rollout was written; that is the
    # "landed" time the recency counters and trend buckets use.
    ts = parse_iso(env.get("stored_at")) or timing.get("start") or 0.0
    wall = None
    if agent.get("start") and agent.get("end"):
        wall = float(agent["end"]) - float(agent["start"])
    else:
        ends = [ph.get("end") for ph in timing.values()
                if isinstance(ph, dict) and ph.get("end")]
        if ends and timing.get("start"):
            wall = max(ends) - float(timing["start"])
    outcome, score = rollout_outcome(trace)
    stop = trace.get("stop_condition")
    timeout = int(is_timeout(trace, wall))
    policy_id = policy.get("id") or ""
    source = env.get("source") or ""
    return (
        env.get("rollout_id") or hashlib.sha1(
            json.dumps(env.get("task"), sort_keys=True).encode()).hexdigest(),
        chunk_key, source, env.get("env_id") or "", groups.get(source, "other"),
        policy_id, harness_label(policy), seat_of(policy_id),
        policy.get("model") or "", digest12_of(policy),
        str(task.get("uid") or ""), str(task.get("sid") or ""),
        str(task.get("repo") or ""), str(task.get("language") or ""),
        float(ts), env.get("stored_at") or "",
        outcome, score, stop, len(trace.get("calls") or []),
        error_type(trace), wall, timeout,
    )


def parse_chunk(blob: bytes, chunk_key: str, groups: dict[str, str]) -> list[tuple]:
    rows = []
    with gzip.GzipFile(fileobj=io.BytesIO(blob)) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(envelope_row(loads(line), chunk_key, groups))
    return rows


# -- ingest ----------------------------------------------------------------------
def load_groups() -> dict[str, str]:
    """source name -> fold group (coding / terminal / math / ...)."""
    if not SOURCES_TOML.exists():
        return {}
    cfg = tomllib.loads(SOURCES_TOML.read_text())
    return {name: str(block.get("group") or "other")
            for name, block in (cfg.get("source") or {}).items()}


def ingest(conn: sqlite3.Connection, fetcher: Fetcher, groups: dict[str, str]) -> dict:
    raw = fetcher.get(MANIFEST_KEY)
    manifest = loads(raw)
    manifest_sha = hashlib.sha256(raw).hexdigest()
    seen = {k: s for k, s in conn.execute("SELECT key, sha256 FROM chunks")}
    todo = [c for c in manifest["chunks"] if seen.get(c["key"]) != c["sha256"]]
    log.info("manifest %s: %d chunks / %d rollouts; %d new (%s)",
             manifest_sha[:12], manifest["n_chunks"], manifest["n_rollouts"],
             len(todo), fetcher.mode)

    def work(c: dict) -> tuple[dict, list[tuple]]:
        # R2 throttles bursts of simultaneous reads (ServiceUnavailable);
        # a chunk that still fails stays unrecorded and is retried next pass.
        last: Exception | None = None
        for attempt in range(4):
            try:
                blob = fetcher.get(c["key"])
                break
            except Exception as e:  # botocore / urllib errors alike
                last = e
                time.sleep(1.5 * 2 ** attempt)
        else:
            raise RuntimeError(f"{c['key']}: {last}")
        got = hashlib.sha256(blob).hexdigest()
        if got != c["sha256"]:
            raise RuntimeError(f"{c['key']}: sha mismatch {got[:12]} != {c['sha256'][:12]}")
        return c, parse_chunk(blob, c["key"], groups)

    n_rows = 0
    failed = 0
    with cf.ThreadPoolExecutor(max_workers=DOWNLOAD_WORKERS) as pool:
        for i, fut in enumerate(cf.as_completed([pool.submit(work, c) for c in todo]), 1):
            try:
                c, rows = fut.result()
            except Exception as e:  # one bad chunk must not stop the pass
                failed += 1
                log.warning("chunk skipped: %s", e)
                continue
            with conn:
                conn.execute("DELETE FROM rollouts WHERE chunk = ?", (c["key"],))
                conn.executemany(
                    "INSERT OR REPLACE INTO rollouts VALUES (" + ",".join("?" * 23) + ")", rows)
                conn.execute(
                    "INSERT OR REPLACE INTO chunks VALUES (?,?,?,?,?)",
                    (c["key"], c["sha256"], c.get("n_rollouts"), c.get("created_at"), time.time()))
            n_rows += len(rows)
            if i % 100 == 0 or i == len(todo):
                log.info("  %d/%d chunks, %d rollouts", i, len(todo), n_rows)
    with conn:
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('manifest_sha', ?)", (manifest_sha,))
        conn.execute("INSERT OR REPLACE INTO meta VALUES ('manifest_published_at', ?)",
                     (manifest.get("published_at") or "",))
    return {"sha256": manifest_sha, "n_chunks": manifest["n_chunks"],
            "n_rollouts": manifest["n_rollouts"],
            "published_at": manifest.get("published_at"),
            "new_chunks": len(todo) - failed, "failed_chunks": failed,
            "new_rollouts": n_rows, "read_mode": fetcher.mode}


# -- stats -------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float | None, float | None, float | None]:
    if n <= 0:
        return None, None, None
    p = k / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return p, max(0.0, (centre - half) / denom), min(1.0, (centre + half) / denom)


class Agg:
    __slots__ = ("n", "solved", "failed", "errored", "unscored", "timeouts", "turns", "walls")

    def __init__(self) -> None:
        self.n = self.solved = self.failed = self.errored = self.unscored = self.timeouts = 0
        self.turns: list[int] = []
        self.walls: list[float] = []

    def add(self, outcome: str, timeout: int, n_calls: int | None, wall: float | None) -> None:
        self.n += 1
        setattr(self, outcome, getattr(self, outcome) + 1)
        self.timeouts += int(timeout or 0)
        if n_calls:
            self.turns.append(int(n_calls))
        if wall is not None and wall > 0:
            self.walls.append(float(wall))

    def merge(self, other: "Agg") -> None:
        for f in ("n", "solved", "failed", "errored", "unscored", "timeouts"):
            setattr(self, f, getattr(self, f) + getattr(other, f))
        self.turns += other.turns
        self.walls += other.walls

    def out(self) -> dict:
        graded = self.solved + self.failed
        rate, lo, hi = wilson(self.solved, graded)
        return {
            "n": self.n, "graded": graded, "solved": self.solved, "failed": self.failed,
            "errored": self.errored, "unscored": self.unscored,
            "rate": rate, "lo": lo, "hi": hi,
            "timeouts": self.timeouts,
            "timeout_rate": (self.timeouts / self.n) if self.n else None,
            "median_turns": statistics.median(self.turns) if self.turns else None,
            "median_wall_s": statistics.median(self.walls) if self.walls else None,
        }


def load_kings() -> list[dict]:
    """Reign list from the validator's state.json (current king + previous)."""
    if not VALIDATOR_STATE.exists():
        return []
    try:
        state = json.loads(VALIDATOR_STATE.read_text())
    except (OSError, ValueError):
        return []
    king = state.get("king") or {}
    out = []
    for k in [king, *(king.get("previous") or [])]:
        rev = str(k.get("revision") or "")
        if not rev:
            continue
        out.append({
            "reign": int(k.get("reign_number") or 0),
            "digest": rev, "digest12": rev[:12],
            "hotkey": k.get("hotkey") or "",
            "repo": k.get("repo") or "",
            "crowned_at": k.get("crowned_at") or "",
            "challenge_id": k.get("challenge_id") or "",
            "current": k is king,
        })
    return out


def compute_stats(conn: sqlite3.Connection, manifest_info: dict,
                  groups: dict[str, str]) -> dict:
    now = time.time()
    cur = conn.execute(
        "SELECT source, env_id, grp, harness, seat, digest12, ts, outcome, stop, "
        "n_calls, wall_s, timeout, policy_id FROM rollouts")
    rows = cur.fetchall()

    kings = load_kings()
    by_digest = {k["digest12"]: k for k in kings}

    env_ids: dict[str, str] = {}
    src_group: dict[str, str] = dict(groups)
    teacher_env: dict[str, Agg] = {}
    teacher_trend: dict[str, dict[int, list[int]]] = {}
    per_king: dict[str, dict] = {}
    stops: dict[str, int] = {}
    recent = {"king_1h": 0, "king_24h": 0, "all_1h": 0, "all_24h": 0}
    last_ts = 0.0
    n_king = n_teacher = 0

    def bucket(ts: float) -> int:
        """0 = the 24 h ending now, 1 = the day before, ..."""
        return int((now - ts) // 86400)

    for (source, env_id, grp, harness, seat, digest12, ts, outcome,
         stop, n_calls, wall, timeout, policy_id) in rows:
        env_ids.setdefault(source, env_id)
        src_group.setdefault(source, grp or "other")
        stops[stop or "none"] = stops.get(stop or "none", 0) + 1
        last_ts = max(last_ts, ts or 0.0)
        age = now - (ts or 0.0)
        if age <= 3600:
            recent["all_1h"] += 1
        if age <= 86400:
            recent["all_24h"] += 1
        if seat == "teacher":
            n_teacher += 1
            teacher_env.setdefault(source, Agg()).add(outcome, timeout, n_calls, wall)
            b = bucket(ts or 0.0)
            if 0 <= b < TREND_BUCKETS and outcome in ("solved", "failed"):
                cell = teacher_trend.setdefault(source, {}).setdefault(b, [0, 0])
                cell[0] += 1
                cell[1] += int(outcome == "solved")
            continue
        if seat != "king" or not digest12:
            continue
        n_king += 1
        k = per_king.setdefault(digest12, {
            "env": {}, "harness": {}, "env_harness": {}, "trend": {},
            "first_ts": ts, "last_ts": ts, "n": 0, "recent_1h": 0, "recent_24h": 0})
        k["n"] += 1
        k["first_ts"] = min(k["first_ts"], ts)
        k["last_ts"] = max(k["last_ts"], ts)
        if age <= 3600:
            k["recent_1h"] += 1
        if age <= 86400:
            k["recent_24h"] += 1
        k["env"].setdefault(source, Agg()).add(outcome, timeout, n_calls, wall)
        k["harness"].setdefault(harness, Agg()).add(outcome, timeout, n_calls, wall)
        k["env_harness"].setdefault((source, harness), Agg()).add(outcome, timeout, n_calls, wall)
        b = bucket(ts or 0.0)
        if 0 <= b < TREND_BUCKETS and outcome in ("solved", "failed"):
            cell = k["trend"].setdefault(source, {}).setdefault(b, [0, 0])
            cell[0] += 1
            cell[1] += int(outcome == "solved")

    current = next((k for k in kings if k["current"]), None)
    if current and current["digest12"] in per_king:
        recent["king_1h"] = per_king[current["digest12"]]["recent_1h"]
        recent["king_24h"] = per_king[current["digest12"]]["recent_24h"]

    def trend_out(tr: dict[str, dict[int, list[int]]]) -> dict:
        out = {}
        for source, cells in tr.items():
            out[source] = [
                {"bucket": b, "end": now - b * 86400, "n": c[0], "solved": c[1],
                 "rate": (c[1] / c[0]) if c[0] else None}
                for b, c in sorted(cells.items())]
        return out

    teacher_rows = {s: a.out() for s, a in teacher_env.items()}
    reigns_out = []
    for digest12, k in sorted(per_king.items(), key=lambda kv: -kv[1]["last_ts"]):
        meta = by_digest.get(digest12, {})
        env_rows = []
        for source, agg in k["env"].items():
            row = agg.out()
            row.update(source=source, env_id=env_ids.get(source, ""),
                       group=src_group.get(source, "other"))
            t = teacher_rows.get(source)
            row["teacher"] = t
            row["delta"] = (row["rate"] - t["rate"]) if (t and t["rate"] is not None
                                                          and row["rate"] is not None) else None
            row["harnesses"] = sorted({h for (s, h) in k["env_harness"] if s == source})
            env_rows.append(row)
        env_rows.sort(key=lambda r: (r["group"], r["source"]))
        harness_rows = []
        for h, agg in k["harness"].items():
            row = agg.out()
            row["harness"] = h
            harness_rows.append(row)
        harness_rows.sort(key=lambda r: -r["n"])
        eh_rows = []
        for (source, h), agg in k["env_harness"].items():
            row = agg.out()
            row.update(source=source, harness=h)
            eh_rows.append(row)
        total = Agg()
        for agg in k["env"].values():
            total.merge(agg)
        reigns_out.append({
            "digest12": digest12,
            "reign": meta.get("reign"),
            "digest": meta.get("digest", ""),
            "hotkey": meta.get("hotkey", ""),
            "crowned_at": meta.get("crowned_at", ""),
            "current": bool(meta.get("current")),
            "n_rollouts": k["n"], "first_ts": k["first_ts"], "last_ts": k["last_ts"],
            "recent_1h": k["recent_1h"], "recent_24h": k["recent_24h"],
            "total": total.out(),
            "envs": env_rows, "harnesses": harness_rows, "env_harness": eh_rows,
            "trend": trend_out(k["trend"]),
        })

    teacher_total = Agg()
    for agg in teacher_env.values():
        teacher_total.merge(agg)

    return {
        "generated_at": datetime.fromtimestamp(now, timezone.utc).isoformat(timespec="seconds"),
        "generated_ts": now,
        "manifest": manifest_info,
        "last_rollout_ts": last_ts or None,
        "counts": {"rollouts": len(rows), "king": n_king, "teacher": n_teacher},
        "recent": recent,
        "king": current,
        "kings": kings,
        "reigns": reigns_out,
        "teacher": {"envs": teacher_rows, "total": teacher_total.out(),
                    "trend": trend_out(teacher_trend)},
        "envs": [{"source": s, "env_id": env_ids.get(s, ""),
                  "group": src_group.get(s, "other")}
                 for s in sorted(env_ids)],
        "stop_conditions": stops,
        "definitions": {
            "outcome": "errored: real error or stop_condition not in {agent_completed, max_turns}; "
                       "solved: primary grade >= 1.0; failed: grade < 1.0 or ungraded at max_turns; "
                       "unscored: no numeric grade",
            "grade_keys": list(PRIMARY_REWARD_KEYS),
            "rate": "solved / (solved + failed); Wilson 95% interval",
            "timeout": f"an error message containing '{TIMEOUT_MARKER}' (the {ROLLOUT_TIMEOUT_S:.0f} s "
                       f"rollout budget) or agent wall >= {ROLLOUT_TIMEOUT_S - TIMEOUT_MARGIN_S:.0f} s; "
                       "timeouts are also counted as errored",
            "turns": "model calls in the trace (trace.calls)",
            "teacher": "policies teacher_* (Qwen/Qwen3.8-27B), all time; the old glm_* seat is not a baseline",
            "time": "rollout stored_at (when the finished rollout landed in the trace store)",
            "trend": f"{TREND_BUCKETS} rolling 24 h buckets ending at generated_at",
        },
    }


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        stream=sys.stderr)
    t0 = time.time()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    groups = load_groups()
    fetcher = Fetcher()
    manifest_info = ingest(conn, fetcher, groups)
    stats = compute_stats(conn, manifest_info, groups)
    stats["build_seconds"] = round(time.time() - t0, 1)
    tmp = STATS_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(stats, separators=(",", ":")))
    tmp.replace(STATS_PATH)
    log.info("stats.json written: %d rollouts (%d king / %d teacher), %d reigns with data, %.1fs",
             stats["counts"]["rollouts"], stats["counts"]["king"], stats["counts"]["teacher"],
             len(stats["reigns"]), stats["build_seconds"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
