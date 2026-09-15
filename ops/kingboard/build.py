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

    errored   real error recorded (the refused-call artifact of ACP
              harnesses at max_turns / loop_guard is not one), or
              stop_condition not in {agent_completed, max_turns,
              loop_guard, no_visible_reply}
    failed    stop_condition no_visible_reply (the model said nothing),
              whatever the grade
    solved    primary grade >= 1.0   (keys tried in order:
              rewards.solved.score, rewards.correct.score,
              rewards.passed_fraction.score)
    failed    primary grade < 1.0, or no grade at the turn cap / loop guard
    unscored  no numeric grade otherwise

(2026-09-13: loop_guard and no_visible_reply were counted as errored here
while the fold graded them; reign 12's 85 affine_agent loop-guard stops,
72 of them graded solved, showed as errors.)

Per row the board also keeps the loop-guard flag (`loop rate` = share of
rollouts the pod's loop guard cut short; loop ONSETS need the baked
conversations and are not computed here) and the policy's sampling
temperature (`policy.temperature`, stamped since 2026-09-13) for the
greedy (T = 0, `king_*_greedy`) vs sampled (T = 0.8) split.

Solve rate = solved / (solved + failed). Errored and unscored rollouts are
counted but excluded from the rate.

The same pass also writes `state/matrix.json` (served as /api/matrix): one
row per model in reign order (teacher, genesis, every crowned king incl.
revoked reigns), one column per held-out benchmark (benchsuite cards in
affine/state/benchsuite/) and per datagen environment, value = average
score 0-100. See build_matrix().

Run once: `python build.py` (env: DATA_R2_ACCESS_KEY_ID /
DATA_R2_SECRET_ACCESS_KEY / DATA_R2_ENDPOINT for S3 reads; without them the
public HTTPS mirror is used). `python build.py --matrix-only` rebuilds the
matrix from the existing stats.json without touching the trace store.
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
import re
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
try:
    import pyarrow.parquet as pq   # corpus index (dataset table); optional
except ImportError:
    pq = None

log = logging.getLogger("kingboard.build")

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STATE_DIR = Path(os.environ.get("KINGBOARD_STATE_DIR", HERE / "state"))
DB_PATH = STATE_DIR / "rollouts.sqlite"
STATS_PATH = STATE_DIR / "stats.json"
MATRIX_PATH = STATE_DIR / "matrix.json"
DATASET_TABLE_PATH = STATE_DIR / "dataset_table.json"
INDEX_STATS_PATH = STATE_DIR / "index_stats.json"     # per (group, source) turns / strata of D, keyed by index sha
CORPUS_MANIFEST_KEY = "corpus/manifest.json"
FOLD_STATS_KEY = "corpus/fold_stats.json"
CURRICULUM_KEY = "curriculum/latest.json"
SOURCES_TOML = Path(os.environ.get(
    "KINGBOARD_SOURCES_TOML", REPO / "rollouts" / "rollouts" / "sources.toml"))
VALIDATOR_STATE = Path(os.environ.get(
    "KINGBOARD_VALIDATOR_STATE", REPO / "affine" / "state" / "state.json"))
VALIDATOR_HISTORY = Path(os.environ.get(
    "KINGBOARD_VALIDATOR_HISTORY", REPO / "affine" / "state" / "history.jsonl"))
BENCHSUITE_DIR = Path(os.environ.get(
    "BENCHSUITE_STATE_DIR", REPO / "affine" / "state" / "benchsuite"))
# ops/benchsuite pass logs: `pass-<run_id>.log` without a `.exit` sibling and
# written recently = a benchmark pass in flight (cells shown as running).
BENCHSUITE_RUNS_DIR = Path(os.environ.get("BENCHSUITE_RUNS_DIR", REPO / "ops" / "benchsuite" / "state"))
BENCHSUITE_SUITE_TOML = Path(os.environ.get("BENCHSUITE_SUITE_TOML", REPO / "ops" / "benchsuite" / "suite.toml"))
CHAT_SET_MODES = {"lium", "prime", "full", "full-king-only", "genesis", "cheap", "challenger"}
INFLIGHT_STALE_S = 2 * 3600      # a pass log untouched this long is dead, not running (a sandbox cell can be silent ~1 h)
DATA_URL = os.environ.get("KINGBOARD_DATA_URL", "https://data.affine.io").rstrip("/")
R2_BUCKET = os.environ.get("DATA_R2_BUCKET", "affine-data")
MANIFEST_KEY = "traces/manifest.json"
# coverage backfill rollouts (operator directive 2026-09-15): same envelope
# schema, own prefix so the fold (which reads traces/manifest.json) never sees them
BACKFILL_MANIFEST_KEY = "traces-backfill/manifest.json"
USER_AGENT = "affine-kingboard/0.1"
DOWNLOAD_WORKERS = int(os.environ.get("KINGBOARD_WORKERS", "8"))
TREND_BUCKETS = 14           # 24 h buckets shown in the trend
ROLLOUT_TIMEOUT_S = 3600.0   # datagen rollout wall cap (rollouts/run.py)
TIMEOUT_MARGIN_S = 100.0     # a rollout this close to the cap is a timeout
TIMEOUT_MARKER = "agent timeout"

TURN_CAP_STOP = "max_turns"
LOOP_GUARD_STOP = "loop_guard"                # rollouts/loopguard.py
NO_VISIBLE_REPLY_STOP = "no_visible_reply"    # pi adapter: reasoning only
CLEAN_STOP_CONDITIONS = {"agent_completed", TURN_CAP_STOP,
                         LOOP_GUARD_STOP, NO_VISIBLE_REPLY_STOP}
# ACP harnesses raise interception's refusal past the cap / on the loop
# guard as their own error ("rollout stopped: <stop>"): not a real error.
REFUSAL_STOPS = {TURN_CAP_STOP, LOOP_GUARD_STOP}
PRIMARY_REWARD_KEYS = ("solved", "correct", "passed_fraction")
# Bump when a row's derivation changes: every chunk is re-read on mismatch.
SCHEMA_VERSION = "2"
GREEDY = "greedy"      # policy.temperature == 0 (king_*_greedy)
SAMPLED = "sampled"    # T > 0, or unstamped rows (all sampled at 0.8 before 2026-09-13)

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

# -- matrix (api/matrix) -----------------------------------------------------------
TEACHER_MODEL = "Qwen/Qwen3.8-27B"
GENESIS_MODEL = "Qwen/Qwen3.6-35B-A3B"
# Reference models: open checkpoints of the genesis family benchmarked for
# comparison (cards with king.label and no reign / duel; not kings, never
# paid). Rendered as a reference row just above Genesis. Notes per label.
REFERENCE_NOTES = {
    "occamy-1.0": "Accio-Lab/occamy-1.0 — Alibaba, Qwen3.6-35B-A3B post-train, admissible reference; not a king",
}
GENESIS_DIGEST12 = "995ad96eacd9"     # HF revision 995ad96e… = reign 0 (seed)
# Held-out benchmarks in display order (benchsuite `[[envs]].id` -> label).
# Cards may carry more ids; unknown ones are appended in card order.
BENCH_COLUMNS = [
    ("mmlu-pro", "MMLU-Pro"), ("math500", "MATH-500"), ("gpqa-diamond", "GPQA"),
    ("aime25", "AIME25"), ("ifbench", "IFBench"), ("ifeval", "IFEval"),
    ("humaneval", "HumanEval"), ("livecodebench", "LiveCodeBench"),
    ("bfcl-v3", "BFCL v3"), ("when2call", "When2Call"),
    ("swebench-verified", "SWE-bench Verified"), ("minif2f", "miniF2F"),
    ("graphwalks", "GraphWalks"), ("mrcr-v2", "MRCR"), ("oolong-synth", "Oolong"),
    # agentic set (benchsuite docs §11, 2026-09-15): its own column group, last
    ("terminal-bench-2", "Terminal-Bench 2"), ("tau2-airline", "τ²-bench airline"),
    ("tau2-retail", "τ²-bench retail"), ("tau2-telecom", "τ²-bench telecom"),
    ("tau3-banking", "τ³-bench banking"), ("swebench-pro", "SWE-bench Pro"),
]
# Benchmarks scored on the rollouts that finished inside the time / context
# budget (card `finished_only`), per the 2026-09-15 directive.
BENCH_FINISHED_ONLY = {"swebench-verified", "swebench-pro"}
BENCH_AGENTIC_GROUP = "agentic"
CAP_BOUND_FRAC = 0.20            # share of replies cut at the completion cap that flags a cell
BENCH_TEMPERATURE = 0.0          # the primary (greedy) card row
# Card modes that are not a king / genesis / teacher measurement.
BENCH_SKIP_MODES = {"challenger", "comparables"}
BENCH_SKIP_STATUS = {"skipped_identical_weights"}
MATRIX_MIN_GRADED = 5            # datagen env cell needs this many graded rollouts
# Kings before this reign are never rows (operator 2026-09-15 20:31 UTC:
# Affine-XI and older are not backfilled); they are listed under `hidden`.
MATRIX_MIN_REIGN = 11
MATRIX_LOW_N = 30                # below this the cell is flagged `low_n` (operator 2026-09-15: an inconsistency)
# Environments scored on the SAMPLED rollouts only (T = 0.8, the standard
# king-seat / teacher sampling). The teacher seat never runs greedy, so a
# king cell pooled over its greedy (T = 0) king_*_greedy rollouts was not
# comparable with the teacher's (2026-09-15 audit: up to 6 pt on swesmith).
MATRIX_ENV_TEMP = SAMPLED
# An env whose rollouts carry a numeric grade less often than this has no
# grader (affine_wiki: 0 of 288 teacher rollouts graded); no row gets a cell.
NO_GRADER_SHARE = 0.05
MATRIX_GROUP_ORDER = ["coding", "terminal", "math", "tool_use", "nl2repo", "general", "agent", "other"]
# Short header labels for the compact matrix (full names travel in `label`).
BENCH_ABBR = {
    "mmlu-pro": "MMLU", "math500": "M500", "gpqa-diamond": "GPQA", "aime25": "AIME",
    "ifbench": "IFB", "ifeval": "IFE", "humaneval": "HE", "livecodebench": "LCB",
    "bfcl-v3": "BFCL", "when2call": "W2C", "swebench-verified": "SWE", "minif2f": "F2F",
    "terminal-bench-2": "TB2", "tau2-airline": "T2A", "tau2-retail": "T2R", "tau2-telecom": "T2T",
    "tau3-banking": "T3B", "swebench-pro": "SWEP",
    "graphwalks": "GW", "mrcr-v2": "MRCR", "oolong-synth": "OOL",
}
# Horizontal header labels for the benchmark table (<= 10 chars).
BENCH_SHORT = {
    "mmlu-pro": "MMLU-Pro", "math500": "MATH-500", "gpqa-diamond": "GPQA", "aime25": "AIME25",
    "ifbench": "IFBench", "ifeval": "IFEval", "humaneval": "HumanEval", "livecodebench": "LCB",
    "bfcl-v3": "BFCL v3", "when2call": "When2Call", "swebench-verified": "SWE-bench",
    "minif2f": "miniF2F", "graphwalks": "GraphWalks", "mrcr-v2": "MRCR", "oolong-synth": "Oolong",
    "terminal-bench-2": "TB2", "tau2-airline": "τ² airline", "tau2-retail": "τ² retail",
    "tau2-telecom": "τ² telecom", "tau3-banking": "τ³ banking", "swebench-pro": "SWE-Pro",
}
GROUP_ABBR = {"coding": "code", "terminal": "term", "math": "math", "tool_use": "tool",
              "nl2repo": "nl2r", "general": "gen", "agent": "agent", "other": "other"}
# Fold groups that are not backed by a datagen source (routed from king /
# teacher rollouts by the fold): extra column groups of the dataset table.
KING_GROUP_ORDER = ["king_fail", "king_loop_onset", "king_pivot", "king_done", "king_tooluse",
                    "king_recoverable", "completion_pre", "completion", "king_coached"]
KING_GROUP_ABBR = {"king_fail": "KFAIL", "king_loop_onset": "KLOOP", "king_pivot": "KPIV",
                   "king_done": "KDONE", "king_tooluse": "KTOOL", "king_recoverable": "KREC",
                   "completion_pre": "CPRE", "completion": "COMPL", "king_coached": "KCOACH"}
# Datagen sources -> 3-5 char headers (operator list 2026-09-15); unknown
# sources fall back to the first 4 letters after `affine_`, upper-cased.
ENV_ABBR = {
    "multiswe": "MSWE", "r2e_gym": "R2E", "scaleswe": "SCSW", "swelego": "SWLG",
    "swerebench_v2": "SWRB", "swesmith": "SWSM", "affine_tmax": "TMAX",
    "terminal_bench_2": "TMB", "terminal_lego": "TML", "affine_i3math": "I3M",
    "affine_math": "MATH", "affine_agent": "AGNT", "affine_notool": "NOTL",
    "affine_when2call": "W2CT", "affine_wiki": "WIKI", "affine_nl2lib": "NL2L",
    "nl2repobench": "NL2R", "affine_autobench": "AUTB", "affine_deshuffle": "DSHF",
    "affine_eog": "EOG", "affine_i3code": "I3C", "affine_ifeval": "IFEV",
    "affine_logic": "LGC", "affine_numina": "NUMI", "affine_oolong": "OOLG",
    "affine_prolog": "PRLG", "affine_pydantic": "PYD", "affine_rcore": "RCOR",
    "affine_rgym": "RGYM", "affine_science": "SCI", "affine_sql": "SQL",
    "affine_trivia": "TRIV", "affine_unscramble": "UNSC", "affine_uuidctf": "UUID",
    "affine_verbatim": "VERB", "affine_wikispeedia": "WKSP",
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
    error_type TEXT, wall_s REAL, timeout INTEGER, temperature REAL,
    backfill INTEGER DEFAULT 0
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
def is_refusal_artifact(err: dict, stop: str | None) -> bool:
    """affine.corpus.trace.is_turn_cap_artifact: the harness surfacing the
    refused model call past max_turns / on the loop guard."""
    return stop in REFUSAL_STOPS and f"rollout stopped: {stop}" in str(err.get("message") or "")


def real_errors(trace: dict) -> list[dict]:
    stop = trace.get("stop_condition")
    return [e for e in (trace.get("errors") or []) if not is_refusal_artifact(e, stop)]


def rollout_outcome(trace: dict) -> tuple[str, float | None]:
    """(outcome, primary score) — same rule as affine.corpus.view."""
    stop = trace.get("stop_condition")
    rewards = trace.get("rewards") or {}
    score = next(((rewards.get(k) or {}).get("score")
                  for k in PRIMARY_REWARD_KEYS if rewards.get(k)), None)
    value: float | None = None
    if not isinstance(score, bool) and isinstance(score, (int, float, str)):
        try:
            value = float(score)
        except (TypeError, ValueError):
            value = None
    if real_errors(trace) or stop not in CLEAN_STOP_CONDITIONS:
        return "errored", value
    if stop == NO_VISIBLE_REPLY_STOP:
        return "failed", value
    if value is None:
        return ("failed" if stop in REFUSAL_STOPS else "unscored"), None
    return ("solved" if value >= 1.0 else "failed"), value


def error_type(trace: dict) -> str | None:
    errors = real_errors(trace)
    if not errors:
        return None
    return errors[0].get("type") or errors[0].get("error") or "unknown"


def temperature_of(policy: dict) -> float | None:
    t = policy.get("temperature")
    if isinstance(t, bool) or not isinstance(t, (int, float)):
        return None
    return float(t)


def temp_class(temperature: float | None) -> str:
    return GREEDY if temperature is not None and temperature <= 0.0 else SAMPLED


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


BACKFILL_PREFIX = "backfill_"      # coverage backfill rollouts (not the live king seat)
BACKFILL_TEACHER = "backfill_teacher_"
_HEX12 = re.compile(r"^[0-9a-f]{12}$")


def backfill_digest12(policy_id: str) -> str | None:
    """`backfill_<digest12>_<harness>` -> digest12 (None for the teacher /
    a non-backfill id). The coverage backfill (operator directive
    2026-09-15) replays every datagen env for a model that the live king
    seat never served (genesis, past kings) and stamps its policies this
    way; the fold ignores the prefix, the board scores it like the seat."""
    if not policy_id.startswith(BACKFILL_PREFIX) or policy_id.startswith(BACKFILL_TEACHER):
        return None
    tail = policy_id[len(BACKFILL_PREFIX):]
    d12 = tail.split("_", 1)[0]
    return d12 if _HEX12.match(d12) else None


def seat_of(policy_id: str) -> str:
    if policy_id.startswith("king_"):
        return "king"
    if policy_id.startswith("teacher_") or policy_id.startswith(BACKFILL_TEACHER):
        return "teacher"
    if backfill_digest12(policy_id):
        return "king"
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
    if tail.startswith("king-") and _HEX12.match(tail[len("king-"):][:12]):
        return tail[len("king-"):][:12]
    return backfill_digest12(policy.get("id") or "")


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
        error_type(trace), wall, timeout, temperature_of(policy),
        int(policy_id.startswith(BACKFILL_PREFIX)),
    )


ROW_WIDTH = 25   # columns of the rollouts table / envelope_row tuple


def parse_chunk(blob: bytes, chunk_key: str, groups: dict[str, str]) -> list[tuple]:
    rows = []
    with gzip.GzipFile(fileobj=io.BytesIO(blob)) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(envelope_row(loads(line), chunk_key, groups))
    return rows


# -- ingest ----------------------------------------------------------------------
def migrate(conn: sqlite3.Connection) -> None:
    """Add columns newer rows carry and, when the row derivation changed
    (SCHEMA_VERSION), forget every processed chunk so the next ingest
    re-reads the whole store with the current rules."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(rollouts)")}
    with conn:
        if "temperature" not in cols:
            conn.execute("ALTER TABLE rollouts ADD COLUMN temperature REAL")
        if "backfill" not in cols:
            # no backfill rollout existed before the column: 0 for every stored row is exact
            conn.execute("ALTER TABLE rollouts ADD COLUMN backfill INTEGER DEFAULT 0")
        row = conn.execute("SELECT v FROM meta WHERE k = 'schema_version'").fetchone()
        if (row[0] if row else None) != SCHEMA_VERSION:
            log.info("row schema %s -> %s: re-reading every chunk",
                     row[0] if row else None, SCHEMA_VERSION)
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM rollouts")
            conn.execute("INSERT OR REPLACE INTO meta VALUES ('schema_version', ?)",
                         (SCHEMA_VERSION,))


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
    # coverage backfill traces live under their own prefix (the fold never
    # reads it); absent until the first backfill chunk lands
    backfill: dict | None = None
    try:
        braw = fetcher.get(BACKFILL_MANIFEST_KEY)
        backfill = loads(braw)
        btodo = [c for c in backfill["chunks"] if seen.get(c["key"]) != c["sha256"]]
        log.info("backfill manifest: %d chunks / %d rollouts; %d new",
                 backfill.get("n_chunks", len(backfill["chunks"])), backfill.get("n_rollouts", 0), len(btodo))
        todo += btodo
    except Exception as e:  # 404 / not published yet / transient
        log.info("no backfill manifest (%s)", str(e)[:80])

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
                    "INSERT OR REPLACE INTO rollouts VALUES (" + ",".join("?" * ROW_WIDTH) + ")", rows)
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
            "new_rollouts": n_rows, "read_mode": fetcher.mode,
            "backfill": ({"n_chunks": backfill.get("n_chunks", len(backfill["chunks"])),
                          "n_rollouts": backfill.get("n_rollouts"),
                          "published_at": backfill.get("published_at")} if backfill else None)}


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
    __slots__ = ("n", "solved", "failed", "errored", "unscored", "timeouts",
                 "loop_guards", "turns", "walls")

    def __init__(self) -> None:
        self.n = self.solved = self.failed = self.errored = self.unscored = self.timeouts = 0
        self.loop_guards = 0
        self.turns: list[int] = []
        self.walls: list[float] = []

    def add(self, outcome: str, timeout: int, n_calls: int | None, wall: float | None,
            stop: str | None = None) -> None:
        self.n += 1
        setattr(self, outcome, getattr(self, outcome) + 1)
        self.timeouts += int(timeout or 0)
        self.loop_guards += int(stop == LOOP_GUARD_STOP)
        if n_calls:
            self.turns.append(int(n_calls))
        if wall is not None and wall > 0:
            self.walls.append(float(wall))

    def merge(self, other: "Agg") -> None:
        for f in ("n", "solved", "failed", "errored", "unscored", "timeouts", "loop_guards"):
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
            "loop_guards": self.loop_guards,
            "loop_guard_rate": (self.loop_guards / self.n) if self.n else None,
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
            "revoked": False,
        })
    return out


def load_revoked_kings() -> list[dict]:
    """Reigns the operator removed after the crown (`crown_revoked` events in
    history.jsonl; the validator rewrites the `crowned` line in place, so
    `at` is the crown time and `revoked_at` the removal). Their reign
    numbers were re-used by later crowns, so rows label by date + digest."""
    if not VALIDATOR_HISTORY.exists():
        return []
    out = []
    try:
        with VALIDATOR_HISTORY.open() as f:
            for line in f:
                if '"crown_revoked"' not in line:
                    continue
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                if ev.get("event") != "crown_revoked":
                    continue
                rev = str(ev.get("revision") or "")
                if not rev:
                    continue
                out.append({
                    "reign": int(ev.get("reign_number") or 0),
                    "digest": rev, "digest12": rev[:12],
                    "hotkey": ev.get("hotkey") or "",
                    "repo": ev.get("repo") or "",
                    "crowned_at": ev.get("at") or "",
                    "challenge_id": ev.get("challenge_id") or "",
                    "current": False,
                    "revoked": True,
                    "revoked_at": ev.get("revoked_at") or "",
                    "revoked_reason": ev.get("revoked_reason") or ev.get("revoked_code") or "removed",
                })
    except OSError:
        return []
    return out


def compute_stats(conn: sqlite3.Connection, manifest_info: dict,
                  groups: dict[str, str]) -> dict:
    now = time.time()
    cur = conn.execute(
        "SELECT source, env_id, grp, harness, seat, digest12, ts, outcome, stop, "
        "n_calls, wall_s, timeout, policy_id, temperature, score, backfill FROM rollouts")
    rows = cur.fetchall()

    kings = load_kings()
    revoked = load_revoked_kings()
    by_digest = {k["digest12"]: k for k in [*revoked, *kings]}

    env_ids: dict[str, str] = {}
    src_group: dict[str, str] = dict(groups)
    teacher_env: dict[str, Agg] = {}
    teacher_trend: dict[str, dict[int, list[int]]] = {}
    per_king: dict[str, dict] = {}
    stops: dict[str, int] = {}
    recent = {"king_1h": 0, "king_24h": 0, "all_1h": 0, "all_24h": 0}
    # per source x seat: every rollout (any outcome), all time and last 24 h
    seat_counts: dict[str, dict[str, dict[str, int]]] = {}
    # per source: rollouts (king + teacher seats) carrying a numeric grade vs
    # all of them -> an env whose grader never writes a score has no cell
    grade_presence: dict[str, list[int]] = {}
    last_ts = 0.0
    n_king = n_teacher = n_backfill = 0

    def bucket(ts: float) -> int:
        """0 = the 24 h ending now, 1 = the day before, ..."""
        return int((now - ts) // 86400)

    for (source, env_id, grp, harness, seat, digest12, ts, outcome,
         stop, n_calls, wall, timeout, policy_id, temperature, score, backfill) in rows:
        env_ids.setdefault(source, env_id)
        src_group.setdefault(source, grp or "other")
        stops[stop or "none"] = stops.get(stop or "none", 0) + 1
        last_ts = max(last_ts, ts or 0.0)
        age = now - (ts or 0.0)
        if age <= 3600:
            recent["all_1h"] += 1
        if age <= 86400:
            recent["all_24h"] += 1
        if seat in ("teacher", "king"):
            sc = seat_counts.setdefault(source, {}).setdefault(
                seat, {"n": 0, "n_24h": 0, "backfill": 0})
            sc["n"] += 1
            if age <= 86400:
                sc["n_24h"] += 1
            if backfill:
                sc["backfill"] += 1
                n_backfill += 1
            if outcome != "errored":
                gp = grade_presence.setdefault(source, [0, 0])
                gp[1] += 1
                gp[0] += int(score is not None)
        if seat == "teacher":
            n_teacher += 1
            teacher_env.setdefault(source, Agg()).add(outcome, timeout, n_calls, wall, stop)
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
            "env_temp": {}, "harness_temp": {}, "temp": {},
            "first_ts": ts, "last_ts": ts, "n": 0, "recent_1h": 0, "recent_24h": 0})
        k["n"] += 1
        k["first_ts"] = min(k["first_ts"], ts)
        k["last_ts"] = max(k["last_ts"], ts)
        if age <= 3600:
            k["recent_1h"] += 1
        if age <= 86400:
            k["recent_24h"] += 1
        k["env"].setdefault(source, Agg()).add(outcome, timeout, n_calls, wall, stop)
        k["harness"].setdefault(harness, Agg()).add(outcome, timeout, n_calls, wall, stop)
        k["env_harness"].setdefault((source, harness), Agg()).add(outcome, timeout, n_calls, wall, stop)
        tc = temp_class(temperature)
        k["env_temp"].setdefault((source, tc), Agg()).add(outcome, timeout, n_calls, wall, stop)
        k["harness_temp"].setdefault((harness, tc), Agg()).add(outcome, timeout, n_calls, wall, stop)
        k["temp"].setdefault(tc, Agg()).add(outcome, timeout, n_calls, wall, stop)
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

    def temp_split(table: dict, key) -> dict:
        """{greedy: Agg.out() | None, sampled: Agg.out() | None} for one key."""
        return {tc: (table[(key, tc)].out() if (key, tc) in table else None)
                for tc in (GREEDY, SAMPLED)}

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
            row["by_temp"] = temp_split(k["env_temp"], source)
            env_rows.append(row)
        env_rows.sort(key=lambda r: (r["group"], r["source"]))
        harness_rows = []
        for h, agg in k["harness"].items():
            row = agg.out()
            row["harness"] = h
            row["by_temp"] = temp_split(k["harness_temp"], h)
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
            "revoked": bool(meta.get("revoked")),
            "revoked_at": meta.get("revoked_at", ""),
            "n_rollouts": k["n"], "first_ts": k["first_ts"], "last_ts": k["last_ts"],
            "recent_1h": k["recent_1h"], "recent_24h": k["recent_24h"],
            "total": total.out(),
            "by_temp": {tc: (k["temp"][tc].out() if tc in k["temp"] else None)
                        for tc in (GREEDY, SAMPLED)},
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
        "counts": {"rollouts": len(rows), "king": n_king, "teacher": n_teacher,
                   "backfill": n_backfill},
        "recent": recent,
        "king": current,
        "kings": kings,
        "revoked_kings": revoked,
        "reigns": reigns_out,
        "teacher": {"envs": teacher_rows, "total": teacher_total.out(),
                    "trend": trend_out(teacher_trend)},
        "envs": [{"source": s, "env_id": env_ids.get(s, ""),
                  "group": src_group.get(s, "other"),
                  # share of (king + teacher, non-errored) rollouts with a numeric grade;
                  # ~0 = the env has no grader (affine_wiki), so no row can be scored on it
                  "graded_share": ((grade_presence[s][0] / grade_presence[s][1])
                                   if grade_presence.get(s, [0, 0])[1] else None)}
                 for s in sorted(env_ids)],
        "seat_counts": seat_counts,
        "stop_conditions": stops,
        "definitions": {
            "outcome": "same rule as the fold (affine.corpus.view.rollout_outcome): errored = a real "
                       "error (the ACP 'rollout stopped: max_turns/loop_guard' artifact is not one) or "
                       "stop_condition not in {agent_completed, max_turns, loop_guard, no_visible_reply}; "
                       "no_visible_reply = failed whatever the grade; solved = primary grade >= 1.0; "
                       "failed = grade < 1.0 or ungraded at max_turns / loop_guard; unscored = no numeric grade",
            "loop rate": "share of the row's rollouts the datagen loop guard cut short "
                         "(stop_condition loop_guard: the same action repeated 6x with the same "
                         "observation; king policies only). Loop ONSETS inside rollouts that ran to the "
                         "end are not counted here (they need the baked conversations), so this is a "
                         "lower bound on looping",
            "greedy / sampled": "policy.temperature stamped on the envelope since 2026-09-13: greedy = "
                                "T 0 (king_*_greedy policies), sampled = T 0.8 (every king policy before "
                                "the stamp existed sampled at 0.8 and counts as sampled). Solve rate and "
                                "n per class",
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


# -- matrix ----------------------------------------------------------------------------
def load_cards() -> list[dict]:
    """Benchmark-suite scorecards (ops/benchsuite/publish.py), newest first."""
    cards = []
    if not BENCHSUITE_DIR.exists():
        return cards
    for p in sorted(BENCHSUITE_DIR.glob("*.json")):
        try:
            card = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        if isinstance(card, dict) and isinstance(card.get("rows"), list):
            cards.append(card)
    cards.sort(key=lambda c: c.get("created_at") or "", reverse=True)
    return cards


def load_inflight_passes() -> list[dict]:
    """Benchmark passes running now, from ops/benchsuite/state/pass-*.log:
    first line `pass <run_id> mode=<m> ref=<ref> label=<label>`, then
    `start king/<env> t=<T>` / `done king/<env> t=<T>` per cell."""
    out = []
    if not BENCHSUITE_RUNS_DIR.is_dir():
        return out
    try:
        suite = tomllib.loads(BENCHSUITE_SUITE_TOML.read_text()) if BENCHSUITE_SUITE_TOML.exists() else {}
    except (OSError, ValueError):
        suite = {}
    chat_envs = list((suite.get("modes") or {}).get("chat_envs") or [])
    now = time.time()
    for log_path in sorted(BENCHSUITE_RUNS_DIR.glob("pass-*.log")):
        if log_path.with_suffix(".exit").exists():
            continue
        try:
            mtime = log_path.stat().st_mtime
            if now - mtime > INFLIGHT_STALE_S:
                continue
            text = log_path.read_text(errors="replace")
        except OSError:
            continue
        run_id = log_path.stem[len("pass-"):]
        head = text.splitlines()[0] if text else ""
        fields = dict(re.findall(r"(\w+)=(\S+)", head))
        started: list[str] = []
        done: set[str] = set()
        started_at = None
        for m in re.finditer(r"^\[\w+\] (\S+) (start|done) king/([\w\-]+) t=([\d.]+)", text, re.M):
            ts, kind, env, temp = m.groups()
            if temp not in ("0", "0.0"):
                continue
            started_at = started_at or ts
            if kind == "start" and env not in started:
                started.append(env)
            elif kind == "done":
                done.add(env)
        m0 = re.search(r"^\[run_pass\] (\S+) pass ", text, re.M)
        mode = fields.get("mode") or ""
        # the standard passes run the chat sets (then sandbox sets when triggered);
        # other modes (agentic, ...) plan only what their log has started
        planned = list(dict.fromkeys([*(chat_envs if mode in CHAT_SET_MODES else []), *started]))
        running_env = next((e for e in reversed(started) if e not in done), None)
        t_first = parse_iso(m0.group(1)) if m0 else mtime
        n_done = len(done)
        eta = None
        if n_done and t_first:
            per_cell = (now - t_first) / n_done
            eta = now + per_cell * max(0, len(planned) - n_done)
        label = fields.get("label") or ""
        ref = fields.get("ref") or ""
        digest12 = None
        m_d = re.search(r"(?:^|-)([0-9a-f]{12})(?:-|\.|$)", run_id)   # ...-<digest12>[-agentic][.attemptN]
        if m_d:
            digest12 = m_d.group(1)
        elif re.fullmatch(r"[0-9a-f]{64}", ref):
            digest12 = ref[:12]
        elif "@" in ref:
            digest12 = ref.rsplit("@", 1)[-1][:12]
        out.append({
            "run_id": run_id, "mode": mode, "label": label, "ref": ref,
            "digest12": digest12, "genesis": label == "genesis" or GENESIS_DIGEST12 in ref,
            "started_at": (m0.group(1) if m0 else None), "log_mtime": mtime,
            "planned": planned, "done": sorted(done), "running_env": running_env,
            "eta_ts": eta, "n_done": n_done, "n_planned": len(planned),
        })
    return out


def card_digest12(card: dict) -> str | None:
    """First 12 hex of the model's identity: the R2 sha256 digest, or for a
    Hugging Face reference (`hf://repo@revision`: the genesis and the HF-era
    kings 1-5) the pinned revision, which is what state.json / history.jsonl
    store as `revision` for those reigns."""
    k = card.get("king") or {}
    rev = str(k.get("hf_revision") or "")
    if rev:
        return rev[:12]
    d = str(k.get("digest") or "")
    if d.startswith("hf-"):
        return None
    return d[:12] if d else None


def is_genesis_card(card: dict) -> bool:
    k = card.get("king") or {}
    d12 = card_digest12(card)
    if d12 == GENESIS_DIGEST12:
        return True
    if k.get("reign") == 0 and d12 is None:
        return True
    ident = " ".join(str(k.get(f) or "") for f in ("label", "model", "hf_repo")).lower()
    return "genesis" in ident or "qwen3.6-35b-a3b" in ident


def is_model_card(card: dict) -> bool:
    """A card measuring a king / genesis (not a challenger or comparable)."""
    return (card.get("mode") not in BENCH_SKIP_MODES
            and card.get("status") not in BENCH_SKIP_STATUS
            and not (card.get("king") or {}).get("duel"))


def bench_value(side: dict | None, env: str) -> dict | None:
    """Score 0-100 + Wilson interval for one card cell (T=0 row)."""
    if not side or side.get("score") is None:
        return None
    src = side
    metric = "score"
    if env in BENCH_FINISHED_ONLY and (side.get("finished_only") or {}).get("score") is not None:
        src = side["finished_only"]
        metric = "finished_only"
    ci = src.get("ci95") or [None, None]
    cap = side.get("finish_length_frac")
    cap = float(cap) if isinstance(cap, (int, float)) and not isinstance(cap, bool) else None
    return {
        "score": round(100.0 * float(src["score"]), 2),
        "lo": None if ci[0] is None else round(100.0 * float(ci[0]), 2),
        "hi": None if ci[1] is None else round(100.0 * float(ci[1]), 2),
        "n": src.get("n") or side.get("n"),
        "metric": metric,
        "all_rollouts": (round(100.0 * float(side["score"]), 2)
                         if metric == "finished_only" else None),
        # share of the model's replies cut at the completion cap (scored 0):
        # above CAP_BOUND_FRAC the number is a lower bound, not a measure
        "cap_frac": None if cap is None else round(cap, 4),
        "cap_bound": bool(cap is not None and cap > CAP_BOUND_FRAC),
    }


def card_cells(cards: list[dict], side: str) -> dict[str, dict]:
    """env -> cell from the best card for each env. Cards are ordered
    best-first by the caller; the first card carrying the env wins, later
    cards only fill envs the earlier ones lack (a parity re-run of five chat
    sets must not override the full pass)."""
    out: dict[str, dict] = {}
    for card in cards:
        for row in card.get("rows") or []:
            env = row.get("env")
            if not env or env in out or row.get("temperature") != BENCH_TEMPERATURE:
                continue
            val = bench_value(row.get(side), env)
            if val is None:
                continue
            val.update(run_id=card.get("run_id"), mode=card.get("mode"),
                       created_at=card.get("created_at"), kind="bench")
            if row.get("graded") == "llm_judge":
                # advisory: an LLM judge graded the rollouts; never part of the score
                val["judge"] = row.get("judge") or {}
                val["graded"] = "llm_judge"
            if side == "teacher" and (row.get("teacher") or {}).get("reused_from"):
                val["reused_from"] = row["teacher"]["reused_from"]
            out[env] = val
    return out


def env_agg(row_agg: dict | None) -> dict | None:
    """The Agg.out() dict a matrix env cell is scored on: the SAMPLED
    (T = 0.8) split when the row carries one (king rows since the
    temperature stamp), else the row itself (teacher rows: never greedy;
    stats.json written by an older builder). Keeps `source`."""
    if not row_agg:
        return None
    split = (row_agg.get("by_temp") or {}).get(MATRIX_ENV_TEMP)
    if split is None:
        return row_agg
    return {**split, "source": row_agg.get("source"), "temp": MATRIX_ENV_TEMP,
            "greedy": (row_agg.get("by_temp") or {}).get(GREEDY)}


def env_cell(agg: dict | None) -> dict | None:
    """Datagen environment cell from an Agg.out() dict (stats.json)."""
    if not agg or not agg.get("graded"):
        return None
    graded = int(agg["graded"])
    if graded < MATRIX_MIN_GRADED:
        return {"score": None, "n": graded, "kind": "env",
                "reason": f"only {graded} graded rollouts (< {MATRIX_MIN_GRADED})"}
    cell = {
        "score": round(100.0 * float(agg["rate"]), 2),
        "lo": None if agg.get("lo") is None else round(100.0 * float(agg["lo"]), 2),
        "hi": None if agg.get("hi") is None else round(100.0 * float(agg["hi"]), 2),
        "n": graded, "kind": "env",
        "solved": agg.get("solved"), "failed": agg.get("failed"),
        "errored": agg.get("errored"), "rollouts": agg.get("n"),
        "low_n": graded < MATRIX_LOW_N,
        "temp": agg.get("temp", MATRIX_ENV_TEMP),
    }
    g = agg.get("greedy")
    if g and g.get("graded"):
        cell["greedy"] = {"score": (round(100.0 * float(g["rate"]), 2) if g.get("rate") is not None else None),
                          "n": int(g["graded"])}
    return cell


def group_cell(aggs: list[dict], group: str) -> dict | None:
    """Pooled solve rate over a fold group's environments (solved / graded
    summed over the envs that clear MATRIX_MIN_GRADED)."""
    solved = graded = rollouts = errored = 0
    envs = []
    for a in aggs:
        if not a or not a.get("graded") or int(a["graded"]) < MATRIX_MIN_GRADED:
            continue
        solved += int(a.get("solved") or 0)
        graded += int(a["graded"])
        rollouts += int(a.get("n") or 0)
        errored += int(a.get("errored") or 0)
        envs.append(a["source"])
    if not graded:
        return None
    rate, lo, hi = wilson(solved, graded)
    return {"score": round(100.0 * rate, 2), "lo": round(100.0 * lo, 2), "hi": round(100.0 * hi, 2),
            "n": graded, "kind": "group", "group": group, "solved": solved, "rollouts": rollouts,
            "errored": errored, "envs": sorted(envs)}


def total_cell(cells: dict[str, dict], keys: list[str]) -> dict | None:
    vals = [(k, cells[k]["score"]) for k in keys if cells.get(k) and cells[k].get("score") is not None]
    if not vals:
        return None
    scores = [v for _, v in vals]
    return {"score": round(sum(scores) / len(scores), 2), "n_cols": len(vals),
            "cols": [k for k, _ in vals], "kind": "total",
            "n_bench": sum(1 for k, _ in vals if k.startswith("bench:")),
            "n_env": sum(1 for k, _ in vals if k.startswith("env:"))}


def matrix_rows_meta(stats: dict) -> list[dict]:
    """Row skeletons in display order: teacher, genesis, kings newest first.
    Reigns the operator revoked are NOT rows (operator directive 2026-09-15);
    the payload lists them under `removed`."""
    kings = [k for k in stats.get("kings") or [] if k.get("digest12") != GENESIS_DIGEST12
             and int(k.get("reign") or 0) != 0]
    genesis = next((k for k in stats.get("kings") or []
                    if k.get("digest12") == GENESIS_DIGEST12 or int(k.get("reign") or 0) == 0), None)
    rows = [{
        "key": "teacher", "kind": "teacher", "label": "Teacher", "model": TEACHER_MODEL,
        "sub": TEACHER_MODEL, "order": 0,
    }, {
        "key": "genesis", "kind": "genesis", "label": "Genesis", "model": GENESIS_MODEL,
        "reign": 0, "digest12": GENESIS_DIGEST12,
        "crowned_at": (genesis or {}).get("crowned_at", ""),
        "sub": f"{GENESIS_MODEL} · reign 0 (seed)", "order": 1,
    }]
    crowned = sorted(kings, key=lambda k: k.get("crowned_at") or "", reverse=True)
    for i, k in enumerate(crowned):
        rows.append({
            "key": k["digest12"], "kind": "king",
            "label": f"King {k['reign']}",
            "reign": k["reign"], "digest12": k["digest12"], "digest": k.get("digest", ""),
            "hotkey": k.get("hotkey", ""), "crowned_at": k.get("crowned_at", ""),
            "challenge_id": k.get("challenge_id", ""), "current": bool(k.get("current")),
            "sub": f"king-{k['digest12']} · crowned {str(k.get('crowned_at') or '')[:16].replace('T', ' ')}",
            "order": 2 + i,
        })
    return rows


def removed_kings_meta(stats: dict) -> list[dict]:
    """Revoked reigns, for the record only (never rendered as rows)."""
    return [{
        "reign": k["reign"], "digest12": k["digest12"], "digest": k.get("digest", ""),
        "crowned_at": k.get("crowned_at", ""), "revoked_at": k.get("revoked_at", ""),
        "reason": k.get("revoked_reason", ""), "challenge_id": k.get("challenge_id", ""),
    } for k in sorted(stats.get("revoked_kings") or [], key=lambda k: k.get("crowned_at") or "", reverse=True)]


def build_matrix(stats: dict, cards: list[dict], inflight: list[dict] | None = None) -> dict:
    now = time.time()
    inflight = inflight or []
    model_cards = [c for c in cards if is_model_card(c)]
    # best-first per digest: the fullest card wins, ties -> newest
    rank = lambda c: (-len({r.get("env") for r in c.get("rows") or []}), c.get("created_at") or "")
    by_digest: dict[str, list[dict]] = {}
    genesis_cards: list[dict] = []
    known_digests = {k.get("digest12") for k in [*(stats.get("kings") or []), *(stats.get("revoked_kings") or [])]}
    reference_cards: dict[str, list[dict]] = {}     # label -> cards (fullest first)
    for c in model_cards:
        if is_genesis_card(c):
            genesis_cards.append(c)
            continue
        kb = c.get("king") or {}
        d12 = card_digest12(c)
        if kb.get("label") and kb.get("reign") is None and d12 not in known_digests:
            reference_cards.setdefault(str(kb["label"]), []).append(c)
            continue
        if d12:
            by_digest.setdefault(d12, []).append(c)
    for lst in by_digest.values():
        lst.sort(key=rank)
    for lst in reference_cards.values():
        lst.sort(key=rank)
    genesis_cards.sort(key=rank)
    # teacher: a card whose teacher cells were measured (not copied) first
    teacher_cards = sorted(
        cards, key=lambda c: (any((r.get("teacher") or {}).get("reused_from") for r in c.get("rows") or []),
                              -len(c.get("rows") or []), c.get("created_at") or ""))

    # -- columns
    bench_envs_seen: list[str] = []
    for c in cards:
        for r in c.get("rows") or []:
            if r.get("temperature") == BENCH_TEMPERATURE and r.get("env") and r["env"] not in bench_envs_seen:
                bench_envs_seen.append(r["env"])
    bench_notes: dict[str, dict] = {}
    for c in cards:
        for r in c.get("rows") or []:
            if r.get("env") and r["env"] not in bench_notes:
                bench_notes[r["env"]] = {"group": r.get("group"), "note": r.get("note"), "n": r.get("n"),
                                         "graded": r.get("graded"), "judge": r.get("judge")}
    known = {b for b, _ in BENCH_COLUMNS}
    is_agentic = lambda e: (bench_notes.get(e, {}).get("group") == BENCH_AGENTIC_GROUP)
    # known order first; unknown card envs appended in card order; the agentic
    # group always forms the last block (cards gain agentic rows by merge, so
    # the column set is re-read on every refresh)
    ordered_bench = [e for e, _ in BENCH_COLUMNS if e in bench_envs_seen and not is_agentic(e)] \
        + [e for e in bench_envs_seen if e not in known and not is_agentic(e)] \
        + [e for e, _ in BENCH_COLUMNS if e in bench_envs_seen and is_agentic(e)] \
        + [e for e in bench_envs_seen if e not in known and is_agentic(e)]
    labels = dict(BENCH_COLUMNS)
    columns = [{"key": "total", "label": "total", "abbr": "total", "kind": "total",
                "note": "unweighted mean of the row's available benchmark and environment cells (0-100)"}]
    for e in ordered_bench:
        meta = bench_notes.get(e, {})
        columns.append({
            "key": f"bench:{e}", "label": labels.get(e, e), "abbr": BENCH_ABBR.get(e, e[:4].upper()),
            "short": BENCH_SHORT.get(e, labels.get(e, e)[:10]),
            "kind": "bench", "env": e,
            "group": meta.get("group"), "n": meta.get("n"), "note": meta.get("note"),
            "metric": "finished_only" if e in BENCH_FINISHED_ONLY else "score",
            "graded": meta.get("graded") or "deterministic",
            "judge": meta.get("judge") if meta.get("graded") == "llm_judge" else None,
            "advisory": meta.get("graded") == "llm_judge",
        })
    env_groups = {e["source"]: e.get("group") or "other" for e in stats.get("envs") or []}
    env_ids = {e["source"]: e.get("env_id") or "" for e in stats.get("envs") or []}
    # envs whose grader never writes a score: a column nobody can fill
    no_grader = {e["source"] for e in stats.get("envs") or []
                 if e.get("graded_share") is not None and e["graded_share"] < NO_GRADER_SHARE}
    gorder = {g: i for i, g in enumerate(MATRIX_GROUP_ORDER)}
    groups_present = sorted({g for g in env_groups.values()}, key=lambda g: (gorder.get(g, 99), g))
    # default view: one pooled column per fold group; the per-env columns
    # sit behind the page's "expand environments" toggle
    for g in groups_present:
        members = sorted(s for s, gg in env_groups.items() if gg == g)
        columns.append({"key": f"group:{g}", "label": f"{g} environments", "abbr": GROUP_ABBR.get(g, g[:4]),
                        "kind": "group", "group": g, "envs": members,
                        "note": f"pooled solve rate over the {g} datagen environments: " + ", ".join(members)})
    for s in sorted(env_groups, key=lambda s: (gorder.get(env_groups[s], 99), s)):
        col = {"key": f"env:{s}", "label": s,
               "abbr": ENV_ABBR.get(s, s.replace("affine_", "")[:4].upper()),
               "kind": "env", "env": s, "group": env_groups[s], "env_id": env_ids.get(s, ""),
               "temp": MATRIX_ENV_TEMP}
        if s in no_grader:
            col["no_grader"] = True
            col["note"] = ("this environment writes no numeric grade (its rollouts are unscored), "
                           "so no model can be scored on it; the column is kept for the rollout counts")
        columns.append(col)
    # `total` = mean over every benchmark + every environment; the page shows two
    # tables, each with its own mean: total:bench and total:env
    value_keys = [c["key"] for c in columns if c["kind"] in ("bench", "env")]
    bench_keys = [c["key"] for c in columns if c["kind"] == "bench"]
    env_keys = [c["key"] for c in columns if c["kind"] == "env"]
    columns.insert(1, {"key": "total:bench", "label": "total", "abbr": "total", "short": "total",
                       "kind": "total_bench", "note": "unweighted mean of the row's available benchmark cells"})
    columns.insert(2, {"key": "total:env", "label": "total", "abbr": "total", "short": "total",
                       "kind": "total_env", "note": "unweighted mean of the row's available environment cells"})

    # -- rows
    teacher_envs = (stats.get("teacher") or {}).get("envs") or {}
    reign_envs = {r["digest12"]: {e["source"]: e for e in r.get("envs") or []}
                  for r in stats.get("reigns") or []}
    rows = matrix_rows_meta(stats)
    for i, (label, lst) in enumerate(sorted(reference_cards.items())):
        kb = lst[0].get("king") or {}
        rev = str(kb.get("hf_revision") or "")
        rows.append({
            "key": f"ref:{label}", "kind": "reference", "label": label[:1].upper() + label[1:],
            "model": kb.get("hf_repo") or kb.get("repo") or label, "hf_revision": rev,
            "digest12": card_digest12(lst[0]),
            "sub": f"{kb.get('hf_repo') or label}{' @ ' + rev[:8] if rev else ''}",
            "tip": REFERENCE_NOTES.get(label, f"{kb.get('hf_repo') or label} — reference model (open checkpoint "
                                              "of the genesis family, benchmarked for comparison); not a king, never paid"),
            "order": 1000 + i,
        })
    for row in rows:
        cells: dict[str, dict] = {}
        if row["kind"] == "teacher":
            bench = card_cells(teacher_cards, "teacher")
            env_aggs = {s: {**a, "source": s} for s, a in teacher_envs.items()}
        elif row["kind"] == "genesis":
            bench = card_cells(genesis_cards, "king")
            env_aggs = reign_envs.get(GENESIS_DIGEST12, {})
        elif row["kind"] == "reference":
            bench = card_cells(reference_cards.get(row["key"][len("ref:"):], []), "king")
            env_aggs = reign_envs.get(row.get("digest12") or "", {})   # blank unless a backfill ran it
        else:
            bench = card_cells(by_digest.get(row["digest12"], []), "king")
            env_aggs = reign_envs.get(row["digest12"], {})
        for e, v in bench.items():
            cells[f"bench:{e}"] = v
        env_aggs = {s: env_agg(a) for s, a in env_aggs.items()}
        for s, a in env_aggs.items():
            if s in no_grader:
                cells[f"env:{s}"] = {"score": None, "kind": "env", "n": 0, "no_grader": True,
                                     "rollouts": (a or {}).get("n"),
                                     "reason": "environment has no grader (rollouts are unscored)"}
                continue
            v = env_cell(a)
            if v is not None:
                cells[f"env:{s}"] = v
        for g in groups_present:
            v = group_cell([a for s, a in env_aggs.items()
                            if env_groups.get(s) == g and s not in no_grader], g)
            if v is not None:
                cells[f"group:{g}"] = v
        for key, keys in (("total", value_keys), ("total:bench", bench_keys), ("total:env", env_keys)):
            tot = total_cell(cells, keys)
            if tot:
                cells[key] = tot
        row["cells"] = cells
        row["n_cells"] = sum(1 for k in value_keys if cells.get(k) and cells[k].get("score") is not None)
        row["cards"] = sorted({v["run_id"] for v in bench.values() if v.get("run_id")})
    # benchmark passes in flight: planned-but-missing cells render as "running"
    for p in inflight:
        target = next((r for r in rows if (r["kind"] == "genesis" and p["genesis"])
                       or (r["kind"] == "king" and p["digest12"] and r["digest12"] == p["digest12"])
                       or (r["kind"] == "reference" and p["label"] and r["key"] == f"ref:{p['label']}")), None)
        if target is None:
            continue
        eta_iso = (datetime.fromtimestamp(p["eta_ts"], timezone.utc).isoformat(timespec="minutes")
                   if p.get("eta_ts") else None)
        target["inflight"] = {**{k: p[k] for k in ("run_id", "mode", "started_at", "planned", "done",
                                                    "running_env", "n_done", "n_planned")}, "eta": eta_iso}
        for e in p["planned"]:
            key = f"bench:{e}"
            if target["cells"].get(key, {}).get("score") is None:
                target["cells"][key] = {"score": None, "running": True, "kind": "bench",
                                        "run_id": p["run_id"], "eta": eta_iso,
                                        "state": ("running now" if e == p["running_env"]
                                                  else "finished, card not published yet" if e in p["done"]
                                                  else "queued in this pass"),
                                        "reason": f"benchmark pass {p['run_id']} in progress"}
    # Row set (operator 2026-09-15 20:27 / 20:31 UTC): teacher, then kings newest
    # first, genesis as the bottom row. Kings before MATRIX_MIN_REIGN are never
    # rows, whatever cards or passes exist -> `hidden`.
    hidden = [r for r in rows if r["kind"] == "king" and int(r.get("reign") or 0) < MATRIX_MIN_REIGN
              and not r["current"]]
    rows = [r for r in rows if r not in hidden]
    genesis_row = next((r for r in rows if r["kind"] == "genesis"), None)
    refs = [r for r in rows if r["kind"] == "reference"]
    rows = [r for r in rows if r["kind"] != "reference" and r is not genesis_row] + refs \
        + ([genesis_row] if genesis_row else [])
    for i, r in enumerate(rows):
        r["order"] = i
    # delta vs the teacher row, per cell. The total compares against the
    # teacher's mean over the SAME columns the row has (rows differ in
    # coverage: a chat-only card has 10 benchmarks, the teacher has 15).
    teacher_cells = rows[0]["cells"]
    for row in rows[1:]:
        for k, v in row["cells"].items():
            if k.startswith("total"):
                same = [teacher_cells[c]["score"] for c in v["cols"]
                        if teacher_cells.get(c) and teacher_cells[c].get("score") is not None]
                if same:
                    v["teacher_same_cols"] = round(sum(same) / len(same), 2)
                    v["n_same_cols"] = len(same)
                    v["delta"] = round(v["score"] - v["teacher_same_cols"], 2)
                continue
            t = teacher_cells.get(k)
            if v.get("score") is not None and t and t.get("score") is not None:
                v["delta"] = round(v["score"] - t["score"], 2)

    return {
        "generated_at": datetime.fromtimestamp(now, timezone.utc).isoformat(timespec="seconds"),
        "generated_ts": now,
        "stats_generated_at": stats.get("generated_at"),
        "columns": columns,
        "rows": rows,
        "removed": removed_kings_meta(stats),
        "hidden": [{k: r.get(k) for k in ("key", "label", "reign", "digest12", "crowned_at", "challenge_id")}
                   for r in hidden],
        "inflight": [{k: p[k] for k in ("run_id", "mode", "label", "digest12", "genesis", "started_at",
                                         "n_done", "n_planned", "running_env")} for p in inflight],
        "n_cards": len(cards),
        "cards": [{"run_id": c.get("run_id"), "mode": c.get("mode"), "status": c.get("status"),
                   "created_at": c.get("created_at"), "digest12": card_digest12(c),
                   "reign": (c.get("king") or {}).get("reign"),
                   "label": (c.get("king") or {}).get("label"),
                   "genesis": is_genesis_card(c), "used": is_model_card(c)}
                  for c in cards],
        "definitions": {
            "rows": "the teacher, then the crowned kings newest first, then the genesis seed (reign 0) "
                    "as the bottom row, with reference models (open checkpoints of the genesis family "
                    "benchmarked for comparison; not kings, never paid) just above it. "
                    f"Kings before reign {MATRIX_MIN_REIGN} (Affine-XI and older, not "
                    "backfilled) are listed under `hidden`, never rows; reigns the operator "
                    "revoked after the crown (history.jsonl crown_revoked) are under `removed`",
            "value": "average score 0-100 per cell. Benchmarks: the card's greedy (T=0) row, "
                     "score = share of tasks passed; SWE-bench Verified uses the finished-only score "
                     "(rollouts inside the time / context budget). Datagen environments: solve rate "
                     "= solved / (solved + failed) over the row's SAMPLED (T = 0.8) rollouts (king "
                     "seat or coverage backfill for kings and the genesis, teacher_* policies for "
                     "the teacher), same outcome rule as the fold; greedy (T = 0) king rollouts are "
                     "reported in the tooltip, not pooled in; a group column pools solved / graded "
                     "over its environments; `low_n` marks a cell on fewer than "
                     f"{MATRIX_LOW_N} graded rollouts",
            "total": "unweighted mean over the row's available benchmark and environment "
                     "columns; the tooltip shows how many entered and the teacher's mean on the "
                     "same columns",
            "blank": f"no measurement (no benchmark card for the model, or fewer than "
                     f"{MATRIX_MIN_GRADED} graded rollouts on the environment); '…' = a benchmark pass "
                     "for the model is running and this cell is planned (ops/benchsuite pass log)",
            "colour": "cell tint = score minus the teacher's score in the same column: green above, "
                      "red below, stronger with the gap",
            "markers": f"‡ = cap-bound: more than {int(CAP_BOUND_FRAC * 100)}% of the model's replies hit the "
                       "completion cap and scored 0, so the number is a lower bound; ⚖ = judge-graded "
                       "(graded = llm_judge): an LLM judge graded the rollouts — advisory, never part of the score",
            "ci": "95% interval: Wilson on graded rollouts (environments) or the card's ci95 "
                  "(benchmarks), shown in the tooltip",
            "cards": "benchmark scorecards from affine/state/benchsuite/ (ops/benchsuite); when a "
                     "model has several cards the fullest one wins per benchmark, others fill gaps; "
                     "challenger and comparables cards are not model rows",
        },
    }


# -- dataset table (api/dataset_table) ---------------------------------------------
def fetch_json(fetcher: Fetcher, key: str) -> dict | None:
    try:
        return loads(fetcher.get(key))
    except Exception as e:  # network / parse: the table renders what it has
        log.warning("%s unavailable: %s", key, e)
        return None


def stratum_group(stratum: str) -> str | None:
    """Fold group encoded in a stratum key (`<group>:...`); None for the
    plain `repo|phase` strata (nl2repo sources), whose group is the source's."""
    if ":" not in stratum:
        return None
    return stratum.split(":", 1)[0]


def index_source_stats(fetcher: Fetcher, manifest: dict | None) -> dict | None:
    """Turns and strata of D per (fold group, source) from the corpus view
    index (one parquet; ~7 MB at epoch 41), cached by the index sha256."""
    idx = (manifest or {}).get("index") or {}
    key, sha = idx.get("key"), idx.get("sha256")
    if not key or not sha:
        return None
    try:
        cached = json.loads(INDEX_STATS_PATH.read_text())
        if cached.get("sha256") == sha:
            return cached
    except (OSError, ValueError):
        pass
    if pq is None:
        log.warning("pyarrow missing: dataset table has no per-source turns / strata")
        return None
    t0 = time.time()
    blob = fetcher.get(key)
    if hashlib.sha256(blob).hexdigest() != sha:
        log.warning("index %s: sha mismatch, skipped", key)
        return None
    table = pq.read_table(io.BytesIO(blob), columns=["source", "stratum", "stratum_src"])
    turns: dict[tuple[str, str], int] = {}
    strata: dict[tuple[str, str], set] = {}
    for source, stratum, stratum_src in zip(table.column("source").to_pylist(),
                                            table.column("stratum").to_pylist(),
                                            table.column("stratum_src").to_pylist()):
        g = stratum_group(stratum or "") or ""       # "" = the source's own group
        k = (g, source or "")
        turns[k] = turns.get(k, 0) + 1
        strata.setdefault(k, set()).add(stratum_src or stratum or "")
    out = {
        "sha256": sha, "key": key, "n_turns": int(table.num_rows), "computed_at": time.time(),
        "cells": [{"group": g, "source": s, "turns": n, "strata": len(strata[(g, s)])}
                  for (g, s), n in sorted(turns.items())],
    }
    INDEX_STATS_PATH.write_text(json.dumps(out))
    log.info("index %s: %d turns -> %d (group, source) cells in %.1fs",
             sha[:12], table.num_rows, len(out["cells"]), time.time() - t0)
    return out


def load_fold_caps() -> dict:
    """Strata caps from rollouts/sources.toml: [strata_budget.buckets] for
    the bucketed teacher groups, strata_buckets x sub_strata for the king
    groups (and [mix] targets)."""
    if not SOURCES_TOML.exists():
        return {"buckets": {}, "sub_strata": {}, "strata_buckets": {}, "mix": {}}
    cfg = tomllib.loads(SOURCES_TOML.read_text())
    budget = cfg.get("strata_budget") or {}
    # math / tool_use: the fold hashes each source into its own
    # [source.<name>].strata_buckets -> the group's cap is their sum
    source_buckets: dict[str, int] = {}
    for name, block in (cfg.get("source") or {}).items():
        if isinstance(block, dict) and block.get("strata_buckets"):
            g = str(block.get("group") or "other")
            source_buckets[g] = source_buckets.get(g, 0) + int(block["strata_buckets"])
    return {
        "buckets": {**source_buckets, **dict(budget.get("buckets") or {})},
        "sub_strata": dict(budget.get("sub_strata") or {}),
        "strata_buckets": {g: int(cfg[g]["strata_buckets"]) for g in KING_GROUP_ORDER
                           if isinstance(cfg.get(g), dict) and cfg[g].get("strata_buckets")},
        "mix": {k: float(v) for k, v in (cfg.get("mix") or {}).items() if isinstance(v, (int, float))},
    }


def fmt_compact(n: float | int | None) -> str | None:
    if n is None:
        return None
    n = float(n)
    if n >= 1e6:
        return f"{n / 1e6:.1f}M"
    if n >= 1e4:
        return f"{n / 1e3:.0f}k"
    if n >= 1e3:
        return f"{n / 1e3:.1f}k"
    return f"{n:.0f}"


def build_dataset_table(stats: dict, matrix: dict, fold: dict | None, curriculum: dict | None,
                        manifest: dict | None, index_stats: dict | None, caps: dict) -> dict:
    now = time.time()
    fold = fold or {}
    fgroups = fold.get("groups") or {}
    curr = (curriculum or {}).get("shares_after_clamp") or {}
    seat_counts = stats.get("seat_counts") or {}
    env_groups = {e["source"]: e.get("group") or "other" for e in stats.get("envs") or []}
    env_cols = [c for c in matrix["columns"] if c["kind"] == "env"]
    current = next((r for r in matrix["rows"] if r.get("current")), None)

    # per (group, source) turns / strata of D
    cells_by = {}
    for c in (index_stats or {}).get("cells") or []:
        cells_by[(c["group"], c["source"])] = c
    def d_cell(group: str, source: str) -> dict:
        own = cells_by.get(("", source), {})            # repo|phase strata (nl2repo)
        tagged = cells_by.get((group, source), {})
        return {"turns": (own.get("turns") or 0) + (tagged.get("turns") or 0),
                "strata": (own.get("strata") or 0) + (tagged.get("strata") or 0)}
    group_raw_strata: dict[str, int] = {}
    for (g, s), c in cells_by.items():
        gg = g or env_groups.get(s, "other")
        group_raw_strata[gg] = group_raw_strata.get(gg, 0) + c["strata"]

    def cap_of(group: str) -> tuple[int | None, str]:
        if group in caps["buckets"]:
            return int(caps["buckets"][group]), "strata_budget.buckets / source strata_buckets"
        if group in caps["strata_buckets"]:
            return int(caps["strata_buckets"][group]) * int(caps["sub_strata"].get(group, 1)), \
                "strata_buckets x sub_strata"
        return None, ""

    def group_meta(group: str) -> dict:
        fg = fgroups.get(group) or {}
        cap, cap_src = cap_of(group)
        strata = fg.get("strata")
        limited = (strata is not None and cap is not None and strata < cap)
        return {
            "group": group, "turns": fg.get("turns"), "strata": strata,
            "draws_per_duel": fg.get("draws_per_duel"), "share": fg.get("share"),
            "static_mix": fg.get("static_mix") if fg.get("static_mix") is not None else caps["mix"].get(group),
            "sub_strata_k": fg.get("sub_strata_k"), "bucket_n": fg.get("bucket_n"),
            "cap": cap, "cap_source": cap_src,
            "supply_limited": limited if (strata is not None and cap is not None) else None,
            "curriculum_share": curr.get(group),
            "raw_strata": group_raw_strata.get(group),
        }

    columns = []
    for c in env_cols:
        s = c["env"]
        g = c["group"]
        gm = group_meta(g)
        dc = d_cell(g, s)
        sc = seat_counts.get(s) or {}
        share_in_group = (dc["strata"] / gm["raw_strata"]) if gm.get("raw_strata") and dc["strata"] else None
        king_cell = (current or {}).get("cells", {}).get(f"env:{s}") if current else None
        columns.append({
            "key": f"env:{s}", "kind": "env", "label": s, "abbr": c["abbr"], "group": g,
            "env_id": c.get("env_id", ""), "group_meta": gm,
            "teacher_24h": (sc.get("teacher") or {}).get("n_24h", 0), "teacher_total": (sc.get("teacher") or {}).get("n", 0),
            "king_24h": (sc.get("king") or {}).get("n_24h", 0), "king_total": (sc.get("king") or {}).get("n", 0),
            "turns": dc["turns"] or None, "strata": dc["strata"] or None,
            "strata_share_in_group": share_in_group,
            # one turn per stratum per duel: the group's draws split by the source's strata share
            "turns_per_duel": (gm["draws_per_duel"] * share_in_group) if (gm.get("draws_per_duel") and share_in_group) else None,
            "turns_per_duel_note": "group draws per duel x the source's share of the group's strata (approximation: "
                                   "bucketed groups merge strata, so a source's exact draw rate is not published)",
            "supply_limited": gm["supply_limited"], "curriculum_share": gm["curriculum_share"],
            "king_solve": king_cell.get("score") if king_cell else None,
            "king_solve_n": king_cell.get("n") if king_cell else None,
        })
    for g in KING_GROUP_ORDER:
        if g not in fgroups and g not in curr:
            continue
        gm = group_meta(g)
        columns.append({
            "key": f"group:{g}", "kind": "king_group", "label": g, "abbr": KING_GROUP_ABBR.get(g, g[:5].upper()),
            "group": g, "group_meta": gm,
            "teacher_24h": None, "teacher_total": None, "king_24h": None, "king_total": None,
            "turns": gm["turns"], "strata": gm["strata"], "strata_share_in_group": None,
            "turns_per_duel": gm["draws_per_duel"], "turns_per_duel_note": "fold_stats draws_per_duel (exact)",
            "supply_limited": gm["supply_limited"], "curriculum_share": gm["curriculum_share"],
            "king_solve": None, "king_solve_n": None,
        })

    rows = [
        {"key": "teacher_24h", "label": "teacher rollouts, 24 h", "short": "teacher 24h", "fmt": "count",
         "note": "teacher_* rollouts on the source that landed in the trace store in the last 24 h (any outcome)"},
        {"key": "teacher_total", "label": "teacher rollouts, total", "short": "teacher total", "fmt": "count",
         "note": "all teacher_* rollouts on the source in the trace store"},
        {"key": "king_24h", "label": "king rollouts, 24 h", "short": "king 24h", "fmt": "count",
         "note": "king_* rollouts (any reign) on the source in the last 24 h"},
        {"key": "king_total", "label": "king rollouts, total", "short": "king total", "fmt": "count",
         "note": "all king_* rollouts on the source in the trace store"},
        {"key": "turns", "label": "turns in D", "short": "turns in D", "fmt": "count",
         "note": "published turns in the duel corpus (view index) whose stratum belongs to this column's group"},
        {"key": "strata", "label": "strata in D", "short": "strata in D", "fmt": "count",
         "note": "distinct strata (index stratum_src, sub-strata merged) for the column; king groups: fold_stats slice keys"},
        {"key": "turns_per_duel", "label": "turns per duel (of 1,300)", "short": "turns / duel", "fmt": "draws", "headline": True,
         "note": "expected turns of this column in one 1,300-turn duel slice (fold_stats draws_per_duel)"},
        {"key": "supply_limited", "label": "supply-limited", "short": "supply-lim.", "fmt": "bool",
         "note": "the column's group has fewer strata than its cap ([strata_budget.buckets] or strata_buckets x sub_strata), so more rollouts would raise its slice share"},
        {"key": "curriculum_share", "label": "curriculum shadow weight", "short": "curric. w", "fmt": "pct",
         "note": "adaptive-curriculum share for the group (v1.2 counted rule, shares_after_clamp, shadow mode: not applied to the fold yet)"},
        {"key": "king_solve", "label": "king solve rate", "short": "king solve %", "fmt": "score",
         "note": "current king's solve rate on the source (same cell as the environments table above)"},
    ]
    rec = fold.get("recurrence") or {}
    return {
        "generated_at": datetime.fromtimestamp(now, timezone.utc).isoformat(timespec="seconds"),
        "generated_ts": now,
        "header": {
            "n_turns": fold.get("n_turns") or (index_stats or {}).get("n_turns"),
            "n_strata": fold.get("n_strata"),
            "epoch": fold.get("epoch") or (manifest or {}).get("corpus_epoch"),
            "manifest_sha12": ((curriculum or {}).get("manifest_sha256") or "")[:12] or None,
            "index_sha12": ((index_stats or {}).get("sha256") or "")[:12] or None,
            "n_per_duel": fold.get("n_per_duel"),
            "recurrence": rec,
            "fold_generated_at": fold.get("generated_at"),
            "curriculum_mode": (curriculum or {}).get("mode"),
            "curriculum_rule": (curriculum or {}).get("knobs", {}).get("counted_rule"),
            "curriculum_for_epoch": (curriculum or {}).get("for_epoch"),
        },
        "rows": rows,
        "columns": columns,
        "groups": {g: group_meta(g) for g in sorted(set(list(fgroups) + list(env_groups.values())))},
        "sources": {"fold_stats": FOLD_STATS_KEY, "curriculum": CURRICULUM_KEY,
                    "index": (index_stats or {}).get("key"), "rollouts": "traces manifest via kingboard stats"},
    }


def write_dataset_table(stats: dict, matrix: dict, fetcher: Fetcher) -> dict:
    manifest = fetch_json(fetcher, CORPUS_MANIFEST_KEY)
    fold = fetch_json(fetcher, FOLD_STATS_KEY)
    curriculum = fetch_json(fetcher, CURRICULUM_KEY)
    try:
        index_stats = index_source_stats(fetcher, manifest)
    except Exception as e:
        log.warning("corpus index stats failed: %s", e)
        index_stats = None
    table = build_dataset_table(stats, matrix, fold, curriculum, manifest, index_stats, load_fold_caps())
    tmp = DATASET_TABLE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(table, separators=(",", ":")))
    tmp.replace(DATASET_TABLE_PATH)
    return table


def write_matrix(stats: dict) -> dict:
    # reign lists re-read from the validator files: cheap, and a stats.json
    # written by an older builder has no revoked_kings
    stats = {**stats, "kings": load_kings() or stats.get("kings") or [],
             "revoked_kings": load_revoked_kings()}
    matrix = build_matrix(stats, load_cards(), load_inflight_passes())
    tmp = MATRIX_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(matrix, separators=(",", ":")))
    tmp.replace(MATRIX_PATH)
    return matrix


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        stream=sys.stderr)
    t0 = time.time()
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    if "--matrix-only" in sys.argv[1:]:
        # derived files only (matrix + dataset table) from the existing stats.json
        stats = json.loads(STATS_PATH.read_text())
        matrix = write_matrix(stats)
        log.info("matrix.json written from existing stats: %d rows x %d columns, %d cards",
                 len(matrix["rows"]), len(matrix["columns"]), matrix["n_cards"])
        table = write_dataset_table(stats, matrix, Fetcher())
        log.info("dataset_table.json written: %d columns, epoch %s",
                 len(table["columns"]), table["header"].get("epoch"))
        return 0
    conn = sqlite3.connect(DB_PATH)
    conn.executescript(SCHEMA)
    migrate(conn)
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
    try:
        matrix = write_matrix(stats)
        log.info("matrix.json written: %d rows x %d columns, %d cards",
                 len(matrix["rows"]), len(matrix["columns"]), matrix["n_cards"])
    except Exception:  # the env stats must still publish if a card is malformed
        log.exception("matrix build failed")
        return 0
    try:
        table = write_dataset_table(stats, matrix, fetcher)
        log.info("dataset_table.json written: %d columns, epoch %s",
                 len(table["columns"]), table["header"].get("epoch"))
    except Exception:
        log.exception("dataset table build failed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
