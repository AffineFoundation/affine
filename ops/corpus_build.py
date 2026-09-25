#!/usr/bin/env python
"""Trace-first corpus fold: published traces -> duel_turns@v4 -> corpus D.

Replaces ops/datagen_refresh.py (which folded pod-sliced turn shards from a
private HF dataset into the Hippius corpus). Since 2026-09-02 the datagen
pod publishes full rollout traces to data.affine.io/traces/**; this script,
run on the validator box (pm2 `affine-corpus-build`, daily), turns them
into the view the duel scores:

  1. read traces/manifest.json (verified against its immutable copy) and
     list chunks not yet folded (state.json next to this script);
  2. derive one view record per rollout (affine.corpus.view: graph paths ->
     baked plain text -> slicer) and validate every turn against the fold
     contract: bench-panel + official SWE-rebench excludes, dialect
     admitted by [dataset].allowed_action_kinds, prefix shape/cap, exactly
     one action, no verbatim leakage; dedupe turn_ids against the live
     index. King-seat rollouts route to `king_fail` (failed only) and,
     since 2026-09-11, single turns are routed to their own groups: the
     first turn of each king loop to `king_loop_onset`, the judge's pivot
     turns to `king_pivot` (leakage rule waived for those two), the reply
     that ended a solved rollout to `completion` (teacher and king);
  3. enforce [mix] group targets from rollouts/rollouts/sources.toml in
     SLICE STRATA (cap_fill: what a duel slice is made of; turn counts are
     not) at ROLLOUT granularity (a rollout's turns enter together, so a
     trajectory is never split across epochs; capped rollouts defer to
     work/deferred_views.jsonl and re-enter next cycle), then
     [lang_mix.coding] over newly selected coding rollouts;
  4. skip when fewer than MIN_NEW_TURNS eligible turns accumulated unless
     the newest unfolded chunk is older than STALE_AFTER_S;
  5. publish in the safe order: view chunks + merged index first, immutable
     manifest revision, pointer last, local state after the manifest. A
     `pending` record makes a crashed publish resumable (same bytes, remote
     sha verified instead of re-uploaded);
  6. announce on the private Arbos ops Discord channel; a failed post is retried next
     cycle.

`--init` bootstraps the first schema-3 revision: imports the live v2 corpus
(epochs 1-13, from data.affine.io/turns/**) as legacy view records and
folds every published trace on top. `--allowed-kinds` overrides the toml
gate (T0 rehearsal); `--publish-prefix staging/` publishes everything under
a staging prefix; `--no-publish` builds locally only.

Credentials: DATA_R2_ACCESS_KEY_ID / DATA_R2_SECRET_ACCESS_KEY (+ optional
DATA_R2_ENDPOINT) and DISCORD_BOT_TOKEN_ARBOS_BITTENSOR from repo .env.
Fail-loud: any error aborts the cycle with state untouched. Never prints a
secret.
"""

from __future__ import annotations

import argparse
import fcntl
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import sqlite3
import random
import statistics
import sys
import tempfile
import tomllib
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "affine"))

from affine import dialects  # noqa: E402
from affine.config import load_config  # noqa: E402
from affine.corpus.completion import bash_body, completion_kind, final_completion  # noqa: E402
from affine.corpus.loops import ESCAPE, IN_LOOP, ONSET, label_loops, norm_ws  # noqa: E402
from affine.corpus.materialize import materialize_turn, node_path, stratum_key  # noqa: E402
from affine.corpus.pack import PackResult  # noqa: E402
from affine.corpus.publish import CorpusPublisher  # noqa: E402
from affine.corpus.trace import (  # noqa: E402
    ToolParityError,
    TraceShapeError,
    message_text,
    trace_conversations,
)
from affine.corpus.view import (  # noqa: E402
    VIEW_SPEC,
    build_view_record,
    legacy_view_record,
    main_root_indices,
    reference_leaks,
    rollout_outcome,
    validate_turns,
    view_turns,
)
from affine.corpus.viewpack import FORMAT, pack_view_records  # noqa: E402
from affine.toolbake import ToolBaker  # noqa: E402
from datagen.slicer import _normalize as normalize_fence  # noqa: E402

STATE_DIR = REPO / "ops" / "corpus_build"
STATE_PATH = STATE_DIR / "state.json"
WORK_DIR = STATE_DIR / "work"
CACHE_DIR = STATE_DIR / "cache"
DEFERRED_PATH = WORK_DIR / "deferred_views.jsonl"
SOURCES_TOML = REPO / "rollouts" / "rollouts" / "sources.toml"
PANEL_PATH = REPO / "affine" / "evalsrv" / "data" / "swe_rebench_lite_ids.json"
OFFICIAL_EXCLUDE_PATH = (REPO / "affine" / "evalsrv" / "data"
                         / "swe_rebench_official_exclude.json")
# Records without a source tag predate the unified pipeline: coding.
DEFAULT_GROUP = "coding"
MIN_NEW_TURNS = 200
STALE_AFTER_S = 48 * 3600
# Private Arbos ops channel. Operator directive 2026-09-12: automated posts
# never go to the public SN120 channel (1381987595881414656) again.
DISCORD_GUILD_ID = "1489753158883344497"
DISCORD_CHANNEL_ID = "1510910974498967613"

LANG_BUCKETS = {
    "python": "python", "py": "python",
    "go": "go",
    "java": "java",
    "rs": "rs", "rust": "rs", "c": "rs", "cpp": "rs", "c++": "rs",
    "ts": "tsjs", "js": "tsjs", "typescript": "tsjs", "javascript": "tsjs",
    "php": "tsjs",
}


def log(msg: str) -> None:
    print(f"{datetime.now(timezone.utc).isoformat(timespec='seconds')} {msg}",
          flush=True)


def fatal(msg: str) -> None:
    log(f"FATAL: {msg}")
    sys.exit(1)


# -- state ---------------------------------------------------------------------
def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text())
    return {"folded_chunks": [], "pending": None, "unannounced": None,
            "history": [], "group_counts": {}, "group_strata": {},
            "lang_strata": {}}


def save_state(state: dict) -> None:
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(STATE_PATH)


def env_value(name: str) -> str:
    if os.environ.get(name):
        return os.environ[name]
    for line in (REPO / ".env").read_text().splitlines():
        if line.startswith(f"{name}="):
            return line.split("=", 1)[1].strip()
    return ""


# -- public reads (anonymous, sha-verified) --------------------------------------
class PublicCorpus:
    def __init__(self, base_url: str):
        self.base = base_url.rstrip("/")
        self.http = httpx.Client(timeout=300, follow_redirects=True)

    def get(self, key: str) -> bytes:
        r = self.http.get(f"{self.base}/{key}")
        r.raise_for_status()
        return r.content

    def manifest(self, pointer_key: str) -> tuple[dict, str]:
        raw = self.get(pointer_key)
        sha = hashlib.sha256(raw).hexdigest()
        frozen = self.get(f"{pointer_key.rsplit('/', 1)[0]}/manifests/{sha}.json")
        if hashlib.sha256(frozen).hexdigest() != sha:
            fatal(f"{pointer_key}: pointer does not match its immutable copy")
        return json.loads(raw), sha

    def cached(self, key: str, sha256: str, *, gz_sha: bool) -> Path:
        """Object on disk under CACHE_DIR, verified. `gz_sha`: the manifest
        sha is over the gzip bytes (trace chunks) rather than the payload
        (corpus chunks)."""
        path = CACHE_DIR / key
        if path.exists():
            return path
        blob = self.get(key)
        got = hashlib.sha256(blob if gz_sha else gzip.decompress(blob)).hexdigest()
        if got != sha256:
            fatal(f"{key}: sha mismatch {got} != {sha256}")
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(blob)
        tmp.replace(path)
        return path


def iter_jsonl_gz(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


# -- panel + mix ---------------------------------------------------------------
def panel_keys() -> tuple[set[str], set[str], set[str]]:
    ids = set(json.loads(PANEL_PATH.read_text())["instance_ids"])
    repos = {i.rsplit("-", 1)[0].replace("__", "/").lower() for i in ids}
    official = json.loads(OFFICIAL_EXCLUDE_PATH.read_text())
    ids |= set(official["instance_ids"])
    repos |= {r.lower() for r in official["repos"]}
    bare = {r.rsplit("/", 1)[-1] for r in repos}
    return ids, repos, bare


def load_king_fail() -> dict:
    """[king_fail] from sources.toml: the fold group for the king seat's
    failed rollouts. {} when the block is absent (feature off)."""
    raw = tomllib.loads(SOURCES_TOML.read_text())
    cfg = raw.get("king_fail") or {}
    if not cfg:
        return {}
    common = load_king_common()
    return {"group": "king_fail",
            "strata_buckets": int(cfg.get("strata_buckets", 0) or 0),
            "policy_prefix": str(cfg.get("policy_prefix") or common["policy_prefix"]),
            "exclude_sources": frozenset(str(x) for x in
                                         (cfg.get("exclude_sources") or []))
            | common["exclude_sources"],
            "common": common}


# Turn-routed fold groups (2026-09-11, data events): derive_chunk splits
# single turns of a rollout off into a second record of the same rollout
# with `fold_group` set and a bucketed stratum `<group>:NNNN`. Precedence
# when one turn qualifies for several: king_recoverable > king_pivot >
# king_loop_onset > completion (completion needs a SOLVED rollout, the king
# groups a FAILED one, so only the king groups can overlap).
KING_LOOP_GROUP = "king_loop_onset"
KING_PIVOT_GROUP = "king_pivot"
KING_RECOVERABLE_GROUP = "king_recoverable"
KING_DONE_GROUP = "king_done"
KING_TOOLUSE_GROUP = "king_tooluse"
COMPLETION_GROUP = "completion"
COMPLETION_PRE_GROUP = "completion_pre"
KING_COACHED_GROUP = "king_coached"
KING_DIVERGENCE_GROUP = "king_divergence"
# Env backfill rollouts (internal/coverage/env-backfill-spec.md) live under
# their own prefix and policy id and never enter D; isolation is the traces
# manifest, this is the belt-and-braces drop (`backfill_excluded`).
BACKFILL_POLICY_PREFIX = "backfill_"
BACKFILL_CHUNK_PREFIX = "traces-backfill/"


# tau2-gen decontamination (2026-09-21, Jacob "admit" 10:51 UTC): the tau2-bench
# `base` ids per domain ship with the data; a generated task is admitted only
# if it carries the [GEN:...] marker and its fingerprint (uid without the
# tau2g-e<epoch>-<domain>- prefix and the [GEN:...] suffix -- telecom bench ids
# ARE such fingerprints) is not a bench id. Airline / retail bench ids are
# bare numbers with no name overlap by construction; the [GEN:] marker is the
# fold-side guard there.
DECONTAM: dict[str, dict] = {}     # source -> {"bench": {domain: set(ids)}, "require_gen": bool}
_GEN_RE = re.compile(r"\[GEN:[^\]]*\]")


def load_decontamination() -> dict[str, dict]:
    raw = tomllib.loads(SOURCES_TOML.read_text()).get("decontamination") or {}
    out: dict[str, dict] = {}
    for src, cfg in raw.items():
        if not isinstance(cfg, dict):
            continue
        bench_ids = str(cfg.get("bench_ids") or "")
        path = REPO / bench_ids
        bench: dict[str, set[str]] = {}
        if bench_ids and path.is_file():
            data = json.loads(path.read_text())
            bench = {str(k): set(map(str, v)) for k, v in data.items() if isinstance(v, list)}
        out[str(src)] = {"bench": bench, "require_gen": bool(cfg.get("require_gen_marker", True)),
                         "domain_field": str(cfg.get("domain_field") or "repo")}
    return out


def decontaminated(env: dict) -> str | None:
    """Drop reason for a generated-benchmark task, or None."""
    cfg = DECONTAM.get(str(env.get("source") or ""))
    if not cfg:
        return None
    task = env.get("task") or {}
    uid = str(task.get("uid") or task.get("sid") or "")
    if cfg["require_gen"] and "[GEN:" not in uid:
        return "decontam_no_gen_marker"
    domain = str(task.get(cfg["domain_field"]) or "").rsplit("/", 1)[-1]
    fp = _GEN_RE.sub("", uid)
    fp = re.sub(r"^tau2g-e\d+-[a-z]+-", "", fp)
    ids = cfg["bench"].get(domain, set())
    if fp in ids:                                   # exact fingerprint (telecom ids are fingerprints)
        return "bench_panel_overlap"
    if domain == "telecom" and fp.split("[PERSONA")[0] in {i.split("[PERSONA")[0] for i in ids}:
        return "bench_panel_overlap"                # same intent + fault composition, any persona
    return None


def is_backfill(env: dict, chunk_key: str = "") -> bool:
    pid = str((env.get("policy") or {}).get("id") or "")
    return pid.startswith(BACKFILL_POLICY_PREFIX) or str(chunk_key).startswith(BACKFILL_CHUNK_PREFIX)


# -- bench_fail: the benchsuite's failed king trials as their own group ----------
# Jacob 2026-09-24 17:53/17:56 UTC ("add the failure runs from the benchmarks
# into the dataset ... upsampling the runs where we are doing badly"; mode (a)
# DIRECT approved: "a fair backward pass at this point in the training").
# ops/bench_fail/ingest.py publishes king-failed / teacher-passed benchmark
# trials as envelopes with source `bench_<suite>` and policy `bench_<harness>`
# into the SEPARATE prefix traces-bench/ (own manifest). The fold reads that
# manifest only when `[decontamination].allow_bench_groups` names the group
# and `[bench_fail].mode = "direct"`; every record of a `bench_*` source is
# routed here (group bench_fail, stratum bench_fail:<suite>:NNNN with the
# per-suite bucket share weighted toward the suites the sitting king does
# worst on), and the corpus manifest stamps `trained_on[<suite>]` from the
# first epoch that admitted rows -- the kingboard shows those columns as
# "trained on since <epoch>". Knob off (default) = fail-closed: a `bench_*`
# record that reaches the fold by any other path is dropped
# (`bench_not_allowed`). Mode "variants" is the alternative (teacher-
# synthesised variants of the failed tasks, like scicomp / terminal_gen):
# direct rows are then refused and the variants enter through their own
# generated source with [GEN:] uids -- not built yet, the knob exists so the
# switch is one line.
BENCH_SOURCE_PREFIX = "bench_"
BENCH_FAIL: dict = {}


def load_bench_fail() -> dict:
    raw = tomllib.loads(SOURCES_TOML.read_text())
    cfg = raw.get("bench_fail") or {}
    allow = [str(g) for g in ((raw.get("decontamination") or {}).get("allow_bench_groups") or [])]
    group = str(cfg.get("group") or "bench_fail")
    return {
        "group": group,
        "enabled": bool(cfg.get("enabled", True)) and group in allow,
        "mode": str(cfg.get("mode") or "direct"),
        "traces_prefix": str(cfg.get("traces_prefix") or "traces-bench/"),
        "strata_buckets": int(cfg.get("strata_buckets") or 0),
        "min_reign": int(cfg.get("min_reign") or 0),
        "suite_weights": cfg.get("suite_weights") or "auto",
        "min_suite_buckets": int(cfg.get("min_suite_buckets") or 20),
        "kingboard_matrix": str(cfg.get("kingboard_matrix") or "https://kings.affine.io/api/matrix.json"),
        # Outcomes the router admits. The ingest already keeps only graded-0,
        # non-infra trials; the fold's generic rollout_outcome files a
        # harness end state it does not know (BFCL "done" / "user_closed",
        # tau2 "user_completed" / "tau2_too_many_errors", a mini-swe trial
        # that hit ContextWindowExceeded) under `errored`. Those are king
        # failures, so "errored" is admitted by default -- 7,778 of the
        # first 13,846 routed trials (2026-09-25).
        "accept_outcomes": frozenset(str(x) for x in (cfg.get("accept_outcomes") or ["failed", "errored"])),
    }


def bench_suite_weights(cfg: dict, suites: list[str]) -> dict[str, float]:
    """Share of the group's buckets per suite. "auto" = proportional to
    (1 - sitting king's score / 100) on that benchmark column (upsample where
    the king does worst; a suite without a cell gets the mean weight); a
    table {suite: w} is used as given. Normalised to sum 1."""
    w: dict[str, float] = {}
    sw = cfg.get("suite_weights")
    if isinstance(sw, dict):
        w = {str(k): float(v) for k, v in sw.items() if str(k) in suites}
    else:
        scores: dict[str, float] = {}
        try:
            import httpx
            m = httpx.get(cfg["kingboard_matrix"], headers={"User-Agent": "affine-fold-bench-fail/0.1"}, timeout=30).json()
            row = next((r for r in m.get("rows", []) if r.get("kind") == "king" and r.get("current")), None)
            cols = {c["key"]: c for c in m.get("columns", []) if c.get("kind") == "bench"}
            for key, c in cols.items():
                v = ((row or {}).get("cells") or {}).get(key) or {}
                if isinstance(v.get("score"), (int, float)):
                    scores[str(c.get("env") or key.split(":", 1)[-1]).replace("-", "_")] = float(v["score"])
        except Exception as e:  # noqa: BLE001
            log(f"bench_fail: kingboard matrix unreachable ({e!r}); equal suite weights")
        mean_gap = (sum(1 - x / 100.0 for x in scores.values()) / len(scores)) if scores else 0.5
        for s0 in suites:
            slug = s0.removeprefix(BENCH_SOURCE_PREFIX)
            w[s0] = max(0.05, 1 - scores[slug] / 100.0) if slug in scores else max(0.05, mean_gap)
    tot = sum(w.values()) or 1.0
    return {k: v / tot for k, v in w.items()}


def route_bench_fail(records: list[dict], cfg: dict, drops: dict[str, int]) -> list[dict]:
    """Records of `bench_*` sources -> group bench_fail with a per-suite
    bucketed stratum; everything else passes through. Fail-closed when the
    knob is off or the mode is not direct."""
    bench = [r for r in records if str(r.get("source") or "").startswith(BENCH_SOURCE_PREFIX)]
    if not bench:
        return records
    out = [r for r in records if not str(r.get("source") or "").startswith(BENCH_SOURCE_PREFIX)]
    if not cfg.get("enabled") or cfg.get("mode") != "direct" or cfg["strata_buckets"] <= 0:
        drops["bench_not_allowed"] = drops.get("bench_not_allowed", 0) + len(bench)
        return out
    suites = sorted({str(r["source"]) for r in bench})
    weights = bench_suite_weights(cfg, suites)
    buckets = {s0: max(cfg["min_suite_buckets"], int(round(cfg["strata_buckets"] * weights.get(s0, 0.0)))) for s0 in suites}
    log(f"bench_fail: suites {suites}, weights { {k: round(v, 3) for k, v in weights.items()} }, buckets {buckets}")
    by_stop: dict[str, int] = {}
    for rec in bench:
        oc = rec.get("outcome") or "unscored"
        if oc not in cfg["accept_outcomes"]:
            drops["bench_not_failed"] = drops.get("bench_not_failed", 0) + 1
            continue
        if oc != "failed":
            k = f"{oc}/{rec.get('stop_condition') or '-'}"
            by_stop[k] = by_stop.get(k, 0) + 1
        reign = (rec.get("task") or {}).get("reign") if isinstance(rec.get("task"), dict) else None
        if cfg["min_reign"] and isinstance(reign, int) and reign < cfg["min_reign"]:
            drops["bench_old_reign"] = drops.get("bench_old_reign", 0) + 1
            continue
        src = str(rec["source"])
        key = str(rec.get("instance_id") or rec.get("traj_id"))
        h = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)
        rec["stratum"] = f"{cfg['group']}:{src.removeprefix(BENCH_SOURCE_PREFIX)}:{h % buckets[src]:04d}"
        rec["fold_group"] = cfg["group"]
        out.append(rec)
    if by_stop:
        log(f"bench_fail: non-`failed` outcomes admitted by stop condition {by_stop}")
    return out
KING_GROUPS = ("king_fail", KING_DONE_GROUP, KING_RECOVERABLE_GROUP, KING_DIVERGENCE_GROUP,
               KING_TOOLUSE_GROUP, KING_PIVOT_GROUP, KING_LOOP_GROUP, COMPLETION_PRE_GROUP,
               KING_COACHED_GROUP)
# Phase 10 (Jacob 2026-09-16): "sample more from steps where the teacher
# stops but the king doesn't" -- the stop-state classes, >= 25 % of the slice.
STOP_STATE_GROUPS = (KING_DONE_GROUP, KING_TOOLUSE_GROUP, COMPLETION_PRE_GROUP,
                     COMPLETION_GROUP, KING_DIVERGENCE_GROUP)
# Precedence order when one turn qualifies for several (king-data spec §3.3).
ROUTED_GROUPS = (KING_DONE_GROUP, KING_RECOVERABLE_GROUP, KING_DIVERGENCE_GROUP, KING_TOOLUSE_GROUP,
                 KING_PIVOT_GROUP, KING_LOOP_GROUP, COMPLETION_GROUP, COMPLETION_PRE_GROUP)


def load_king_common() -> dict:
    """[king_common] (king-data spec, 2026-09-13): rules every king_* group
    inherits. `exclude_sources` are unioned into each group's own list;
    `one_reply_ok = false` keeps one-reply rollouts (the state is just the
    task prompt, already in D through the teacher) out of every king group;
    `first_onset_only` keeps one loop onset per rollout; `max_turns_per_
    rollout` caps king_fail; `kind_by_teacher` is a KNOB ONLY (not
    implemented -- it would change which parser scores a turn; Jacob's
    call)."""
    raw = tomllib.loads(SOURCES_TOML.read_text()).get("king_common") or {}
    return {"policy_prefix": str(raw.get("policy_prefix") or "king_"),
            "exclude_sources": frozenset(str(x) for x in (raw.get("exclude_sources") or [])),
            "one_reply_ok": bool(raw.get("one_reply_ok", False)),
            "first_onset_only": bool(raw.get("first_onset_only", True)),
            "max_turns_per_rollout": int(raw.get("max_turns_per_rollout", 0) or 0),
            "kind_by_teacher": bool(raw.get("kind_by_teacher", False)),
            "retire_excluded_published": bool(raw.get("retire_excluded_published", False)),
            "retire_later_onsets": bool(raw.get("retire_later_onsets", False))}


def _group_cfg(group: str) -> dict:
    """Common keys of a `[<group>]` block in sources.toml; {} when absent."""
    raw = tomllib.loads(SOURCES_TOML.read_text())
    cfg = raw.get(group) or {}
    if not cfg:
        return {}
    out = {"group": group,
           "strata_buckets": int(cfg.get("strata_buckets", 0) or 0),
           "policy_prefix": str(cfg.get("policy_prefix") or ""),
           "exclude_sources": frozenset(str(s) for s in
                                        (cfg.get("exclude_sources") or [])),
           "leak_exempt": bool(cfg.get("leak_exempt", False)),
           "raw": cfg}
    if group in KING_GROUPS:
        common = dict(load_king_common())
        out["policy_prefix"] = out["policy_prefix"] or common["policy_prefix"]
        # `inherit_exclude = false`: the group keeps only its own list (a tool
        # group must see the tool sources the others exclude). `lift_exclude`:
        # sources taken back out of the inherited list.
        if cfg.get("inherit_exclude", True):
            out["exclude_sources"] = out["exclude_sources"] | common["exclude_sources"]
        out["exclude_sources"] = out["exclude_sources"] - frozenset(
            str(x) for x in (cfg.get("lift_exclude") or []))
        if "one_reply_ok" in cfg:
            common["one_reply_ok"] = bool(cfg["one_reply_ok"])
        out["common"] = common
    return out


def load_king_loop_onset() -> dict:
    """[king_loop_onset]: the first turn of every loop in the king seat's
    failed rollouts. The leak rule is always waived for this group."""
    cfg = _group_cfg(KING_LOOP_GROUP)
    if cfg:
        cfg["policy_prefix"] = cfg["policy_prefix"] or "king_"
        cfg["leak_exempt"] = True
    return cfg


def _load_side_table(cfg: dict, *, default_dir: str, row_ok, file_ok=None) -> dict:
    """Per-king side-tables `<digest>.jsonl` under `side_table_dir`: one JSON
    line per (rollout_id, turn_idx); rows passing `row_ok` become
    `table[rollout_id][turn_idx]`. Every table in the directory is read:
    an earlier king's states stay valid prefixes."""
    raw = cfg["raw"]
    side_dir = REPO / str(raw.get("side_table_dir") or default_dir)
    table: dict[str, dict[int, dict]] = {}
    n_rows = n_files = 0
    for path in sorted(side_dir.glob("*.jsonl")) if side_dir.is_dir() else []:
        if file_ok is not None and not file_ok(path):
            continue
        n_files += 1
        for line in path.read_text().split("\n"):
            if not line.strip():
                continue
            row = json.loads(line)
            if "rollout_id" not in row or "turn_idx" not in row:
                continue
            n_rows += 1
            if row_ok(row):
                table.setdefault(str(row["rollout_id"]), {})[int(row["turn_idx"])] = row
    cfg.update(table=table, side_table_dir=str(side_dir), n_files=n_files,
               n_rows=n_rows, leak_exempt=True,
               policy_prefix=cfg["policy_prefix"] or "king_",
               readmit_published=bool(raw.get("readmit_published", False)))
    return cfg


def load_king_pivot() -> dict:
    """[king_pivot]: the turns an LLM judge marked as the decision point of
    a failed king rollout (ops/king-review, PR #8). Routed: `admit == true`
    at or above `min_confidence`, category not excluded. Leak rule waived
    as for king_loop_onset."""
    cfg = _group_cfg(KING_PIVOT_GROUP)
    if not cfg:
        return {}
    raw = cfg["raw"]
    excluded = {str(c) for c in (raw.get("exclude_categories") or [])}
    min_conf = float(raw.get("min_confidence", 0.7))
    return _load_side_table(
        cfg, default_dir="affine/state/king_pivots",
        row_ok=lambda row: (bool(row.get("admit"))
                            and str(row.get("failure_category")) not in excluded
                            and float(row.get("confidence") or 0) >= min_conf))


def load_king_recoverable() -> dict:
    """[king_recoverable] (phase 5, 2026-09-12): failure states of the king
    the TEACHER recovers from -- ops/recoverable (PR #13) replays the king's
    prefix and lets the teacher continue; `admit` = teacher solved from the
    state AND its first action differs from the king's. Any state kind
    (loop onset, pivot). Wins over king_pivot / king_loop_onset for the same
    turn. Leak rule waived as for the other king groups."""
    cfg = _group_cfg(KING_RECOVERABLE_GROUP)
    if not cfg:
        return {}
    return _load_side_table(cfg, default_dir="affine/state/recoverable",
                            row_ok=lambda row: bool(row.get("admit")))


def load_king_divergence() -> dict:
    """[king_divergence] (phase 10, Jacob 2026-09-16): the king's FIRST
    out-of-reference action -- the step where the teacher's references stop
    or go elsewhere and the king keeps going its own way (improvement-loop
    worker's divergence side-table, one row per (rollout_id, turn_idx),
    `admit` = the state passed the worker's checks). Precedence below
    king_done / king_recoverable, above king_tooluse / pivot / onset. Leak
    rule waived like every king group; teacher-probe gate applies."""
    cfg = _group_cfg(KING_DIVERGENCE_GROUP)
    if not cfg:
        return {}
    raw = cfg["raw"]
    cfg["leak_exempt"] = True
    min_valid = int(raw.get("min_ref_valid", 2) or 2)
    stop_refs_for_text = int(raw.get("stop_refs_for_text", 2) or 2)
    require_stop = bool(raw.get("require_stop_eligible", False))

    def row_ok(row: dict) -> bool:
        if row.get("admit") is False or str(row.get("outcome") or "failed") != "failed":
            return False
        if require_stop and not row.get("stop_eligible"):
            return False
        return int(row.get("ref_n_valid") or 0) >= min_valid and not row.get("ref_unanimous")

    out = _load_side_table(cfg, default_dir="affine/state/king_divergence", row_ok=row_ok,
                           file_ok=lambda p: ".turns" not in p.name)   # <digest12>.jsonl only
    # The row's teacher references ARE a P4 probe at the state (3 samples,
    # same teacher): register them so the gate does not re-sample. Where
    # >= stop_refs_for_text references stopped (prose / finish), the turn is
    # scored as `text` (kind_by_teacher, as with king_tooluse).
    n_text = 0
    for rid, turns in out["table"].items():
        for ti, row in turns.items():
            refs = row.get("refs") or []
            stops = [r for r in refs if r.get("stop")]
            stop_texts = {norm_ws(str(r.get("visible") or "")) for r in stops if str(r.get("visible") or "").strip()}
            hint = row.get("fold_hint")
            if hint == "text_kind":
                row["_kind"] = dialects.TEXT_KIND
            elif hint == "waive_stored_reply_parse":
                row["_kind"] = None            # teacher acts: keep the policy dialect
            else:
                row["_kind"] = dialects.TEXT_KIND if len(stops) >= stop_refs_for_text else None
            n_text += row["_kind"] is not None
            SIDE_PROBE_ROWS[str(row.get("turn_id") or f"{row.get('traj_id')}:{ti}")] = {
                "turn_id": row.get("turn_id"), "group": KING_DIVERGENCE_GROUP,
                "kind": row["_kind"] or row.get("kind"), "n": len(refs),
                "n_valid": len(stops) if row["_kind"] else int(row.get("ref_n_valid") or 0),
                "identical": (len(stop_texts) <= 1) if row["_kind"] else bool(row.get("ref_unanimous")),
                "n_distinct": len(stop_texts) if row["_kind"] else None,
                "text_valid": len(stops), "text_distinct": len(stop_texts),
                "sample_kinds": ["text" if r.get("stop") else str(row.get("kind")) for r in refs],
                "source": "king_divergence_side_table", "probed_at": row.get("probed_at")}
    out["n_text_kind"] = n_text
    return out


_EXAMPLE_RE = re.compile(r"""(?:such as|e\.g\.|for example|example[s]?:?)\s*['"`]([^'"`\s]{3,64})['"`]""", re.I)


def schema_example_values(trace: dict, prefix_text: str = "") -> set[str]:
    """String values a tool schema offers as EXAMPLES (`example` /
    `examples` / `default` fields, and "such as 'sara_doe_496'" in
    descriptions), from the trace's tool schemas and the baked prefix."""
    out: set[str] = set()

    def walk(x):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ("example", "examples", "default") and isinstance(v, str) and 3 <= len(v) <= 64:
                    out.add(v)
                elif k in ("example", "examples") and isinstance(v, list):
                    out.update(str(e) for e in v if isinstance(e, (str, int)) and 3 <= len(str(e)) <= 64)
                elif k == "description" and isinstance(v, str):
                    out.update(_EXAMPLE_RE.findall(v))
                else:
                    walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(trace.get("tools") or [])
    if prefix_text:
        out.update(_EXAMPLE_RE.findall(prefix_text))
    return out


def action_string_args(action: str) -> set[str]:
    """String argument values of a normalised tool-call action (JSON list of
    {name, arguments}) or of a raw <tool_call> body."""
    vals: set[str] = set()
    try:
        calls = json.loads(action)
    except (ValueError, TypeError):
        calls = None
    if isinstance(calls, list):
        for c in calls:
            args = c.get("arguments") if isinstance(c, dict) else None
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (ValueError, TypeError):
                    args = None
            if isinstance(args, dict):
                vals.update(str(v) for v in args.values() if isinstance(v, (str, int)))
        return vals
    vals.update(re.findall(r"<parameter=[^>]+>\s*([^<\n]{3,64}?)\s*</parameter>", action))
    return vals


def divergence_sublabel(row: dict, env: dict, prefix_text: str = "") -> str | None:
    """`schema_example_where_teacher_asked`: at least one teacher reference
    stopped with a question and the king's action carries an identifier
    that is an example value from the tool schema (tau2-airline read
    2026-09-18: `get_user_details(user_id="sara_doe_496")`).
    `false_confirmation` (tau2-gen admission 2026-09-21): the king STOPPED
    with prose (confirms / reports) where every teacher reference acts --
    it claims an outcome it has not produced."""
    refs = row.get("refs") or []
    if row.get("king_stop") and refs and not any(r.get("stop") for r in refs) \
            and all(r.get("valid") for r in refs):
        return "false_confirmation"
    asked = any(r.get("stop") and "?" in str(r.get("visible") or "") for r in refs)
    if not asked:
        return None
    examples = schema_example_values(env.get("trace") or {}, prefix_text)
    if not examples:
        return None
    args = action_string_args(str(row.get("king_action") or ""))
    return "schema_example_where_teacher_asked" if args & examples else None


def load_king_done() -> dict:
    """[king_done] (king-data spec §2.2, 2026-09-13): done-blind states.
    The king's reply at turn k-1 was completion-eligible by the harness
    rule (affine.corpus.completion: `submit`, a finish tool, Terminus
    `task_complete`, a prose report with no action) yet the rollout went on
    for >= `min_more_turns` more replies on the main root; the routed state
    is turn k -- the first turn where the king kept going instead of
    stopping. One per rollout (the first). Leak rule waived like every king
    group."""
    cfg = _group_cfg(KING_DONE_GROUP)
    if cfg:
        cfg["leak_exempt"] = True
        cfg["min_more_turns"] = int(cfg["raw"].get("min_more_turns", 2) or 2)
        # Wave 5 (affine_mrcr, 2026-09-20): the king writes the right answer,
        # then keeps calling tools -- every loop reply is a `bash` tool call,
        # never prose, so the completion rule sees no "done" reply and the
        # loop labeler is skipped because the rollout is graded SOLVED. With
        # `solved_onset`, the FIRST loop onset of a solved king rollout is a
        # king_done state (the teacher's reference there is the prose stop).
        cfg["solved_onset"] = bool(cfg["raw"].get("solved_onset", True))
        # Looser form (coordinator 2026-09-20 19:02 UTC): in a SOLVED king
        # rollout, the first reply AFTER the answer artefact was last written
        # whose command head repeats an earlier post-write command is the done
        # state -- the textbased kings re-read the answer with near-identical
        # `python3 -c` / `cat` commands before submitting. Guards: the final
        # reply must be the submit (completion-eligible) and the post-write
        # span >= post_write_min_span replies.
        # Default OFF (held 2026-09-20 19:40 UTC): on the published textbased
        # mrcr kings the post-write span is 0-1 replies (one `cat answer.txt`,
        # then submit) -- the rule fired on 1 of 36 solved rollouts and the
        # teacher did not stop there (1 of 3). Turn on once the bash-tool
        # batches land and a 24-state teacher sample shows the prose stop.
        cfg["post_write_repeat"] = bool(cfg["raw"].get("post_write_repeat", False))
        cfg["post_write_min_span"] = int(cfg["raw"].get("post_write_min_span", 2) or 2)
        # king_bad_finish (auto-research 2026-09-22): on sources whose GRADE
        # reads the final reply (affine_sql: "one ```sql block + the submit
        # fence in one reply"), a FAILED king rollout that ended on a
        # completion-eligible reply (submit fence / finish tool /
        # task_complete) is a done state at its final turn, kind `text`, so
        # the whole reply -- block + fence -- is the scored action against the
        # teacher's correct final reply (reign 21: 72 of 80 sql finishes were
        # the bare submit fence; under `bash` the refs' action is the fence
        # alone, identical across refs, R == 0).
        cfg["final_output_sources"] = frozenset(str(x) for x in (cfg["raw"].get("final_output_sources") or []))
        # Duel-time kind: at a done state the teacher stops -- a prose report
        # or a finish tool call; `text` parses both (Jacob 2026-09-13).
        cfg["kind"] = str(cfg["raw"].get("kind") or dialects.TEXT_KIND)
    return cfg


def load_king_tooluse() -> dict:
    """[king_tooluse] (improvement loop P1; Jacob 2026-09-13 14:58 UTC:
    "route based on king so we get points where the king failed"). Selected
    by the KING's behaviour alone:
      (a) one-shot: on a prose-answer prompt set served with tool schemas
          (`prose_sources`; the datagen worker's P1 source), the king's first
          reply is a tool call -> the king's turn 0;
      (b) persist: on a tool source (`tool_sources`), the king repeats the
          same tool call after an error / empty / nudge observation -> that
          turn (`affine.corpus.loops` `persist`).
    Duel-time kind: `text` (`kind`) -- it parses a whole visible reply, tool
    XML included, so the teacher's references stay parseable whether the
    teacher answers in prose or calls a tool; under `tool_call` a prose
    reference would drop and zero the group. One-reply rollouts allowed
    (the prompt-with-tools IS the state); the common exclusions are not
    inherited (they exclude exactly these sources)."""
    cfg = _group_cfg(KING_TOOLUSE_GROUP)
    if not cfg:
        return {}
    raw = cfg["raw"]
    cfg.update(leak_exempt=True,
               prose_sources=frozenset(str(x) for x in (raw.get("prose_sources") or [])),
               tool_sources=frozenset(str(x) for x in (raw.get("tool_sources") or [])),
               # Rule (a) skips rollouts whose task repo is listed: When2Call
               # stamps `repo = when2call/<label>`, and on `tool_call` items a
               # first-reply call is the RIGHT move (datagen worker, 2026-09-14).
               prose_skip_repos=frozenset(str(x) for x in (raw.get("prose_skip_repos") or [])),
               kind=str(raw.get("kind") or dialects.TEXT_KIND))
    cfg["sources"] = cfg["prose_sources"] | cfg["tool_sources"]
    return cfg


def load_completion_pre() -> dict:
    """[completion_pre] (improvement loop P3, king-selected per Jacob
    2026-09-13): the KING finished on its own (`agent_completed`, the final
    main-root reply completion-eligible) and the env graded the rollout
    FAILED -- a premature finish. The routed states are the `n_before` turns
    right before that final reply: where the king should have verified
    (run the tests, check the diff) instead of finishing. Duel-time kind
    `kind` (default `text`, so a teacher that verifies with a tool and a
    teacher that finishes in prose both parse). A king group: leak rule
    waived, common exclusions inherited."""
    cfg = _group_cfg(COMPLETION_PRE_GROUP)
    if cfg:
        cfg.update(leak_exempt=True,
                   n_before=int(cfg["raw"].get("n_before", 2) or 2),
                   kind=str(cfg["raw"].get("kind") or dialects.TEXT_KIND))
    return cfg


def load_completion() -> dict:
    """[completion]: the reply that ended a SOLVED rollout on purpose
    (affine.corpus.completion), teacher and king alike. `policy_prefix`
    is empty = any policy. `min_replies` (default 2): a one-reply rollout
    (math, single-shot answer envs) has no "decide to stop" state -- its
    only turn is the answer, and routing it would drain the source group's
    growth into this one."""
    cfg = _group_cfg(COMPLETION_GROUP)
    if cfg:
        cfg["min_replies"] = int(cfg["raw"].get("min_replies", 2) or 0)
    return cfg


def _policy_ok(env: dict, cfg: dict) -> bool:
    if not cfg or cfg["strata_buckets"] <= 0:
        return False
    pid = str((env.get("policy") or {}).get("id") or "")
    if cfg["policy_prefix"] and not pid.startswith(cfg["policy_prefix"]):
        return False
    return str(env.get("source") or "") not in cfg["exclude_sources"]


def main_reply_count(env: dict) -> int:
    return len(main_root_indices(env["trace"]))


def king_multi_turn(env: dict, cfg: dict) -> bool:
    """One-reply rollouts never route to a king group unless the common
    block says so (`one_reply_ok`)."""
    common = (cfg or {}).get("common") or {}
    return bool(common.get("one_reply_ok")) or main_reply_count(env) >= 2


def king_loop_candidate(env: dict, cfg: dict) -> bool:
    """A rollout the loop labeler runs on: played by a king policy, graded
    FAILED by its env (the same test as `route_king_fail`), from a source
    the group admits. tool_use sources are excluded by config: the teacher's
    next tool call is near-deterministic there, so centered R is ~0 and a
    loop prefix carries no signal (wvk-11 findings)."""
    return (_policy_ok(env, cfg) and king_multi_turn(env, cfg)
            and rollout_outcome(env["trace"]) == "failed")


def king_done_candidate(env: dict, cfg: dict) -> bool:
    """king_done needs no failed grade: the state is "the work is done and
    the king kept going", which a SOLVED rollout shows just as well (bash-
    tool harness 2026-09-14: 49/90 loop-guard stops graded solved). Errored
    / unscored rollouts stay out."""
    return (_policy_ok(env, cfg) and king_multi_turn(env, cfg)
            and rollout_outcome(env["trace"]) in ("failed", "solved"))


def side_table_turns(env: dict, cfg: dict) -> dict[int, dict]:
    """Admitted side-table rows for this rollout, {turn_idx: row}; {} when
    the rollout has none or is not a failed king rollout of an admitted
    source (king_pivot, king_recoverable)."""
    if not cfg or not _policy_ok(env, cfg) or not king_multi_turn(env, cfg):
        return {}
    rows = cfg["table"].get(str(env.get("rollout_id") or ""))
    if not rows or rollout_outcome(env["trace"]) != "failed":
        return {}
    return dict(rows)


ANSWER_WRITE_RE = re.compile(
    r"""(?:>>?\s*|\btee\s+(?:-a\s+)?|\bcp\s+\S+\s+|\bmv\s+\S+\s+|open\(\s*['"]|<parameter=path>\s*|"path":\s*")[^\n'"<]*answer""",
    re.I)
INTERPRETERS = frozenset({"python", "python3", "bash", "sh", "node", "perl", "ruby"})


def reply_command(reply: str, kind: str) -> str | None:
    """The shell command a reply carries (bash fence body, bash-tool
    `<parameter=command>` body, or a JSON `command` argument); None when
    the reply carries no action."""
    acts = dialects.get(kind).actions(reply)
    if not acts:
        return None
    a = acts[-1]
    if kind == dialects.DEFAULT_KIND:
        return bash_body(a)
    m = re.search(r"<parameter=command>\s*(.*?)\s*</parameter>", a, re.S)
    if m:
        return m.group(1)
    m = re.search(r'"command"\s*:\s*"((?:[^"\\]|\\.)*)"', a)
    if m:
        try:
            return json.loads('"' + m.group(1) + '"')
        except ValueError:
            return m.group(1)
    return a


def command_head(cmd: str) -> str:
    toks = cmd.strip().split()
    if not toks:
        return ""
    head = toks[0].rsplit("/", 1)[-1]
    if head in INTERPRETERS and len(toks) > 1:
        return f"{head} {toks[1]}"
    return head


def post_write_repeat_turn(main_convs: list[list[dict]], kind: str, min_span: int) -> int | None:
    """Position (within the main-root replies) of the first post-answer-write
    reply whose command head repeats an earlier post-write reply's head, when
    the rollout ends on a completion-eligible reply and the post-write span
    holds >= min_span replies. None otherwise."""
    replies = [conv[-1]["content"] if conv and conv[-1]["role"] == "assistant" else "" for conv in main_convs]
    if len(replies) < 3 or completion_kind(replies[-1], kind) is None:
        return None
    cmds = [reply_command(r, kind) for r in replies]
    last_write = None
    for j, c in enumerate(cmds[:-1]):
        if c and ANSWER_WRITE_RE.search(c):
            last_write = j
    if last_write is None:
        return None
    span = list(range(last_write + 1, len(replies) - 1))     # exclude the final submit
    if len(span) < min_span:
        return None
    seen: set[str] = set()
    for j in span:
        h = command_head(cmds[j] or "")
        if not h:
            continue
        if h in seen:
            return j
        seen.add(h)
    return None


def king_done_turn(main_convs: list[list[dict]], kind: str, min_more: int,
                   interactive: bool = False) -> int | None:
    """Position (within the main-root replies) of the first turn that
    follows a completion-eligible reply while >= `min_more` replies follow
    it -- the king said/attempted "done" and kept going. None if no such
    turn. In an interactive harness a prose reply is a message to the user,
    not a completion (tau2-gen 2026-09-21: 75 of 66 king rollouts' asks read
    as "done"), so only real finishes (submit / finish tool / task_complete)
    count there."""
    for k in range(1, len(main_convs)):
        if len(main_convs) - k < min_more:
            return None
        prev = main_convs[k - 1]
        if prev and prev[-1]["role"] == "assistant":
            ck = completion_kind(prev[-1]["content"], kind)
            if ck is not None and not (interactive and ck == "text"):
                return k
    return None


def first_reply_is_tool_call(convs: list[list[dict]], main: list[int], kind: str) -> bool | None:
    """Does the first main-root reply carry a tool call (`<tool_call>` block
    or native tool_calls baked into the content)? None when there is no
    reply."""
    if not main:
        return None
    reply = convs[main[0]][-1]["content"]
    return len(dialects.get("tool_call").actions(reply)) >= 1


def cap_king_fail_turns(idx: set[int], escapes: set[int], cap: int) -> set[int]:
    """At most `cap` king_fail turns per rollout: labeler escape turns first
    (recovery states), then the rest spread evenly over depth."""
    if cap <= 0 or len(idx) <= cap:
        return set(idx)
    keep = sorted(i for i in idx if i in escapes)[:cap]
    rest = sorted(i for i in idx if i not in keep)
    room = cap - len(keep)
    if room > 0 and rest:
        step = len(rest) / room
        keep += [rest[min(len(rest) - 1, int(j * step))] for j in range(room)]
    return set(keep)


king_pivot_turns = side_table_turns


def completion_candidate(env: dict, cfg: dict) -> bool:
    """A SOLVED rollout the agent ended itself (`agent_completed`) after at
    least `min_replies` replies."""
    if not (_policy_ok(env, cfg)
            and env["trace"].get("stop_condition") == "agent_completed"
            and rollout_outcome(env["trace"]) == "solved"):
        return False
    n_replies = sum(1 for nd in env["trace"].get("nodes") or []
                    if nd.get("sampled")
                    and (nd.get("message") or {}).get("role") == "assistant")
    return n_replies >= cfg["min_replies"]


def group_stratum(rec: dict, cfg: dict) -> str:
    key = str(rec.get("instance_id") or rec.get("traj_id"))
    h = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)
    return f"{cfg['group']}:{h % cfg['strata_buckets']:04d}"


def drop_excluded_routed(records: list[dict], cfgs: dict[str, dict],
                         drops: dict[str, int]) -> list[dict]:
    """Routed records (deferred carryover included) whose source the
    group's `exclude_sources` now names are dropped (`<group>_excluded_source`);
    derive_chunk never creates new ones, this catches the backlog."""
    out: list[dict] = []
    for rec in records:
        g = rec.get("fold_group")
        cfg = cfgs.get(g) if g else None
        if cfg and str(rec.get("source") or "") in cfg["exclude_sources"]:
            _count(drops, f"{g}_excluded_source")
            continue
        out.append(rec)
    return out


def stamp_routed_groups(records: list[dict], cfgs: dict[str, dict]) -> dict[str, int]:
    """Bucketed stratum `<group>:NNNN` on the records derive split off.
    Runs after `assign_bucket_strata` and `route_king_fail` so it wins over
    both; idempotent for deferred carryover. Returns records stamped per
    group."""
    n: dict[str, int] = {}
    for rec in records:
        g = rec.get("fold_group")
        if g in cfgs and cfgs[g]:
            rec["stratum"] = group_stratum(rec, cfgs[g])
            n[g] = n.get(g, 0) + 1
    return n


def route_king_fail(records: list[dict], king: dict, drops: dict[str, int]
                    ) -> list[dict]:
    """The king seat (2026-09-10). Rollouts played by a `king_*` policy are
    kept only when the env graded them FAILED (`outcome == "failed"`,
    affine.corpus.view.rollout_outcome); they move to the `king_fail`
    group with a bucketed stratum `king_fail:NNNN` (sha256(instance_id) %
    strata_buckets) so the group holds its own slice share instead of
    adding within-stratum variety to the teacher's repo strata. Successful,
    errored (harness/API failure) and unscored king rollouts are dropped
    here (`king_not_failed` / `king_errored` / `king_unscored`). Non-king
    records pass through untouched. Without a
    [king_fail] block every king record is dropped (fail-closed: the seat's
    data never lands unlabelled in the teacher groups). A king record
    another fold step already routed (ROUTED_GROUPS, split off in
    derive_chunk) passes through untouched too."""
    prefix = (king or {}).get("policy_prefix") or "king_"
    n = int((king or {}).get("strata_buckets") or 0)
    out: list[dict] = []
    for rec in records:
        pid = str((rec.get("policy") or {}).get("id") or "")
        if not pid.startswith(prefix):
            out.append(rec)
            continue
        if rec.get("fold_group") in ROUTED_GROUPS:
            out.append(rec)
            continue
        if not king or n <= 0:
            drops["king_no_fold_group"] = drops.get("king_no_fold_group", 0) + 1
            continue
        if str(rec.get("source") or "") in king.get("exclude_sources", ()):
            # 2026-09-12: wiki / agent / math king failures carry no R
            # signal (teacher refs identical there); they leave the seat.
            drops["king_excluded_source"] = drops.get("king_excluded_source", 0) + 1
            continue
        if (not (king.get("common") or {}).get("one_reply_ok")
                and int(rec.get("n_replies", 99)) < 2):
            drops["king_one_reply"] = drops.get("king_one_reply", 0) + 1
            continue
        outcome = rec.get("outcome") or "unscored"
        if outcome == "solved":
            drops["king_not_failed"] = drops.get("king_not_failed", 0) + 1
            continue
        if outcome == "errored":
            # Harness / API failure graded 0 by the env: not the king's doing.
            drops["king_errored"] = drops.get("king_errored", 0) + 1
            continue
        if outcome != "failed":
            drops["king_unscored"] = drops.get("king_unscored", 0) + 1
            continue
        key = str(rec.get("instance_id") or rec.get("traj_id"))
        h = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)
        rec["stratum"] = f"{king['group']}:{h % n:04d}"
        rec["fold_group"] = king["group"]
        out.append(rec)
    return out


def load_mix(*, ignore_fold_mix: bool = False
             ) -> tuple[dict[str, float], dict[str, str], dict[str, float], dict[str, int]]:
    raw = tomllib.loads(SOURCES_TOML.read_text())
    # [fold_mix] overrides [mix] during a dialect notice period; the T0
    # commit deletes the block (rehearsals pass --ignore-fold-mix).
    table = raw.get("mix", {}) if ignore_fold_mix else (raw.get("fold_mix") or raw.get("mix", {}))
    mix = {g: float(v) for g, v in table.items()}
    if abs(sum(mix.values()) - 1.0) > 0.01:
        fatal(f"[fold_mix]/[mix] in {SOURCES_TOML} must sum to 1.0")
    src2grp = {name: cfg["group"] for name, cfg in raw.get("source", {}).items()}
    lang_mix = {b: float(v) for b, v in
                (raw.get("lang_mix", {}).get("coding") or {}).items()}
    if lang_mix and abs(sum(lang_mix.values()) - 1.0) > 0.01:
        fatal(f"[lang_mix.coding] in {SOURCES_TOML} must sum to 1.0")
    buckets = {name: (int(cfg.get("strata_buckets", 0) or 0),
                      int(cfg.get("strata_offset", 0) or 0))
               for name, cfg in raw.get("source", {}).items()}
    return mix, src2grp, lang_mix, buckets


def assign_bucket_strata(records: list[dict], buckets: dict[str, tuple[int, int]],
                         src2grp: dict[str, str]) -> int:
    """Fold-owned slice strata for sources with `strata_buckets` (math,
    tool_use). Datagen stamps the same shape (catalog.bucket_stratum) but
    from the bucket count at generation time; the fold recomputes from the
    CURRENT toml value so the group's slice share follows one setting.
    Bucket = offset + sha256(instance_id) % n, so every rollout of a task
    shares a stratum. Raising n later only adds bucket names above the old
    range (old shards are immutable and stay in the low buckets), so the
    share can be re-derived upward without a rewrite. Bucket names are
    per GROUP, so a second bucketed source in one group must set
    `strata_offset` past the first source's range or the two collide into
    the same strata (affine_wiki 0-469, affine_agent 470-969, 2026-09-07).
    Returns records touched."""
    n_set = 0
    for rec in records:
        if rec.get("fold_group"):
            continue        # a routed group (king_coached, ...) owns its stratum
        n, offset = buckets.get(rec.get("source") or "", (0, 0))
        if n <= 0:
            continue
        group = src2grp.get(rec["source"], DEFAULT_GROUP)
        key = str(rec.get("instance_id") or rec.get("traj_id"))
        h = int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)
        rec["stratum"] = f"{group}:{offset + h % n:04d}"
        n_set += 1
    return n_set


def group_of(rec: dict, src2grp: dict[str, str], mix: dict[str, float]) -> str:
    # A fold-assigned group (king_fail) wins over the source's group.
    g = rec.get("fold_group") or src2grp.get(rec.get("source") or "", DEFAULT_GROUP)
    return g if g in mix else DEFAULT_GROUP


def lang_bucket(rec: dict) -> str:
    return LANG_BUCKETS.get(str(rec.get("language") or "").lower(), "python")


# -- strata budget (phase 9, Jacob 2026-09-14 23:25 UTC: "sample from the
# dataset more aggressively") ------------------------------------------------
# The duel slicer draws one turn per stratum, uniformly over strata, so a
# group's slice share IS its strata share. Two levers, both index-side:
#   buckets:    teacher-trajectory groups (coding, terminal, general,
#               tool_use) are merged into N fixed strata per group
#               (`<group>:b<sha(original stratum) % N>`); no turn leaves D,
#               each is just drawn less often.
#   sub_strata: supply-limited king groups (+ completion) split each task
#               stratum into up to k sub-strata by turn (`<stratum>#<sha(
#               turn_id) % k>`), so a duel may draw up to k different turns
#               of the same task. INTERFACE for the adaptive curriculum
#               (docs/adaptive-curriculum-plan.md): the ledger strips `#k`
#               to the base stratum; `[curriculum] mode = "apply"` sets k
#               per group from the published weights. RT-6 trade-off: a task recurs across duels
#               k times as often; fresh per-duel teacher refs and the
#               block-hash-seeded slice stay the defense, and the fold logs
#               the simulated per-duel recurrence in the announce.
# The original key is kept in the index column `stratum_src`, so the
# mapping is idempotent and re-tunable without a rewrite of chunks.
STRATA_BUDGET: dict = {}
SRC2GRP_GLOBAL: dict[str, str] = {}
MIX_GROUPS_GLOBAL: frozenset[str] = frozenset()   # every [mix] group: a published stratum namespace wins over the source's current group
FOLD_STATS_PATH = STATE_DIR / "fold_stats.json"
FOLD_STATS_KEY = "corpus/fold_stats.json"


def load_floors() -> dict[str, tuple[float, tuple[str, ...]]]:
    """Published slice-share floors, one place: `[curriculum].<name>_floor`
    with `[curriculum].<name>_groups` (MiMo item 3; the stop-state floor from
    phase 10 is the first). Enforced in apply mode by rescaling the vector;
    reported (`floor_status`) every fold."""
    raw = tomllib.loads(SOURCES_TOML.read_text()).get("curriculum") or {}
    out: dict[str, tuple[float, tuple[str, ...]]] = {}
    for k, v in raw.items():
        if k.endswith("_floor") and isinstance(v, (int, float)) and k not in ("floor_coding_terminal",):
            name = k[:-len("_floor")]
            groups = raw.get(f"{name}_groups")
            if isinstance(groups, list) and groups:
                out[name] = (float(v), tuple(str(g) for g in groups))
    return out


def floor_status(after: dict[str, int], floors: dict) -> dict[str, dict]:
    tot = sum(after.values()) or 1
    return {name: {"floor": fl, "groups": list(groups),
                   "share": round(sum(after.get(g, 0) for g in groups) / tot, 4),
                   "ok": sum(after.get(g, 0) for g in groups) / tot >= fl}
            for name, (fl, groups) in floors.items()}


def apply_floors(mix: dict[str, float], floors: dict) -> dict[str, float]:
    """Rescale a group-share vector so every floor block holds (iterate: a
    lift of one block dilutes the others a little)."""
    mix = dict(mix)
    for _ in range(5):
        moved = False
        for name, (fl, groups) in floors.items():
            block = sum(mix.get(g, 0.0) for g in groups)
            if 0 < block < fl:
                up = fl / block
                down = (1 - fl) / max(1e-9, 1 - block)
                mix = {g: v * (up if g in groups else down) for g, v in mix.items()}
                log(f"curriculum apply: {name} block {block:.3f} < floor {fl:.2f}; rescaled")
                moved = True
        if not moved:
            break
    return mix


def load_curriculum() -> dict:
    """[curriculum] (adaptive curriculum, docs/adaptive-curriculum-plan.md;
    hook spec internal/curriculum/hooks-for-fold.md). mode: off | shadow |
    apply. `weights_path` is the published group vector the curriculum job
    writes (JSON: {"groups": {<group>: {"share": x, "m": k}}} or a flat
    {<group>: share}); `shadow` logs it in the announce next to the static
    [mix]; `apply` uses the shares as the group targets and `m` as the
    sub-strata count. A missing / unreadable vector falls back to the static
    [mix] (mode reported as `fallback`)."""
    raw = tomllib.loads(SOURCES_TOML.read_text()).get("curriculum") or {}
    mode = str(raw.get("mode") or "off")
    out = {"mode": mode, "raw": raw, "groups": {}, "m": {}, "path": None, "error": None}
    if mode == "off":
        return out
    # Contract guard (ops/health/contract_compat.py, 2026-09-19): while the
    # curriculum is FROZEN (its inputs' units no longer match the live
    # score_mode, or an operator froze it) the fold falls back to the static
    # [mix] exactly like a missing vector, and the announce line says why.
    # Kept in place by ops/fold/ensure_frozen_hook.py — do not remove.
    frozen_path = REPO / "affine" / "state" / "curriculum" / "FROZEN.json"
    if frozen_path.exists():
        try:
            fz = json.loads(frozen_path.read_text())
        except (OSError, ValueError):
            fz = {}
        out["error"] = f"frozen since {fz.get('frozen_at', '?')} by {fz.get('by', '?')}: {fz.get('reason', 'no reason recorded')}"
        out["frozen"] = True
        return out
    path = REPO / str(raw.get("weights_path") or "ops/curriculum/out/groups.json")
    out["path"] = str(path)
    if not path.exists():
        out["error"] = "missing"
        return out
    try:
        data = json.loads(path.read_text())
        groups = data.get("groups", data) if isinstance(data, dict) else {}
        # Decision 2026-09-15 (coordinator): the fold reads the rule summed
        # over SLICE KEYS (phase-9 buckets / sub-strata), not base strata --
        # per-group `share_raw_slice_keys`, else the top-level table of the
        # same name; then the apply / shadow shares as published.
        top_slice = data.get("share_raw_slice_keys") if isinstance(data, dict) else None
        for g, v in groups.items():
            if isinstance(v, dict):
                share = v.get("share_raw_slice_keys")
                if share is None and isinstance(top_slice, dict):
                    share = top_slice.get(g)
                if share is None:
                    share = v.get("share_applied", v.get("share", v.get("share_shadow")))
                if share is not None:
                    out["groups"][str(g)] = float(share)
                if v.get("m") is not None:
                    out["m"][str(g)] = int(v["m"])
            else:
                out["groups"][str(g)] = float(v)
        out["vector"] = ("share_raw_slice_keys" if any(isinstance(v, dict) and "share_raw_slice_keys" in v
                                                        for v in groups.values()) or isinstance(top_slice, dict)
                         else "share")
        blk = data.get("manifest_curriculum_block") if isinstance(data, dict) else None
        if isinstance(blk, dict) and all(k in blk for k in ("rule_version", "mode", "ledger_sha256",
                                                            "weights_sha256", "manifest_sha256")):
            out["manifest_block"] = {"rule_version": int(blk["rule_version"]), "mode": str(blk["mode"]),
                                     "ledger_sha256": str(blk["ledger_sha256"]),
                                     "weights_sha256": str(blk["weights_sha256"]),
                                     "manifest_sha256": str(blk["manifest_sha256"])}
        if str(data.get("mode") or mode) != mode:
            out["error"] = f"mode mismatch (vector {data.get('mode')} vs toml {mode})"
        age_h = (datetime.now(timezone.utc) - datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)).total_seconds() / 3600
        max_age = float(raw.get("max_vector_age_h", 24) or 24)
        if age_h > max_age:
            out["error"] = f"vector older than {max_age:.0f} h ({age_h:.0f} h)"
        out["meta"] = {k: data.get(k) for k in ("epoch", "ledger_sha256", "weights_sha256", "rule_version", "generated_at")
                       if isinstance(data, dict) and k in data}
        tot = sum(out["groups"].values())
        if not out["groups"] or abs(tot - 1.0) > 0.02:
            out["error"] = f"bad vector (sum {tot:.3f})"
    except (OSError, ValueError, TypeError) as e:
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def curriculum_line(cur: dict, static_mix: dict[str, float]) -> str:
    if not cur or cur["mode"] == "off":
        return ""
    if cur.get("error"):
        return f"curriculum: mode {cur['mode']} -> fallback to static [mix] ({cur['error']}; {cur.get('path')})"
    moves = sorted(((g, cur["groups"].get(g, 0.0) - static_mix.get(g, 0.0)) for g in set(cur["groups"]) | set(static_mix)),
                   key=lambda kv: -abs(kv[1]))[:3]
    meta = cur.get("meta") or {}
    epoch = meta.get("epoch")
    return (f"curriculum: mode {cur['mode']}, rule v{meta.get('rule_version', '?')}, ledger "
            f"{str(meta.get('ledger_sha256') or '')[:12] or 'n/a'}, weights "
            f"{str(meta.get('weights_sha256') or '')[:12] or 'n/a'} ({cur.get('vector', 'share')}); "
            "top moves vs static [mix]: " + ", ".join(f"{g} {d:+.3f}" for g, d in moves)
            + (f"; m>1 on {sorted(g for g, k in cur['m'].items() if k > 1)}" if cur.get("m") else "")
            + (f"; diff https://data.affine.io/curriculum/{epoch}/diff.md" if epoch else ""))


def group_caps(src2grp: dict[str, str]) -> dict[str, int]:
    """Strata ceiling per group (Jacob 2026-09-15, dataset table): the bucket
    N for bucketed groups; strata_buckets x sub-strata k for the fold-routed
    groups; the sum of the sources' strata_buckets otherwise (math)."""
    raw = tomllib.loads(SOURCES_TOML.read_text())
    caps: dict[str, int] = dict((STRATA_BUDGET.get("buckets") or {}))
    sub = STRATA_BUDGET.get("sub_strata") or {}
    for g in (*KING_GROUPS, COMPLETION_GROUP):
        n = int((raw.get(g) or {}).get("strata_buckets", 0) or 0)
        if n and g not in caps:
            caps[g] = n * int(sub.get(g, 1))
    per_src: dict[str, int] = {}
    for name, cfg in (raw.get("source") or {}).items():
        g = src2grp.get(name, DEFAULT_GROUP)
        per_src[g] = per_src.get(g, 0) + int(cfg.get("strata_buckets", 0) or 0)
    for g, n in per_src.items():
        if g not in caps and n:
            caps[g] = n
    return caps


def source_draws(rows: list[tuple[str, str, str]], n_strata: int, n: int = 1300) -> dict[str, dict]:
    """Exact expected draws per duel per source under the evalsrv sampler
    (every stratum drawn with probability n / N, one turn uniformly inside
    it): sum over strata of the source's share of the stratum's turns."""
    per_stratum: dict[str, dict[str, int]] = {}
    for _tid, st, src in rows:
        d = per_stratum.setdefault(st, {})
        d[src] = d.get(src, 0) + 1
    p = min(1.0, n / max(1, n_strata))
    out: dict[str, dict] = {}
    for st, d in per_stratum.items():
        tot = sum(d.values())
        for src, c in d.items():
            o = out.setdefault(src, {"draws_per_duel": 0.0, "turns": 0, "strata": 0})
            o["draws_per_duel"] += p * c / tot
            o["turns"] += c
            o["strata"] += 1
    for o in out.values():
        o["draws_per_duel"] = round(o["draws_per_duel"], 3)
        o["draws_per_turn_per_duel"] = round(o["draws_per_duel"] / o["turns"], 6) if o["turns"] else None
    return out


def yield_report(after: dict[str, int], mix: dict[str, float], group_turns: dict[str, int],
                 src2grp: dict[str, str]) -> dict:
    """MiMo item 3 -- per source: envelopes seen, accepted (records / turns
    at derive), top-3 drop reasons; per group: accepted turns this fold,
    strata now vs target strata (mix share x total), accepted / target."""
    tot = sum(after.values()) or 1
    groups = {}
    for g in sorted(set(after) | set(mix)):
        target = mix.get(g, 0.0) * tot
        groups[g] = {"strata": after.get(g, 0), "target_strata": round(target, 1),
                     "strata_over_target": round(after.get(g, 0) / target, 3) if target else None,
                     "accepted_turns_this_fold": group_turns.get(g, 0)}
    sources = {}
    for src, y in sorted(YIELD.items()):
        top = sorted(y["drops"].items(), key=lambda kv: -kv[1])[:3]
        sources[src] = {"group": src2grp.get(src, DEFAULT_GROUP), "seen": y["seen"], "records": y["records"],
                        "accepted_turns": y["accepted_turns"],
                        "accepted_per_seen": round(y["records"] / y["seen"], 3) if y["seen"] else None,
                        "top_drops": [{"reason": k, "n": v} for k, v in top]}
    subl = {k[len("king_divergence_sublabel_"):]: v for k, v in (NOTES_GLOBAL or {}).items()
            if k.startswith("king_divergence_sublabel_")}
    upstream_fetch_turns = sum(y["drops"].get("upstream_fetch", 0) for y in YIELD.values())
    return {"groups": groups, "sources": sources, "by_king": dict(YIELD_BY_KING),
            # tau2-airline read (2026-09-18): "acted with a schema example value
            # where the teacher asked", per fold and per king digest, so the
            # rate can be tracked reign over reign.
            "divergence_sublabels": subl,
            "interactive_prose_turns": int((NOTES_GLOBAL or {}).get("interactive_prose_turns", 0)),
            # Turns whose prefix already held a successful GitHub / upstream
            # fetch (affine.corpus.upstream). New derivations drop them.
            "upstream_fetch_turns": upstream_fetch_turns,
            # Records of sources not listed in [source.*]: held, never admitted.
            "unknown_source_turns": int((NOTES_GLOBAL or {}).get("unknown_source_turns", 0)),
            "unknown_sources": (NOTES_GLOBAL or {}).get("unknown_sources") or {}}


def write_fold_stats(epoch: int, after: dict[str, int], turns_by_group: dict[str, int],
                     recurrence: dict | None, cur: dict | None, mix: dict[str, float],
                     n_turns: int, publisher, sim_rows: list[tuple[str, str, str]] | None = None,
                     src2grp: dict[str, str] | None = None, extra: dict | None = None) -> None:
    """Per-group / per-source draw statistics for the curriculum job and the
    dataset table (published next to the manifest as corpus/fold_stats.json):
    strata, turns, slice share, expected draws per 1,300-turn duel, draws per
    turn per duel, sub-strata k / bucket N, cap + supply_limited (strata <
    cap), per-source draws, and the simulated per-duel recurrence."""
    tot = sum(after.values()) or 1
    caps = group_caps(src2grp or SRC2GRP_GLOBAL)
    groups = {}
    for g, n_strata in sorted(after.items()):
        share = n_strata / tot
        draws = 1300 * share
        nt = turns_by_group.get(g, 0)
        cap = caps.get(g)
        groups[g] = {"strata": n_strata, "turns": nt, "share": round(share, 5),
                     "draws_per_duel": round(draws, 2),
                     "draws_per_turn_per_duel": round(draws / nt, 6) if nt else None,
                     "sub_strata_k": (STRATA_BUDGET.get("sub_strata") or {}).get(g, 1),
                     "bucket_n": (STRATA_BUDGET.get("buckets") or {}).get(g),
                     "static_mix": mix.get(g),
                     "cap": cap,
                     "supply_limited": (n_strata < cap) if cap else None,
                     "below_static_mix": (share < float(mix.get(g) or 0)) if g in mix else None}
    doc = {"epoch": epoch, "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "n_turns": n_turns, "n_strata": tot, "n_per_duel": 1300,
           "sub_strata_separator": "#", "bucket_prefix": "b",
           "supply_limited_rule": "strata < cap; cap = bucket N (bucketed groups), strata_buckets x k (routed groups), sum of source strata_buckets (others)",
           "groups": groups,
           "sources": source_draws(sim_rows, tot) if sim_rows else None,
           "recurrence": recurrence,
           **(extra or {}),
           "curriculum": {"mode": cur["mode"], "error": cur.get("error"), "groups": cur.get("groups"),
                          "m": cur.get("m"), "meta": cur.get("meta")} if cur else None}
    FOLD_STATS_PATH.write_text(json.dumps(doc, indent=1, sort_keys=True))
    if publisher is not None:
        publisher.put(FOLD_STATS_KEY, json.dumps(doc, sort_keys=True).encode(), "application/json",
                      cache_control="public, max-age=60")
    log(f"fold stats written: {FOLD_STATS_PATH}" + (f" + {FOLD_STATS_KEY}" if publisher is not None else ""))


def load_strata_budget() -> dict:
    raw = tomllib.loads(SOURCES_TOML.read_text()).get("strata_budget") or {}
    if not raw or not raw.get("enabled", False):
        return {}
    return {"buckets": {str(k): int(v) for k, v in (raw.get("buckets") or {}).items() if int(v) > 0},
            "sub_strata": {str(k): int(v) for k, v in (raw.get("sub_strata") or {}).items() if int(v) > 1},
            # the separator / naming version is part of the signature so a
            # rename re-keys the index once
            "signature": json.dumps({"raw": raw, "sub_sep": "#", "v": 2}, sort_keys=True)}


def budget_stratum(group: str, stratum_src: str, turn_id: str, cfg: dict | None = None) -> str:
    """The slice stratum a turn gets under the budget; `stratum_src` is the
    ORIGINAL (pre-budget) key. Identity for groups the budget does not list."""
    cfg = STRATA_BUDGET if cfg is None else cfg
    if not cfg:
        return stratum_src
    n = cfg["buckets"].get(group)
    if n:
        h = int(hashlib.sha256(stratum_src.encode("utf-8")).hexdigest()[:8], 16)
        return f"{group}:b{h % n:05d}"
    k = cfg["sub_strata"].get(group)
    if k:
        # `<stratum>#<k>` -- the interface the curriculum ledger strips
        # (docs/adaptive-curriculum-plan.md §1, §7.1). Stable; do not rename.
        h = int(hashlib.sha256(turn_id.encode("utf-8")).hexdigest()[:8], 16)
        return f"{stratum_src}#{h % k}"
    return stratum_src


BUCKETED_TEACHER_GROUPS = ("math", "tool_use", "general", "agentic_ops", "long_context")


def group_from_row(stratum_src: str, source: str, src2grp: dict[str, str]) -> str:
    """A published row's group. Bucketed strata carry their group as the
    namespace (`general:2660`), and that namespace wins over the source's
    CURRENT group: a source moved to another group (autobench / eog ->
    agentic_ops, mrcr / oolong -> long_context, 2026-09-21) keeps its
    already-published rows where they were folded."""
    ns = str(stratum_src).split(":")[0]
    return ns if ns in ROUTED_GROUPS or ns in KING_GROUPS or ns in BUCKETED_TEACHER_GROUPS or ns in MIX_GROUPS_GLOBAL \
        else src2grp.get(str(source), DEFAULT_GROUP)


def live_rows_for_budget(pub: PublicCorpus, live: dict | None) -> list[tuple[str, str, str]]:
    """(turn_id, ORIGINAL stratum, source) for every live index row -- the
    pre-budget key comes from `stratum_src` once the index carries it."""
    if not live or not live.get("index"):
        return []
    raw = pub.get(live["index"]["key"])
    if hashlib.sha256(raw).hexdigest() != live["index"]["sha256"]:
        fatal(f"live index sha mismatch for {live['index']['key']}")
    t = pq.read_table(io.BytesIO(raw))
    src = t.column("stratum_src").to_pylist() if "stratum_src" in t.column_names \
        else t.column("stratum").to_pylist()
    if SRC_OVERRIDE and "rollout_id" in t.column_names:
        src = [SRC_OVERRIDE.get(str(r), s0) for r, s0 in zip(t.column("rollout_id").to_pylist(), src)]
    return list(zip(t.column("turn_id").to_pylist(), [str(x) for x in src],
                    [str(x) for x in t.column("source").to_pylist()]))


SRC_OVERRIDE: dict[str, str] = {}      # rollout_id -> stratum_src (routed envelopes)


def apply_budget_table(table: pa.Table, src2grp: dict[str, str], cfg: dict) -> pa.Table:
    """Index table -> same rows with `stratum` = budget key and `stratum_src`
    = original key (added when missing). SRC_OVERRIDE re-keys whole rollouts
    (the epoch-41 king_coached rows were stamped with their source group's
    stratum before the routed-group guard existed)."""
    if not cfg:
        return table
    has_src = "stratum_src" in table.column_names
    src_col = table.column("stratum_src").to_pylist() if has_src else table.column("stratum").to_pylist()
    if SRC_OVERRIDE and "rollout_id" in table.column_names:
        rids = table.column("rollout_id").to_pylist()
        src_col = [SRC_OVERRIDE.get(str(r), s0) for r, s0 in zip(rids, src_col)]
    tids = table.column("turn_id").to_pylist()
    sources = table.column("source").to_pylist()
    new = [budget_stratum(group_from_row(s0, src, src2grp), str(s0), tid, cfg)
           for s0, src, tid in zip(src_col, sources, tids)]
    i = table.column_names.index("stratum")
    table = table.set_column(i, "stratum", pa.array(new, pa.string()))
    src_arr = pa.array([str(x) for x in src_col], pa.string())
    return table.set_column(table.column_names.index("stratum_src"), "stratum_src", src_arr) \
        if has_src else table.append_column("stratum_src", src_arr)


def simulate_recurrence(rows: list[tuple[str, str]], n: int = 1300, n_slices: int = 4,
                        seed0: int = 20260914) -> dict:
    """Mean pairwise overlap between simulated duel slices (the evalsrv
    round-robin sampler: seed-shuffled strata, one turn per stratum): share
    of turn ids and of rollouts (traj_id) a slice shares with another."""
    by: dict[str, list[str]] = {}
    for tid, st in rows:
        by.setdefault(st, []).append(tid)
    slices: list[set[str]] = []
    for s in range(n_slices):
        rng = random.Random(seed0 + s)
        keys = sorted(by)
        rng.shuffle(keys)
        picked: list[str] = []
        for k in keys:
            picked.append(rng.choice(by[k]))
            if len(picked) >= n:
                break
        slices.append(set(picked))
    def rollouts(sl: set[str]) -> set[str]:
        return {t.rsplit(":", 1)[0] for t in sl}
    pairs = [(a, b) for i, a in enumerate(slices) for b in slices[i + 1:]]
    t_ov = statistics.mean(len(a & b) / n for a, b in pairs)
    r_ov = statistics.mean(len(rollouts(a) & rollouts(b)) / len(rollouts(a)) for a, b in pairs)
    return {"turn_overlap": round(t_ov, 4), "rollout_overlap": round(r_ov, 4),
            "n_strata": len(by), "n": n, "n_slices": n_slices}


def record_strata(rec: dict) -> set[str]:
    """Slice strata this record's turns land in (affine.corpus.materialize.
    stratum_key on the index row: explicit bucket for math / tool_use,
    repo|phase from traj_id otherwise), under the strata budget when one is
    configured."""
    g = rec.get("fold_group") or SRC2GRP_GLOBAL.get(rec.get("source") or "", DEFAULT_GROUP)
    out: set[str] = set()
    for m in rec["turns"]:
        base = stratum_key({"stratum": m.get("stratum") or rec.get("stratum"),
                            "traj_id": rec.get("traj_id")})
        out.add(budget_stratum(g, base, f"{rec['traj_id']}:{m['turn_idx']}"))
    return out


# Keys with a target at or above this are "anchors" for the GROUP mix: only
# their supply can serve as the reference the other keys are capped
# against (coding 0.43, terminal 0.22). Epoch 25 (2026-09-12) showed why:
# `completion` entered with 759 strata at target 0.03 -- a supply ratio of
# 25,300 against coding's 14,000 -- and became the reference, so terminal's
# cap rose from 0.2162 x 14,000 = 3,025 to 0.2162 x 25,300 = 5,470 and
# 2,443 deferred terminal strata (19,738 turns) entered at once; the slice
# went coding 52 / terminal 26 % -> 41 / 37 %. Small groups can be far
# "over-supplied" relative to a small target while being tiny in absolute
# terms; they must never set the scale of the big ones.
ANCHOR_MIN_TARGET = 0.1


def cap_fill(records: list[dict], keyf, have: dict[str, set[str]],
             targets: dict[str, float], *, anchor_min_target: float = 0.0,
             max_new: dict[str, int] | None = None
             ) -> tuple[list[dict], list[dict], dict[str, set[str]]]:
    """Mix enforcement in SLICE STRATA, not turns.

    sample_slice draws round-robin over strata and n_turns (1300) is far
    below the strata count, so a duel slice holds one turn per stratum: a
    group's share of what miners are scored on is its share of strata, and
    its turn count is irrelevant. Counting turns (the fold until 2026-09-03)
    gave a 1-turn math rollout the weight of a 40-turn coding rollout; the
    traces-only rehearsal selected coding 9,677 turns = 186 strata against
    math 1,935 = 1,935 strata -- coding 4% of the slice.

    Rule: over-supply of key k is (strata available) / target_k -- the
    corpus size k alone could support at its target. The reference is the
    second-highest over-supply among the ANCHOR keys (target >=
    `anchor_min_target`; 0.0 = every key, the rule until 2026-09-12), and
    every key is capped at target_k x reference. With every key an anchor
    only the single most over-supplied key can exceed its cap, so exhausted
    keys (math, tool_use, small languages) never throttle the others (the
    strict waterfill froze D at the first exhausted key) while the one
    flood (terminal 8.6k tasks vs coding 5.3k; python vs the other
    languages) is held to its target ratio against the next-largest
    supply. With anchors restricted to the big groups (the group stage,
    ANCHOR_MIN_TARGET), a small group with a tiny target cannot become the
    reference and lift the big groups' caps, and is itself held to
    target_k x reference. Trimmed rollouts defer and re-enter as the
    reference key grows. Keys without a positive target are deferred
    whole, as before. `max_new` (2026-09-12): per-key ceiling on the strata
    opened in this fold -- the catch-up budget that keeps every group's
    share move under the shift guard, so a backlog enters over several
    folds instead of tripping the guard (general: 111 chunks landed between
    a dry run and its real fold and the group jumped +5.4 points)."""
    pools: dict[str, list[dict]] = {}
    for rec in records:
        pools.setdefault(keyf(rec), []).append(rec)
    keyed = {k: v for k, v in pools.items() if targets.get(k, 0.0) > 0}
    # A key held at 0 (env wave 1: `general = 0.0` in [fold_mix]) is deferred
    # whole below; it has no supply ratio and must not enter the cap math.
    positive = {k: v for k, v in targets.items() if v > 0}
    avail: dict[str, set[str]] = {k: set(have.get(k, ())) for k in positive}
    for k, pool in keyed.items():
        for rec in pool:
            avail[k] |= record_strata(rec)
    anchors = [k for k, v in positive.items() if v >= anchor_min_target] or list(positive)
    supply = sorted((len(avail[k]) / positive[k] for k in anchors), reverse=True)
    ref_total = supply[1] if len(supply) > 1 else float("inf")
    cap = {k: positive[k] * ref_total for k in positive}
    selected: list[dict] = []
    deferred: list[dict] = []
    added: dict[str, set[str]] = {}
    for k, pool in pools.items():
        if k not in keyed:
            deferred.extend(pool)
            continue
        strata = set(have.get(k, ()))
        opened = 0
        limit = (max_new or {}).get(k)
        for rec in pool:
            new = record_strata(rec) - strata
            # A rollout in strata the corpus already holds adds within-stratum
            # variety and moves no share; a rollout opening new strata must fit
            # under the cap and under this fold's budget.
            if new and (len(strata) + len(new) > cap[k] + 1e-9
                        or (limit is not None and opened + len(new) > limit)):
                deferred.append(rec)
                continue
            selected.append(rec)
            strata |= new
            opened += len(new)
            added.setdefault(k, set()).update(new)
    return selected, deferred, added


# Phase 4 (2026-09-12): while coding is BELOW its group target the language
# targets are soft. The hard language cap (python held to 0.25/0.20 of go's
# supply) starved coding: epoch 26 kept 0 of 4,139 coding rollouts, so the
# group could never grow back toward 0.44/0.27 against terminal. Below
# target every language is admitted, python included, subject to a
# per-language ceiling of LANG_SOFT_CEILING of coding's strata and to a
# per-fold budget that keeps coding's share move under the shift guard.
# At or above target the hard cap_fill rule applies again. The ceiling was
# raised 0.45 -> 0.55 the same day so coding reaches its ratio target from
# the existing (python-heavy) backlog in ~3 folds; language diversity is
# to come from non-python coding supply, not from the fold.
LANG_SOFT_CEILING = 0.55   # 0.45 -> 0.55, operator decision 2026-09-12 06:04 UTC
CATCHUP_SHIFT_BUDGET = 0.045   # points of slice share per fold, under MAX_SHARE_SHIFT


def coding_below_target(group_strata: dict[str, list], targets: dict[str, float]
                        ) -> bool:
    """Coding is below target while it is the group stage's reference
    anchor: its supply ratio (strata / target) is not the highest among
    the anchors -- the same test that caps terminal against coding."""
    anchors = {k: v for k, v in targets.items()
               if v >= ANCHOR_MIN_TARGET and k != "coding"}
    if "coding" not in targets or not anchors:
        return False
    coding_supply = len(group_strata.get("coding", [])) / targets["coding"]
    return coding_supply < max(len(group_strata.get(k, [])) / v for k, v in anchors.items())


def catchup_budget(group_strata: dict[str, list], group: str) -> int:
    """New strata `group` may open this fold without moving its share of
    all strata by more than CATCHUP_SHIFT_BUDGET points."""
    total = sum(len(v) for v in group_strata.values()) or 1
    have = len(group_strata.get(group, []))
    target_share = have / total + CATCHUP_SHIFT_BUDGET
    if target_share >= 1:
        return 10 ** 9
    return max(0, int((target_share * total - have) / (1 - target_share)))


def soft_lang_fill(records: list[dict], have: dict[str, set[str]], *,
                   ceiling: float, budget: int
                   ) -> tuple[list[dict], list[dict], dict[str, set[str]]]:
    """Admit coding rollouts of every language; a rollout that opens new
    strata must keep its language at or under `ceiling` of coding's strata
    (after admission) and fit the fold's `budget` of new coding strata.
    Non-python rollouts go first so the scarce languages are never crowded
    out by the budget."""
    counts = {b: len(v) for b, v in have.items()}
    strata = {b: set(v) for b, v in have.items()}
    total = sum(counts.values())
    selected: list[dict] = []
    deferred: list[dict] = []
    added: dict[str, set[str]] = {}
    used = 0
    order = sorted(records, key=lambda r: lang_bucket(r) == "python")
    for rec in order:
        b = lang_bucket(rec)
        new = record_strata(rec) - strata.setdefault(b, set())
        if not new:
            selected.append(rec)
            continue
        n = len(new)
        if used + n > budget or (counts.get(b, 0) + n) > ceiling * (total + n) + 1e-9:
            deferred.append(rec)
            continue
        selected.append(rec)
        strata[b] |= new
        counts[b] = counts.get(b, 0) + n
        total += n
        used += n
        added.setdefault(b, set()).update(new)
    return selected, deferred, added


# -- derive ----------------------------------------------------------------------
# The prefix cap in the unit that binds at duel time: the serving window is
# max_model_len = 131072 tokens minus 1792 generated. MAX_PREFIX_CHARS (300k,
# datagen/slicer.py) is the coarse cut; prefixes above TOKEN_GUARD_FROM_CHARS
# are measured with the teacher tokenizer and dropped past MAX_PREFIX_TOKENS
# (a 2026-09-10 data event; 300k chars ~ 78k tokens p50 / 90k p10).
MAX_PREFIX_TOKENS = 110_000
TOKEN_GUARD_FROM_CHARS = 120_000


class PrefixTokenCache:
    """sha256(prefix text) -> token count, on disk (sqlite under CACHE_DIR).

    Tokenizing a 100-300k-char prefix with the teacher tokenizer costs
    ~0.5 s and the same prefixes come back on every dry run / re-derive
    (2026-09-12: a 213-chunk multi-root back-fill ran > 1.5 h in
    tokenization alone before its real run). The count is a pure function
    of the text and the pinned tokenizer, so caching it changes nothing."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), timeout=60)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("CREATE TABLE IF NOT EXISTS tok (k TEXT PRIMARY KEY, n INTEGER)")
        self.conn.commit()
        self.hits = self.misses = 0

    def count(self, text: str, baker: ToolBaker) -> int:
        k = hashlib.sha256(text.encode("utf-8")).hexdigest()
        row = self.conn.execute("SELECT n FROM tok WHERE k = ?", (k,)).fetchone()
        if row is not None:
            self.hits += 1
            return int(row[0])
        n = len(baker.tok(text, add_special_tokens=False)["input_ids"])
        self.conn.execute("INSERT OR IGNORE INTO tok (k, n) VALUES (?, ?)", (k, n))
        self.conn.commit()
        self.misses += 1
        return n


_TOKEN_CACHE: PrefixTokenCache | None = None


def token_cache() -> PrefixTokenCache:
    global _TOKEN_CACHE
    if _TOKEN_CACHE is None:
        _TOKEN_CACHE = PrefixTokenCache(CACHE_DIR / "prefix_tokens.sqlite")
    return _TOKEN_CACHE


def prefix_over_token_cap(turn: dict, baker: ToolBaker) -> bool:
    if int(turn.get("n_prefix_chars") or 0) <= TOKEN_GUARD_FROM_CHARS:
        return False
    text = "\n".join(m.get("content", "") for m in turn.get("prefix") or [])
    n = token_cache().count(text, baker)
    return n + 8 * len(turn.get("prefix") or []) > MAX_PREFIX_TOKENS


def _count(counter: dict[str, int], key: str, n: int = 1) -> None:
    counter[key] = counter.get(key, 0) + n


def derive_chunk(path: Path, baker: ToolBaker, panel, allowed_kinds,
                 published: set[str], drops: dict[str, int],
                 king_loop: dict | None = None,
                 king_pivot: dict | None = None,
                 completion: dict | None = None,
                 king_recoverable: dict | None = None,
                 king_divergence: dict | None = None,
                 king_done: dict | None = None,
                 king_fail_cfg: dict | None = None,
                 notes: dict[str, int] | None = None,
                 published_king_ns: dict[str, str] | None = None,
                 reclaimed: dict[str, set[str]] | None = None,
                 probe_text: frozenset[str] = frozenset(),
                 king_tooluse: dict | None = None,
                 completion_pre: dict | None = None,
                 leak_exempt_all: bool = False,
                 chunk_key: str = "") -> list[dict]:
    """View records for one trace chunk, with only the turns that pass the
    fold contract and are not yet published. Records with no surviving
    turn are dropped.

    Turn-routed groups (2026-09-11; each `{}`/None = off):
    `king_loop`  [king_loop_onset] -- a failed king rollout's replies are
                 loop-labelled (affine.corpus.loops); loop-onset turns route
                 to `king_loop_onset`, in-loop turns are dropped
                 (`king_in_loop`), the rest keep the `king_fail` path.
    `king_pivot` [king_pivot] -- the judge's admitted pivot turns of a
                 failed king rollout route to `king_pivot`; a turn that is
                 both a pivot and an onset is a pivot.
    `completion` [completion] -- the final reply of a SOLVED
                 `agent_completed` rollout that satisfies the per-harness
                 completion rule (affine.corpus.completion) routes to
                 `completion`, teacher and king alike.
    Routed turns of groups with `leak_exempt` are sliced and validated
    WITHOUT the reference-leakage rule (only that rule; caps, dialect gate,
    panel, action count, token cap still apply) and land in one extra
    record per group of the same rollout (`fold_group`, bucketed stratum).
    Every other turn is derived exactly as before. `notes` collects
    telemetry that is not a drop (`<group>_leak_exempt`: routed turns
    admitted that the leak rule would have refused; `<group>_leaked`:
    candidate turns the leak rule did refuse because the group is not
    exempt).
    `published_king_ns` ({turn_id: king group it is published under}) lets a
    routed king group RECLAIM a turn already in D under a lower-precedence
    king group: the turn is admitted here and its id recorded in
    `reclaimed[<old group>]` so the caller retires the old index row in the
    same revision (one group per turn, precedence wins)."""
    notes = notes if notes is not None else {}
    cfgs = {KING_LOOP_GROUP: king_loop, KING_PIVOT_GROUP: king_pivot,
            COMPLETION_GROUP: completion, KING_RECOVERABLE_GROUP: king_recoverable,
            KING_DIVERGENCE_GROUP: king_divergence,
            KING_DONE_GROUP: king_done, KING_TOOLUSE_GROUP: king_tooluse,
            COMPLETION_PRE_GROUP: completion_pre}
    common = (king_fail_cfg or {}).get("common") or load_king_common()
    out: list[dict] = []
    _prev: tuple | None = None      # (source, drops snapshot, len(out)) of the previous envelope

    def _settle(prev):
        if prev is None:
            return
        src0, snap, n0 = prev
        y = YIELD.setdefault(src0, {"seen": 0, "accepted_turns": 0, "records": 0, "drops": {}})
        for k, v in drops.items():
            d = v - snap.get(k, 0)
            if d > 0:
                y["drops"][k] = y["drops"].get(k, 0) + d
        for r in out[n0:]:
            y["records"] += 1
            y["accepted_turns"] += len(r["turns"])
            pid = str((r.get("policy") or {}).get("id") or "")
            if pid.startswith("king_"):
                kd = r.get("king_digest") or str((r.get("policy") or {}).get("model") or "").rsplit("king-", 1)[-1][:12] or "unknown"
                ky = YIELD_BY_KING.setdefault(kd, {"seen": 0, "records": 0, "turns": 0, "groups": {}, "sources": {}})
                ky["records"] += 1
                ky["turns"] += len(r["turns"])
                g0 = r.get("fold_group") or "king_fail?"
                ky["groups"][g0] = ky["groups"].get(g0, 0) + len(r["turns"])

    for env in iter_jsonl_gz(path):
        _settle(_prev)
        _prev = None
        if is_backfill(env, chunk_key):
            _count(drops, "backfill_excluded")
            continue
        _dc = decontaminated(env)
        if _dc:
            _count(drops, _dc)
            _count(drops, f"{_dc}_{env.get('source')}")
            continue
        _src = str(env.get("source") or "")
        YIELD.setdefault(_src, {"seen": 0, "accepted_turns": 0, "records": 0, "drops": {}})["seen"] += 1
        _pid = str((env.get("policy") or {}).get("id") or "")
        if _pid.startswith("king_"):
            _kd = str((env.get("policy") or {}).get("model") or "").rsplit("king-", 1)[-1][:12] or "unknown"
            _ky = YIELD_BY_KING.setdefault(_kd, {"seen": 0, "records": 0, "turns": 0, "groups": {}, "sources": {}})
            _ky["seen"] += 1
            _ky["sources"][_src] = _ky["sources"].get(_src, 0) + 1
        _prev = (_src, dict(drops), len(out))
        convs = None
        route: dict[int, str] = {}          # turn_idx -> group (final)
        extra: dict[int, dict] = {}         # turn_idx -> meta fields to stamp
        in_loop: set[int] = set()
        kind = (env.get("policy") or {}).get("action_kind") or "bash"
        want_loop = bool(king_loop) and king_loop_candidate(env, king_loop)
        want_done_onset = (bool(king_done) and king_done.get("solved_onset") and _policy_ok(env, king_done)
                           and king_multi_turn(env, king_done) and rollout_outcome(env["trace"]) == "solved")
        want_bad_finish = (bool(king_done) and str(env.get("source") or "") in king_done.get("final_output_sources", ())
                           and _policy_ok(env, king_done) and king_multi_turn(env, king_done)
                           and rollout_outcome(env["trace"]) == "failed"
                           and env["trace"].get("stop_condition") == "agent_completed")
        pivots = side_table_turns(env, king_pivot) if king_pivot else {}
        recoverable = side_table_turns(env, king_recoverable) if king_recoverable else {}
        divergence = side_table_turns(env, king_divergence) if king_divergence else {}
        divergence_text: dict[int, str] = {}
        # auto-research 2026-09-20: rows whose KING reply has no action while
        # the teacher's refs act. The duel scores prefix + fresh samples, never
        # the stored reply, so the slicer's one-action check is waived for
        # them: the reply is admitted through the text fallback and the turn
        # keeps the policy dialect (kind_stamp back to `kind`).
        divergence_waive: set[int] = set()
        want_done = bool(king_done) and king_done_candidate(env, king_done)
        want_tooluse = (bool(king_tooluse) and _policy_ok(env, king_tooluse)
                        and str(env.get("source") or "") in king_tooluse["sources"])
        # P3 (king-selected): the king finished by itself and still failed.
        want_pre = (bool(completion_pre) and _policy_ok(env, completion_pre)
                    and king_multi_turn(env, completion_pre)
                    and env["trace"].get("stop_condition") == "agent_completed"
                    and rollout_outcome(env["trace"]) == "failed")
        is_king_fail = (bool(king_fail_cfg) and _policy_ok(env, king_fail_cfg)
                        and rollout_outcome(env["trace"]) == "failed")
        one_reply_king = (is_king_fail and not king_multi_turn(env, king_fail_cfg))
        escapes: set[int] = set()
        want_completion = bool(completion) and completion_candidate(env, completion)
        interactive = str(env.get("source") or "") in INTERACTIVE_SOURCES
        if want_loop or want_completion or want_done or want_tooluse or want_pre or interactive or want_done_onset or want_bad_finish:
            try:
                convs = trace_conversations(env["trace"], baker)
            except (ToolParityError, TraceShapeError) as e:
                _count(drops, type(e).__name__)
                continue
        # Side conversations (other roots: WebFetch summaries, sub-agents,
        # compaction) are real turns but not part of the main loop, so loop
        # labels and the completion rule see the MAIN root's replies only.
        main = main_root_indices(env["trace"]) if convs else []
        main_convs = [convs[i] for i in main] if convs else []
        if convs and len(main) != len(convs):
            _count(notes, "multi_root_rollouts")
        if want_completion and main_convs:
            ckind = final_completion(main_convs, kind)
            _count(notes, "completion_candidates")
            if ckind:
                i = main[-1]
                route[i] = COMPLETION_GROUP
                extra[i] = {"completion_kind": ckind}
                _count(notes, f"completion_kind_{ckind}")

        later_onsets: set[int] = set()
        loop_labels = None
        if (want_loop or want_tooluse or want_done_onset) and main_convs:
            loop_labels = label_loops(main_convs, kind)
        if want_loop and main_convs:
            n_on = 0
            for j, lab in enumerate(loop_labels):
                i = main[j]
                if lab.label == ONSET:
                    n_on += 1
                    if n_on > 1 and common["first_onset_only"]:
                        # Later onsets are near-duplicate prefixes of the same
                        # wreck (first onsets recover 28 %, later 13 %).
                        later_onsets.add(i)
                        continue
                    route[i] = KING_LOOP_GROUP
                    extra[i] = {"loop_onset_of": int(main[int(lab.repeats)]),
                                "onset_rank": n_on}
                elif lab.label == IN_LOOP:
                    in_loop.add(i)
                elif lab.label == ESCAPE:
                    escapes.add(i)
            _count(notes, "king_loop_labelled_rollouts")
            _count(notes, "king_loop_onset_labels", n_on)
            _count(notes, "king_loop_in_loop_labels", len(in_loop))
            _count(notes, "king_later_onset_labels", len(later_onsets))
        if want_done and main_convs:
            k = king_done_turn(main_convs, kind, king_done["min_more_turns"], interactive=interactive)
            if k is not None:
                i = main[k]
                _count(notes, "king_done_states")
                if route.get(i) == KING_LOOP_GROUP:
                    _count(notes, "king_done_over_onset")
                done_route = i
            else:
                done_route = None
        else:
            done_route = None
        done_rule = "completion_then_continued"
        if done_route is None and want_done_onset and loop_labels:
            # Solved rollout that looped: the first onset is the done state.
            for j, lab in enumerate(loop_labels):
                if lab.label == ONSET:
                    done_route = main[j]
                    done_rule = "solved_loop_onset"
                    _count(notes, "king_done_states")
                    _count(notes, "king_done_solved_onset")
                    _count(notes, f"king_done_solved_onset_{env.get('source')}")
                    break
        if done_route is None and want_bad_finish and main_convs:
            final_reply = main_convs[-1][-1]["content"] if main_convs[-1] and main_convs[-1][-1]["role"] == "assistant" else ""
            if completion_kind(final_reply, kind) is not None:
                done_route = main[-1]
                done_rule = "king_bad_finish"
                _count(notes, "king_done_states")
                _count(notes, "king_done_bad_finish")
                _count(notes, f"king_done_bad_finish_{env.get('source')}")
        if done_route is None and want_done_onset and king_done.get("post_write_repeat") and main_convs:
            j = post_write_repeat_turn(main_convs, kind, king_done["post_write_min_span"])
            if j is not None:
                done_route = main[j]
                done_rule = "post_write_repeat"
                _count(notes, "king_done_states")
                _count(notes, "king_done_post_write_repeat")
                _count(notes, f"king_done_post_write_repeat_{env.get('source')}_{(env.get('policy') or {}).get('harness')}")
        if pivots:
            _count(notes, "king_pivot_rollouts")
            for i, row in pivots.items():
                if route.get(i) == KING_LOOP_GROUP:
                    _count(notes, "king_pivot_over_onset")
                route[i] = KING_PIVOT_GROUP
                in_loop.discard(i)
                later_onsets.discard(i)
                extra[i] = {"pivot": {
                    "category": row.get("failure_category"),
                    "confidence": row.get("confidence"),
                    "judge": row.get("judge_model"),
                    "prompt_hash": row.get("prompt_hash")}}
        if recoverable:
            _count(notes, "king_recoverable_rollouts")
            for i, row in recoverable.items():
                if route.get(i) in (KING_LOOP_GROUP, KING_PIVOT_GROUP):
                    _count(notes, f"king_recoverable_over_{route[i]}")
                route[i] = KING_RECOVERABLE_GROUP
                in_loop.discard(i)
                extra[i] = {"recoverable": {
                    "state_kind": row.get("state_kind"),
                    "teacher_turns": row.get("teacher_turns"),
                    "teacher_first_action_kind": row.get("teacher_first_action_kind"),
                    # PR #13: "same_task" = the ACP same-task proxy (the teacher
                    # solved the task, not necessarily from this state); the
                    # pipeline caps it at RECOVERABLE_ACP_MAX_SHARE. Tagged so
                    # the duel telemetry can compare proxy vs continuation rows.
                    "proxy": row.get("proxy"),
                    "timestamp": row.get("timestamp")}}
                _count(notes, f"king_recoverable_proxy_{row.get('proxy') or 'continuation'}")
        if divergence:
            _count(notes, "king_divergence_rollouts")
            for i, row in divergence.items():
                if route.get(i) in (KING_DONE_GROUP, KING_RECOVERABLE_GROUP):
                    continue        # higher precedence keeps the turn
                if route.get(i) in (KING_LOOP_GROUP, KING_PIVOT_GROUP):
                    _count(notes, f"king_divergence_over_{route[i]}")
                route[i] = KING_DIVERGENCE_GROUP
                in_loop.discard(i)
                extra[i] = {"divergence": {k: row.get(k) for k in (
                    "divergence_kind", "stop_eligible", "ref_stop", "king_stop", "ref_n_valid",
                    "ref_unanimous", "king", "probed_at") if k in row}}
                sub = divergence_sublabel(row, env)
                if sub:
                    extra[i]["divergence"]["sublabel"] = sub
                    _count(notes, f"king_divergence_sublabel_{sub}")
                    _count(notes, f"king_divergence_sublabel_{sub}_{row.get('king') or 'king'}")
                if row.get("_kind"):
                    divergence_text[i] = row["_kind"]
                elif row.get("fold_hint") == "waive_stored_reply_parse":
                    divergence_waive.add(i)
                    _count(notes, "king_divergence_waived_stored_reply")
        kind_stamp: dict[int, str] = dict(divergence_text)     # turn -> duel-time action_kind
        if interactive and convs and kind != dialects.TEXT_KIND:
            # Mid-trajectory prose replies of an interactive harness are
            # scorable `text` turns (the teacher asks the user, then acts).
            d_pol = dialects.get(kind)
            for i in main:
                reply = convs[i][-1]["content"] if convs[i] and convs[i][-1]["role"] == "assistant" else ""
                if reply.strip() and not d_pol.actions(reply) and not dialects.get("tool_call").actions(reply):
                    if i not in route:
                        kind_stamp.setdefault(i, dialects.TEXT_KIND)
                        _count(notes, "interactive_prose_turns")
        if want_tooluse and convs and main:
            src = str(env.get("source") or "")
            # (a) one-shot on a prose-answer prompt set: the king opened with a tool call.
            repo = str((env.get("task") or {}).get("repo") or "")
            if src in king_tooluse["prose_sources"] and repo in king_tooluse["prose_skip_repos"]:
                _count(notes, "king_tooluse_skip_tool_call_label")
            elif src in king_tooluse["prose_sources"] and first_reply_is_tool_call(convs, main, kind):
                i = main[0]
                if route.get(i) not in (KING_DONE_GROUP, KING_RECOVERABLE_GROUP):
                    route[i] = KING_TOOLUSE_GROUP
                    in_loop.discard(i); later_onsets.discard(i)
                    extra[i] = {"tooluse": {"rule": "tool_call_on_prose_prompt"}}
                    kind_stamp[i] = king_tooluse["kind"]
                    _count(notes, "king_tooluse_one_shot")
            # (b) persist on a tool source: the same tool call again after a
            # bad observation (the king's own labels; no teacher involved).
            if src in king_tooluse["tool_sources"]:
                for j, lab in enumerate(loop_labels or []):
                    i = main[j]
                    if not lab.persist or route.get(i) in (KING_DONE_GROUP, KING_RECOVERABLE_GROUP):
                        continue
                    if not dialects.get("tool_call").actions(convs[i][-1]["content"]):
                        continue
                    route[i] = KING_TOOLUSE_GROUP
                    in_loop.discard(i); later_onsets.discard(i)
                    extra[i] = {"tooluse": {"rule": "persist_after_bad_obs", "prev_obs": lab.prev_obs}}
                    kind_stamp[i] = king_tooluse["kind"]
                    _count(notes, "king_tooluse_persist")
        if want_pre and main_convs and len(main) >= 2:
            final = main[-1]
            if completion_kind(main_convs[-1][-1]["content"], kind) is not None:
                for back in range(1, completion_pre["n_before"] + 1):
                    j = len(main) - 1 - back
                    if j < 0:
                        break
                    i = main[j]
                    if i in route:
                        # Lowest king precedence: never displaces an onset,
                        # pivot, recoverable, tooluse or done state.
                        continue
                    route[i] = COMPLETION_PRE_GROUP
                    in_loop.discard(i); later_onsets.discard(i)
                    extra[i] = {"pre_finish": {"final_turn": int(final), "rank": back}}
                    kind_stamp[i] = completion_pre["kind"]
                    _count(notes, "completion_pre_states")
        if done_route is not None:
            if route.get(done_route) in (KING_RECOVERABLE_GROUP, KING_PIVOT_GROUP):
                _count(notes, f"king_done_over_{route[done_route]}")
            route[done_route] = KING_DONE_GROUP
            in_loop.discard(done_route)
            later_onsets.discard(done_route)
            extra[done_route] = {"done": {"after_turn": int(done_route) - 1, "rule": done_rule}}
            kind_stamp[done_route] = king_done["kind"]
        if one_reply_king and not any(g == KING_TOOLUSE_GROUP for g in route.values()):
            # The state is the task prompt, which the teacher's own rollout
            # already puts in D: nothing of this rollout enters a king group
            # (king_tooluse is the exception: the prompt-with-tools IS the state).
            _count(drops, "king_one_reply")
            continue
        leak_exempt = frozenset(i for i, g in route.items() if cfgs[g]["leak_exempt"])
        if leak_exempt_all:
            # every reply index; convs may not have been built (no routing)
            leak_exempt = frozenset(range(10_000))
        # Turns scored under `text` may be recorded from a reply with no action
        # in the policy dialect (the affine_sql king answers with a bare
        # ```sql block; 26 of 28 refused king_done states, 2026-09-13).
        text_replies = frozenset(i for i, k in kind_stamp.items() if k == dialects.TEXT_KIND) | frozenset(divergence_waive)
        for i in divergence_waive:
            kind_stamp[i] = kind           # slicer admits via text; meta reverts to the policy dialect
        try:
            rec = build_view_record(env, baker=baker,
                                    generated_at=env.get("stored_at"),
                                    convs=convs, leak_exempt=leak_exempt,
                                    text_replies=text_replies)
        except (ToolParityError, TraceShapeError) as e:
            _count(drops, type(e).__name__)
            continue
        if rec is None:
            if route and convs:
                _count_leaked(route, convs, kind, leak_exempt, notes)
            _count(drops, "no_scorable_turn")
            continue
        _pid0 = str((env.get("policy") or {}).get("id") or "")
        if _pid0.startswith("king_"):
            # Served king digest (policy.model `king/king-<digest12>`) on the
            # record, so admissions can be reported per king (2026-09-19).
            rec["king_digest"] = str((env.get("policy") or {}).get("model") or "").rsplit("king-", 1)[-1][:12] or None
        if kind_stamp:
            # Duel-time kind per routed turn (Jacob 2026-09-13: whatever keeps
            # the teacher's references parseable at the state). Stamped on
            # the meta so the record, the index and the duel all see it.
            # `text` needs a system message in the prefix (its mandate check);
            # Terminus prefixes have none, so those keep the policy dialect.
            for m in rec["turns"]:
                k_new = kind_stamp.get(m["turn_idx"])
                if not k_new or k_new == m["action_kind"]:
                    continue
                prefix = [{"role": nd["role"], "content": nd["content"]}
                          for nd in node_path(rec["nodes"], int(m["node_id"]))[:-1]]
                if not dialects.get(k_new).mandate_ok(prefix):
                    _count(notes, f"{route.get(m['turn_idx'], 'interactive')}_kind_kept_{m['action_kind']}")
                    continue
                m["action_kind"] = k_new
                _count(notes, f"{route.get(m['turn_idx'], 'interactive')}_kind_{k_new}")
        turns = view_turns(rec)
        present = {t["turn_idx"] for t in turns}
        if route and convs:
            _count_leaked({i: g for i, g in route.items() if i not in present},
                          convs, kind, leak_exempt, notes)
        rest = [t for t in turns if t["turn_idx"] not in route
                and t["turn_idx"] not in in_loop and t["turn_idx"] not in later_onsets]
        routed = [t for t in turns if t["turn_idx"] in route]
        n_later = sum(1 for t in turns if t["turn_idx"] in later_onsets
                      and t["turn_idx"] not in route)
        n_in_loop = len(turns) - len(routed) - len(rest) - n_later
        if n_in_loop:
            _count(drops, "king_in_loop", n_in_loop)
        if n_later:
            _count(drops, "king_later_onset", n_later)
        kept, d = validate_turns(rest, panel=panel, allowed_kinds=allowed_kinds,
                                 leak_check=not leak_exempt_all)
        for k, v in d.items():
            _count(drops, k, v)
        kept_routed: list[dict] = []
        for g in ROUTED_GROUPS:
            gturns = [t for t in routed if route[t["turn_idx"]] == g]
            if not gturns:
                continue
            kg, d = validate_turns(gturns, panel=panel, allowed_kinds=allowed_kinds,
                                   leak_check=not cfgs[g]["leak_exempt"])
            kept_routed += kg
            for k, v in d.items():
                _count(drops, k, v)
        keep_idx: set[int] = set()
        keep_routed: dict[str, set[int]] = {}
        for t in [*kept, *kept_routed]:
            tid = f"{t['traj_id']}:{t['turn_idx']}"
            g = route.get(t["turn_idx"])
            if tid in published:
                old_ns = (published_king_ns or {}).get(tid)
                rank = {grp: i for i, grp in enumerate(ROUTED_GROUPS)}
                if (g in KING_GROUPS and old_ns in KING_GROUPS and old_ns != g
                        and rank.get(g, 99) < rank.get(old_ns, 99)
                        and reclaimed is not None):
                    reclaimed.setdefault(old_ns, set()).add(tid)
                    _count(notes, f"{g}_reclaimed_from_{old_ns}")
                elif (g in KING_GROUPS and old_ns in KING_GROUPS and tid in probe_text
                        and reclaimed is not None):
                    # Probe says the teacher answers in prose here: the row is
                    # re-published with kind `text` (chunk records are
                    # immutable, so a new record replaces the old row).
                    reclaimed.setdefault(old_ns, set()).add(tid)
                    _count(notes, f"{g}_restamp_reclaimed")
                else:
                    _count(drops, "already_published")
                    if g is not None:
                        _count(notes, f"{g}_already_published")
                    continue
            if prefix_over_token_cap(t, baker):
                _count(drops, "prefix_too_many_tokens")
                continue
            if g is None:
                keep_idx.add(t["turn_idx"])
                continue
            keep_routed.setdefault(g, set()).add(t["turn_idx"])
            leaks = reference_leaks(t["prefix"], dialects.last_action(
                t["reference_turn"], t["action_kind"]))
            _count(notes, f"{g}_leak_exempt" if leaks else f"{g}_not_leaking")
        if is_king_fail and common["max_turns_per_rollout"] \
                and len(keep_idx) > common["max_turns_per_rollout"]:
            capped = cap_king_fail_turns(keep_idx, escapes, common["max_turns_per_rollout"])
            _count(drops, "king_fail_cap", len(keep_idx) - len(capped))
            keep_idx = capped
        metas = rec["turns"]
        rec["turns"] = [m for m in metas if m["turn_idx"] in keep_idx]
        if is_king_fail:
            rec["n_replies"] = len(main)
        if rec["turns"]:
            out.append(rec)
        for g in ROUTED_GROUPS:
            idx = keep_routed.get(g)
            if not idx:
                continue
            grec = dict(rec)
            grec["turns"] = [{**m, **extra.get(m["turn_idx"], {})}
                             for m in metas if m["turn_idx"] in idx]
            grec["fold_group"] = g
            grec["stratum"] = group_stratum(grec, cfgs[g])
            out.append(grec)
    _settle(_prev)
    return out


def _count_leaked(missing: dict[int, str], convs: list[list[dict]], kind: str,
                  leak_exempt: frozenset[int], notes: dict[str, int]) -> None:
    """Routed turns the slicer did not admit: was it the leak rule? (Only
    non-exempt groups can lose a turn to it; the count answers "does this
    group need the exemption".)"""
    for i, g in missing.items():
        if i in leak_exempt or i >= len(convs):
            continue
        conv = convs[i]
        acts = dialects.get(kind).actions(normalize_fence(conv[-1]["content"]))
        if len(acts) != 1:
            _count(notes, f"{g}_missing_other")
            continue
        prefix = [{"role": m["role"], "content": m["content"]} for m in conv[:-1]]
        _count(notes, f"{g}_leaked" if reference_leaks(prefix, acts[0])
               else f"{g}_missing_other")


def legacy_records(pub: PublicCorpus, turns_manifest: dict) -> list[dict]:
    """The live v2 corpus (index row order) as v4 legacy records."""
    idx = turns_manifest["index"]
    raw = pub.get(idx["key"])
    if hashlib.sha256(raw).hexdigest() != idx["sha256"]:
        fatal("live v2 index sha mismatch")
    table = pq.read_table(io.BytesIO(raw), columns=["traj_id", "chunk_key"])
    order: list[tuple[str, str]] = []
    seen: set[str] = set()
    for tid, ck in zip(table.column("traj_id").to_pylist(),
                       table.column("chunk_key").to_pylist()):
        if tid not in seen:
            seen.add(tid)
            order.append((tid, ck))
    by_key = {s["key"]: s for s in turns_manifest["shards"]
              if s.get("active") and s.get("format") == "traj_v1"}
    trajs: dict[str, dict] = {}
    for key, shard in by_key.items():
        epoch = int(key.rsplit("chunk_", 1)[1].split("_")[0])
        for traj in iter_jsonl_gz(pub.cached(key, shard["sha256"], gz_sha=False)):
            trajs[traj["traj_id"]] = legacy_view_record(traj, legacy_epoch=epoch)
    missing = [tid for tid, _ in order if tid not in trajs]
    if missing:
        fatal(f"{len(missing)} indexed trajectories missing from active chunks")
    return [trajs[tid] for tid, _ in order]


def published_turn_ids(pub: PublicCorpus, manifest: dict | None) -> set[str]:
    if not manifest or not manifest.get("index"):
        return set()
    idx = manifest["index"]
    raw = pub.get(idx["key"])
    if hashlib.sha256(raw).hexdigest() != idx["sha256"]:
        fatal(f"live index sha mismatch for {idx['key']}")
    table = pq.read_table(io.BytesIO(raw), columns=["turn_id"])
    return set(table.column("turn_id").to_pylist())


def index_table(pub: PublicCorpus, manifest: dict | None,
                columns: list[str]) -> pa.Table | None:
    """The live index (sha-verified) with the given columns; None when the
    corpus has no schema-3 manifest yet."""
    if not manifest or not manifest.get("index"):
        return None
    idx = manifest["index"]
    raw = pub.get(idx["key"])
    if hashlib.sha256(raw).hexdigest() != idx["sha256"]:
        fatal(f"live index sha mismatch for {idx['key']}")
    return pq.read_table(io.BytesIO(raw), columns=columns)


# -- math re-source (2026-09-12, data plan phase 3) -----------------------------
# Math is 92 % dead on the R leg: the teacher boxes the same answer on 75 %
# of math turns, so its k = 3 references are identical and centered R is
# exactly 0. The 8 % of turns where the teacher disagrees with itself carry
# R = 0.145 per byte, 15x a coding turn (docs/duel-signal-by-group.md). The
# fold cannot see duel-time references, so it uses the traces as a proxy:
# a problem is KEPT when the teacher's own datagen rollouts on it show
# disagreement (>= 2 distinct normalized \boxed{} strings across its
# samples, or the teacher failed it at least once) or the king failed it.
# Everything else is deterministic for the teacher and leaves D: new
# candidates are dropped (`math_deterministic`) and, with
# `retire_published`, the published turns of those problems are removed
# from the INDEX (the chunk objects stay; old manifests replay unchanged).
TRAJ_SHA8_RE = re.compile(r"\.([0-9a-f]{8})\.pr_")


def load_math_filter() -> dict:
    raw = tomllib.loads(SOURCES_TOML.read_text())
    cfg = raw.get("math_filter") or {}
    if not cfg or not cfg.get("enabled", True):
        return {}
    sources = cfg.get("sources") or [cfg.get("source") or "affine_math"]
    return {"source": str(sources[0]),
            "sources": frozenset(str(x) for x in sources),
            # P4 (2026-09-14): the teacher must have >= this many boxed answers
            # that fit the duel's reference cap among its datagen samples;
            # 0 disables. cap_chars approximates 1,792 tokens.
            "min_boxed_within_cap": int(cfg.get("min_boxed_within_cap", 0) or 0),
            "cap_chars": int(cfg.get("cap_chars", 7000) or 7000),
            "retire_published": bool(cfg.get("retire_published", True)),
            "min_surviving_strata": int(cfg.get("min_surviving_strata", 100) or 0),
            "teacher_prefix": str(cfg.get("teacher_prefix") or "teacher_"),
            "king_prefix": str(cfg.get("king_prefix") or "king_")}


def boxed_answer(trace: dict) -> str | None:
    """Normalized content of the LAST \boxed{} in the rollout's final reply."""
    nodes = trace.get("nodes") or []
    final = next((nd for nd in reversed(nodes)
                  if nd.get("sampled") and (nd.get("message") or {}).get("role") == "assistant"),
                 None)
    if final is None:
        return None
    acts = dialects.get("boxed").actions(message_text((final["message"] or {}).get("content")))
    if not acts:
        return None
    body = acts[-1].strip()
    if body.startswith("\\boxed{") and body.endswith("}"):
        body = body[len("\\boxed{"):-1]
    return " ".join(body.split())


def final_reply_chars(trace: dict) -> int:
    nodes = trace.get("nodes") or []
    final = next((nd for nd in reversed(nodes)
                  if nd.get("sampled") and (nd.get("message") or {}).get("role") == "assistant"),
                 None)
    if final is None:
        return 0
    m = final["message"] or {}
    return len(message_text(m.get("content"))) + len(str(m.get("reasoning_content") or ""))


def math_keep_set(pub: PublicCorpus, traces_manifest: dict, cfg: dict
                  ) -> tuple[set[str], dict]:
    """Problems (task sid) the proxy keeps, plus the survey."""
    per: dict[str, dict] = {}
    n_chunks = 0
    for c in traces_manifest["chunks"]:
        name = c["key"].rsplit("/", 1)[-1]
        if not any(name.startswith(f"{src}-") for src in cfg["sources"]):
            continue
        n_chunks += 1
        for env in iter_jsonl_gz(pub.cached(c["key"], c["sha256"], gz_sha=True)):
            if str(env.get("source") or "") not in cfg["sources"]:
                continue
            sid = str((env.get("task") or {}).get("sid") or "")
            pid = str((env.get("policy") or {}).get("id") or "")
            outcome = rollout_outcome(env["trace"])
            row = per.setdefault(sid, {"answers": set(), "teacher_failed": False,
                                       "king_failed": False, "n_teacher": 0, "n_king": 0,
                                       "n_boxed_in_cap": 0})
            if pid.startswith(cfg["teacher_prefix"]):
                if outcome in ("solved", "failed"):
                    row["n_teacher"] += 1
                    ans = boxed_answer(env["trace"])
                    if ans is not None:
                        row["answers"].add(ans)
                        if final_reply_chars(env["trace"]) <= cfg["cap_chars"]:
                            row["n_boxed_in_cap"] += 1
                    row["teacher_failed"] |= outcome == "failed"
            elif pid.startswith(cfg["king_prefix"]):
                if outcome in ("solved", "failed"):
                    row["n_king"] += 1
                    row["king_failed"] |= outcome == "failed"
    base = {sid for sid, r in per.items()
            if len(r["answers"]) >= 2 or r["teacher_failed"] or r["king_failed"]}
    # In-cap rule (P4): the teacher must box within the duel cap on at least
    # min(min_boxed_within_cap, its sample count) of its samples -- most
    # problems have a single teacher sample, so "2 of 1" cannot be asked.
    def in_cap_ok(r: dict) -> bool:
        need = min(cfg["min_boxed_within_cap"], max(1, r["n_teacher"]))
        return r["n_boxed_in_cap"] >= need
    keep = {sid for sid in base if in_cap_ok(per[sid])} if cfg["min_boxed_within_cap"] else base
    stats = {"chunks": n_chunks, "problems": len(per), "kept": len(keep), "kept_base_rule": len(base),
             "in_cap_ok": sum(in_cap_ok(r) for r in per.values()),
             "multi_sample": sum(r["n_teacher"] >= 2 for r in per.values()),
             "disagree": sum(len(r["answers"]) >= 2 for r in per.values()),
             "teacher_failed": sum(r["teacher_failed"] for r in per.values()),
             "king_failed": sum(r["king_failed"] for r in per.values())}
    stats["_base_keep"] = base
    return keep, stats


def sha8_of(instance_id: str) -> str:
    return hashlib.sha256(instance_id.encode()).hexdigest()[:8]


def math_retire_plan(pub: PublicCorpus, live: dict | None, cfg: dict,
                     keep: set[str], group: str) -> tuple[list[str], set[str], set[str]]:
    """(turn ids to retire from the live index, surviving published math
    strata, retired math strata). Only rows in the math GROUP's own strata
    (`<group>:NNNN`) are considered: a king math failure published under
    `king_fail:NNNN` stays where it is (forward-only, phase 3 decision)."""
    table = index_table(pub, live, ["turn_id", "traj_id", "source", "stratum"])
    if table is None:
        return [], set(), set()
    keep_sha8 = {sha8_of(sid) for sid in keep}
    retire: list[str] = []
    surviving: set[str] = set()
    retired_strata: set[str] = set()
    for tid, traj, src, stratum in zip(*(table.column(c).to_pylist()
                                          for c in ("turn_id", "traj_id", "source", "stratum"))):
        if src not in cfg["sources"] or not str(stratum).startswith(f"{group}:"):
            continue
        m = TRAJ_SHA8_RE.search(traj or "")
        if m and m.group(1) in keep_sha8:
            surviving.add(stratum)
        else:
            retire.append(tid)
            retired_strata.add(stratum)
    return retire, surviving, retired_strata - surviving


def readmit_plan(pub: PublicCorpus, live: dict | None, cfg: dict,
                 from_groups: tuple[str, ...]) -> dict[str, list[str]]:
    """Retire-and-readmit (2026-09-12): side-table turns already published
    under one of `from_groups` (`<group>:*` strata). Returns
    {from_group: [turn ids]}. The caller removes those ids from `published`
    and re-derives their chunks so derive_chunk readmits them under the
    side-table's group; a turn not readmitted in the same run stays where
    it is (its id is dropped from the retire list)."""
    admitted = {str(row.get("turn_id") or "")
                for rows in cfg["table"].values() for row in rows.values()}
    table = index_table(pub, live, ["turn_id", "stratum"])
    plan: dict[str, list[str]] = {g: [] for g in from_groups}
    if table is None:
        return plan
    for tid, stratum in zip(table.column("turn_id").to_pylist(),
                            table.column("stratum").to_pylist()):
        ns = str(stratum).split(":")[0]
        if ns in plan and tid in admitted:
            plan[ns].append(tid)
    return plan


def strata_after_retire(pub: PublicCorpus, live: dict | None, group: str,
                        retire: set[str]) -> set[str]:
    """Strata of `group:*` index rows that keep at least one turn, keyed
    under the CURRENT budget from the row's original stratum (`stratum_src`;
    hashing the already-bucketed `stratum` again under-counted a bucketed
    group's survivors -- coding 3,553 -> 2,780 in the 2026-09-22 band
    dry run)."""
    table = index_table(pub, live, ["turn_id", "stratum", "stratum_src"])
    out: set[str] = set()
    if table is None:
        return out
    src_col = table.column("stratum_src").to_pylist() if "stratum_src" in table.column_names \
        else table.column("stratum").to_pylist()
    for tid, stratum, src in zip(table.column("turn_id").to_pylist(),
                                 table.column("stratum").to_pylist(), src_col):
        if str(stratum).startswith(f"{group}:") and tid not in retire:
            out.add(budget_stratum(group, str(src or stratum), tid))
    return out


def king_fail_source_retire(pub: PublicCorpus, live: dict | None,
                            excluded: frozenset[str]) -> list[str]:
    """Published `king_fail:*` rows whose source the king groups now exclude
    (wiki / agent / math and the one-reply general sources): the state is
    the task prompt or an R-dead tool loop (king-data spec §1.2)."""
    table = index_table(pub, live, ["turn_id", "stratum", "source"])
    if table is None:
        return []
    return [tid for tid, stratum, src in zip(table.column("turn_id").to_pylist(),
                                              table.column("stratum").to_pylist(),
                                              table.column("source").to_pylist())
            if str(stratum).startswith("king_fail:") and str(src) in excluded]


def later_onset_retire(pub: PublicCorpus, live: dict | None) -> list[str]:
    """Published `king_loop_onset:*` rows that are not the FIRST onset of
    their rollout (lowest turn_idx per traj_id): near-duplicate prefixes of
    the same wreck (king-data spec §1.3)."""
    table = index_table(pub, live, ["turn_id", "traj_id", "turn_idx", "stratum"])
    if table is None:
        return []
    first: dict[str, tuple[int, str]] = {}
    rows: list[tuple[str, str, int]] = []
    for tid, traj, tix, stratum in zip(*(table.column(c).to_pylist()
                                         for c in ("turn_id", "traj_id", "turn_idx", "stratum"))):
        if not str(stratum).startswith("king_loop_onset:"):
            continue
        rows.append((tid, traj, int(tix)))
        if traj not in first or int(tix) < first[traj][0]:
            first[traj] = (int(tix), tid)
    return [tid for tid, traj, _ in rows if first[traj][1] != tid]


# -- king_coached (coached-teacher recovery, 2026-09-15) ------------------------
def load_king_coached() -> dict:
    """[king_coached] (docs/coached-recovery.md; Jacob 2026-09-15, lever 1/2):
    the teacher's HINT-FREE continuation from a king failure state where the
    coach was DECISIVE -- the coached teacher solved >= `min_coached_solved`
    of 3 continuations while the plain teacher solved <= `max_plain_solved`
    of 3. Envelopes (datagen schema + a top-level `privileged` block with the
    coach's per-turn notes) sit in `envelopes_dir`; the fold drops the
    `privileged` block before deriving, so the notes never reach a prefix
    (Jacob's rule: miners do not see hints). The trace itself is hint-free,
    so the prefixes are what a miner would see: king trajectory + teacher
    continuation. Stratum = king_coached:<sha256(king state id) % n>; the
    teacher-probe gate applies like every king group."""
    cfg = _group_cfg(KING_COACHED_GROUP)
    if cfg:
        raw = cfg["raw"]
        cfg["leak_exempt"] = True
        cfg["envelopes_dir"] = REPO / str(raw.get("envelopes_dir") or "affine/state/king_coached")
        # Handoff (internal/hints/coached/king-coached-handoff.md): fold iff
        # origin.hint_decisive (coached solved >= 1 of 3, plain 0 of >= 2);
        # the thresholds below are the fallback when the flag is absent.
        # rule "plain_0_of_n" (2026-09-15 19:58 UTC, the 3-draw label was
        # ~2/3 noise): the plain teacher solved 0 of >= min_plain_n draws
        # and the coached teacher solved >= min_coached_solved. Draw counts
        # come from the envelope's origin block, overridden by the verdict
        # side-table (`verdict_table`: {"keep": [state_id], "demote": [...],
        # "missing": [...]}); `keep_missing` keeps untestable states.
        cfg["rule"] = str(raw.get("rule") or "hint_decisive")
        cfg["min_coached_solved"] = int(raw.get("min_coached_solved", 1) or 1)
        cfg["max_plain_solved"] = int(raw.get("max_plain_solved", 0) or 0)
        cfg["min_plain_n"] = int(raw.get("min_plain_n", 6) or 6)
        cfg["keep_missing"] = bool(raw.get("keep_missing", True))
        pp = raw.get("policy_prefix") or "coached_"
        cfg["policy_prefixes"] = tuple(str(x) for x in (pp if isinstance(pp, list) else [pp]))
        cfg["policy_prefix"] = cfg["policy_prefixes"][0]
        cfg["verdict"] = {}
        vt = raw.get("verdict_table")
        if vt and (REPO / str(vt)).exists():
            v = json.loads((REPO / str(vt)).read_text())
            for label in ("keep", "demote", "missing"):
                for sid in v.get(label) or []:
                    cfg["verdict"][str(sid)] = label
        ids = raw.get("envelope_ids")
        cfg["envelope_ids"] = None
        if ids and (REPO / str(ids)).exists():
            cfg["envelope_ids"] = {l.strip() for l in (REPO / str(ids)).read_text().split("\n") if l.strip()}
    return cfg


def coached_decisive(env: dict, cfg: dict) -> bool:
    o = ((env.get("privileged") or {}).get("origin") or {})
    pid = str((env.get("policy") or {}).get("id") or "")
    if not pid.startswith(cfg["policy_prefixes"]) or not o:
        return False
    if cfg["envelope_ids"] is not None and str(env.get("rollout_id")) not in cfg["envelope_ids"]:
        return False
    if rollout_outcome(env["trace"]) != "solved":
        return False
    if cfg["rule"] == "plain_0_of_n":
        if int(o.get("coached_n_solved") or 0) < cfg["min_coached_solved"]:
            return False
        label = cfg["verdict"].get(str(o.get("state_id")))
        if label == "demote":
            return False
        if label == "keep":
            return True
        if label == "missing":
            return cfg["keep_missing"]
        return (int(o.get("plain_n") or 0) >= cfg["min_plain_n"]
                and int(o.get("plain_n_solved") or 0) <= cfg["max_plain_solved"])
    if cfg["rule"] == "hint_decisive" and "hint_decisive" in o:
        if not bool(o["hint_decisive"]):
            return False
    elif cfg["rule"] == "hint_decisive_strict" and "hint_decisive_strict" in o:
        if not bool(o["hint_decisive_strict"]):
            return False
    else:
        if int(o.get("coached_n_solved") or 0) < cfg["min_coached_solved"]:
            return False
        if int(o.get("plain_n_solved") or 0) > cfg["max_plain_solved"] or int(o.get("plain_n") or 0) < 2:
            return False
    return rollout_outcome(env["trace"]) == "solved"


def coached_retire_ids(cfg: dict, pub: PublicCorpus, live: dict | None) -> tuple[list[str], set[str]]:
    """Published king_coached rows whose envelope no longer passes the
    admission rule (a demoted state): (turn ids to retire, demoted state ids)."""
    files = sorted(cfg["envelopes_dir"].glob("*.jsonl.gz")) if cfg["envelopes_dir"].exists() else []
    failing_rollouts: dict[str, str] = {}
    for f in files:
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                env = json.loads(line)
                if not coached_decisive(env, cfg):
                    sid = str(((env.get("privileged") or {}).get("origin") or {}).get("state_id") or "")
                    failing_rollouts[str(env.get("rollout_id"))] = sid
    if not failing_rollouts or not live or not live.get("index"):
        return [], set()
    t = index_table(pub, live, ["turn_id", "rollout_id", "stratum"])
    ids = [tid for tid, rid, st in zip(t.column("turn_id").to_pylist(), t.column("rollout_id").to_pylist(),
                                       t.column("stratum").to_pylist())
           if str(rid) in failing_rollouts and str(st).startswith(f"{KING_COACHED_GROUP}:")]
    states = {failing_rollouts[str(rid)] for rid in t.column("rollout_id").to_pylist() if str(rid) in failing_rollouts}
    return ids, states


def derive_coached(cfg: dict, baker: ToolBaker, panel, allowed_kinds, published: set[str],
                   drops: dict[str, int], notes: dict[str, int], folded: set[str]) -> list[dict]:
    """View records for the decisive coached continuations: filter, strip
    `privileged`, derive like a trace chunk, route to king_coached."""
    out: list[dict] = []
    files = sorted(cfg["envelopes_dir"].glob("*.jsonl.gz")) if cfg["envelopes_dir"].exists() else []
    tmp_dir = WORK_DIR / "coached"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    n_buckets = cfg["strata_buckets"]
    for f in files:
        kept_envs: list[tuple[dict, dict]] = []
        n_all = 0
        with gzip.open(f, "rt", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                env = json.loads(line)
                n_all += 1
                if not coached_decisive(env, cfg):
                    _count(notes, "king_coached_not_decisive")
                    continue
                sid0 = str(((env.get("privileged") or {}).get("origin") or {}).get("state_id") or env.get("rollout_id"))
                h0 = int(hashlib.sha256(sid0.encode("utf-8")).hexdigest()[:8], 16)
                SRC_OVERRIDE[str(env.get("rollout_id"))] = f"{KING_COACHED_GROUP}:{h0 % n_buckets:04d}"
                if str(env.get("rollout_id")) in folded:
                    continue      # derived by an earlier fold (published or in the carryover)
                origin = dict(env["privileged"]["origin"])
                env.pop("privileged", None)          # never in a record or prefix
                kept_envs.append((env, origin))
        if not kept_envs:
            log(f"king_coached: {f.name}: no new decisive envelopes ({n_all} in file, "
                f"{sum(1 for _ in folded)} already folded)")
            continue
        stripped = tmp_dir / f.name
        with gzip.open(stripped, "wt", encoding="utf-8") as fh:
            for env, _ in kept_envs:
                fh.write(json.dumps(env, ensure_ascii=False) + "\n")
        by_rid = {str(env.get("rollout_id")): origin for env, origin in kept_envs}
        recs = derive_chunk(stripped, baker, panel, allowed_kinds, published, drops,
                            notes=notes, leak_exempt_all=True)
        n = cfg["strata_buckets"]
        for rec in recs:
            origin = by_rid.get(str(rec.get("rollout_id")), {})
            sid = str(origin.get("state_id") or rec.get("instance_id") or rec["traj_id"])
            h = int(hashlib.sha256(sid.encode("utf-8")).hexdigest()[:8], 16)
            rec["fold_group"] = KING_COACHED_GROUP
            rec["stratum"] = f"{KING_COACHED_GROUP}:{h % n:04d}"
            rec["coached"] = {"state_id": sid, "king_digest": origin.get("king_digest"),
                              "king_turn_idx": origin.get("king_turn_idx"),
                              "state_kind": origin.get("state_kind"),
                              "coached_n_solved": origin.get("coached_n_solved"),
                              "plain_n_solved": origin.get("plain_n_solved"),
                              "hint_decisive_strict": origin.get("hint_decisive_strict"),
                              "harness": (rec.get("policy") or {}).get("harness")}
            for m in rec["turns"]:
                m["stratum"] = rec["stratum"]
                published.add(f"{rec['traj_id']}:{m['turn_idx']}")
            _count(notes, "king_coached_rollouts")
            _count(notes, "king_coached_turns", len(rec["turns"]))
        folded.update(str(env.get("rollout_id")) for env, _ in kept_envs)
        log(f"king_coached: {f.name}: {len(kept_envs)} of {n_all} envelopes decisive -> "
            f"{len(recs)} records / {sum(len(r['turns']) for r in recs)} new turns "
            f"({len({r['stratum'] for r in recs})} states)")
        out.extend(recs)
    return out


# -- two-sided admission gate (MiMo item 1, Jacob 2026-09-17 09:08 UTC) ---------
# A king-derived state enters D only if (a) the king failed there AND (b) the
# teacher recovers it. (b) from the signals we already have, cheapest first:
#   state level -- the recoverable pipeline's continuations at the exact
#                  state (majority of 3 solved = recovers; proxy rows are
#                  task-level and fall through);
#   task level  -- `task.teacher_solved` stamped on the envelope, else the
#                  traces: any teacher_* rollout solved the task (cached per
#                  traces manifest).
# A state with neither signal is HELD (deferred, `gate_unverified`), not
# admitted. Dead-reference drop: turns whose teacher references were dead in
# stored verdicts (curriculum ledger rows: < 2 valid refs or all identical)
# are dropped / retired. Groups listed in `apply_groups` are enforced; the
# others are measured and reported only (the operator's per-group ok).
TEACHER_SOLVED_CACHE = CACHE_DIR / "teacher_solved_tasks.json"
GATE_SHADOW_RETIRE: dict[str, list[str]] = {}   # rows a shadow group WOULD retire under the recovery rule
YIELD: dict[str, dict] = {}     # per source: envelopes seen, records / turns accepted at derive, drop reasons
YIELD_BY_KING: dict[str, dict] = {}   # per served king digest: king rollouts seen, records / turns at derive, by routed group / source
NOTES_GLOBAL: dict[str, int] = {}   # the fold's `notes` counters, for the yield report
# Sources whose harness talks to a (simulated) user mid-trajectory
# (`[source.<name>] interactive = true`, tau2-airline read 2026-09-18): a
# prose reply with no action ("Could you provide your user ID?") is a real,
# scorable turn there, not only at the end of the rollout. The fold admits
# those replies with kind `text`. Every other source keeps the final-reply
# rule, so existing records do not change.
INTERACTIVE_SOURCES: frozenset[str] = frozenset()


def load_interactive_sources() -> frozenset[str]:
    raw = tomllib.loads(SOURCES_TOML.read_text())
    return frozenset(name for name, cfg in (raw.get("source") or {}).items() if cfg.get("interactive"))


def load_admission_gate() -> dict:
    raw = tomllib.loads(SOURCES_TOML.read_text()).get("admission_gate") or {}
    if not raw or not raw.get("enabled", False):
        return {}
    return {"raw": raw,
            "groups": tuple(str(g) for g in (raw.get("groups") or KING_GROUPS)),
            "apply_groups": frozenset(str(g) for g in (raw.get("apply_groups") or [])),
            "failed_exempt": frozenset(str(g) for g in (raw.get("king_failed_exempt") or [KING_DONE_GROUP])),
            "state_dir": REPO / str(raw.get("state_table_dir") or "affine/state/recoverable"),
            "majority_min": int(raw.get("majority_min_solved", 2) or 2),
            "majority_n": int(raw.get("majority_min_continuations", 3) or 3),
            "task_signal": bool(raw.get("task_teacher_solved", True)),
            "dead_refs": bool(raw.get("dead_refs", True)),
            "ledger_dir": REPO / str(raw.get("ledger_dir") or "affine/state/curriculum/ledger"),
            "min_refs_valid": int(raw.get("min_refs_valid", 2) or 2),
            # 2026-09-17 09:51 UTC decision: dead-reference drop on EVERY gated
            # group; the recovery condition enforced on `apply_groups` + the
            # groups the fold auto-promoted (post-gate strata >= quota, never
            # flips back); `recovery_exempt` groups (king_coached: the coached
            # teacher IS the recovery) get the dead-reference drop only.
            "recovery_exempt": frozenset(str(g) for g in (raw.get("recovery_exempt") or [KING_COACHED_GROUP])),
            "auto_promote": bool(raw.get("auto_promote", True)),
            # Policies whose solved rollouts count as "a strong model solves
            # the task" for the task-level signal. 2026-09-20: the 45 gate-
            # unverified coding tasks had only GLM-era teacher runs
            # (`glm_*`, engy/glm-5.2, Aug 2026) -- the teacher of wvk <= 9,
            # not backfill; counted by default, drop `glm_` here to revert.
            "teacher_prefixes": tuple(str(x) for x in (raw.get("teacher_policy_prefixes") or ["teacher_", "glm_"]))}


def load_state_recovery(cfg: dict) -> dict[tuple[str, int], bool]:
    """(rollout_id, turn_idx) -> the teacher recovers (majority of >= n
    continuations solved). Proxy (same-task) rows are not state evidence."""
    out: dict[tuple[str, int], bool] = {}
    if not cfg["state_dir"].is_dir():
        return out
    for path in sorted(cfg["state_dir"].glob("*.jsonl")):
        for line in path.read_text().split("\n"):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("proxy"):
                continue
            conts = [c for c in (row.get("continuations") or []) if c.get("status") == "ok"]
            n = len(conts) if conts else int(row.get("n_continuations") or 0)
            if n < cfg["majority_n"]:
                continue
            solved = sum(1 for c in conts if c.get("solved")) if conts else int(row.get("n_solved") or 0)
            out[(str(row["rollout_id"]), int(row["turn_idx"]))] = solved >= cfg["majority_min"]
    return out


def teacher_solved_tasks(pub: PublicCorpus, traces_manifest: dict,
                         prefixes: tuple[str, ...] = ("teacher_",)) -> tuple[set[str], set[str]]:
    """(solved instance ids, seen instance ids) from every rollout of a
    teacher-side policy (`prefixes`) in the traces; cached per traces
    manifest + prefixes. Backfill rollouts never enter the manifest."""
    key = hashlib.sha256(json.dumps([c["key"] for c in traces_manifest["chunks"]] + list(prefixes)).encode()).hexdigest()[:16]
    if TEACHER_SOLVED_CACHE.exists():
        try:
            c = json.loads(TEACHER_SOLVED_CACHE.read_text())
            if c.get("key") == key:
                return set(c["solved"]), set(c["seen"])
        except (OSError, ValueError):
            pass
    solved: set[str] = set()
    seen: set[str] = set()
    for c in traces_manifest["chunks"]:
        path = pub.cached(c["key"], c["sha256"], gz_sha=True)
        for env in iter_jsonl_gz(path):
            pid = str((env.get("policy") or {}).get("id") or "")
            if not pid.startswith(prefixes) or is_backfill(env):
                continue
            sid = str((env.get("task") or {}).get("sid") or "")
            if not sid:
                continue
            seen.add(sid)
            if rollout_outcome(env["trace"]) == "solved":
                solved.add(sid)
    TEACHER_SOLVED_CACHE.parent.mkdir(parents=True, exist_ok=True)
    TEACHER_SOLVED_CACHE.write_text(json.dumps({"key": key, "solved": sorted(solved), "seen": sorted(seen)}))
    return solved, seen


def dead_reference_turns(cfg: dict) -> set[str]:
    """Turn ids whose teacher references were dead in a stored verdict."""
    out: set[str] = set()
    if not cfg["dead_refs"] or not cfg["ledger_dir"].is_dir():
        return out
    files = sorted(cfg["ledger_dir"].glob("*.rows.parquet"))
    if not files:
        return out
    t = pq.read_table(files[-1], columns=["turn_id", "n_refs_valid", "refs_identical"])
    for tid, nv, ident in zip(t.column("turn_id").to_pylist(), t.column("n_refs_valid").to_pylist(),
                              t.column("refs_identical").to_pylist()):
        if (nv is not None and int(nv) < cfg["min_refs_valid"]) or bool(ident):
            out.add(str(tid))
    return out


def gate_turn(rec: dict, m: dict, g: str, cfg: dict, state_rec: dict, task_solved: set[str],
              task_seen: set[str], dead: set[str]) -> str:
    """admit | king_not_failed | not_recovered | dead_refs | unverified"""
    tid = f"{rec['traj_id']}:{m['turn_idx']}"
    if tid in dead:
        return "dead_refs"
    if g not in cfg["failed_exempt"] and str(rec.get("outcome") or "") != "failed" \
            and str((rec.get("policy") or {}).get("id") or "").startswith("king_"):
        return "king_not_failed"
    st = state_rec.get((str(rec.get("rollout_id")), int(m["turn_idx"])))
    if st is not None:
        return "admit" if st else "not_recovered"
    task = (rec.get("task") or {})
    ts = task.get("teacher_solved") if isinstance(task, dict) else None
    if isinstance(ts, str):       # datagen stamps it as text ('True' / 'False')
        ts = ts.strip().lower() in ("true", "1", "yes")
    if ts is False:
        ts = None                 # 'False' = not known solved at stamp time; let the traces decide
    if ts is None and cfg["task_signal"]:
        sid = str(rec.get("instance_id") or "")
        if sid in task_solved:
            ts = True
        elif sid in task_seen:
            ts = False
    if ts is None:
        return "unverified"
    return "admit" if ts else "not_recovered"


def admission_gate(records: list[dict], cfg: dict, src2grp, mix, state_rec, task_solved, task_seen, dead,
                   drops: dict[str, int]) -> tuple[list[dict], list[dict], dict[str, dict[str, int]]]:
    """(ready, held, per-group tally). Enforced only for `apply_groups`; the
    tally covers every gated group so the dry run reports the shadow."""
    ready: list[dict] = []
    held: list[dict] = []
    tally: dict[str, dict[str, int]] = {}
    for rec in records:
        g = group_of(rec, src2grp, mix)
        if g not in cfg["groups"]:
            ready.append(rec)
            continue
        keep, pending = [], []
        enforced = g in cfg["apply_groups"]
        for m in rec["turns"]:
            v = gate_turn(rec, m, g, cfg, state_rec, task_solved, task_seen, dead)
            tally.setdefault(g, {})[v] = tally.setdefault(g, {}).get(v, 0) + 1
            if v == "dead_refs":                       # every gated group
                _count(drops, "gate_dead_refs"); _count(drops, f"gate_dead_refs_{g}")
            elif v == "admit" or not enforced or g in cfg["recovery_exempt"]:
                keep.append(m)
            elif v == "unverified":
                pending.append(m)
            else:
                _count(drops, f"gate_{v}")
                _count(drops, f"gate_{v}_{g}")
        if pending:
            rec["turns"] = keep + pending
            held.append(rec)
        elif keep:
            rec["turns"] = keep
            ready.append(rec)
        else:
            _count(drops, f"gate_emptied_{g}")
    return ready, held, tally


def gate_published(pub: PublicCorpus, live: dict | None, cfg: dict, state_rec, task_solved, task_seen,
                   dead, src2grp) -> tuple[dict[str, list[str]], dict[str, dict[str, int]]]:
    """Published rows of the gated groups: ({group: turn ids to retire},
    per-group tally). Task-level signal via the traj_id's sha8 of the
    instance id; unverified rows stay."""
    if not live or not live.get("index"):
        return {}, {}
    sha8_solved = {hashlib.sha256(s.encode()).hexdigest()[:8] for s in task_solved}
    sha8_seen = {hashlib.sha256(s.encode()).hexdigest()[:8] for s in task_seen}
    t = index_table(pub, live, ["turn_id", "traj_id", "rollout_id", "turn_idx", "stratum", "source"])
    retire: dict[str, list[str]] = {}
    tally: dict[str, dict[str, int]] = {}
    shadow_retire = GATE_SHADOW_RETIRE
    shadow_retire.clear()
    for tid, traj, rid, ti, st, src in zip(*(t.column(c).to_pylist() for c in
                                             ("turn_id", "traj_id", "rollout_id", "turn_idx", "stratum", "source"))):
        g = group_from_row(str(st), str(src), src2grp)
        if g not in cfg["groups"]:
            continue
        if tid in dead:
            v = "dead_refs"
        else:
            s_ = state_rec.get((str(rid), int(ti)))
            if s_ is not None:
                v = "admit" if s_ else "not_recovered"
            else:
                parts = str(traj).split(".")
                sha8 = parts[1] if len(parts) > 2 else ""
                v = "admit" if sha8 in sha8_solved else ("not_recovered" if sha8 in sha8_seen else "unverified")
        tally.setdefault(g, {})[v] = tally.setdefault(g, {}).get(v, 0) + 1
        if v == "dead_refs":
            retire.setdefault(g, []).append(tid)
        elif v == "not_recovered" and g in cfg["apply_groups"] and g not in cfg["recovery_exempt"]:
            retire.setdefault(g, []).append(tid)
        elif v == "not_recovered":
            shadow_retire.setdefault(g, []).append(tid)
    return retire, tally


# -- band filter (AA gap-fill plan §5 / §1.1, Jacob 2026-09-22 15:24 UTC; ------
#    widened to every source 2026-09-22 21:05 UTC, Jacob "fold go")
# A TEACHER source's task folds only inside the signal band: the teacher
# solved it in >= teacher_min_solved of >= teacher_min_attempts attempts
# (<= teacher_max_solved when set -- drop the saturated end) AND the king
# seat did NOT: with >= king_min_attempts seat attempts, drop when the seat
# solved > king_max_solved of them (absolute m) or > king_max_solved_frac
# of them (per-source m/n as a fraction; the default 0.5 = "the king
# mostly fails": 1/1, 2/2, 2/3 drop; 1/2, 1/3 stay). Attempt counts come
# from the traces (every graded teacher_* / king_* rollout of the task,
# keyed by source + task.sid), cached per trace chunk (chunks are
# immutable, so a fold only scans the new ones). Missing teacher attempts
# -> hold; missing king attempts -> hold when king_missing = "hold", admit
# when "admit" (a source is never held hostage to seat coverage);
# king_min_attempts = 0 turns the king side off. Records whose own rollout
# is ungraded (no grader: affine_wiki) bypass the band. Published teacher-
# side rows outside the band retire from the index (retire_published;
# chunks and old manifests untouched, old verdicts replay). King-derived
# records keep their own groups' two-sided admission gate.
# Why every source (2026-09-22): the teacher-vs-king control went negative
# on the last 30 verdicts -- the meter is saturated on turns the king
# already handles -- while reign 21 fell on the envs. The lever left is
# slice composition: keep the turns of tasks the teacher can do and the
# king cannot.
# [band_filter.defaults] applies to every [source.*] (all run the king
# seat) minus exclude_sources; [band_filter.<source>] overrides fields.
BAND_CACHE = CACHE_DIR / "band_attempts.jsonl"     # one line per trace chunk
BAND_FIELDS = ("teacher_min_solved", "teacher_min_attempts", "teacher_max_solved",
               "king_max_solved", "king_max_solved_frac", "king_min_attempts", "king_missing",
               "retire_published")


def _band_cfg(cfg: dict, base: dict | None = None) -> dict:
    b = dict(base or {})
    out = {
        "teacher_min_solved": int(cfg.get("teacher_min_solved", b.get("teacher_min_solved", 1)) or 0),
        "teacher_min_attempts": int(cfg.get("teacher_min_attempts", b.get("teacher_min_attempts", 1)) or 1),
        "teacher_max_solved": cfg.get("teacher_max_solved", b.get("teacher_max_solved")),
        "king_max_solved": cfg.get("king_max_solved", b.get("king_max_solved")),
        "king_max_solved_frac": cfg.get("king_max_solved_frac", b.get("king_max_solved_frac")),
        "king_min_attempts": int(cfg.get("king_min_attempts", b.get("king_min_attempts", 0)) or 0),
        "king_missing": str(cfg.get("king_missing", b.get("king_missing", "hold"))),
        "retire_published": bool(cfg.get("retire_published", b.get("retire_published", True)))}
    if out["king_max_solved"] is None and out["king_max_solved_frac"] is None and out["king_min_attempts"] > 0:
        out["king_max_solved"] = 1
    return out


def load_band_filters() -> dict[str, dict]:
    toml = tomllib.loads(SOURCES_TOML.read_text())
    raw = toml.get("band_filter") or {}
    out: dict[str, dict] = {}
    d = raw.get("defaults")
    if isinstance(d, dict) and d.get("enabled", True):
        base = _band_cfg(d)
        excl = {str(x) for x in (d.get("exclude_sources") or [])}
        for src in (toml.get("source") or {}):
            if src not in excl:
                out[str(src)] = dict(base)
    for src, cfg in raw.items():
        if src == "defaults" or not isinstance(cfg, dict):
            continue
        if not cfg.get("enabled", True):
            out.pop(str(src), None)
            continue
        out[str(src)] = _band_cfg(cfg, out.get(str(src)))
    return out


def load_band_backfill() -> dict:
    """[band_filter.defaults] backfill_outcomes = "<key under the public
    base>" (+ backfill_seats, backfill_kings = "all" | "current"): the env
    backfill's per-task outcome table (datagen worker, internal/king-seat-
    replay-band-coverage.md: one row per (source, sid, seat model) with n /
    n_solved / n_errored). Its king rows merge into the seat counts k_n /
    k_s -- more coverage for the "king did not solve it" side without
    waiting for the seat replay. Only drops can come of it (no backfill
    turn enters D). Empty key = off."""
    d = (tomllib.loads(SOURCES_TOML.read_text()).get("band_filter") or {}).get("defaults") or {}
    key = str(d.get("backfill_outcomes") or "").strip("/")
    return {"key": key,
            "seats": frozenset(str(x) for x in (d.get("backfill_seats") or ["king"])),
            "kings": str(d.get("backfill_kings") or "all"),
            "current_digest": None}


def current_king_digest() -> str | None:
    """The validator's current king (affine/state/state.json revision[:12]),
    for backfill_kings = "current"."""
    try:
        k = json.loads((REPO / "affine" / "state" / "state.json").read_text()).get("king") or {}
        rev = str(k.get("revision") or "")
        return rev[:12] or None
    except (OSError, ValueError):
        return None


def merge_band_backfill(pub: PublicCorpus, stats: dict[str, list[int]], cfg: dict,
                        current_digest: str | None = None) -> dict:
    """Add the backfill table's rows to k_n / k_s in place. Returns a report
    (rows merged, tasks touched, sha) or {"error": ...}; a sha mismatch with
    the meta file (hourly republish race) skips the merge for this fold."""
    if not cfg.get("key"):
        return {}
    try:
        raw = pub.get(cfg["key"])
        meta = json.loads(pub.get(cfg["key"].rsplit(".", 1)[0] + ".meta.json").decode())
    except Exception as ex:  # noqa: BLE001 -- network / 404: the band runs on the seat alone
        log(f"band filter: backfill outcomes unavailable ({type(ex).__name__}: {ex}); seat counts only")
        return {"error": f"{type(ex).__name__}: {ex}"}
    sha = hashlib.sha256(raw).hexdigest()
    if meta.get("sha256") and meta["sha256"] != sha:
        log(f"band filter: backfill outcomes sha {sha[:12]} != meta {str(meta.get('sha256'))[:12]} (republish race?); seat counts only")
        return {"error": "sha_mismatch", "sha256": sha}
    rows = 0
    touched: set[str] = set()
    new_tasks = 0
    by_src: dict[str, int] = {}
    for line in raw.decode().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if str(r.get("seat") or "") not in cfg["seats"]:
            continue
        if cfg["kings"] == "current" and current_digest and str(r.get("king_digest") or "") != current_digest:
            continue
        n, ns = int(r.get("n") or 0), int(r.get("n_solved") or 0)
        if n <= 0:
            continue
        key = f"{r.get('source') or ''}\t{r.get('sid') or ''}"
        st = stats.get(key)
        if st is None:
            st = stats[key] = [0, 0, 0, 0]
            new_tasks += 1
        st[2] += n
        st[3] += ns
        rows += 1
        touched.add(key)
        by_src[key.split("\t", 1)[0]] = by_src.get(key.split("\t", 1)[0], 0) + 1
    rep = {"key": cfg["key"], "sha256": sha, "generated_at": meta.get("generated_at"),
           "traces_backfill_manifest_sha256": meta.get("traces_backfill_manifest_sha256"),
           "seats": sorted(cfg["seats"]), "kings": cfg["kings"], "rows_merged": rows,
           "tasks_touched": len(touched), "tasks_new": new_tasks, "rows_by_source": by_src}
    log(f"band filter: merged {rows} backfill outcome rows ({len(touched)} tasks, {new_tasks} not in the traces) "
        f"from {cfg['key']} sha {sha[:12]} (generated {meta.get('generated_at')})")
    return rep


def band_coverage(stats: dict[str, list[int]], bands: dict[str, dict]) -> dict[str, dict]:
    """Per source: teacher-covered tasks, share with >= 1 king attempt, share
    the king side would drop -- for the seat-only vs merged comparison."""
    out: dict[str, dict] = {}
    per: dict[str, list] = {}
    for k, v in stats.items():
        src = k.split("\t", 1)[0]
        if src not in bands or v[0] <= 0:
            continue
        p = per.setdefault(src, [0, 0, 0])
        p[0] += 1
        p[1] += v[2] > 0
        p[2] += band_verdict(v, bands[src]) == "king_solved"
    for src, (n, cov, ks) in per.items():
        out[src] = {"teacher_tasks": n, "king_covered": round(cov / n, 3), "king_solved_share": round(ks / n, 3)}
    return out


def _band_scan_chunk(path: Path) -> dict[str, list[int]]:
    """{source\\tsid: [t_n, t_s, k_n, k_s]} for one trace chunk."""
    stats: dict[str, list[int]] = {}
    for env in iter_jsonl_gz(path):
        if is_backfill(env):
            continue
        pid = str((env.get("policy") or {}).get("id") or "")
        side = 0 if pid.startswith(("teacher_", "glm_")) else (2 if pid.startswith("king_") else None)
        if side is None:
            continue
        outcome = rollout_outcome(env["trace"])
        if outcome not in ("solved", "failed"):
            continue
        key = f"{env.get('source') or ''}\t{(env.get('task') or {}).get('sid') or ''}"
        st = stats.setdefault(key, [0, 0, 0, 0])
        st[side] += 1
        st[side + 1] += int(outcome == "solved")
    return stats


def task_attempts(pub: PublicCorpus, traces_manifest: dict, sources: frozenset[str]) -> dict[str, list[int]]:
    """{source\\tsid: [t_n, t_s, k_n, k_s]} over every graded teacher_* /
    king_* rollout in the traces. Per-chunk cache; only unseen chunks are
    scanned (a process pool: the work is gzip + json)."""
    cached: dict[str, dict[str, list[int]]] = {}
    if BAND_CACHE.exists():
        with BAND_CACHE.open() as fh:
            for line in fh:
                try:
                    row = json.loads(line)
                    cached[row["key"]] = row["stats"]
                except (ValueError, KeyError):
                    continue
    want = [c for c in traces_manifest["chunks"] if c["key"] not in cached]
    if want:
        paths = {c["key"]: pub.cached(c["key"], c["sha256"], gz_sha=True) for c in want}
        BAND_CACHE.parent.mkdir(parents=True, exist_ok=True)
        with ProcessPoolExecutor(max_workers=8) as ex, BAND_CACHE.open("a") as fh:
            for key, st in zip(paths, ex.map(_band_scan_chunk, paths.values())):
                cached[key] = st
                fh.write(json.dumps({"key": key, "stats": st}) + "\n")
        log(f"band filter: scanned {len(want)} new trace chunk(s) ({len(cached)} cached)")
    live_keys = {c["key"] for c in traces_manifest["chunks"]}
    stats: dict[str, list[int]] = {}
    for key, st in cached.items():
        if key not in live_keys:
            continue
        for k, v in st.items():
            if k.split("\t", 1)[0] not in sources:
                continue
            a = stats.setdefault(k, [0, 0, 0, 0])
            for i in range(4):
                a[i] += v[i]
    return stats


def band_verdict(st: list[int] | None, cfg: dict) -> str:
    """admit | hold | teacher_unsolved | teacher_saturated | king_solved"""
    t_n, t_s, k_n, k_s = st or (0, 0, 0, 0)
    if t_n < cfg["teacher_min_attempts"]:
        return "hold"
    if t_s < cfg["teacher_min_solved"]:
        return "teacher_unsolved"
    if cfg["teacher_max_solved"] is not None and t_s > int(cfg["teacher_max_solved"]):
        return "teacher_saturated"
    if cfg["king_min_attempts"] > 0:
        if k_n < cfg["king_min_attempts"]:
            return "hold" if cfg["king_missing"] == "hold" else "admit"
        if cfg["king_max_solved"] is not None:
            if k_s > int(cfg["king_max_solved"]):
                return "king_solved"
        elif cfg["king_max_solved_frac"] is not None and k_s > float(cfg["king_max_solved_frac"]) * k_n:
            return "king_solved"
    return "admit"


def band_filter_records(records: list[dict], bands: dict[str, dict], stats: dict[str, list[int]],
                        drops: dict[str, int]) -> tuple[list[dict], list[dict], dict[str, dict[str, int]]]:
    """Teacher-side records of banded sources -> (ready, held, tally by
    source: verdict -> turns). King-derived records (king_* policies) keep
    their own groups' gates; ungraded rollouts bypass the band."""
    ready: list[dict] = []
    held: list[dict] = []
    tally: dict[str, dict[str, int]] = {}
    for rec in records:
        src = str(rec.get("source") or "")
        pid = str((rec.get("policy") or {}).get("id") or "")
        if src not in bands or pid.startswith("king_"):
            ready.append(rec)
            continue
        n = len(rec.get("turns") or [])
        t = tally.setdefault(src, {})
        if rec.get("outcome") not in ("solved", "failed"):
            t["ungraded"] = t.get("ungraded", 0) + n
            ready.append(rec)
            continue
        v = band_verdict(stats.get(f"{src}\t{rec.get('instance_id') or ''}"), bands[src])
        t[v] = t.get(v, 0) + n
        if v == "admit":
            ready.append(rec)
        elif v == "hold":
            held.append(rec)
        else:
            _count(drops, f"band_{v}", n)
            _count(drops, f"band_{v}_{src}", n)
    return ready, held, tally


def band_published_retire(pub: PublicCorpus, live: dict | None, bands: dict[str, dict],
                          stats: dict[str, list[int]], src2grp: dict[str, str]
                          ) -> tuple[dict[str, list[str]], dict[str, dict[str, int]]]:
    """Published teacher-side rows of banded sources that fail the band now:
    ({group: [turn ids]}, {source: {kept, retired}}). King-group rows are
    left to the admission gate; a row whose task has no attempt record
    (sid not recoverable from traj_id) is kept."""
    if not live or not live.get("index"):
        return {}, {}
    t = index_table(pub, live, ["turn_id", "traj_id", "stratum_src", "source"])
    # sid is not an index column; traj_id = <stem>.<sha256(sid)[:8]>.pr_<n>__<run>
    sha8: dict[str, dict[str, str]] = {}
    for k in stats:
        src, sid = k.split("\t", 1)
        sha8.setdefault(src, {})[hashlib.sha256(sid.encode()).hexdigest()[:8]] = k
    out: dict[str, list[str]] = {}
    by_src: dict[str, dict[str, int]] = {}
    cols = [t.column(c).to_pylist() for c in ("turn_id", "traj_id", "stratum_src", "source")]
    for tid, traj, st, src in zip(*cols):
        src = str(src)
        if src not in bands or not bands[src]["retire_published"]:
            continue
        g = group_from_row(str(st), src, src2grp)
        if g in KING_GROUPS:
            continue
        m = sha8.get(src) or {}
        key = next((m[p] for p in str(traj).split(".") if p in m), None)
        v = band_verdict(stats.get(key) if key else None, bands[src])
        b = by_src.setdefault(src, {"kept": 0, "retired": 0})
        if v in ("teacher_unsolved", "teacher_saturated", "king_solved"):
            out.setdefault(g, []).append(str(tid))
            b["retired"] += 1
            b[f"retired_{v}"] = b.get(f"retired_{v}", 0) + 1
        else:
            b["kept"] += 1
    return out, by_src


def band_report_per_source(bands: dict[str, dict], stats: dict[str, list[int]],
                           tally: dict[str, dict[str, int]], published: dict[str, dict[str, int]]) -> dict[str, dict]:
    per: dict[str, dict] = {}
    for src, cfg in bands.items():
        rows = [v for k, v in stats.items() if k.split("\t", 1)[0] == src]
        if not rows and src not in tally and src not in published:
            continue
        kept = [v for v in rows if band_verdict(v, cfg) == "admit"]
        t_all = sum(v[0] for v in rows)
        t_kept = sum(v[0] for v in kept)
        per[src] = {"tasks_seen": len(rows), "tasks_kept": len(kept),
                    "tasks_king_covered": sum(1 for v in rows if v[2] > 0),
                    "teacher_solve_rate_all": round(sum(v[1] for v in rows) / t_all, 3) if t_all else None,
                    "teacher_solve_rate_kept": round(sum(v[1] for v in kept) / t_kept, 3) if t_kept else None,
                    "king_solve_rate_all": round(sum(v[3] for v in rows) / max(1, sum(v[2] for v in rows)), 3),
                    "new_turns": tally.get(src) or {},
                    "published": published.get(src) or {}}
    return per


# -- teacher probe gate (improvement loop P4, 2026-09-14) -----------------------
# ~25 % of king-group strata were dead for every miner: the teacher itself
# gave <= 1 parseable reference or forfeited there. The gate: a turn of a
# king group enters D only with a probe row (ops/teacher_probe/probe.py:
# 3 teacher samples at the prefix under the duel's cap and parser) showing
# >= `min_valid` parsed actions that are not all identical. Turns without a
# row are held (record deferred, turn written to pending.jsonl for the
# probe job); failing turns are dropped (`probe_failed`); published king
# rows with a failing probe are retired once.
PROBE_STATE_DIR = REPO / "affine" / "state" / "teacher_probe"
SIDE_PROBE_ROWS: dict[str, dict] = {}   # probe evidence carried by side-tables (king_divergence)


def load_teacher_probe() -> dict:
    raw = tomllib.loads(SOURCES_TOML.read_text()).get("teacher_probe") or {}
    if not raw or not raw.get("enabled", False):
        return {}
    path = REPO / str(raw.get("side_table") or "affine/state/teacher_probe/probes.jsonl")
    rows: dict[str, dict] = {}
    if path.exists():
        for line in path.read_text().split("\n"):
            if line.strip():
                row = json.loads(line)
                rows[str(row["turn_id"])] = row     # last row wins (re-probes)
    for tid, row in SIDE_PROBE_ROWS.items():
        rows.setdefault(tid, row)          # a real probe row wins
    return {"rows": rows, "path": str(path),
            "groups": frozenset(str(g) for g in (raw.get("groups") or KING_GROUPS)),
            "min_valid": int(raw.get("min_valid", 2) or 2),
            "require_distinct": bool(raw.get("require_distinct", True)),
            "retire_failed_published": bool(raw.get("retire_failed_published", True)),
            "restamp_text": bool(raw.get("restamp_text", True)),
            "pending_path": REPO / str(raw.get("pending") or "affine/state/teacher_probe/pending.jsonl")}


def probe_verdict(cfg: dict, turn_id: str) -> str:
    """pass | pass_text | fail | missing.
    pass_text (Jacob's rule, 2026-09-14: select by king failure, label by
    whatever keeps the teacher's references parseable): >= min_valid of the
    teacher's samples are prose with no action in the turn's dialect and
    not all identical -- the teacher answers / says "done" where the king
    repeated a tool call. The turn is admitted with kind `text` instead of
    dropped. Rows probed before the text fields existed with >= 2 prose
    samples are `missing` (re-probed), not failed."""
    row = cfg["rows"].get(turn_id)
    if row is None:
        return "missing"
    n_valid = int(row.get("n_valid") or 0)
    if n_valid >= cfg["min_valid"] and not (cfg["require_distinct"] and row.get("identical")):
        return "pass"
    if cfg["restamp_text"] and row.get("kind") != dialects.TEXT_KIND:
        if "text_distinct" in row:
            if int(row["text_valid"]) >= cfg["min_valid"] and \
                    (int(row["text_distinct"]) >= 2 or not cfg["require_distinct"]):
                return "pass_text"
        elif sum(k == dialects.TEXT_KIND for k in (row.get("sample_kinds") or [])) >= cfg["min_valid"]:
            return "missing"      # probed before the text fields existed
    return "fail"


def probe_gate(records: list[dict], cfg: dict, drops: dict[str, int],
               src2grp: dict[str, str], mix: dict[str, float]
               ) -> tuple[list[dict], list[dict], list[dict]]:
    """(records ready for the mix, records held for probing, pending turns).
    Runs after routing on new and carryover records alike; a record whose
    turns all pass proceeds, failing turns are dropped, and a record with
    any unprobed turn is held whole."""
    ready: list[dict] = []
    held: list[dict] = []
    pending: list[dict] = []
    for rec in records:
        g = group_of(rec, src2grp, mix)
        if g not in cfg["groups"]:
            ready.append(rec)
            continue
        keep, missing = [], []
        for m in rec["turns"]:
            tid = f"{rec['traj_id']}:{m['turn_idx']}"
            v = probe_verdict(cfg, tid)
            if v == "pass_text":
                prefix = [{"role": nd["role"], "content": nd["content"]}
                          for nd in node_path(rec["nodes"], int(m["node_id"]))[:-1]]
                if dialects.get(dialects.TEXT_KIND).mandate_ok(prefix):
                    m["action_kind"] = dialects.TEXT_KIND
                    _count(drops, f"probe_restamped_text_{g}")
                    keep.append(m)
                else:
                    _count(drops, "probe_failed")
                    _count(drops, f"probe_failed_{g}_text_mandate")
            elif v == "pass":
                keep.append(m)
            elif v == "fail":
                _count(drops, "probe_failed")
                _count(drops, f"probe_failed_{g}")
            else:
                missing.append(m)
        if missing:
            for m in missing:
                t = materialize_turn(rec, m)
                pending.append({"turn_id": f"{rec['traj_id']}:{m['turn_idx']}", "group": g,
                                "kind": m.get("action_kind") or rec.get("action_kind"),
                                "prefix": t["prefix"]})
            rec["turns"] = keep + missing
            held.append(rec)
            continue
        if keep:
            rec["turns"] = keep
            ready.append(rec)
        else:
            _count(drops, f"probe_emptied_{g}")
    return ready, held, pending


def failed_published(pub: PublicCorpus, live: dict | None, cfg: dict) -> dict[str, list[str]]:
    """Published rows of the gated groups whose probe row fails: {group: [turn ids]}."""
    table = index_table(pub, live, ["turn_id", "stratum"])
    out: dict[str, list[str]] = {}
    if table is None:
        return out
    for tid, stratum in zip(table.column("turn_id").to_pylist(), table.column("stratum").to_pylist()):
        ns = str(stratum).split(":")[0]
        if ns in cfg["groups"] and probe_verdict(cfg, tid) == "fail":
            out.setdefault(ns, []).append(tid)
    return out


# -- composition guard ----------------------------------------------------------
MAX_SHARE_SHIFT = 0.05
# docs/auto-research-loop.md §2: with folds every 6 h the per-fold guard
# alone allows a 30-point daily swing, so any group's slice share may move
# at most DAILY_SHIFT_CAP points against its share 24 h ago (state
# `share_history`: one snapshot per published epoch).
DAILY_SHIFT_CAP = 0.10
DAILY_WINDOW_S = 24 * 3600


def composition_table(before: dict[str, int], after: dict[str, int]) -> list[tuple]:
    """(group, strata before, strata after, share before, share after, delta)."""
    tb = sum(before.values()) or 1
    ta = sum(after.values()) or 1
    rows = []
    for g in sorted(set(before) | set(after), key=lambda k: -after.get(k, 0)):
        b, a = before.get(g, 0), after.get(g, 0)
        rows.append((g, b, a, b / tb, a / ta, a / ta - b / tb))
    return rows


# -- publish -------------------------------------------------------------------
def pack_pending(records: list[dict], epoch: int) -> PackResult:
    pack_dir = WORK_DIR / f"pack_{epoch:04d}"
    if pack_dir.exists():
        shutil.rmtree(pack_dir)
    return pack_view_records(records, pack_dir, epoch=epoch, view_spec=VIEW_SPEC)


def resume_pack(pending: dict) -> PackResult:
    pack_dir = Path(pending["pack_dir"])
    epoch = int(pending["epoch"])
    chunk_paths = sorted(pack_dir.glob(f"view_{epoch:04d}_*.jsonl"))
    if not chunk_paths:
        fatal(f"pending pack dir {pack_dir} has no chunks -- operator check")
    chunk_meta = []
    for p in chunk_paths:
        raw = p.read_bytes()
        recs = [json.loads(l) for l in raw.split(b"\n") if l.strip()]
        i = int(p.stem.rsplit("_", 1)[-1])
        chunk_meta.append({
            "key": f"views/{VIEW_SPEC}/chunks/view_{epoch:04d}_{i:04d}.jsonl.gz",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "n_trajectories": len(recs),
            "n_turns": sum(len(r["turns"]) for r in recs),
            "format": FORMAT, "active": True,
        })
    index_path = pack_dir / f"turns_{epoch:04d}.parquet"
    return PackResult(chunk_paths=chunk_paths, chunk_meta=chunk_meta,
                      index_path=index_path,
                      index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(),
                      n_turns=pq.read_metadata(index_path).num_rows,
                      n_trajectories=sum(m["n_trajectories"] for m in chunk_meta))


def merge_index(pack: PackResult, publisher: CorpusPublisher,
                prev: dict | None, epoch: int,
                retire_turn_ids: list[str] | None = None) -> None:
    """Previous active index + the new pack's rows -> one parquet the
    manifest points at (evalsrv reads exactly one index). `retire_turn_ids`
    (math re-source, 2026-09-12): rows of the previous index dropped from
    the merged one -- the turns leave D while their chunk objects stay."""
    if not prev or not prev.get("index"):
        return
    prev_raw = publisher.get(prev["index"]["key"])
    if hashlib.sha256(prev_raw).hexdigest() != prev["index"]["sha256"]:
        fatal("previous index sha mismatch on the bucket")
    prev_table = pq.read_table(io.BytesIO(prev_raw))
    if retire_turn_ids:
        mask = pc.invert(pc.is_in(prev_table.column("turn_id"),
                                  value_set=pa.array(retire_turn_ids, pa.string())))
        kept = prev_table.filter(mask)
        log(f"index: retired {prev_table.num_rows - kept.num_rows} of "
            f"{len(retire_turn_ids)} listed turn rows from the previous index")
        prev_table = kept
    new_table = pq.read_table(pack.index_path)
    if STRATA_BUDGET:
        prev_table = apply_budget_table(prev_table, SRC2GRP_GLOBAL, STRATA_BUDGET)
        new_table = apply_budget_table(new_table, SRC2GRP_GLOBAL, STRATA_BUDGET)
    elif "stratum_src" in prev_table.column_names and "stratum_src" not in new_table.column_names:
        new_table = new_table.append_column("stratum_src", new_table.column("stratum"))
    merged = pa.concat_tables([prev_table, new_table], promote_options="default")
    merged_path = pack.index_path.with_name(f"turns_{epoch:04d}_merged.parquet")
    pq.write_table(merged, merged_path, compression="zstd")
    pack.index_path = merged_path
    pack.index_sha256 = hashlib.sha256(merged_path.read_bytes()).hexdigest()
    pack.n_turns = merged.num_rows


def publish_pending(state: dict, publisher: CorpusPublisher,
                    traces_sha: str, legacy_sha: str | None) -> tuple[dict, str]:
    pending = state["pending"]
    epoch = int(pending["epoch"])
    prev, prev_sha = publisher.current_manifest()
    if prev and int(prev["corpus_epoch"]) >= epoch:
        log(f"pending epoch {epoch} already in manifest; finalizing state only")
        return prev, prev_sha
    pack = resume_pack(pending)
    if pending.get("src_override"):
        SRC_OVERRIDE.update(pending["src_override"])
    merge_index(pack, publisher, prev, epoch,
                retire_turn_ids=pending.get("retire_turn_ids") or [])
    if prev is not None:
        prev_manifest = f"{publisher.manifests_prefix}/{prev_sha}.json"
        prev_shards = list(prev["shards"])
    else:
        # First schema-3 revision: chain into the Hippius history so the
        # lineage back to epoch 1 stays walkable.
        prev_manifest = f"turns/manifests/{legacy_sha}.json" if legacy_sha else None
        prev_shards = []
    extra = {"traces_manifest_sha256": traces_sha,
             "allowed_action_kinds": pending["allowed_kinds"]}
    if legacy_sha:
        extra["legacy_turns_manifest_sha256"] = legacy_sha
    # Top-level totals for the dataset table (index.n_turns stays the SSOT).
    extra["n_turns"] = int(pack.n_turns)
    extra["n_strata"] = int(pending.get("n_strata_after") or 0) or None
    extra["published_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if pending.get("floors"):
        extra["floors"] = pending["floors"]
    if pending.get("yield_groups"):
        extra["yield"] = {"groups": pending["yield_groups"],
                          "sources": {k: {kk: vv for kk, vv in v.items() if kk != "top_drops"} | {
                              "top_drops": v["top_drops"]} for k, v in (pending.get("yield_sources") or {}).items()}}
    if pending.get("admission_gate"):
        ag = pending["admission_gate"]
        extra["admission_gate"] = {"apply_groups": ag.get("apply_groups"), "retired": ag.get("retired"),
                                   "gate_state": ag.get("gate_state"), "gate_reason": ag.get("gate_reason"),
                                   "flips_this_fold": ag.get("flips_this_fold"), "projection": ag.get("projection"),
                                   "published_tally": ag.get("published"), "signals": ag.get("signals"),
                                   "rule": "admit iff king failed AND teacher recovers (state majority-of-3, else task teacher_solved); dead-reference turns dropped; unverified held"}
    if pending.get("band_filter"):
        bf = pending["band_filter"]
        extra["band_filter"] = {"per_source": bf.get("per_source"), "retired": bf.get("retired"),
                                "retired_by_source": bf.get("retired_by_source"),
                                "tally": bf.get("tally"), "rules": bf.get("rules"), "held_turns": bf.get("held_turns"),
                                "backfill": bf.get("backfill"), "coverage": bf.get("coverage"),
                                "rule": "teacher-side turns of a task fold iff the teacher solved it (>= k of n attempts) and the king seat did not (> m of n seat solves drop; attempts from the traces); rows outside the band retire"}
    if pending.get("curriculum_block"):
        # Adaptive curriculum stamp (plan §2.3 / §3.4; evalsrv reads it into
        # slice.curriculum_version). manifest_sha256 = the manifest the
        # weights were computed AGAINST.
        extra["curriculum"] = pending["curriculum_block"]
    return publisher.publish_revision(
        pack, epoch=epoch, view_spec=VIEW_SPEC, prev_manifest=prev_manifest,
        prev_shards=prev_shards, extra=extra)


def finalize(state: dict, manifest: dict, mhash: str) -> None:
    pending = state["pending"]
    state["folded_chunks"] = sorted(set(state["folded_chunks"])
                                    | set(pending["folded_chunks"]))
    for group, n in (pending.get("group_turns") or {}).items():
        state["group_counts"][group] = int(state["group_counts"].get(group, 0)) + int(n)
    for group, keys in (pending.get("group_strata_after_retire") or {}).items():
        state["group_strata"][group] = sorted(set(keys))
    for group, keys in (pending.get("group_strata_added") or {}).items():
        state["group_strata"][group] = sorted(
            set(state["group_strata"].get(group, [])) | set(keys))
    for bucket, keys in (pending.get("lang_strata_added") or {}).items():
        state["lang_strata"][bucket] = sorted(
            set(state["lang_strata"].get(bucket, [])) | set(keys))
    tot_s = sum(len(v) for v in state["group_strata"].values()) or 1
    hist = [h for h in (state.get("share_history") or [])
            if datetime.now(timezone.utc).timestamp() - float(h["at"]) <= 3 * DAILY_WINDOW_S]
    hist.append({"epoch": int(pending["epoch"]), "at": datetime.now(timezone.utc).timestamp(),
                 "iso": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "shares": {g: round(len(v) / tot_s, 5) for g, v in state["group_strata"].items()}})
    state["share_history"] = hist
    state["unannounced"] = {
        "epoch": int(pending["epoch"]), "n_added": int(pending["n_turns"]),
        "total": int(manifest["index"]["n_turns"]), "manifest_sha256": mhash,
        "by_dialect": pending.get("by_dialect") or {},
        "by_group": pending.get("group_turns") or {},
        "strata": {g: len(v) for g, v in state["group_strata"].items()},
        "init": bool(pending.get("init")),
        "n_retired": len(pending.get("retire_turn_ids") or []),
        "n_backfill_excluded": int(pending.get("n_backfill_excluded") or 0),
        "recurrence": pending.get("recurrence"),
        "budget_migrated": bool(pending.get("budget_migrated")),
        "strata_raw_before_budget": pending.get("strata_raw_before_budget"),
        "recurrence_before_budget": pending.get("recurrence_before_budget"),
        "curriculum_line": pending.get("curriculum_line"),
        "floors": pending.get("floors"),
        "yield_groups": pending.get("yield_groups"),
        "yield_extra": pending.get("yield_extra"),
        "admission_gate": pending.get("admission_gate"),
        "band_filter": pending.get("band_filter"),
    }
    if pending.get("coached_folded") is not None:
        state["coached_folded"] = pending["coached_folded"]
    if pending.get("gate_enforced") is not None:
        state["gate_enforced"] = sorted(set(state.get("gate_enforced") or []) | set(pending["gate_enforced"]))
    if pending.get("budget_signature"):
        state["strata_budget_signature"] = pending["budget_signature"]
        state.pop("group_strata_raw_before_budget", None)
        state.pop("recurrence_before_budget", None)
    state["history"].append({
        "epoch": int(pending["epoch"]), "n_turns": int(pending["n_turns"]),
        "n_chunks": len(pending["folded_chunks"]),
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "manifest_sha256": mhash,
    })
    state["pending"] = None
    save_state(state)


def budget_note(info: dict) -> str:
    rec = info.get("recurrence")
    if not rec:
        return ""
    out = ""
    if info.get("budget_migrated") and info.get("strata_raw_before_budget"):
        raw = info["strata_raw_before_budget"]; tr = sum(raw.values()) or 1
        top = sorted(raw.items(), key=lambda kv: -kv[1])[:6]
        out += ("**Strata budget (phase 9, operator directive 2026-09-14):** slice re-weighted toward "
                "the king's failure states (fixed buckets for teacher groups, up to 3 turns per task "
                "for king groups; no turn left D). Before: " + ", ".join(
                    f"{k} {100 * v / tr:.0f}%" for k, v in top) + ", ...\n")
    before = info.get("recurrence_before_budget")
    out += (f"Simulated per-duel recurrence: {100 * rec['turn_overlap']:.1f}% turn ids / "
            f"{100 * rec['rollout_overlap']:.1f}% rollouts shared between two duels"
            + (f" (before {100 * before['turn_overlap']:.1f}% / {100 * before['rollout_overlap']:.1f}%)"
               if before else "")
            + " -- RT-6 watch item.\n")
    return out


def floors_line(info: dict) -> str:
    f = info.get("floors")
    if not f:
        return ""
    return "Floors: " + ", ".join(f"{k} {100 * v['share']:.1f}%{'' if v['ok'] else ' BELOW'} (>= {100 * v['floor']:.1f}%)"
                                  for k, v in f.items()) + ".\n"


def yield_line(info: dict) -> str:
    y = info.get("yield_groups")
    if not y:
        return ""
    short = sorted(((g, v) for g, v in y.items() if v.get("strata_over_target") is not None),
                   key=lambda kv: kv[1]["strata_over_target"])[:6]
    return "Yield (strata / target): " + ", ".join(
        f"{g} {v['strata']}/{v['target_strata']:.0f} ({v['strata_over_target']:.2f})" for g, v in short) + \
        "; full per-source table in corpus/fold_stats.json.\n"


def upstream_line(info: dict) -> str:
    extra = info.get("yield_extra") or {}
    if "upstream_fetch_turns" not in extra:
        return ""
    n = int(extra.get("upstream_fetch_turns") or 0)
    return (f"Upstream fetch: {n} new turns dropped because the prefix already "
            "held a GitHub fetch, an upstream clone, or a package download.\n")


def sublabel_line(info: dict) -> str:
    y = info.get("yield_extra") or {}
    subl = y.get("divergence_sublabels") or {}
    parts = []
    if subl:
        parts.append("king_divergence sub-labels this fold: " + ", ".join(f"{k} {v}" for k, v in sorted(subl.items())))
    if y.get("interactive_prose_turns"):
        parts.append(f"interactive prose turns admitted as text: {y['interactive_prose_turns']}")
    bk = y.get("by_king") or {}
    if bk:
        parts.append("king candidates by digest: " + ", ".join(
            f"{d} {v['seen']} rollouts/{v['turns']} turns" for d, v in sorted(bk.items(), key=lambda kv: -kv[1]['seen'])[:5]))
    return ("; ".join(parts) + ".\n") if parts else ""


def gate_line(info: dict) -> str:
    ag = info.get("admission_gate")
    if not ag:
        return ""
    ret = ag.get("retired") or {}
    st = ag.get("gate_state") or {}
    return ("Admission gate (dead refs out everywhere; recovery rule): enforced "
            f"{sorted(g for g, v in st.items() if v == 'enforced')}, shadow "
            f"{sorted(g for g, v in st.items() if v == 'shadow')}, exempt "
            f"{sorted(g for g, v in st.items() if v == 'exempt')}"
            + (f"; PROMOTED this fold: {ag['flips_this_fold']}" if ag.get("flips_this_fold") else "")
            + (f"; retired {sum(ret.values())} rows" if ret else "") + ".\n")


def announce(state: dict, public_base: str) -> None:
    info = state["unannounced"]
    epoch = info["epoch"]
    dialects_line = ", ".join(f"{k} {v:,}" for k, v in
                              sorted(info.get("by_dialect", {}).items())) or "n/a"
    groups_line = ", ".join(f"{k} {v:,}" for k, v in
                            sorted(info.get("by_group", {}).items())) or "n/a"
    strata = info.get("strata") or {}
    tot = sum(strata.values()) or 1
    strata_line = ", ".join(f"{k} {100 * v / tot:.0f}%" for k, v in
                            sorted(strata.items(), key=lambda kv: -kv[1])) or "n/a"
    head = ("**Corpus D is now the schema-3 trace view — epoch "
            f"{epoch} is live.**" if info.get("init") else
            f"**Corpus refresh: epoch {epoch} is live.**")
    content = (
        f"{head}\n\n"
        f"{info['n_added']:,} new turns folded into the production turn corpus D "
        f"(corpus total: {info['total']:,} turns). New turns by dialect: "
        f"{dialects_line}; by group: {groups_line}.\n"
        f"Slice composition (share of strata = share of every duel slice): "
        f"{strata_line}.\n"
        + (f"Retired from the index: {info['n_retired']:,} turns (chunks unchanged; see llms.txt).\n"
           if info.get("n_retired") else "")
        + (f"Env backfill rollouts excluded from D: {info['n_backfill_excluded']:,}.\n"
           if info.get("n_backfill_excluded") else "")
        + budget_note(info)
        + (f"{info['curriculum_line']}\n" if info.get("curriculum_line") else "")
        + floors_line(info) + yield_line(info) + upstream_line(info) + gate_line(info) + sublabel_line(info)
        + "\n"
        f"- corpus_epoch: **{epoch}**\n"
        f"- schema_version: **3** (view `{VIEW_SPEC}`: one record per rollout "
        "holding the message graph the model saw + turn metas; prefix = "
        "root→parent path of the turn's node)\n"
        f"- manifest: `{public_base}/corpus/manifest.json` "
        f"(sha256 `{info['manifest_sha256']}`)\n"
        f"- index: `views/{VIEW_SPEC}/index/turns_{epoch:04d}.parquet`; "
        f"chunks: `views/{VIEW_SPEC}/chunks/view_{epoch:04d}_*.jsonl.gz`\n"
        f"- full rollout traces (what the turns were cut from): "
        f"`{public_base}/traces/manifest.json`\n\n"
        "Miners: fetch the manifest, read the Parquet index, pull only the "
        "chunk objects you need. Layout: https://affine.io/llms.txt. Eval "
        "pods pick the new manifest up automatically."
    )
    if len(content) > 1990:
        # Discord caps a message at 2,000 characters; keep the head (the
        # numbers) and drop the tail (the layout boilerplate) rather than fail.
        content = content[:1980].rsplit("\n", 1)[0] + "\n…"
    r = httpx.post(
        f"https://discord.com/api/v10/channels/{DISCORD_CHANNEL_ID}/messages",
        headers={"Authorization": f"Bot {env_value('DISCORD_BOT_TOKEN_ARBOS_BITTENSOR')}"},
        json={"content": content}, timeout=30)
    if r.status_code >= 300:
        log(f"announce failed (HTTP {r.status_code}: {r.text[:200]}); will retry next cycle")
        return
    mid = r.json().get("id")
    log(f"announced epoch {epoch}: https://discord.com/channels/"
        f"{DISCORD_GUILD_ID}/{DISCORD_CHANNEL_ID}/{mid}")
    state["unannounced"] = None
    save_state(state)


# -- main ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--init", action="store_true",
                    help="first schema-3 revision: import the live v2 corpus "
                         "as legacy records, then fold every published trace")
    ap.add_argument("--no-legacy", action="store_true",
                    help="with --init: do not import the v2 epochs; D starts "
                         "from the traces alone and the mix waterfill counts "
                         "from zero (the v2 history stays at turns/** for "
                         "replay and is still chained via prev_manifest)")
    ap.add_argument("--allowed-kinds", default=None,
                    help="comma list overriding [dataset].allowed_action_kinds")
    ap.add_argument("--publish-prefix", default="",
                    help="publish under this key prefix (e.g. staging/)")
    ap.add_argument("--no-publish", action="store_true",
                    help="build the pack locally, publish nothing, keep state")
    ap.add_argument("--no-announce", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="publish even below MIN_NEW_TURNS")
    ap.add_argument("--ignore-fold-mix", action="store_true",
                    help="use [mix] even while [fold_mix] exists (T0 rehearsal)")
    ap.add_argument("--rederive", action="store_true",
                    help="re-derive EVERY published trace chunk, not just the "
                         "unfolded ones, and drop the deferred carryover (it is "
                         "regenerated). Already-published turn ids are still "
                         "skipped, so only turns the previous contract could not "
                         "admit enter. Used once at the wvk 13 flip (2026-09-09) "
                         "to back-fill the `text` turns of trajectories folded "
                         "under wvk 11/12.")
    ap.add_argument("--rederive-since", default=None, metavar="ISO8601",
                    help="like --rederive, but only for published chunks "
                         "created at or after this time (plus the unfolded "
                         "ones); deferred rollouts from other chunks are kept. "
                         "A full --rederive re-tokenizes every deferred coding "
                         "prefix (hours); a data event that touches only recent "
                         "traces (king_loop_onset, 2026-09-11: the king seat "
                         "went live 2026-09-10T13:00Z) needs only these.")
    ap.add_argument("--rederive-chunks", default=None, metavar="FILE",
                    help="like --rederive-since, for the published chunk keys "
                         "listed in FILE (one per line): re-derive exactly those "
                         "chunks and drop their deferred copies. Used 2026-09-12 "
                         "to back-fill the multi-root rollouts (Claude Code "
                         "WebFetch / Kimi sub-agent / pi compaction side chats) "
                         "the fold had dropped as TraceShapeError.")
    ap.add_argument("--allow-shift", action="store_true",
                    help="publish even if a group's slice share (share of "
                         f"strata) moves by more than {MAX_SHARE_SHIFT:.0%} in "
                         "this epoch (guard added after the epoch-25 terminal "
                         "flood, 2026-09-12)")
    args = ap.parse_args()

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # One fold at a time: the pm2 cron and an operator's manual run share
    # state.json, the pack dir and the deferred file. The lock lives for the
    # process; a second instance exits at once instead of racing.
    lock = open(STATE_DIR / "fold.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fatal("another fold is running (ops/corpus_build/fold.lock held); exiting")
    cfg = load_config()
    public_base = cfg.data_r2["public_base_url"].rstrip("/")
    pub = PublicCorpus(public_base)
    sec = cfg.secrets
    ak, sk = env_value("DATA_R2_ACCESS_KEY_ID"), env_value("DATA_R2_SECRET_ACCESS_KEY")
    if not (ak and sk) and not args.no_publish:
        fatal("DATA_R2_ACCESS_KEY_ID / DATA_R2_SECRET_ACCESS_KEY missing")
    prefix = args.publish_prefix.strip("/")
    publisher = None if args.no_publish else CorpusPublisher(
        bucket=cfg.data_r2["bucket"],
        endpoint=env_value("DATA_R2_ENDPOINT") or sec.data_r2_endpoint,
        access_key_id=ak, secret_access_key=sk,
        key_prefix=f"{prefix}/" if prefix else "", log=log)
    if prefix:
        # Staging publishes must not touch production state.
        global STATE_PATH, DEFERRED_PATH
        STATE_PATH = STATE_DIR / f"state.{prefix.replace('/', '_')}.json"
        DEFERRED_PATH = WORK_DIR / f"deferred_views.{prefix.replace('/', '_')}.jsonl"
    state = load_state()

    traces_manifest, traces_sha = pub.manifest(cfg.dataset.traces_manifest_key)
    BENCH_FAIL.clear(); BENCH_FAIL.update(load_bench_fail())
    bench_sha = None
    if BENCH_FAIL.get("enabled") and BENCH_FAIL.get("mode") == "direct":
        try:
            bench_manifest, bench_sha = pub.manifest(BENCH_FAIL["traces_prefix"] + "manifest.json")
            extra_chunks = [c for c in bench_manifest.get("chunks", [])
                            if c["key"] not in {x["key"] for x in traces_manifest["chunks"]}]
            traces_manifest["chunks"] = list(traces_manifest["chunks"]) + extra_chunks
            log(f"bench_fail: direct mode ON -- {len(bench_manifest.get('chunks', []))} bench trace chunk(s) "
                f"({bench_manifest.get('n_rollouts')} rollouts) from {BENCH_FAIL['traces_prefix']} joined the fold "
                f"(manifest {str(bench_sha)[:12]})")
        except Exception as e:  # noqa: BLE001
            log(f"bench_fail: {BENCH_FAIL['traces_prefix']}manifest.json unreadable ({e!r}); no bench rows this fold")
    else:
        log(f"bench_fail: off (allow_bench_groups={'on' if BENCH_FAIL.get('enabled') else 'off'}, mode={BENCH_FAIL.get('mode')})")
    legacy_manifest, legacy_sha = pub.manifest("turns/manifest.json")

    if state["pending"]:
        if publisher is None:
            fatal("pending publish exists; rerun without --no-publish")
        log(f"resuming pending publish for epoch {state['pending']['epoch']}")
        # merge_index needs the budget + source->group map even on a resume
        _b = load_strata_budget()
        STRATA_BUDGET.clear(); STRATA_BUDGET.update(_b)
        SRC2GRP_GLOBAL.clear(); SRC2GRP_GLOBAL.update(load_mix(ignore_fold_mix=args.ignore_fold_mix)[1])
        finalize(state, *publish_pending(state, publisher, traces_sha, legacy_sha))
    if state["unannounced"] and not args.no_announce and publisher is not None:
        announce(state, public_base)

    if publisher is not None:
        live, live_sha = publisher.current_manifest()
    elif not args.init and not prefix:
        # --no-publish preview of the production fold: read the live manifest
        # anonymously so the dry run skips already-published turns and
        # numbers the epoch as the real cycle would. Before 2026-09-11 a dry
        # run re-admitted every turn of a re-derived chunk (epoch "14").
        live, live_sha = pub.manifest(cfg.dataset.manifest_key)
    else:
        live, live_sha = None, None
    if args.init and live is not None:
        fatal(f"--init but a corpus manifest already exists (epoch {live['corpus_epoch']})")
    if not args.init and live is None and not args.no_publish:
        fatal("no schema-3 corpus manifest yet; run with --init first")

    allowed = (tuple(k.strip() for k in args.allowed_kinds.split(",") if k.strip())
               if args.allowed_kinds else tuple(cfg.dataset.allowed_action_kinds))
    log(f"allowed action kinds: {list(allowed)}")

    if sum(bool(x) for x in (args.rederive, args.rederive_since, args.rederive_chunks)) > 1:
        fatal("--rederive, --rederive-since and --rederive-chunks are exclusive")
    since = (datetime.fromisoformat(args.rederive_since)
             if args.rederive_since else None)
    if since is not None and since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)
    listed: set[str] = set()
    if args.rederive_chunks:
        listed = {l.strip() for l in Path(args.rederive_chunks).read_text().split("\n")
                  if l.strip()}
        known = {c["key"] for c in traces_manifest["chunks"]}
        if listed - known:
            fatal(f"--rederive-chunks: {len(listed - known)} key(s) not in the traces manifest")
    unfolded = [c for c in traces_manifest["chunks"]
                if args.rederive or c["key"] not in state["folded_chunks"]
                or c["key"] in listed
                or (since is not None
                    and datetime.fromisoformat(c["created_at"]) >= since)]
    # split("\n"), not splitlines(): JSON strings may carry U+2028 / U+0085.
    carryover = ([json.loads(l) for l in DEFERRED_PATH.read_text().split("\n")
                  if l.strip()] if DEFERRED_PATH.exists() else [])
    if args.rederive:
        log(f"--rederive: all {len(unfolded)} chunks re-derived; "
            f"{len(carryover)} deferred rollouts dropped (regenerated from traces)")
        carryover = []
    if since is not None or listed:
        # The deferred copies of rollouts in a re-derived chunk are stale
        # (they were cut under the previous contract); the chunk regenerates
        # them, so drop them here or the pack would hold each turn twice.
        rederived_rollouts: set[str] = set()
        for c in unfolded:
            if c["key"] in state["folded_chunks"]:
                path = pub.cached(c["key"], c["sha256"], gz_sha=True)
                rederived_rollouts |= {str(e.get("rollout_id") or "")
                                       for e in iter_jsonl_gz(path)}
        n0 = len(carryover)
        carryover = [r for r in carryover
                     if str(r.get("rollout_id") or "") not in rederived_rollouts]
        log(f"--rederive-since {since.isoformat() if since else '-'} / "
            f"--rederive-chunks {len(listed)}: {len(unfolded)} chunk(s) "
            f"derived ({sum(c['key'] in state['folded_chunks'] for c in unfolded)} "
            f"already folded); {n0 - len(carryover)} deferred rollouts from those "
            f"chunks dropped (regenerated from traces), {len(carryover)} kept")
    if not unfolded and not carryover and not args.init:
        log(f"no unfolded trace chunks ({len(traces_manifest['chunks'])} total) "
            "and no deferred rollouts; done")
        return
    log(f"{len(unfolded)} unfolded trace chunk(s), {len(carryover)} deferred "
        f"rollout(s)")

    published = published_turn_ids(
        PublicCorpus(f"{public_base}/{prefix}" if prefix else public_base), live)
    legacy: list[dict] = []
    if args.no_legacy and not args.init:
        fatal("--no-legacy only applies to --init")
    if args.init and args.no_legacy:
        log("--no-legacy: v2 epochs not imported; D restarts from the traces")
    if args.init and not args.no_legacy:
        legacy = legacy_records(pub, legacy_manifest)
        for rec in legacy:
            for m in rec["turns"]:
                published.add(f"{rec['traj_id']}:{m['turn_idx']}")
        log(f"legacy import: {len(legacy)} trajectories, "
            f"{sum(len(r['turns']) for r in legacy)} turns from v2 epochs")

    mix, src2grp, lang_mix, buckets = load_mix(ignore_fold_mix=args.ignore_fold_mix)
    global INTERACTIVE_SOURCES
    INTERACTIVE_SOURCES = load_interactive_sources()      # before any derive_chunk call
    DECONTAM.clear(); DECONTAM.update(load_decontamination())
    if DECONTAM:
        log("decontamination: " + "; ".join(f"{src} bench ids {sum(len(v) for v in c['bench'].values())} "
                                            f"({', '.join(f'{d} {len(v)}' for d, v in c['bench'].items())})"
                                            for src, c in DECONTAM.items()))
    if INTERACTIVE_SOURCES:
        log(f"interactive sources (mid-trajectory prose replies admitted as text): {sorted(INTERACTIVE_SOURCES)}")
    routed = {KING_LOOP_GROUP: load_king_loop_onset(),
              KING_PIVOT_GROUP: load_king_pivot(),
              KING_RECOVERABLE_GROUP: load_king_recoverable(),
              KING_DIVERGENCE_GROUP: load_king_divergence(),
              KING_DONE_GROUP: load_king_done(),
              KING_TOOLUSE_GROUP: load_king_tooluse(),
              COMPLETION_GROUP: load_completion(),
              COMPLETION_PRE_GROUP: load_completion_pre()}
    king = load_king_fail()
    log(f"king_common: {load_king_common()}")
    for g, cfg in routed.items():
        if cfg and mix.get(g, 0.0) <= 0:
            # Fail closed: group_of would file the routed records under coding.
            fatal(f"[{g}] is configured but the fold mix has no {g} share")
        shown = {k: v for k, v in (cfg or {}).items() if k not in ("raw", "table")}
        log(f"{g}: {'off' if not cfg else shown}")
    king_loop, king_pivot, completion, king_recoverable, king_done, king_tooluse, completion_pre = (
        routed[g] for g in (KING_LOOP_GROUP, KING_PIVOT_GROUP, COMPLETION_GROUP,
                            KING_RECOVERABLE_GROUP, KING_DONE_GROUP, KING_TOOLUSE_GROUP,
                            COMPLETION_PRE_GROUP))
    king_divergence = routed[KING_DIVERGENCE_GROUP]
    if completion:
        # A group the fold mix holds at 0 contributes nothing to D -- not
        # even its finals (env wave 1: `general` = 0.0 until the operator
        # reads the first fold telemetry).
        zero = {src for src, g in src2grp.items() if mix.get(g, 0.0) <= 0}
        completion["exclude_sources"] = completion["exclude_sources"] | zero
        log(f"{COMPLETION_GROUP}: excluding sources {sorted(completion['exclude_sources'])} "
            f"(config + zero-share groups); min_replies {completion['min_replies']}")
    for g, cfg in ((KING_PIVOT_GROUP, king_pivot), (KING_RECOVERABLE_GROUP, king_recoverable),
                   (KING_DIVERGENCE_GROUP, king_divergence)):
        if cfg:
            log(f"{g}: {cfg['n_rows']} side-table rows in {cfg['n_files']} file(s) -> "
                f"admitted on {len(cfg['table'])} rollouts / "
                f"{sum(len(v) for v in cfg['table'].values())} turns")

    # Retire-and-readmit plans: side-table groups whose turns were published
    # under another king group before the side-table existed.
    readmit_from = {KING_PIVOT_GROUP: ("king_fail",),
                    KING_RECOVERABLE_GROUP: ("king_fail", KING_LOOP_GROUP, KING_PIVOT_GROUP),
                    KING_DIVERGENCE_GROUP: ("king_fail", KING_LOOP_GROUP, KING_PIVOT_GROUP, KING_TOOLUSE_GROUP)}
    readmits: dict[str, dict[str, list[str]]] = {}
    for g, cfg in ((KING_PIVOT_GROUP, king_pivot), (KING_RECOVERABLE_GROUP, king_recoverable),
                   (KING_DIVERGENCE_GROUP, king_divergence)):
        if cfg and cfg.get("readmit_published"):
            readmits[g] = readmit_plan(pub, live, cfg, readmit_from[g])
            ids = {t for v in readmits[g].values() for t in v}
            log(f"{g}: readmit -- {len(ids)} admitted turns are published under "
                f"{ {k: len(v) for k, v in readmits[g].items()} }; unpublishing them for this run")
            published -= ids

    # King rows in the live index by group: lets a higher-precedence king
    # group reclaim a turn published under a lower one (king_done over
    # king_fail, ...); the old row is retired in this revision.
    published_king_ns: dict[str, str] = {}
    kt = index_table(pub, live, ["turn_id", "stratum"])
    if kt is not None:
        for tid, stratum in zip(kt.column("turn_id").to_pylist(),
                                kt.column("stratum").to_pylist()):
            ns = str(stratum).split(":")[0]
            if ns in KING_GROUPS:
                published_king_ns[tid] = ns
    reclaimed: dict[str, set[str]] = {}
    probe_early = load_teacher_probe()
    probe_text = frozenset(tid for tid in (probe_early.get("rows") or {})
                           if probe_verdict(probe_early, tid) == "pass_text") if probe_early else frozenset()
    if probe_text:
        log(f"teacher probe: {len(probe_text)} turns pass as `text` (teacher answers in prose)")

    baker = ToolBaker.from_pretrained()
    panel = panel_keys()
    drops: dict[str, int] = {}
    notes: dict[str, int] = NOTES_GLOBAL
    notes.clear()
    candidates: list[dict] = list(carryover)
    for i, c in enumerate(unfolded, 1):
        path = pub.cached(c["key"], c["sha256"], gz_sha=True)
        if str(c["key"]).startswith(BACKFILL_CHUNK_PREFIX):
            n_bf = sum(1 for _ in iter_jsonl_gz(path))
            _count(drops, "backfill_excluded", n_bf)
            log(f"backfill chunk {c['key']} excluded ({n_bf} rollouts)")
            continue
        recs = derive_chunk(path, baker, panel, allowed, published, drops,
                            chunk_key=str(c["key"]),
                            king_loop=king_loop, king_pivot=king_pivot,
                            completion=completion, king_recoverable=king_recoverable,
                            king_divergence=king_divergence,
                            king_done=king_done, king_fail_cfg=king, notes=notes,
                            published_king_ns=published_king_ns, reclaimed=reclaimed,
                            probe_text=probe_text,
                            king_tooluse=king_tooluse, completion_pre=completion_pre)
        for rec in recs:
            for m in rec["turns"]:
                published.add(f"{rec['traj_id']}:{m['turn_idx']}")
        candidates.extend(recs)
        if i % 100 == 0 or i == len(unfolded):
            log(f"derived {i}/{len(unfolded)} chunks: {len(candidates)} rollouts, "
                f"{sum(len(r['turns']) for r in candidates)} turns")
    # One record per turn id across the candidate set: the carryover can hold
    # two copies of a rollout (a --rederive-chunks pass re-derived chunks whose
    # deferred copies were not all dropped, 2026-09-14) and a datagen chunk can
    # repeat a rollout; the pack refuses duplicates, so drop them here.
    seen_tids: set[str] = set()
    deduped: list[dict] = []
    n_dup_turns = n_dup_recs = 0
    n_bf_carry = sum(1 for rec in candidates if is_backfill(rec))
    if n_bf_carry:
        _count(drops, "backfill_excluded", n_bf_carry)
        candidates = [rec for rec in candidates if not is_backfill(rec)]
    for rec in candidates:
        keep = []
        for m in rec["turns"]:
            tid = f"{rec['traj_id']}:{m['turn_idx']}"
            if tid in seen_tids:
                n_dup_turns += 1
                continue
            seen_tids.add(tid)
            keep.append(m)
        if not keep:
            n_dup_recs += 1
            continue
        rec["turns"] = keep
        deduped.append(rec)
    if n_dup_turns:
        log(f"candidates: dropped {n_dup_turns} duplicate turn(s) / {n_dup_recs} whole duplicate record(s)")
    candidates = deduped
    king_coached = load_king_coached()
    coached_folded: set[str] = set(state.get("coached_folded") or [])
    if king_coached:
        candidates.extend(derive_coached(king_coached, baker, panel, allowed, published, drops, notes,
                                         coached_folded))
    log(f"drops: {drops or 'none'}")
    if _TOKEN_CACHE is not None:
        log(f"prefix token cache: {_TOKEN_CACHE.hits} hits / {_TOKEN_CACHE.misses} misses")
    if king_loop:
        log(f"king loop onsets: labelled {notes.get('king_loop_labelled_rollouts', 0)} "
            f"failed king rollouts -> {notes.get('king_loop_onset_labels', 0)} onset / "
            f"{notes.get('king_loop_in_loop_labels', 0)} in-loop labels; "
            f"admitted onsets {notes.get(f'{KING_LOOP_GROUP}_leak_exempt', 0)} leak-exempt + "
            f"{notes.get(f'{KING_LOOP_GROUP}_not_leaking', 0)} not leaking; "
            f"king_in_loop dropped {drops.get('king_in_loop', 0)}")
    if king_pivot:
        log(f"king pivots: {notes.get('king_pivot_rollouts', 0)} rollouts with admitted "
            f"pivots seen; admitted {notes.get(f'{KING_PIVOT_GROUP}_leak_exempt', 0)} "
            f"leak-exempt + {notes.get(f'{KING_PIVOT_GROUP}_not_leaking', 0)} not leaking; "
            f"{notes.get(f'{KING_PIVOT_GROUP}_already_published', 0)} already published "
            f"(stay in their current group); {notes.get('king_pivot_over_onset', 0)} "
            f"took precedence over an onset")
    if king_done:
        log(f"king done: {notes.get('king_done_states', 0)} done-blind states; admitted "
            f"{notes.get(f'{KING_DONE_GROUP}_leak_exempt', 0)} leak-exempt + "
            f"{notes.get(f'{KING_DONE_GROUP}_not_leaking', 0)} not leaking; "
            f"{notes.get(f'{KING_DONE_GROUP}_already_published', 0)} already published; "
            f"over onset {notes.get('king_done_over_onset', 0)} / recoverable "
            f"{notes.get(f'king_done_over_{KING_RECOVERABLE_GROUP}', 0)} / pivot "
            f"{notes.get(f'king_done_over_{KING_PIVOT_GROUP}', 0)}; later onsets dropped "
            f"{drops.get('king_later_onset', 0)} (labels {notes.get('king_later_onset_labels', 0)}); "
            f"one-reply king rollouts dropped {drops.get('king_one_reply', 0)}; king_fail "
            f"per-rollout cap dropped {drops.get('king_fail_cap', 0)} turns")
    if king_tooluse:
        log(f"king tooluse (king-selected): one-shot {notes.get('king_tooluse_one_shot', 0)} + "
            f"persist {notes.get('king_tooluse_persist', 0)} states; kind stamped "
            f"{ {k: v for k, v in notes.items() if k.startswith('king_tooluse_kind_')} }; admitted "
            f"{notes.get(f'{KING_TOOLUSE_GROUP}_leak_exempt', 0) + notes.get(f'{KING_TOOLUSE_GROUP}_not_leaking', 0)}; "
            f"already published {notes.get(f'{KING_TOOLUSE_GROUP}_already_published', 0)}")
    if completion_pre:
        log(f"completion_pre (king premature finish): {notes.get('completion_pre_states', 0)} states; "
            f"kind stamped { {k: v for k, v in notes.items() if k.startswith('completion_pre_kind_')} }; admitted "
            f"{notes.get(f'{COMPLETION_PRE_GROUP}_leak_exempt', 0) + notes.get(f'{COMPLETION_PRE_GROUP}_not_leaking', 0)}; "
            f"already published {notes.get(f'{COMPLETION_PRE_GROUP}_already_published', 0)}")
    if king_done:
        log(f"king done kind stamped { {k: v for k, v in notes.items() if k.startswith('king_done_kind_')} }")
    if king_recoverable:
        log(f"king recoverable: {notes.get('king_recoverable_rollouts', 0)} rollouts with admitted "
            f"states seen; admitted {notes.get(f'{KING_RECOVERABLE_GROUP}_leak_exempt', 0)} "
            f"leak-exempt + {notes.get(f'{KING_RECOVERABLE_GROUP}_not_leaking', 0)} not leaking; "
            f"{notes.get(f'{KING_RECOVERABLE_GROUP}_already_published', 0)} already published; "
            f"took precedence over onset {notes.get(f'king_recoverable_over_{KING_LOOP_GROUP}', 0)} / "
            f"pivot {notes.get(f'king_recoverable_over_{KING_PIVOT_GROUP}', 0)}")
    if completion:
        kinds = {k[len('completion_kind_'):]: v for k, v in notes.items()
                 if k.startswith('completion_kind_')}
        log(f"completion: {notes.get('completion_candidates', 0)} solved agent_completed "
            f"rollouts -> final replies by kind {kinds}; admitted "
            f"{notes.get(f'{COMPLETION_GROUP}_not_leaking', 0)} not leaking + "
            f"{notes.get(f'{COMPLETION_GROUP}_leak_exempt', 0)} leak-exempt; "
            f"refused by the leak rule {notes.get(f'{COMPLETION_GROUP}_leaked', 0)}; "
            f"missing for other reasons {notes.get(f'{COMPLETION_GROUP}_missing_other', 0)}; "
            f"{notes.get(f'{COMPLETION_GROUP}_already_published', 0)} already published")

    # Math re-source (phase 3): drop candidates of teacher-deterministic
    # problems and plan the retirement of their published turns.
    math_cfg = load_math_filter()
    retire_ids: list[str] = []
    math_surviving: set[str] = set()
    math_retired_strata: set[str] = set()
    if math_cfg:
        keep, mstats = math_keep_set(pub, traces_manifest, math_cfg)
        base_keep = mstats.pop("_base_keep")
        log(f"math re-source: {mstats}")
        retire_ids, math_surviving, math_retired_strata = math_retire_plan(
            pub, live, math_cfg, keep, src2grp.get(math_cfg["source"], DEFAULT_GROUP))
        cand_math = [r for r in candidates if str(r.get("source") or "") in math_cfg["sources"]]
        cand_keep = [r for r in cand_math if str(r.get("instance_id")) in keep]
        # Survival = strata the kept published turns hold + what kept
        # candidates would open (bucket assignment happens below; recompute
        # the bucket here from the source's setting).
        cand_strata: set[str] = set()
        for r in cand_keep:
            n_b, off = buckets.get(str(r.get("source") or ""), (0, 0))
            if n_b:
                grp = src2grp.get(str(r.get("source") or ""), DEFAULT_GROUP)
                h = int(hashlib.sha256(str(r["instance_id"]).encode()).hexdigest()[:8], 16)
                cand_strata.add(f"{grp}:{off + h % n_b:04d}")
        surviving_total = len(math_surviving | cand_strata)
        log(f"math re-source: published math turns {len(retire_ids) + 0} to retire, "
            f"surviving published strata {len(math_surviving)}, retired strata "
            f"{len(math_retired_strata)}; candidates {len(cand_math)} -> kept {len(cand_keep)}; "
            f"surviving strata incl. candidates {surviving_total}")
        if surviving_total < math_cfg["min_surviving_strata"] and keep != base_keep:
            # The in-cap rule alone would empty the group: fall back to the
            # phase-3 proxy (disagreement / failure), never to "no filter".
            log(f"math re-source: only {surviving_total} strata would survive the in-cap rule "
                f"(< {math_cfg['min_surviving_strata']}); falling back to the base proxy")
            keep = base_keep
            retire_ids, math_surviving, math_retired_strata = math_retire_plan(
                pub, live, math_cfg, keep, src2grp.get(math_cfg["source"], DEFAULT_GROUP))
            cand_keep = [r for r in cand_math if str(r.get("instance_id")) in keep]
            surviving_total = len(math_surviving | {
                f"{src2grp.get(str(r.get('source') or ''), DEFAULT_GROUP)}:{buckets.get(str(r.get('source') or ''), (0, 0))[1] + int(hashlib.sha256(str(r['instance_id']).encode()).hexdigest()[:8], 16) % buckets.get(str(r.get('source') or ''), (1, 0))[0]:04d}"
                for r in cand_keep if buckets.get(str(r.get("source") or ""), (0, 0))[0]})
            log(f"math re-source (base proxy): retire {len(retire_ids)}, surviving strata {surviving_total}")
        if surviving_total < math_cfg["min_surviving_strata"]:
            log(f"math re-source: only {surviving_total} strata would survive "
                f"(< {math_cfg['min_surviving_strata']}); keeping the old math pool")
            retire_ids, math_surviving, math_retired_strata = [], set(), set()
        else:
            n0 = len(candidates)
            keep_ids = {id(r) for r in cand_keep}
            candidates = [r for r in candidates
                          if str(r.get("source") or "") not in math_cfg["sources"] or id(r) in keep_ids]
            _count(drops, "math_deterministic", n0 - len(candidates))
            if not math_cfg["retire_published"]:
                retire_ids, math_surviving, math_retired_strata = [], set(), set()
    n_bucketed = assign_bucket_strata(candidates, buckets, src2grp)
    log(f"bucket strata assigned on {n_bucketed} rollouts "
        f"({ {k: (n if not off else f'{n}@{off}') for k, (n, off) in buckets.items() if n} })")
    # King seat: after the source buckets so `king_fail:NNNN` wins for king
    # rollouts on bucketed sources (math / tool_use) too.
    n_before = len(candidates)
    king_drops: dict[str, int] = {}
    candidates = route_king_fail(candidates, king, king_drops)
    n_king = sum(1 for r in candidates if r.get("fold_group") == "king_fail")
    log(f"king seat: {n_king} failed king rollouts -> king_fail "
        f"({sum(len(r['turns']) for r in candidates if r.get('fold_group') == 'king_fail')} turns), "
        f"dropped {n_before - len(candidates)} {king_drops or ''}")
    for k, v in king_drops.items():
        drops[k] = drops.get(k, 0) + v
    n_before = len(candidates)
    bench_drops: dict[str, int] = {}
    candidates = route_bench_fail(candidates, BENCH_FAIL, bench_drops)
    n_bench = sum(1 for r in candidates if r.get("fold_group") == BENCH_FAIL.get("group"))
    if n_bench or bench_drops:
        log(f"bench_fail: {n_bench} failed benchmark trials -> {BENCH_FAIL.get('group')} "
            f"({sum(len(r['turns']) for r in candidates if r.get('fold_group') == BENCH_FAIL.get('group'))} turns), "
            f"dropped {n_before - len(candidates)} {bench_drops or ''}")
    for k, v in bench_drops.items():
        drops[k] = drops.get(k, 0) + v
    n_before = len(candidates)
    # Unknown-source gate (2026-09-24, hygiene hole found by the datagen
    # worker): a record whose source is not a [source.*] entry -- and is not
    # a bench_* record the bench_fail router just claimed -- used to fall into
    # DEFAULT_GROUP (coding) as a teacher-side row. It is now HELD (deferred,
    # re-enters when the toml lists the source) and logged per source; it is
    # never admitted.
    known_sources = set(src2grp)
    unknown_held: list[dict] = []
    kept_known: list[dict] = []
    unknown_by_src: dict[str, int] = {}
    for rec in candidates:
        src0 = str(rec.get("source") or "")
        bench_ok = (rec.get("fold_group") == BENCH_FAIL.get("group") and src0.startswith(BENCH_SOURCE_PREFIX)
                    and BENCH_FAIL.get("enabled"))
        if src0 in known_sources or bench_ok:
            kept_known.append(rec)
        else:
            unknown_held.append(rec)
            unknown_by_src[src0] = unknown_by_src.get(src0, 0) + len(rec.get("turns") or [])
    if unknown_held:
        log(f"unknown source: {len(unknown_held)} records held, never admitted (turns by source {unknown_by_src}); "
            f"list the source in [source.*] to admit it")
        NOTES_GLOBAL["unknown_source_turns"] = sum(unknown_by_src.values())
        NOTES_GLOBAL["unknown_sources"] = unknown_by_src  # type: ignore[assignment]
    candidates = kept_known
    candidates = drop_excluded_routed(candidates, routed, drops)
    if len(candidates) != n_before:
        log(f"routed groups: dropped {n_before - len(candidates)} carryover records "
            f"from excluded sources")
    stamped = stamp_routed_groups(candidates, routed)
    extra_retire: dict[str, set[str]] = {}   # from_group -> retired ids
    # Reclaimed turns: admitted above under a higher-precedence king group;
    # retire their old rows (only those actually kept in a candidate record).
    kept_now = {f"{r['traj_id']}:{m['turn_idx']}" for r in candidates
                if r.get("fold_group") in KING_GROUPS for m in r["turns"]}
    for old_ns, ids in reclaimed.items():
        ok = {t for t in ids if t in kept_now}
        extra_retire.setdefault(old_ns, set()).update(ok)
        log(f"{old_ns}: {len(ok)} published rows reclaimed by higher-precedence king groups")
    common = king.get("common") or load_king_common()
    if common.get("retire_excluded_published") and king:
        ids = king_fail_source_retire(pub, live, king["exclude_sources"])
        extra_retire.setdefault("king_fail", set()).update(ids)
        log(f"king_fail: retiring {len(ids)} published rows from excluded sources "
            f"{sorted(king['exclude_sources'])}")
    if common.get("retire_later_onsets"):
        ids = later_onset_retire(pub, live)
        extra_retire.setdefault(KING_LOOP_GROUP, set()).update(ids)
        log(f"{KING_LOOP_GROUP}: retiring {len(ids)} published later-onset rows (first onset per rollout kept)")
    for g, plan in readmits.items():
        readmitted = {f"{r['traj_id']}:{m['turn_idx']}" for r in candidates
                      if r.get("fold_group") == g for m in r["turns"]}
        n_missing = 0
        for src_g, ids in plan.items():
            kept = [t for t in ids if t in readmitted]
            n_missing += len(ids) - len(kept)
            extra_retire.setdefault(src_g, set()).update(kept)
        if n_missing:
            log(f"{g}: {n_missing} retire candidates were not readmitted in this run "
                f"-> they stay where they are")
        log(f"{g}: retiring {sum(len([t for t in ids if t in readmitted]) for ids in plan.values())} "
            f"index rows, readmitted as {g}")
    retire_surviving: dict[str, set[str]] = {}
    for src_g, ids in extra_retire.items():
        if ids:
            retire_surviving[src_g] = strata_after_retire(pub, live, src_g, ids)
            log(f"{src_g}: {len(retire_surviving[src_g])} strata survive the retirement")
    pivot_retire = sorted(set().union(*extra_retire.values())) if extra_retire else []
    budget_cfg = load_strata_budget()
    STRATA_BUDGET.clear(); STRATA_BUDGET.update(budget_cfg)
    SRC2GRP_GLOBAL.clear(); SRC2GRP_GLOBAL.update(src2grp)
    global MIX_GROUPS_GLOBAL
    MIX_GROUPS_GLOBAL = frozenset(mix)
    static_mix = dict(mix)
    curriculum = load_curriculum()
    if curriculum["mode"] != "off":
        log(curriculum_line(curriculum, static_mix))
    if curriculum["mode"] == "apply" and not curriculum.get("error"):
        # The published vector becomes the group targets; `m` the sub-strata
        # count (changes the budget signature -> re-key + --allow-shift).
        mix = {g: float(v) for g, v in curriculum["groups"].items() if float(v) > 0}
        # Bootstrap (2026-09-23): a group the static [mix] targets but the
        # curriculum vector does not know yet (absent, or 0 because no
        # verdict has ever carried it) keeps its static share, the known
        # groups scale down to make room. Without this `group_of` mapped
        # every record of such a group to DEFAULT_GROUP (coding, at its cap)
        # and the group could never enter a slice -- science / sci_code /
        # agentic_ops / long_context sat at 0 % through epochs 64-73 while
        # the fold "accepted" their turns. The curriculum can only measure a
        # group once it is in slices, so the static target is the prior.
        boot = {g: float(v) for g, v in static_mix.items() if float(v) > 0 and mix.get(g, 0.0) <= 0}
        if boot:
            room = max(0.0, 1.0 - sum(boot.values()))
            tot = sum(mix.values()) or 1.0
            mix = {g: v * room / tot for g, v in mix.items()}
            mix.update(boot)
            log(f"curriculum apply: bootstrapped {boot} from the static [mix] (no curriculum signal yet); "
                f"known groups scaled by {room:.3f}")
        # Published floors ([curriculum].<name>_floor / _groups: stop_state,
        # notool, chat, ...) hold under any applied vector -- belt-and-braces
        # next to the rule's own bonuses; the excess comes from the others.
        mix = apply_floors(mix, load_floors())
        if curriculum["m"] and budget_cfg:
            # The rule's m only RAISES a group's sub-strata k: the static k
            # (phase 9/10, the floors' supply lever) is the lower bound, and
            # bucketed groups keep their buckets (m is moot there).
            raised = {g: k for g, k in curriculum["m"].items()
                      if k > budget_cfg["sub_strata"].get(g, 1) and g not in budget_cfg["buckets"]}
            if raised:
                budget_cfg["sub_strata"].update(raised)
                budget_cfg["signature"] = json.dumps({**json.loads(budget_cfg["signature"]), "m": raised}, sort_keys=True)
                STRATA_BUDGET.clear(); STRATA_BUDGET.update(budget_cfg)
                log(f"curriculum apply: sub-strata raised by the rule {raised}")
    budget_migrated = False
    rename_only = False
    if budget_cfg and state.get("strata_budget_signature") != budget_cfg["signature"]:
        # A naming-version change alone (same [strata_budget] raw table) re-keys
        # the index without moving any share: no guard, no announce note.
        try:
            old_sig = json.loads(state.get("strata_budget_signature") or "null")
            new_sig = json.loads(budget_cfg["signature"])
            old_raw = old_sig.get("raw", old_sig) if isinstance(old_sig, dict) else None
            rename_only = (old_raw is not None and old_raw == new_sig.get("raw")
                           and (old_sig.get("m") if isinstance(old_sig, dict) else None) == new_sig.get("m"))
        except (ValueError, TypeError):
            rename_only = False
        # First fold under this budget: re-key the mix state from the live
        # index (the deliberate composition shift; --allow-shift required).
        live_rows = live_rows_for_budget(pub, live)
        if live_rows:
            raw_groups: dict[str, set[str]] = {}
            new_groups: dict[str, set[str]] = {}
            for tid, s0, src in live_rows:
                g = group_from_row(str(s0), str(src), src2grp)
                raw_groups.setdefault(g, set()).add(str(s0))
                new_groups.setdefault(g, set()).add(budget_stratum(g, str(s0), tid, budget_cfg))
            tr = sum(len(v) for v in raw_groups.values()) or 1
            tn = sum(len(v) for v in new_groups.values()) or 1
            raw_rec = simulate_recurrence([(tid, s0) for tid, s0, _ in live_rows])
            log(f"strata budget: per-duel recurrence BEFORE the budget {raw_rec}")
            state["recurrence_before_budget"] = raw_rec
            log("strata budget: live index re-keyed -- " + "; ".join(
                f"{g} {len(raw_groups.get(g, ()))}->{len(new_groups.get(g, ()))} "
                f"({100 * len(raw_groups.get(g, ())) / tr:.1f}% -> {100 * len(new_groups.get(g, ())) / tn:.1f}%)"
                for g in sorted(new_groups, key=lambda k: -len(new_groups[k]))))
            state["group_strata"] = {g: sorted(v) for g, v in new_groups.items()}
            state["group_strata_raw_before_budget"] = {g: len(v) for g, v in raw_groups.items()}
            budget_migrated = not rename_only
            if rename_only:
                log("strata budget: naming-version change only (shares unchanged); re-keyed without a guard")
            if not args.allow_shift and not rename_only:
                msg = "strata budget re-keys the live index (deliberate composition shift); rerun with --allow-shift"
                if args.no_publish:
                    log(f"GUARD (dry run): {msg}")
                else:
                    fatal(msg)
    bands = load_band_filters()
    band_report: dict = {}
    band_held: list[dict] = []
    if bands:
        band_stats = task_attempts(pub, traces_manifest, frozenset(bands))
        backfill_cfg = load_band_backfill()
        coverage_seat = band_coverage(band_stats, bands)
        backfill_rep = {}
        if backfill_cfg.get("key"):
            backfill_rep = merge_band_backfill(pub, band_stats, backfill_cfg,
                                               current_digest=current_king_digest())
        coverage_merged = band_coverage(band_stats, bands)
        candidates, band_held, band_tally = band_filter_records(candidates, bands, band_stats, drops)
        band_retire, band_pub = band_published_retire(pub, live, bands, band_stats, src2grp)
        for g, ids in band_retire.items():
            extra_retire.setdefault(g, set()).update(ids)
            retire_surviving[g] = strata_after_retire(pub, live, g, extra_retire[g])
        if band_retire:
            pivot_retire = sorted(set().union(*extra_retire.values()))
        per_src = band_report_per_source(bands, band_stats, band_tally, band_pub)
        rules = {}
        for src, cfg in bands.items():
            rules.setdefault(json.dumps(cfg, sort_keys=True), []).append(src)
        band_report = {"tally": band_tally, "retired": {g: len(v) for g, v in band_retire.items()},
                       "retired_by_source": {s0: v.get("retired", 0) for s0, v in band_pub.items() if v.get("retired")},
                       "per_source": per_src,
                       "rules": [{"sources": sorted(v), **json.loads(k)} for k, v in rules.items()],
                       "held_turns": sum(len(r.get("turns") or []) for r in band_held),
                       "backfill": backfill_rep or None,
                       "coverage": {src: {"teacher_tasks": m["teacher_tasks"],
                                          "king_covered_seat": coverage_seat.get(src, {}).get("king_covered"),
                                          "king_covered": m["king_covered"],
                                          "king_solved_share_seat": coverage_seat.get(src, {}).get("king_solved_share"),
                                          "king_solved_share": m["king_solved_share"]}
                                    for src, m in coverage_merged.items()}}
        if backfill_rep and not backfill_rep.get("error"):
            moved = sorted(((src, c["king_covered_seat"], c["king_covered"], c["king_solved_share_seat"], c["king_solved_share"])
                            for src, c in band_report["coverage"].items()
                            if c["king_covered_seat"] is not None and c["king_covered"] != c["king_covered_seat"]),
                           key=lambda x: -(x[2] - x[1]))
            log("band filter: king coverage seat -> merged (king_solved share seat -> merged): " + ", ".join(
                f"{src} {a:.2f}->{b:.2f} ({c:.2f}->{d:.2f})" for src, a, b, c, d in moved[:25]))
        log(f"band filter: retired by group {band_report['retired']}; held {band_report['held_turns']} turns; "
            f"by source: " + ", ".join(
                f"{s0} kept {v['published'].get('kept', 0)}/retired {v['published'].get('retired', 0)}"
                for s0, v in sorted(per_src.items(), key=lambda kv: -kv[1]['published'].get('retired', 0))
                if v["published"]))
        log(f"band filter: {json.dumps(band_report, sort_keys=True)}")
    gate = load_admission_gate()
    gate_report: dict = {}
    gate_held: list[dict] = []
    gate_flips: list[str] = []
    if gate:
        # enforced = toml apply_groups + earlier auto-promotions (sticky)
        gate["apply_groups"] = frozenset(gate["apply_groups"]) | frozenset(state.get("gate_enforced") or [])
        state_rec = load_state_recovery(gate)
        task_solved, task_seen = (teacher_solved_tasks(pub, traces_manifest, gate["teacher_prefixes"])
                                  if gate["task_signal"] else (set(), set()))
        dead = dead_reference_turns(gate)
        log(f"admission gate: {len(state_rec)} state-level recovery rows "
            f"({sum(state_rec.values())} recover), {len(task_solved)} teacher-solved tasks of {len(task_seen)} seen, "
            f"{len(dead)} dead-reference turn ids; enforced on {sorted(gate['apply_groups']) or 'none (shadow)'}")
        candidates, gate_held, cand_tally = admission_gate(
            candidates, gate, src2grp, mix, state_rec, task_solved, task_seen, dead, drops)
        pub_retire, pub_tally = gate_published(pub, live, gate, state_rec, task_solved, task_seen, dead, src2grp)
        for g, ids in pub_retire.items():
            extra_retire.setdefault(g, set()).update(ids)
            retire_surviving[g] = strata_after_retire(pub, live, g, extra_retire[g])
        if pub_retire:
            pivot_retire = sorted(set().union(*extra_retire.values()))
        # Auto-promote: a shadow group whose post-gate strata (published rows
        # minus what the recovery rule would retire, plus this fold's admitted
        # candidates) reach its quota flips to enforced -- for this fold too.
        if gate["auto_promote"]:
            live_rows_g = live_rows_for_budget(pub, live)
            strata_by_g: dict[str, set[str]] = {}
            would = {g: set(v) for g, v in GATE_SHADOW_RETIRE.items()}
            dead_rows = {g: set(v) for g, v in pub_retire.items()}
            for tid, s0, src in live_rows_g:
                g0 = group_from_row(s0, src, src2grp)
                if g0 in gate["groups"] and tid not in would.get(g0, set()) and tid not in dead_rows.get(g0, set()):
                    strata_by_g.setdefault(g0, set()).add(budget_stratum(g0, s0, tid))
            for rec in candidates:
                g0 = group_of(rec, src2grp, mix)
                if g0 in gate["groups"]:
                    for m in rec["turns"]:
                        if gate_turn(rec, m, g0, gate, state_rec, task_solved, task_seen, dead) == "admit":
                            base = stratum_key({"stratum": m.get("stratum") or rec.get("stratum"), "traj_id": rec["traj_id"]})
                            strata_by_g.setdefault(g0, set()).add(budget_stratum(g0, base, f"{rec['traj_id']}:{m['turn_idx']}"))
            tot_now = sum(len(v) for v in (state.get("group_strata") or {}).values()) or 1
            gate_projection: dict[str, dict] = {}
            for g0 in gate["groups"]:
                if g0 in gate["recovery_exempt"]:
                    continue
                post = len(strata_by_g.get(g0, ()))
                quota = mix.get(g0, 0.0) * tot_now
                admits_fold = cand_tally.get(g0, {}).get("admit", 0)
                gate_projection[g0] = {"post_gate_strata": post, "quota_strata": round(quota),
                                       "admits_this_fold": admits_fold,
                                       "folds_to_quota": (0 if post >= quota else
                                                          (round((quota - post) / admits_fold, 1) if admits_fold else None))}
                if g0 not in gate["apply_groups"] and post >= quota and quota > 0:
                    gate_flips.append(g0)
            if gate_flips:
                gate["apply_groups"] = gate["apply_groups"] | frozenset(gate_flips)
                log(f"admission gate: auto-promoted {gate_flips} (post-gate strata >= quota); re-running the gate enforced")
                # rows the shadow rule would have retired now retire; candidates re-gated
                for g0 in gate_flips:
                    ids = GATE_SHADOW_RETIRE.get(g0) or []
                    if ids:
                        pub_retire.setdefault(g0, []).extend(ids)
                candidates, held2, cand_tally = admission_gate(
                    candidates, gate, src2grp, mix, state_rec, task_solved, task_seen, dead, drops)
                gate_held.extend(held2)
                for g0, ids in pub_retire.items():
                    extra_retire.setdefault(g0, set()).update(ids)
                    retire_surviving[g0] = strata_after_retire(pub, live, g0, extra_retire[g0])
                pivot_retire = sorted(set().union(*extra_retire.values()))
        else:
            gate_projection = {}
        gate_state = {g0: ("exempt" if g0 in gate["recovery_exempt"] else
                           "enforced" if g0 in gate["apply_groups"] else "shadow")
                      for g0 in gate["groups"]}
        gate_reason = {g0: ("dead-reference drop only: the coached teacher is the recovery signal" if st == "exempt"
                            else "recovery rule enforced (toml or auto-promoted at quota)" if st == "enforced"
                            else "recovery rule measured only; auto-promotes when post-gate strata >= quota")
                       for g0, st in gate_state.items()}
        gate_report = {"candidates": cand_tally, "published": pub_tally,
                       "apply_groups": sorted(gate["apply_groups"]),
                       "gate_state": gate_state, "gate_reason": gate_reason,
                       "flips_this_fold": gate_flips, "projection": gate_projection,
                       "retired": {g: len(v) for g, v in pub_retire.items()},
                       "signals": {"state_rows": len(state_rec), "teacher_solved_tasks": len(task_solved),
                                   "teacher_seen_tasks": len(task_seen), "dead_ref_turns": len(dead)}}
        for g in sorted(set(cand_tally) | set(pub_tally)):
            log(f"admission gate [{g}]{' ENFORCED' if g in gate['apply_groups'] else ' shadow'}: "
                f"published {pub_tally.get(g, {})}; candidates {cand_tally.get(g, {})}")
    probe = load_teacher_probe()
    probe_held: list[dict] = []
    if probe:
        candidates, probe_held, pending = probe_gate(candidates, probe, drops, src2grp, mix)
        probe["pending_path"].parent.mkdir(parents=True, exist_ok=True)
        probe["pending_path"].write_text("".join(json.dumps(p) + "\n" for p in pending))
        log(f"teacher probe: {len(probe['rows'])} rows; held {len(probe_held)} records / "
            f"{len(pending)} unprobed turns -> {probe['pending_path']}; "
            f"dropped { {k: v for k, v in drops.items() if k.startswith('probe_')} }")
        if probe["retire_failed_published"]:
            fp = failed_published(pub, live, probe)
            for src_g, ids in fp.items():
                extra_retire.setdefault(src_g, set()).update(ids)
            if fp:
                log(f"teacher probe: retiring published rows that fail the probe "
                    f"{ {k: len(v) for k, v in fp.items()} }")
                for src_g in fp:
                    retire_surviving[src_g] = strata_after_retire(
                        pub, live, src_g, extra_retire[src_g])
                    log(f"{src_g}: {len(retire_surviving[src_g])} strata survive the retirement")
                pivot_retire = sorted(set().union(*extra_retire.values()))
    if king_coached:
        cr_ids, cr_states = coached_retire_ids(king_coached, pub, live)
        if cr_ids:
            extra_retire.setdefault(KING_COACHED_GROUP, set()).update(cr_ids)
            retire_surviving[KING_COACHED_GROUP] = strata_after_retire(
                pub, live, KING_COACHED_GROUP, extra_retire[KING_COACHED_GROUP])
            pivot_retire = sorted(set().union(*extra_retire.values()))
            log(f"king_coached: retiring {len(cr_ids)} published rows of {len(cr_states)} demoted states; "
                f"{len(retire_surviving[KING_COACHED_GROUP])} strata survive")
    if not state.get("mix_seeded"):
        state["mix_seeded"] = True
        # Mix state is the set of slice strata each group / language bucket
        # already holds in D (see cap_fill). Legacy import seeds it from the
        # imported records; --no-legacy starts empty (rehearsal 2026-09-03:
        # with the 60k legacy turns counted, coding 67%, the fold admitted
        # zero new coding/terminal rollouts).
        state["group_strata"] = {}
        state["lang_strata"] = {}
        for rec in legacy:
            g = group_of(rec, src2grp, mix)
            state["group_strata"].setdefault(g, [])
            state["group_strata"][g] = sorted(set(state["group_strata"][g])
                                              | record_strata(rec))
            if g == "coding":
                b = lang_bucket(rec)
                state["lang_strata"][b] = sorted(set(state["lang_strata"].get(b, []))
                                                 | record_strata(rec))
        log("mix state seeded: strata per group "
            f"{ {g: len(v) for g, v in state['group_strata'].items()} }")
    # Language cap first, inside coding, so the group stage sizes terminal /
    # math / tool_use against the coding strata that actually enter D.
    lang_added: dict[str, set[str]] = {}
    lang_deferred: list[dict] = []
    if lang_mix:
        coding = [r for r in candidates if group_of(r, src2grp, mix) == "coding"]
        other = [r for r in candidates if group_of(r, src2grp, mix) != "coding"]
        have_langs = {b: set(v) for b, v in (state.get("lang_strata") or {}).items()}
        gs = state.get("group_strata") or {}
        if coding_below_target(gs, mix):
            budget = catchup_budget(gs, "coding")
            chosen, lang_deferred, lang_added = soft_lang_fill(
                coding, have_langs, ceiling=LANG_SOFT_CEILING, budget=budget)
            mode = (f"soft (coding below target: budget {budget} new strata, "
                    f"per-language ceiling {LANG_SOFT_CEILING:.0%})")
        else:
            chosen, lang_deferred, lang_added = cap_fill(
                coding, lang_bucket, have_langs, lang_mix)
            mode = "hard (coding at/above target)"
        candidates = other + chosen
        log(f"lang mix [{mode}]: kept {len(chosen)}/{len(coding)} coding rollouts "
            f"(+{ {b: len(v) for b, v in lang_added.items()} } strata), "
            f"deferred {len(lang_deferred)}")
    have_groups = {g: set(v) for g, v in (state.get("group_strata") or {}).items()}
    if retire_ids:
        # The retired math strata are gone from D; the cap sees what survives.
        have_groups[src2grp.get(math_cfg["source"], DEFAULT_GROUP)] = set(math_surviving)
    for src_g, surv in retire_surviving.items():
        have_groups[src_g] = set(surv)
    budgets = {g: catchup_budget(state.get("group_strata") or {}, g)
               for g, v in mix.items() if v > 0}
    selected, deferred, group_added = cap_fill(
        candidates, lambda r: group_of(r, src2grp, mix), have_groups, mix,
        anchor_min_target=ANCHOR_MIN_TARGET, max_new=budgets)
    deferred += lang_deferred + probe_held + gate_held + band_held + unknown_held
    log(f"mix: selected {len(selected)} rollouts (+{ {g: len(v) for g, v in group_added.items()} } "
        f"strata), deferred {len(deferred)}")
    # Language strata credited only for coding rollouts that made it through
    # the group stage too.
    if lang_mix and STRATA_BUDGET.get("buckets", {}).get("coding"):
        lang_mix = {}     # coding strata are fixed buckets; language mix by strata is moot
    if lang_mix:
        kept_ids = {id(r) for r in selected}
        lang_added = {}
        have_langs2 = {b: set(v) for b, v in (state.get("lang_strata") or {}).items()}
        for r in chosen:
            if id(r) in kept_ids:
                b = lang_bucket(r)
                new = record_strata(r) - have_langs2.get(b, set())
                lang_added.setdefault(b, set()).update(new)
    group_turns: dict[str, int] = {}
    for r in selected:
        g = group_of(r, src2grp, mix)
        group_turns[g] = group_turns.get(g, 0) + len(r["turns"])
    log(f"mix: turns by group {group_turns}")

    before = {g: len(v) for g, v in (state.get("group_strata") or {}).items()}
    after = dict(before)
    if retire_ids:
        mg = src2grp.get(math_cfg["source"], DEFAULT_GROUP)
        after[mg] = len(math_surviving)
    for src_g, surv in retire_surviving.items():
        after[src_g] = len(surv)
    for g, keys in group_added.items():
        after[g] = after.get(g, 0) + len(keys)
    rows = composition_table(before, after)
    log("projected slice composition (share of strata): " + "; ".join(
        f"{g} {b}->{a} ({100 * sb:.1f}% -> {100 * sa:.1f}%, {100 * d:+.1f})"
        for g, b, a, sb, sa, d in rows))
    shifted = [(g, d) for g, _, _, _, _, d in rows if abs(d) > MAX_SHARE_SHIFT]
    if shifted and not args.allow_shift:
        msg = (f"slice share of {shifted} would move by more than "
               f"{MAX_SHARE_SHIFT:.0%} in one epoch; rerun with --allow-shift to accept")
        if args.no_publish:
            log(f"GUARD (dry run): {msg}")
        else:
            fatal(msg)
    # 24 h cumulative cap: compare against the oldest snapshot inside the window
    # (or the newest before it) so six 5-point steps cannot add up to thirty.
    now_ts = datetime.now(timezone.utc).timestamp()
    hist = [h for h in (state.get("share_history") or []) if now_ts - float(h["at"]) <= DAILY_WINDOW_S]
    older = [h for h in (state.get("share_history") or []) if now_ts - float(h["at"]) > DAILY_WINDOW_S]
    base = (hist[0] if hist else (older[-1] if older else None))
    if base:
        ta = sum(after.values()) or 1
        daily = [(g, after.get(g, 0) / ta - float(base["shares"].get(g, 0.0)))
                 for g in set(after) | set(base["shares"])]
        over = [(g, round(d, 3)) for g, d in daily if abs(d) > DAILY_SHIFT_CAP]
        if over and not args.allow_shift:
            msg = (f"24 h cumulative slice-share move of {over} exceeds {DAILY_SHIFT_CAP:.0%} "
                   f"(baseline epoch {base.get('epoch')} at {base.get('iso')}); rerun with --allow-shift")
            if args.no_publish:
                log(f"GUARD (dry run): {msg}")
            else:
                fatal(msg)
        log(f"24 h shift cap: max |move| {max((abs(d) for _, d in daily), default=0):.3f} vs "
            f"baseline epoch {base.get('epoch')} ({len(hist)} snapshots in window)")

    recurrence = None
    if STRATA_BUDGET:
        live_rows = live_rows_for_budget(pub, live)
        retired_now = set(retire_ids) | set(pivot_retire)
        sim_rows: list[tuple[str, str]] = []
        sim_rows_src: list[tuple[str, str, str]] = []
        for tid, s0, src in live_rows:
            if tid in retired_now:
                continue
            key = budget_stratum(group_from_row(s0, src, src2grp), s0, tid)
            sim_rows.append((tid, key))
            sim_rows_src.append((tid, key, src))
        for r in selected:
            g = group_of(r, src2grp, mix)
            for m in r["turns"]:
                tid = f"{r['traj_id']}:{m['turn_idx']}"
                base = stratum_key({"stratum": m.get("stratum") or r.get("stratum"), "traj_id": r["traj_id"]})
                key = budget_stratum(g, base, tid)
                sim_rows.append((tid, key))
                sim_rows_src.append((tid, key, str(r.get("source") or "")))
        recurrence = simulate_recurrence(sim_rows)
        per_duel = {g: round(1300 * a / (sum(after.values()) or 1), 1) for g, a in after.items()}
        log(f"strata budget: simulated per-duel recurrence {recurrence}; projected turns per duel {per_duel}")
        turns_by_group: dict[str, int] = {}
        for tid, s0, src in live_rows:
            if tid not in retired_now:
                g = group_from_row(s0, src, src2grp)
                turns_by_group[g] = turns_by_group.get(g, 0) + 1
        for r in selected:
            g = group_of(r, src2grp, mix)
            turns_by_group[g] = turns_by_group.get(g, 0) + len(r["turns"])
        floors = load_floors()
        fstat = floor_status(after, floors)
        yrep = yield_report(after, mix, group_turns, src2grp)
        log("floors: " + "; ".join(f"{k} {100 * v['share']:.1f}% {'>=' if v['ok'] else '<'} {100 * v['floor']:.1f}%"
                                   for k, v in fstat.items()))
        log("yield (accepted turns this fold / strata over target): " + ", ".join(
            f"{g} {v['accepted_turns_this_fold']}/{v['strata_over_target']}" for g, v in yrep["groups"].items()
            if v["accepted_turns_this_fold"] or (v["strata_over_target"] or 0) < 1))
        # trained_on: a bench_* suite flips when the fold PUBLISHES its rows
        # (records in `selected`, i.e. past cap_fill), not when derive_chunk
        # accepts them -- a suite whose records the mix stage deferred is not
        # trained on yet. Once flipped, the first epoch stays.
        epoch_next = (int(live["corpus_epoch"]) if live else 0) + 1
        bench_selected: dict[str, int] = {}
        for r in selected:
            if r.get("fold_group") == BENCH_FAIL.get("group") and str(r.get("source") or "").startswith(BENCH_SOURCE_PREFIX):
                bench_selected[str(r["source"])] = bench_selected.get(str(r["source"]), 0) + len(r["turns"])
        if bench_selected:
            log(f"bench_fail: published this fold by suite (turns) {dict(sorted(bench_selected.items()))}")
        if BENCH_FAIL.get("enabled"):
            tr = dict(state.get("trained_on") or {})
            for src0, n0 in bench_selected.items():
                if n0 > 0:
                    suite = str(src0).removeprefix(BENCH_SOURCE_PREFIX)
                    tr.setdefault(suite, {"since_epoch": epoch_next, "since": datetime.now(timezone.utc).date().isoformat(),
                                          "group": BENCH_FAIL.get("group"), "mode": BENCH_FAIL.get("mode")})
            if tr != (state.get("trained_on") or {}):
                log(f"trained_on: {sorted(set(tr) - set(state.get('trained_on') or {}))} flip this fold")
            state["trained_on"] = tr
        write_fold_stats((int(live["corpus_epoch"]) if live else 0) + 1, after, turns_by_group, recurrence,
                         curriculum, mix, sum(turns_by_group.values()), None if args.no_publish else publisher,
                         sim_rows=sim_rows_src, src2grp=src2grp,
                         extra={"floors": fstat,
                                "yield": {**yrep, "gate_state": (gate_report or {}).get("gate_state"),
                                          "gate_reason": (gate_report or {}).get("gate_reason")},
                                "admission_gate": gate_report or None,
                                "band_filter": band_report or None,
                                "trained_on": state.get("trained_on") or {}})

    n_new = sum(len(r["turns"]) for r in selected)
    stale = False
    if unfolded:
        newest = max(datetime.fromisoformat(c["created_at"]) for c in unfolded)
        stale = (datetime.now(timezone.utc) - newest).total_seconds() >= STALE_AFTER_S
    if n_new < MIN_NEW_TURNS and not stale and not args.force and not args.init \
            and not retire_ids and not pivot_retire and not budget_migrated:
        log(f"only {n_new} mix-eligible new turns (< {MIN_NEW_TURNS}); skipping")
        return
    if not selected and not legacy:
        log("nothing to publish")
        return

    epoch = (int(live["corpus_epoch"]) if live else int(legacy_manifest["corpus_epoch"])) + 1
    records = legacy + selected
    turn_ids = [f"{r['traj_id']}:{m['turn_idx']}" for r in records for m in r["turns"]]
    if len(turn_ids) != len(set(turn_ids)):
        fatal(f"{len(turn_ids) - len(set(turn_ids))} duplicate turn id(s) in the "
              "pack (deferred copy + re-derived record?) -- operator check")
    by_dialect: dict[str, int] = {}
    for r in selected:
        for m in r["turns"]:
            by_dialect[m["action_kind"]] = by_dialect.get(m["action_kind"], 0) + 1
    pack = pack_pending(records, epoch)
    log(f"epoch {epoch}: {pack.n_trajectories} records / {pack.n_turns} turns in "
        f"{len(pack.chunk_paths)} chunks; new by dialect {by_dialect}")
    if args.no_publish:
        log(f"--no-publish: pack left in {pack.chunk_paths[0].parent}")
        return

    tmp = DEFERRED_PATH.with_suffix(".jsonl.tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in deferred))
    tmp.replace(DEFERRED_PATH)
    state["pending"] = {
        "epoch": epoch, "pack_dir": str(pack.chunk_paths[0].parent),
        "n_turns": n_new, "group_turns": group_turns,
        "group_strata_added": {g: sorted(v) for g, v in group_added.items()},
        "recurrence": recurrence,
        "coached_folded": sorted(coached_folded),
        "src_override": dict(SRC_OVERRIDE),
        "n_strata_after": int(sum(after.values())),
        "n_backfill_excluded": int(drops.get("backfill_excluded", 0)),
        "floors": fstat if STRATA_BUDGET else None,
        "yield_groups": yrep["groups"] if STRATA_BUDGET else None,
        "yield_extra": {"divergence_sublabels": yrep.get("divergence_sublabels"),
                        "interactive_prose_turns": yrep.get("interactive_prose_turns"),
                        "by_king": yrep.get("by_king"),
                        "upstream_fetch_turns": yrep.get("upstream_fetch_turns", 0),
                        "unknown_source_turns": yrep.get("unknown_source_turns", 0),
                        "unknown_sources": yrep.get("unknown_sources") or {}} if STRATA_BUDGET else None,
        "yield_sources": yrep["sources"] if STRATA_BUDGET else None,
        "admission_gate": gate_report or None,
        "band_filter": band_report or None,
        # Benchmark columns whose failed trials enter D directly (bench_fail):
        # {suite: {since_epoch, since, group, mode}} -- persists from the first
        # admitting fold on; the kingboard renders "trained on since <epoch>".
        "trained_on": state.get("trained_on") or {},
        "bench_traces_manifest_sha256": bench_sha,
        "gate_enforced": sorted(gate["apply_groups"]) if gate else None,
        "budget_signature": budget_cfg.get("signature") if budget_cfg else None,
        "budget_migrated": budget_migrated,
        "strata_raw_before_budget": state.get("group_strata_raw_before_budget") if budget_migrated else None,
        "recurrence_before_budget": state.get("recurrence_before_budget") if budget_migrated else None,
        "curriculum_line": curriculum_line(curriculum, static_mix) if curriculum["mode"] != "off" else None,
        "curriculum_block": curriculum.get("manifest_block") if curriculum["mode"] != "off" and not curriculum.get("error") else None,
        "lang_strata_added": {b: sorted(v) for b, v in lang_added.items()},
        "by_dialect": by_dialect, "allowed_kinds": list(allowed),
        "folded_chunks": [c["key"] for c in unfolded], "init": bool(args.init),
        "retire_turn_ids": retire_ids + pivot_retire,
        "group_strata_after_retire": {
            **({src2grp.get(math_cfg["source"], DEFAULT_GROUP): sorted(math_surviving)}
               if retire_ids else {}),
            **{src_g: sorted(surv) for src_g, surv in retire_surviving.items()},
        },
    }
    save_state(state)
    finalize(state, *publish_pending(state, publisher, traces_sha, legacy_sha))
    if not args.no_announce:
        announce(state, public_base)
    log(f"cycle complete: epoch {epoch}, +{n_new} turns ({len(deferred)} rollouts deferred)")


if __name__ == "__main__":
    main()
