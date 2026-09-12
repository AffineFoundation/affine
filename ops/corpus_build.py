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
  6. announce on the SN120 Discord channel; a failed post is retried next
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
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import tomllib
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
from affine.corpus.completion import final_completion  # noqa: E402
from affine.corpus.loops import IN_LOOP, ONSET, label_loops  # noqa: E402
from affine.corpus.materialize import stratum_key  # noqa: E402
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
DISCORD_GUILD_ID = "799672011265015819"
DISCORD_CHANNEL_ID = "1381987595881414656"

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
    return {"group": "king_fail",
            "strata_buckets": int(cfg.get("strata_buckets", 0) or 0),
            "policy_prefix": str(cfg.get("policy_prefix") or "king_"),
            "exclude_sources": frozenset(str(x) for x in
                                         (cfg.get("exclude_sources") or []))}


# Turn-routed fold groups (2026-09-11, data events): derive_chunk splits
# single turns of a rollout off into a second record of the same rollout
# with `fold_group` set and a bucketed stratum `<group>:NNNN`. Precedence
# when one turn qualifies for several: king_pivot > king_loop_onset >
# completion (a turn can only be one of them by construction: the first two
# need a FAILED king rollout, completion a SOLVED one).
KING_LOOP_GROUP = "king_loop_onset"
KING_PIVOT_GROUP = "king_pivot"
COMPLETION_GROUP = "completion"
ROUTED_GROUPS = (KING_PIVOT_GROUP, KING_LOOP_GROUP, COMPLETION_GROUP)


def _group_cfg(group: str) -> dict:
    """Common keys of a `[<group>]` block in sources.toml; {} when absent."""
    raw = tomllib.loads(SOURCES_TOML.read_text())
    cfg = raw.get(group) or {}
    if not cfg:
        return {}
    return {"group": group,
            "strata_buckets": int(cfg.get("strata_buckets", 0) or 0),
            "policy_prefix": str(cfg.get("policy_prefix") or ""),
            "exclude_sources": frozenset(str(s) for s in
                                         (cfg.get("exclude_sources") or [])),
            "leak_exempt": bool(cfg.get("leak_exempt", False)),
            "raw": cfg}


def load_king_loop_onset() -> dict:
    """[king_loop_onset]: the first turn of every loop in the king seat's
    failed rollouts. The leak rule is always waived for this group."""
    cfg = _group_cfg(KING_LOOP_GROUP)
    if cfg:
        cfg["policy_prefix"] = cfg["policy_prefix"] or "king_"
        cfg["leak_exempt"] = True
    return cfg


def load_king_pivot() -> dict:
    """[king_pivot]: the turns an LLM judge marked as the decision point of
    a failed king rollout (ops/king-review, PR #8). Rows come from the
    per-king side-tables under `side_table_dir` (`<digest>.jsonl`, one JSON
    line per (rollout_id, turn_idx)); only `admit == true` rows at or above
    `min_confidence` and outside `exclude_categories` route. Every table in
    the directory is read: an earlier king's pivots stay valid states. The
    leak rule is waived as for king_loop_onset. `table` =
    {rollout_id: {turn_idx: row}}."""
    cfg = _group_cfg(KING_PIVOT_GROUP)
    if not cfg:
        return {}
    raw = cfg["raw"]
    cfg["policy_prefix"] = cfg["policy_prefix"] or "king_"
    cfg["leak_exempt"] = True
    excluded = {str(c) for c in (raw.get("exclude_categories") or [])}
    min_conf = float(raw.get("min_confidence", 0.7))
    side_dir = REPO / str(raw.get("side_table_dir") or "affine/state/king_pivots")
    table: dict[str, dict[int, dict]] = {}
    n_rows = n_files = 0
    for path in sorted(side_dir.glob("*.jsonl")) if side_dir.is_dir() else []:
        n_files += 1
        for line in path.read_text().split("\n"):
            if not line.strip():
                continue
            row = json.loads(line)
            n_rows += 1
            if not row.get("admit") or str(row.get("failure_category")) in excluded:
                continue
            if float(row.get("confidence") or 0) < min_conf:
                continue
            table.setdefault(str(row["rollout_id"]), {})[int(row["turn_idx"])] = row
    cfg.update(table=table, side_table_dir=str(side_dir), n_files=n_files,
               n_rows=n_rows)
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


def king_loop_candidate(env: dict, cfg: dict) -> bool:
    """A rollout the loop labeler runs on: played by a king policy, graded
    FAILED by its env (the same test as `route_king_fail`), from a source
    the group admits. tool_use sources are excluded by config: the teacher's
    next tool call is near-deterministic there, so centered R is ~0 and a
    loop prefix carries no signal (wvk-11 findings)."""
    return _policy_ok(env, cfg) and rollout_outcome(env["trace"]) == "failed"


def king_pivot_turns(env: dict, cfg: dict) -> dict[int, dict]:
    """Admitted pivot rows for this rollout, {turn_idx: row}; {} when the
    rollout has none or is not a failed king rollout."""
    if not _policy_ok(env, cfg):
        return {}
    rows = cfg["table"].get(str(env.get("rollout_id") or ""))
    if not rows or rollout_outcome(env["trace"]) != "failed":
        return {}
    return dict(rows)


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


def record_strata(rec: dict) -> set[str]:
    """Slice strata this record's turns land in (affine.corpus.materialize.
    stratum_key on the index row: explicit bucket for math / tool_use,
    repo|phase from traj_id otherwise)."""
    return {stratum_key({"stratum": m.get("stratum") or rec.get("stratum"),
                         "traj_id": rec.get("traj_id")}) for m in rec["turns"]}


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
             targets: dict[str, float], *, anchor_min_target: float = 0.0
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
    whole, as before."""
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
        for rec in pool:
            new = record_strata(rec) - strata
            # A rollout in strata the corpus already holds adds within-stratum
            # variety and moves no share; a rollout opening new strata must fit
            # under the cap.
            if new and len(strata) + len(new) > cap[k] + 1e-9:
                deferred.append(rec)
                continue
            selected.append(rec)
            strata |= new
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
                 notes: dict[str, int] | None = None) -> list[dict]:
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
    exempt)."""
    notes = notes if notes is not None else {}
    cfgs = {KING_LOOP_GROUP: king_loop, KING_PIVOT_GROUP: king_pivot,
            COMPLETION_GROUP: completion}
    out: list[dict] = []
    for env in iter_jsonl_gz(path):
        convs = None
        route: dict[int, str] = {}          # turn_idx -> group (final)
        extra: dict[int, dict] = {}         # turn_idx -> meta fields to stamp
        in_loop: set[int] = set()
        kind = (env.get("policy") or {}).get("action_kind") or "bash"
        want_loop = bool(king_loop) and king_loop_candidate(env, king_loop)
        pivots = king_pivot_turns(env, king_pivot) if king_pivot else {}
        want_completion = bool(completion) and completion_candidate(env, completion)
        if want_loop or want_completion:
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
        if want_loop and main_convs:
            n_on = 0
            for j, lab in enumerate(label_loops(main_convs, kind)):
                i = main[j]
                if lab.label == ONSET:
                    route[i] = KING_LOOP_GROUP
                    extra[i] = {"loop_onset_of": int(main[int(lab.repeats)])}
                    n_on += 1
                elif lab.label == IN_LOOP:
                    in_loop.add(i)
            _count(notes, "king_loop_labelled_rollouts")
            _count(notes, "king_loop_onset_labels", n_on)
            _count(notes, "king_loop_in_loop_labels", len(in_loop))
        if pivots:
            _count(notes, "king_pivot_rollouts")
            for i, row in pivots.items():
                if route.get(i) == KING_LOOP_GROUP:
                    _count(notes, "king_pivot_over_onset")
                route[i] = KING_PIVOT_GROUP
                in_loop.discard(i)
                extra[i] = {"pivot": {
                    "category": row.get("failure_category"),
                    "confidence": row.get("confidence"),
                    "judge": row.get("judge_model"),
                    "prompt_hash": row.get("prompt_hash")}}
        leak_exempt = frozenset(i for i, g in route.items() if cfgs[g]["leak_exempt"])
        try:
            rec = build_view_record(env, baker=baker,
                                    generated_at=env.get("stored_at"),
                                    convs=convs, leak_exempt=leak_exempt)
        except (ToolParityError, TraceShapeError) as e:
            _count(drops, type(e).__name__)
            continue
        if rec is None:
            if route and convs:
                _count_leaked(route, convs, kind, leak_exempt, notes)
            _count(drops, "no_scorable_turn")
            continue
        turns = view_turns(rec)
        present = {t["turn_idx"] for t in turns}
        if route and convs:
            _count_leaked({i: g for i, g in route.items() if i not in present},
                          convs, kind, leak_exempt, notes)
        rest = [t for t in turns if t["turn_idx"] not in route
                and t["turn_idx"] not in in_loop]
        routed = [t for t in turns if t["turn_idx"] in route]
        n_in_loop = len(turns) - len(routed) - len(rest)
        if n_in_loop:
            _count(drops, "king_in_loop", n_in_loop)
        kept, d = validate_turns(rest, panel=panel, allowed_kinds=allowed_kinds)
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
        metas = rec["turns"]
        rec["turns"] = [m for m in metas if m["turn_idx"] in keep_idx]
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
    return {"source": str(cfg.get("source") or "affine_math"),
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


def math_keep_set(pub: PublicCorpus, traces_manifest: dict, cfg: dict
                  ) -> tuple[set[str], dict]:
    """Problems (task sid) the proxy keeps, plus the survey."""
    per: dict[str, dict] = {}
    n_chunks = 0
    for c in traces_manifest["chunks"]:
        if not c["key"].rsplit("/", 1)[-1].startswith(f"{cfg['source']}-"):
            continue
        n_chunks += 1
        for env in iter_jsonl_gz(pub.cached(c["key"], c["sha256"], gz_sha=True)):
            if str(env.get("source") or "") != cfg["source"]:
                continue
            sid = str((env.get("task") or {}).get("sid") or "")
            pid = str((env.get("policy") or {}).get("id") or "")
            outcome = rollout_outcome(env["trace"])
            row = per.setdefault(sid, {"answers": set(), "teacher_failed": False,
                                       "king_failed": False, "n_teacher": 0, "n_king": 0})
            if pid.startswith(cfg["teacher_prefix"]):
                if outcome in ("solved", "failed"):
                    row["n_teacher"] += 1
                    ans = boxed_answer(env["trace"])
                    if ans is not None:
                        row["answers"].add(ans)
                    row["teacher_failed"] |= outcome == "failed"
            elif pid.startswith(cfg["king_prefix"]):
                if outcome in ("solved", "failed"):
                    row["n_king"] += 1
                    row["king_failed"] |= outcome == "failed"
    keep = {sid for sid, r in per.items()
            if len(r["answers"]) >= 2 or r["teacher_failed"] or r["king_failed"]}
    stats = {"chunks": n_chunks, "problems": len(per), "kept": len(keep),
             "multi_sample": sum(r["n_teacher"] >= 2 for r in per.values()),
             "disagree": sum(len(r["answers"]) >= 2 for r in per.values()),
             "teacher_failed": sum(r["teacher_failed"] for r in per.values()),
             "king_failed": sum(r["king_failed"] for r in per.values())}
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
        if src != cfg["source"] or not str(stratum).startswith(f"{group}:"):
            continue
        m = TRAJ_SHA8_RE.search(traj or "")
        if m and m.group(1) in keep_sha8:
            surviving.add(stratum)
        else:
            retire.append(tid)
            retired_strata.add(stratum)
    return retire, surviving, retired_strata - surviving


# -- composition guard ----------------------------------------------------------
MAX_SHARE_SHIFT = 0.05


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
    merged = pa.concat_tables([prev_table, pq.read_table(pack.index_path)])
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
    state["unannounced"] = {
        "epoch": int(pending["epoch"]), "n_added": int(pending["n_turns"]),
        "total": int(manifest["index"]["n_turns"]), "manifest_sha256": mhash,
        "by_dialect": pending.get("by_dialect") or {},
        "by_group": pending.get("group_turns") or {},
        "strata": {g: len(v) for g, v in state["group_strata"].items()},
        "init": bool(pending.get("init")),
        "n_retired": len(pending.get("retire_turn_ids") or []),
    }
    state["history"].append({
        "epoch": int(pending["epoch"]), "n_turns": int(pending["n_turns"]),
        "n_chunks": len(pending["folded_chunks"]),
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "manifest_sha256": mhash,
    })
    state["pending"] = None
    save_state(state)


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
        + (f"Retired from the index: {info['n_retired']:,} math turns of problems the "
           "teacher answers deterministically (chunks unchanged; see llms.txt).\n"
           if info.get("n_retired") else "")
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
    r = httpx.post(
        f"https://discord.com/api/v10/channels/{DISCORD_CHANNEL_ID}/messages",
        headers={"Authorization": f"Bot {env_value('DISCORD_BOT_TOKEN_ARBOS_BITTENSOR')}"},
        json={"content": content}, timeout=30)
    if r.status_code >= 300:
        log(f"announce failed (HTTP {r.status_code}); will retry next cycle")
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
    legacy_manifest, legacy_sha = pub.manifest("turns/manifest.json")

    if state["pending"]:
        if publisher is None:
            fatal("pending publish exists; rerun without --no-publish")
        log(f"resuming pending publish for epoch {state['pending']['epoch']}")
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
    routed = {KING_LOOP_GROUP: load_king_loop_onset(),
              KING_PIVOT_GROUP: load_king_pivot(),
              COMPLETION_GROUP: load_completion()}
    for g, cfg in routed.items():
        if cfg and mix.get(g, 0.0) <= 0:
            # Fail closed: group_of would file the routed records under coding.
            fatal(f"[{g}] is configured but the fold mix has no {g} share")
        shown = {k: v for k, v in (cfg or {}).items() if k not in ("raw", "table")}
        log(f"{g}: {'off' if not cfg else shown}")
    king_loop, king_pivot, completion = (routed[g] for g in
                                         (KING_LOOP_GROUP, KING_PIVOT_GROUP, COMPLETION_GROUP))
    if completion:
        # A group the fold mix holds at 0 contributes nothing to D -- not
        # even its finals (env wave 1: `general` = 0.0 until the operator
        # reads the first fold telemetry).
        zero = {src for src, g in src2grp.items() if mix.get(g, 0.0) <= 0}
        completion["exclude_sources"] = completion["exclude_sources"] | zero
        log(f"{COMPLETION_GROUP}: excluding sources {sorted(completion['exclude_sources'])} "
            f"(config + zero-share groups); min_replies {completion['min_replies']}")
    if king_pivot:
        log(f"{KING_PIVOT_GROUP}: {king_pivot['n_rows']} side-table rows in "
            f"{king_pivot['n_files']} file(s) -> admitted pivots on "
            f"{len(king_pivot['table'])} rollouts / "
            f"{sum(len(v) for v in king_pivot['table'].values())} turns")

    baker = ToolBaker.from_pretrained()
    panel = panel_keys()
    drops: dict[str, int] = {}
    notes: dict[str, int] = {}
    candidates: list[dict] = list(carryover)
    for i, c in enumerate(unfolded, 1):
        path = pub.cached(c["key"], c["sha256"], gz_sha=True)
        recs = derive_chunk(path, baker, panel, allowed, published, drops,
                            king_loop=king_loop, king_pivot=king_pivot,
                            completion=completion, notes=notes)
        for rec in recs:
            for m in rec["turns"]:
                published.add(f"{rec['traj_id']}:{m['turn_idx']}")
        candidates.extend(recs)
        if i % 100 == 0 or i == len(unfolded):
            log(f"derived {i}/{len(unfolded)} chunks: {len(candidates)} rollouts, "
                f"{sum(len(r['turns']) for r in candidates)} turns")
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
        log(f"math re-source: {mstats}")
        retire_ids, math_surviving, math_retired_strata = math_retire_plan(
            pub, live, math_cfg, keep, src2grp.get(math_cfg["source"], DEFAULT_GROUP))
        cand_math = [r for r in candidates if str(r.get("source") or "") == math_cfg["source"]]
        cand_keep = [r for r in cand_math if str(r.get("instance_id")) in keep]
        # Survival = strata the kept published turns hold + what kept
        # candidates would open (bucket assignment happens below; recompute
        # the bucket here from the source's setting).
        n_b, off = buckets.get(math_cfg["source"], (0, 0))
        grp = src2grp.get(math_cfg["source"], DEFAULT_GROUP)
        cand_strata = {f"{grp}:{off + int(hashlib.sha256(str(r['instance_id']).encode()).hexdigest()[:8], 16) % n_b:04d}"
                       for r in cand_keep} if n_b else set()
        surviving_total = len(math_surviving | cand_strata)
        log(f"math re-source: published math turns {len(retire_ids) + 0} to retire, "
            f"surviving published strata {len(math_surviving)}, retired strata "
            f"{len(math_retired_strata)}; candidates {len(cand_math)} -> kept {len(cand_keep)}; "
            f"surviving strata incl. candidates {surviving_total}")
        if surviving_total < math_cfg["min_surviving_strata"]:
            log(f"math re-source: only {surviving_total} strata would survive "
                f"(< {math_cfg['min_surviving_strata']}); keeping the old math pool")
            retire_ids, math_surviving, math_retired_strata = [], set(), set()
        else:
            n0 = len(candidates)
            keep_ids = {id(r) for r in cand_keep}
            candidates = [r for r in candidates
                          if str(r.get("source") or "") != math_cfg["source"] or id(r) in keep_ids]
            _count(drops, "math_deterministic", n0 - len(candidates))
            if not math_cfg["retire_published"]:
                retire_ids, math_surviving, math_retired_strata = [], set(), set()
    n_bucketed = assign_bucket_strata(candidates, buckets, src2grp)
    log(f"bucket strata assigned on {n_bucketed} rollouts "
        f"({ {k: (n if not off else f'{n}@{off}') for k, (n, off) in buckets.items() if n} })")
    # King seat: after the source buckets so `king_fail:NNNN` wins for king
    # rollouts on bucketed sources (math / tool_use) too.
    king = load_king_fail()
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
    candidates = drop_excluded_routed(candidates, routed, drops)
    if len(candidates) != n_before:
        log(f"routed groups: dropped {n_before - len(candidates)} carryover records "
            f"from excluded sources")
    stamped = stamp_routed_groups(candidates, routed)
    for g in ROUTED_GROUPS:
        if not routed[g]:
            continue
        recs_g = [r for r in candidates if r.get("fold_group") == g]
        log(f"{g}: {stamped.get(g, 0)} records ({sum(len(r['turns']) for r in recs_g)} "
            f"turns, {len({r['stratum'] for r in recs_g})} strata)")
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
    selected, deferred, group_added = cap_fill(
        candidates, lambda r: group_of(r, src2grp, mix), have_groups, mix,
        anchor_min_target=ANCHOR_MIN_TARGET)
    deferred += lang_deferred
    log(f"mix: selected {len(selected)} rollouts (+{ {g: len(v) for g, v in group_added.items()} } "
        f"strata), deferred {len(deferred)}")
    # Language strata credited only for coding rollouts that made it through
    # the group stage too.
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

    n_new = sum(len(r["turns"]) for r in selected)
    stale = False
    if unfolded:
        newest = max(datetime.fromisoformat(c["created_at"]) for c in unfolded)
        stale = (datetime.now(timezone.utc) - newest).total_seconds() >= STALE_AFTER_S
    if n_new < MIN_NEW_TURNS and not stale and not args.force and not args.init \
            and not retire_ids:
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
        "lang_strata_added": {b: sorted(v) for b, v in lang_added.items()},
        "by_dialect": by_dialect, "allowed_kinds": list(allowed),
        "folded_chunks": [c["key"] for c in unfolded], "init": bool(args.init),
        "retire_turn_ids": retire_ids,
        "group_strata_after_retire": (
            {src2grp.get(math_cfg["source"], DEFAULT_GROUP): sorted(math_surviving)}
            if retire_ids else {}),
    }
    save_state(state)
    finalize(state, *publish_pending(state, publisher, traces_sha, legacy_sha))
    if not args.no_announce:
        announce(state, public_base)
    log(f"cycle complete: epoch {epoch}, +{n_new} turns ({len(deferred)} rollouts deferred)")


if __name__ == "__main__":
    main()
