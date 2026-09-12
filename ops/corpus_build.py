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
     index;
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
import gzip
import hashlib
import io
import json
import os
import shutil
import sys
import tempfile
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "affine"))

from affine.config import load_config  # noqa: E402
from affine.corpus.materialize import stratum_key  # noqa: E402
from affine.corpus.pack import PackResult  # noqa: E402
from affine.corpus.publish import CorpusPublisher  # noqa: E402
from affine.corpus.trace import ToolParityError, TraceShapeError  # noqa: E402
from affine.corpus.view import (  # noqa: E402
    VIEW_SPEC,
    build_view_record,
    legacy_view_record,
    validate_turns,
    view_turns,
)
from affine.corpus.viewpack import FORMAT, pack_view_records  # noqa: E402
from affine.toolbake import ToolBaker  # noqa: E402

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
    return {"group": "king_fail",
            "strata_buckets": int(cfg.get("strata_buckets", 0) or 0),
            "policy_prefix": str(cfg.get("policy_prefix") or "king_")}


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
    data never lands unlabelled in the teacher groups)."""
    prefix = (king or {}).get("policy_prefix") or "king_"
    n = int((king or {}).get("strata_buckets") or 0)
    out: list[dict] = []
    for rec in records:
        pid = str((rec.get("policy") or {}).get("id") or "")
        if not pid.startswith(prefix):
            out.append(rec)
            continue
        if not king or n <= 0:
            drops["king_no_fold_group"] = drops.get("king_no_fold_group", 0) + 1
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


def cap_fill(records: list[dict], keyf, have: dict[str, set[str]],
             targets: dict[str, float]
             ) -> tuple[list[dict], list[dict], dict[str, set[str]]]:
    """Mix enforcement in SLICE STRATA, not turns.

    sample_slice draws round-robin over strata and n_turns (1300) is far
    below the strata count, so a duel slice holds one turn per stratum: a
    group's share of what miners are scored on is its share of strata, and
    its turn count is irrelevant. Counting turns (the fold until 2026-09-03)
    gave a 1-turn math rollout the weight of a 40-turn coding rollout; the
    traces-only rehearsal selected coding 9,677 turns = 186 strata against
    math 1,935 = 1,935 strata -- coding 4% of the slice.

    Rule: every key takes all its candidates except the single most
    over-supplied one, which is capped at the share it would hold if the
    second-most over-supplied key were exactly on target. Over-supply of key
    k is (strata available) / target_k -- the corpus size k alone could
    support at its target. Exhausted keys (math, tool_use, small languages)
    therefore never throttle the others (the strict waterfill froze D at the
    first exhausted key), while the one flood (terminal 8.6k tasks vs coding
    5.3k; python vs the other languages) is held to its target ratio against
    the next-largest supply. At most one key is ever trimmed; trimmed
    rollouts defer and re-enter as the reference key grows.
    Keys without a positive target are deferred whole, as before."""
    pools: dict[str, list[dict]] = {}
    for rec in records:
        pools.setdefault(keyf(rec), []).append(rec)
    keyed = {k: v for k, v in pools.items() if targets.get(k, 0.0) > 0}
    avail: dict[str, set[str]] = {k: set(have.get(k, ())) for k in targets}
    for k, pool in keyed.items():
        for rec in pool:
            avail[k] |= record_strata(rec)
    supply = sorted((len(avail[k]) / targets[k] for k in targets), reverse=True)
    ref_total = supply[1] if len(supply) > 1 else float("inf")
    cap = {k: targets[k] * ref_total for k in targets}
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


# -- derive ----------------------------------------------------------------------
# The prefix cap in the unit that binds at duel time: the serving window is
# max_model_len = 131072 tokens minus 1792 generated. MAX_PREFIX_CHARS (300k,
# datagen/slicer.py) is the coarse cut; prefixes above TOKEN_GUARD_FROM_CHARS
# are measured with the teacher tokenizer and dropped past MAX_PREFIX_TOKENS
# (a 2026-09-10 data event; 300k chars ~ 78k tokens p50 / 90k p10).
MAX_PREFIX_TOKENS = 110_000
TOKEN_GUARD_FROM_CHARS = 120_000


def prefix_over_token_cap(turn: dict, baker: ToolBaker) -> bool:
    if int(turn.get("n_prefix_chars") or 0) <= TOKEN_GUARD_FROM_CHARS:
        return False
    text = "\n".join(m.get("content", "") for m in turn.get("prefix") or [])
    n = len(baker.tok(text, add_special_tokens=False)["input_ids"])
    return n + 8 * len(turn.get("prefix") or []) > MAX_PREFIX_TOKENS


def derive_chunk(path: Path, baker: ToolBaker, panel, allowed_kinds,
                 published: set[str], drops: dict[str, int]) -> list[dict]:
    """View records for one trace chunk, with only the turns that pass the
    fold contract and are not yet published. Records with no surviving
    turn are dropped."""
    out: list[dict] = []
    for env in iter_jsonl_gz(path):
        try:
            rec = build_view_record(env, baker=baker,
                                    generated_at=env.get("stored_at"))
        except (ToolParityError, TraceShapeError) as e:
            drops[type(e).__name__] = drops.get(type(e).__name__, 0) + 1
            continue
        if rec is None:
            drops["no_scorable_turn"] = drops.get("no_scorable_turn", 0) + 1
            continue
        kept, d = validate_turns(view_turns(rec), panel=panel,
                                 allowed_kinds=allowed_kinds)
        for k, v in d.items():
            drops[k] = drops.get(k, 0) + v
        keep_idx = set()
        for t in kept:
            tid = f"{t['traj_id']}:{t['turn_idx']}"
            if tid in published:
                drops["already_published"] = drops.get("already_published", 0) + 1
                continue
            if prefix_over_token_cap(t, baker):
                drops["prefix_too_many_tokens"] = drops.get("prefix_too_many_tokens", 0) + 1
                continue
            keep_idx.add(t["turn_idx"])
        rec["turns"] = [m for m in rec["turns"] if m["turn_idx"] in keep_idx]
        if rec["turns"]:
            out.append(rec)
    return out


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
                prev: dict | None, epoch: int) -> None:
    """Previous active index + the new pack's rows -> one parquet the
    manifest points at (evalsrv reads exactly one index)."""
    if not prev or not prev.get("index"):
        return
    prev_raw = publisher.get(prev["index"]["key"])
    if hashlib.sha256(prev_raw).hexdigest() != prev["index"]["sha256"]:
        fatal("previous index sha mismatch on the bucket")
    merged = pa.concat_tables([pq.read_table(io.BytesIO(prev_raw)),
                               pq.read_table(pack.index_path)])
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
    merge_index(pack, publisher, prev, epoch)
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
        f"{strata_line}.\n\n"
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

    live, live_sha = (publisher.current_manifest() if publisher else (None, None))
    if args.init and live is not None:
        fatal(f"--init but a corpus manifest already exists (epoch {live['corpus_epoch']})")
    if not args.init and live is None and not args.no_publish:
        fatal("no schema-3 corpus manifest yet; run with --init first")

    allowed = (tuple(k.strip() for k in args.allowed_kinds.split(",") if k.strip())
               if args.allowed_kinds else tuple(cfg.dataset.allowed_action_kinds))
    log(f"allowed action kinds: {list(allowed)}")

    unfolded = [c for c in traces_manifest["chunks"]
                if args.rederive or c["key"] not in state["folded_chunks"]]
    # split("\n"), not splitlines(): JSON strings may carry U+2028 / U+0085.
    carryover = ([json.loads(l) for l in DEFERRED_PATH.read_text().split("\n")
                  if l.strip()] if DEFERRED_PATH.exists() else [])
    if args.rederive:
        log(f"--rederive: all {len(unfolded)} chunks re-derived; "
            f"{len(carryover)} deferred rollouts dropped (regenerated from traces)")
        carryover = []
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

    baker = ToolBaker.from_pretrained()
    panel = panel_keys()
    drops: dict[str, int] = {}
    candidates: list[dict] = list(carryover)
    for i, c in enumerate(unfolded, 1):
        path = pub.cached(c["key"], c["sha256"], gz_sha=True)
        recs = derive_chunk(path, baker, panel, allowed, published, drops)
        for rec in recs:
            for m in rec["turns"]:
                published.add(f"{rec['traj_id']}:{m['turn_idx']}")
        candidates.extend(recs)
        if i % 100 == 0 or i == len(unfolded):
            log(f"derived {i}/{len(unfolded)} chunks: {len(candidates)} rollouts, "
                f"{sum(len(r['turns']) for r in candidates)} turns")
    log(f"drops: {drops or 'none'}")

    mix, src2grp, lang_mix, buckets = load_mix(ignore_fold_mix=args.ignore_fold_mix)
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
        chosen, lang_deferred, lang_added = cap_fill(
            coding, lang_bucket, have_langs, lang_mix)
        candidates = other + chosen
        log(f"lang mix: kept {len(chosen)}/{len(coding)} coding rollouts "
            f"(+{ {b: len(v) for b, v in lang_added.items()} } strata), "
            f"deferred {len(lang_deferred)}")
    have_groups = {g: set(v) for g, v in (state.get("group_strata") or {}).items()}
    selected, deferred, group_added = cap_fill(
        candidates, lambda r: group_of(r, src2grp, mix), have_groups, mix)
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

    n_new = sum(len(r["turns"]) for r in selected)
    stale = False
    if unfolded:
        newest = max(datetime.fromisoformat(c["created_at"]) for c in unfolded)
        stale = (datetime.now(timezone.utc) - newest).total_seconds() >= STALE_AFTER_S
    if n_new < MIN_NEW_TURNS and not stale and not args.force and not args.init:
        log(f"only {n_new} mix-eligible new turns (< {MIN_NEW_TURNS}); skipping")
        return
    if not selected and not legacy:
        log("nothing to publish")
        return

    epoch = (int(live["corpus_epoch"]) if live else int(legacy_manifest["corpus_epoch"])) + 1
    records = legacy + selected
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
    }
    save_state(state)
    finalize(state, *publish_pending(state, publisher, traces_sha, legacy_sha))
    if not args.no_announce:
        announce(state, public_base)
    log(f"cycle complete: epoch {epoch}, +{n_new} turns ({len(deferred)} rollouts deferred)")


if __name__ == "__main__":
    main()
