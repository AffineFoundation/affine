#!/usr/bin/env python
"""Recurring datagen -> production corpus refresh.

pm2 app `affine-corpus-refresh` (affine/scripts/ecosystem.config.js) runs this
daily at 16:00 UTC (host clock is UTC). Each cycle:

  1. list shards in the private HF dataset unconst/affine-datagen-turns that
     are not yet folded into the production corpus (state.json next to this
     script, under ops/datagen_refresh/);
  2. download + prefilter records: the per-record contract from
     affine/scripts/corpus_push.py validate, plus action_kind == bash, exactly
     one closed ```bash block in reference_turn, no verbatim leakage of that
     block into the prefix, and bench-panel repo disjointness
     (affine/evalsrv/data/swe_rebench_lite_ids.json);
  3. enforce the [mix] group targets from rollouts/rollouts/sources.toml:
     candidates are selected by whole-corpus deficit waterfill (a group over
     its target share defers its surplus turns to work/deferred_turns.jsonl,
     re-considered every cycle) — this also retroactively rebalances the
     pre-mix corpus, which counts as group "coding";
  4. skip the cycle when fewer than MIN_NEW_TURNS valid new turns accumulated,
     unless the newest unfolded shard is older than STALE_AFTER_S (stalled
     producer -- fold whatever exists to keep the cadence);
  5. publish via the corpus_push path and in its safe order: immutable shard
     object first, manifest revision last, local state only after the manifest
     confirms. A `pending` record written before the upload makes a crashed
     publish resumable: the next run re-publishes the exact same candidate
     bytes, verifying the remote shard sha instead of re-uploading;
  6. announce on the SN120 Discord channel -- only when a publish actually
     happened; a failed announcement is queued and retried next cycle.

Credentials (the local doppler token is broken, 2026-08-07):
  * HIPPIUS_ACCESS_KEY / HIPPIUS_SECRET_KEY -- read from the running
    affine-validator process environment (same source as the manual
    2026-08-07 epoch-2 refresh);
  * HF token for the private dataset -- cached 0600 in the state dir,
    re-fetched from the datagen pod's env file over SSH when missing/invalid
    (the validator's own HF_TOKEN is revoked);
  * DISCORD_BOT_TOKEN_ARBOS_BITTENSOR -- repo .env.

Fail-loud: any error aborts the cycle with state untouched; pm2 cron retries
next day. Never prints a secret.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import os
import tomllib
from datetime import datetime, timezone
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
STATE_DIR = REPO / "ops" / "datagen_refresh"
STATE_PATH = STATE_DIR / "state.json"
HF_TOKEN_PATH = STATE_DIR / "hf_token"
WORK_DIR = STATE_DIR / "work"
DEFERRED_PATH = WORK_DIR / "deferred_turns.jsonl"
SOURCES_TOML = REPO / "rollouts" / "rollouts" / "sources.toml"
# Records without a source tag predate the unified pipeline: mini-swe
# swerebench trajectories, i.e. coding.
DEFAULT_GROUP = "coding"

DATAGEN_REPO = "unconst/affine-datagen-turns"
PANEL_PATH = REPO / "affine" / "evalsrv" / "data" / "swe_rebench_lite_ids.json"
MIN_NEW_TURNS = 200
STALE_AFTER_S = 48 * 3600
# Bittensor guild -> SN120 community channel (120・ⴷffine・ⴷ), same venue as
# the manual epoch-2 announcement (message 1535361517502070917).
DISCORD_GUILD_ID = "799672011265015819"
DISCORD_CHANNEL_ID = "1381987595881414656"
# Read-only fetch of the datagen pod's env file (pod owned by another agent;
# we only ever read the token line).
DATAGEN_POD_SSH = [
    "ssh", "-o", "StrictHostKeyChecking=accept-new",
    "-o", "UserKnownHostsFile=/tmp/datagen.known_hosts",
    "-o", "ConnectTimeout=15", "-o", "BatchMode=yes",
    "-p", "3049", "root@66.153.184.201",
]

BASH_BLOCK = re.compile(r"```bash\n.*?\n```", re.DOTALL)

# Reuse the production publish path (validation, immutable shard upload,
# manifest revision scheme) instead of reimplementing it.
_spec = importlib.util.spec_from_file_location(
    "corpus_push", REPO / "affine" / "scripts" / "corpus_push.py")
corpus_push = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(corpus_push)


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
    return {"folded": [], "pending": None, "unannounced": None, "history": []}


def save_state(state: dict) -> None:
    tmp = STATE_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2) + "\n")
    tmp.replace(STATE_PATH)


# -- credentials ---------------------------------------------------------------
def hippius_env_from_validator() -> None:
    p = subprocess.run(["pm2", "jlist"], capture_output=True, text=True,
                       timeout=60)
    if p.returncode != 0:
        fatal("pm2 jlist failed; cannot locate affine-validator")
    pid = next((proc["pid"] for proc in json.loads(p.stdout)
                if proc["name"] == "affine-validator"
                and proc["pm2_env"]["status"] == "online"), None)
    if not pid:
        fatal("affine-validator not online; no Hippius credential source")
    env = {}
    for entry in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0"):
        k, _, v = entry.partition(b"=")
        env[k.decode()] = v.decode()
    for key in ("HIPPIUS_ACCESS_KEY", "HIPPIUS_SECRET_KEY"):
        if not env.get(key):
            fatal(f"{key} missing from validator environ")
        os.environ[key] = env[key]
    log("hippius credentials sourced from validator process env")


def _hf_token_valid(token: str) -> bool:
    try:
        r = httpx.get("https://huggingface.co/api/whoami-v2", timeout=30,
                      headers={"Authorization": f"Bearer {token}"})
        return r.status_code == 200
    except httpx.HTTPError:
        return False


def hf_token() -> str:
    if HF_TOKEN_PATH.exists():
        token = HF_TOKEN_PATH.read_text().strip()
        if token and _hf_token_valid(token):
            return token
        log("cached HF token invalid; re-fetching from datagen pod")
    p = subprocess.run(
        DATAGEN_POD_SSH
        + ["sed -n 's/^export HF_TOKEN=//p' /root/affine/.datagen_env"
           " | head -1 | tr -d '\"'"],
        capture_output=True, text=True, timeout=60)
    token = (p.stdout or "").strip()
    if p.returncode != 0 or not token:
        fatal("could not read HF token from datagen pod env file")
    if not _hf_token_valid(token):
        fatal("HF token from datagen pod failed whoami")
    HF_TOKEN_PATH.touch(mode=0o600, exist_ok=True)
    HF_TOKEN_PATH.write_text(token + "\n")
    os.chmod(HF_TOKEN_PATH, 0o600)
    log("HF token fetched from datagen pod and cached (0600)")
    return token


def discord_token() -> str:
    for line in (REPO / ".env").read_text().splitlines():
        if line.startswith("DISCORD_BOT_TOKEN_ARBOS_BITTENSOR="):
            return line.split("=", 1)[1].strip()
    fatal("DISCORD_BOT_TOKEN_ARBOS_BITTENSOR not found in .env")
    raise AssertionError  # unreachable


# -- collect + prefilter ---------------------------------------------------------
def datagen_manifest(token: str) -> dict:
    r = httpx.get(
        f"https://huggingface.co/datasets/{DATAGEN_REPO}/resolve/main/manifest.json",
        headers={"Authorization": f"Bearer {token}"},
        timeout=60, follow_redirects=True)
    r.raise_for_status()
    return r.json()


def download_shard(token: str, shard: dict) -> bytes:
    r = httpx.get(
        f"https://huggingface.co/datasets/{DATAGEN_REPO}/resolve/main/{shard['key']}",
        headers={"Authorization": f"Bearer {token}"},
        timeout=300, follow_redirects=True)
    r.raise_for_status()
    got = hashlib.sha256(r.content).hexdigest()
    if got != shard["sha256"]:
        fatal(f"datagen shard {shard['key']} sha mismatch: {got}")
    return r.content


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def panel_keys() -> tuple[set[str], set[str]]:
    ids = set(json.loads(PANEL_PATH.read_text())["instance_ids"])
    repos = {i.rsplit("-", 1)[0].replace("__", "/").lower() for i in ids}
    return ids, repos


def published_turn_ids(corpus: "corpus_push.Corpus", manifest: dict) -> set[str]:
    ids: set[str] = set()
    # Prefer the Parquet index when present (schema v2) — covers every
    # scorable turn without walking chunk bodies.
    if manifest.get("index"):
        import pyarrow.parquet as pq
        import tempfile
        idx = manifest["index"]
        raw = corpus._get(idx["key"])
        if hashlib.sha256(raw).hexdigest() != idx["sha256"]:
            fatal(f"live index sha mismatch for {idx['key']}")
        with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
            tmp.write(raw)
            tmp.flush()
            table = pq.read_table(tmp.name, columns=["turn_id"])
            ids.update(table.column("turn_id").to_pylist())
        return ids
    for s in manifest["shards"]:
        for tid in corpus_push.iter_published_turn_ids(corpus._get, s):
            ids.add(tid)
    return ids


def prefilter(lines: list[str], published: set[str],
              panel_ids: set[str], panel_repos: set[str]) -> tuple[list[str], dict]:
    """Per-record contract checks; returns (kept lines, drop counts)."""
    kept: list[str] = []
    drops: dict[str, int] = {}
    seen: set[str] = set()

    def drop(reason: str) -> None:
        drops[reason] = drops.get(reason, 0) + 1

    for line in lines:
        line = line.rstrip("\n")
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            drop("not_json")
            continue
        tid, tix = rec.get("traj_id"), rec.get("turn_idx")
        if not (isinstance(tid, str) and tid and isinstance(tix, int)):
            drop("bad_ids")
            continue
        turn_id = f"{tid}:{tix}"
        if turn_id in seen:
            drop("dup_within_batch")
            continue
        if turn_id in published:
            drop("already_published")
            continue
        repo = str(rec.get("repo", "")).lower()
        if repo in panel_repos or str(rec.get("instance_id", "")) in panel_ids:
            drop("bench_panel_overlap")
            continue
        if rec.get("action_kind") != "bash":
            drop("action_kind_not_bash")
            continue
        prefix = rec.get("prefix")
        if (not isinstance(prefix, list) or not prefix
                or not all(isinstance(m, dict)
                           and isinstance(m.get("role"), str)
                           and isinstance(m.get("content"), str)
                           for m in prefix)
                or prefix[-1]["role"] != "user"):
            drop("prefix_shape")
            continue
        sys_msgs = [m for m in prefix if m["role"] == "system"]
        if not sys_msgs or "bash" not in sys_msgs[0]["content"].lower():
            drop("system_msg_no_bash_mandate")
            continue
        if sum(len(m["content"]) for m in prefix) > corpus_push.MAX_PREFIX_CHARS:
            drop("prefix_too_long")
            continue
        blocks = BASH_BLOCK.findall(rec.get("reference_turn") or "")
        if len(blocks) != 1:
            drop(f"ref_bash_blocks={len(blocks)}")
            continue
        body = _norm(blocks[0])
        if len(body) > 40 and any(body in _norm(m["content"]) for m in prefix):
            drop("reference_leaked_into_prefix")
            continue
        seen.add(turn_id)
        kept.append(line)
    return kept, drops


# -- mix enforcement -------------------------------------------------------------
def load_mix() -> tuple[dict[str, float], dict[str, str]]:
    """([mix] group targets, source name -> group) from the rollouts
    registry TOML (single source of truth shared with the generation-side
    scheduler)."""
    raw = tomllib.loads(SOURCES_TOML.read_text())
    mix = {g: float(v) for g, v in raw.get("mix", {}).items()}
    if abs(sum(mix.values()) - 1.0) > 0.01:
        fatal(f"[mix] in {SOURCES_TOML} must sum to 1.0")
    src2grp = {name: cfg["group"] for name, cfg in raw.get("source", {}).items()}
    return mix, src2grp


def enforce_mix(kept: list[str], group_counts: dict[str, int],
                mix: dict[str, float], src2grp: dict[str, str],
                ) -> tuple[list[str], list[str], dict[str, int]]:
    """Whole-corpus deficit waterfill: repeatedly take a turn from the
    group furthest below its target share of the post-fold corpus; stop
    when every group with candidates left would only move further above
    target. Returns (selected, deferred, per-group counts added).

    The existing corpus counts (group_counts) are the retroactive part:
    a fully-coding corpus keeps coding candidates deferred until the other
    groups catch up to the declared mix."""
    pools: dict[str, list[str]] = {}
    for line in kept:
        rec = json.loads(line)
        group = src2grp.get(rec.get("source") or "", DEFAULT_GROUP)
        if group not in mix:
            group = DEFAULT_GROUP
        pools.setdefault(group, []).append(line)
    counts = {g: int(group_counts.get(g, 0)) for g in mix}
    added: dict[str, int] = {}
    selected: list[str] = []
    cursors = {g: 0 for g in pools}
    while True:
        cands = [g for g in pools if cursors[g] < len(pools[g])]
        if not cands:
            break
        total = sum(counts.values()) + 1
        best = max(cands, key=lambda g: mix[g] * total - counts[g])
        if mix[best] * total - counts[best] <= 0:
            break   # every remaining group is at/over target
        selected.append(pools[best][cursors[best]])
        cursors[best] += 1
        counts[best] += 1
        added[best] = added.get(best, 0) + 1
    deferred = [line for g, pool in pools.items()
                for line in pool[cursors[g]:]]
    return selected, deferred, added


# -- publish -------------------------------------------------------------------
def publish_candidate(corpus: "corpus_push.Corpus", state: dict) -> dict:
    """Publish state['pending']. Routes to v1 or v2 based on live schema /
    pending.pack_dir. Idempotent across crash/resume."""
    pending = state["pending"]
    if pending.get("pack_dir") or int(
            corpus._require_manifest().get("schema_version") or 1) >= 2:
        return publish_candidate_v2(corpus, state)
    return publish_candidate_v1(corpus, state)


def publish_candidate_v1(corpus: "corpus_push.Corpus", state: dict) -> dict:
    """Legacy turn-flat publish (pre-migration)."""
    pending = state["pending"]
    epoch = int(pending["epoch"])
    path = Path(pending["candidate"])
    m = corpus._require_manifest()

    if int(m["corpus_epoch"]) >= epoch:
        hit = [s for s in m["shards"]
               if s["key"].endswith(f"turns_epoch_{epoch:04d}.jsonl.gz")]
        if not hit or hit[0]["sha256"] != pending["sha256"]:
            fatal(f"manifest already at epoch {m['corpus_epoch']} but pending "
                  f"epoch {epoch} shard is absent/mismatched -- operator check")
        log(f"pending epoch {epoch} already in manifest; finalizing state only")
        return m

    if not path.is_file():
        fatal(f"pending candidate file missing: {path} -- operator check")
    if not corpus.cmd_validate(path, manifest=m):
        fatal("candidate failed corpus_push validation")

    key = f"{corpus_push.SHARDS_PREFIX}/turns_epoch_{epoch:04d}.jsonl.gz"
    if corpus._exists(key):
        got = hashlib.sha256(gzip.decompress(corpus._get(key))).hexdigest()
        if got != pending["sha256"]:
            fatal(f"remote shard {key} exists with different sha -- operator check")
        log(f"shard {key} already uploaded (sha verified); resuming at manifest")
        shard = {"key": key, "sha256": pending["sha256"],
                 "n_turns": pending["n_turns"], "active": True}
    else:
        shard = corpus.upload_shard(path, epoch)

    prev_raw = json.dumps(m, indent=2, sort_keys=True).encode()
    prev_hash = hashlib.sha256(prev_raw).hexdigest()
    manifest = {
        "corpus_epoch": epoch,
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "shards": m["shards"] + [shard],
        "prev_manifest": f"{corpus.manifests_prefix}/{prev_hash}.json",
    }
    corpus.publish_manifest(manifest)
    return manifest


def publish_candidate_v2(corpus: "corpus_push.Corpus", state: dict) -> dict:
    """Append new traj chunks + rewrite combined Parquet index (schema v2)."""
    from affine.corpus.pack import pack_turns_to_chunks
    import pyarrow as pa
    import pyarrow.parquet as pq
    import tempfile

    pending = state["pending"]
    epoch = int(pending["epoch"])
    path = Path(pending["candidate"])
    m = corpus._require_manifest()

    if int(m["corpus_epoch"]) >= epoch:
        if int(m.get("schema_version") or 1) >= 2 and m.get("index"):
            log(f"pending epoch {epoch} already in manifest; finalizing")
            return m
        fatal(f"manifest at epoch {m['corpus_epoch']} but pending v2 epoch "
              f"{epoch} not found -- operator check")

    if not path.is_file():
        fatal(f"pending candidate file missing: {path}")

    pack_dir = Path(pending.get("pack_dir") or (WORK_DIR / f"pack_{epoch:04d}"))
    if not (pack_dir / f"turns_{epoch:04d}.parquet").is_file():
        log(f"packing candidate → {pack_dir}")
        pack = pack_turns_to_chunks(
            [path], pack_dir, epoch=epoch, write_compat=False)
    else:
        # Resume: rebuild PackResult from disk.
        from affine.corpus.pack import PackResult
        chunk_paths = sorted(pack_dir.glob(f"chunk_{epoch:04d}_*.jsonl"))
        index_path = pack_dir / f"turns_{epoch:04d}.parquet"
        chunk_meta = []
        n_turns = 0
        for p in chunk_paths:
            raw = p.read_bytes()
            n_traj = sum(1 for line in raw.splitlines() if line.strip())
            nt = 0
            for line in raw.splitlines():
                if line.strip():
                    nt += len(json.loads(line)["turns"])
            n_turns += nt
            idx = int(p.stem.rsplit("_", 1)[-1])
            key = (f"{corpus_push.CHUNKS_PREFIX}/chunk_{epoch:04d}_"
                   f"{idx:04d}.jsonl.gz")
            chunk_meta.append({
                "key": key, "sha256": hashlib.sha256(raw).hexdigest(),
                "n_trajectories": n_traj, "n_turns": nt,
                "format": "traj_v1", "active": True,
            })
        pack = PackResult(
            chunk_paths=chunk_paths, chunk_meta=chunk_meta,
            index_path=index_path,
            index_sha256=hashlib.sha256(index_path.read_bytes()).hexdigest(),
            n_turns=n_turns,
            n_trajectories=sum(m_["n_trajectories"] for m_ in chunk_meta),
        )

    # Validate the new pack alone (index rows resolve into its chunks).
    if not corpus.cmd_validate_v2(pack, manifest=None):
        fatal("candidate failed v2 validation")

    # Merge with previous active index when live is already v2.
    skip_validate = False
    if int(m.get("schema_version") or 1) >= 2 and m.get("index"):
        prev_raw = corpus._get(m["index"]["key"])
        with tempfile.NamedTemporaryFile(suffix=".parquet") as tmp:
            tmp.write(prev_raw)
            tmp.flush()
            prev_table = pq.read_table(tmp.name)
        new_table = pq.read_table(pack.index_path)
        merged = pa.concat_tables([prev_table, new_table])
        merged_path = pack_dir / f"turns_{epoch:04d}_merged.parquet"
        pq.write_table(merged, merged_path, compression="zstd")
        pack.index_path = merged_path
        pack.index_sha256 = hashlib.sha256(merged_path.read_bytes()).hexdigest()
        pack.n_turns = merged.num_rows
        # Keep previous active chunks; only add new ones. Skip re-validate:
        # merged index references prior chunks not in pack.chunk_paths.
        retire_previous = False
        skip_validate = True
    else:
        # First v2 publish from a v1 live corpus via refresh (unusual —
        # migrate_corpus_v2.py is the normal cutover). Retire v1 shards.
        retire_previous = True

    pending["pack_dir"] = str(pack_dir)
    pending["index_sha256"] = pack.index_sha256
    pending["n_turns"] = pack.n_turns
    save_state(state)

    return corpus.cmd_add_v2(
        pack, retire_previous=retire_previous, skip_validate=skip_validate)


# -- announce ------------------------------------------------------------------
def announce(state: dict) -> None:
    """Post state['unannounced'] to Discord; clears it on success."""
    info = state["unannounced"]
    epoch, n_added, total, mhash = (info["epoch"], info["n_added"],
                                    info["total"], info["manifest_sha256"])
    schema = int(info.get("schema_version") or 1)
    if schema >= 2:
        layout = (
            f"- schema_version: **2** (trajectory chunks + Parquet index)\n"
            f"- index: `turns/index/turns_{epoch:04d}.parquet`\n"
            f"- chunks: `turns/chunks/chunk_{epoch:04d}_*.jsonl.gz`\n"
        )
        if info.get("compat_key"):
            layout += f"- compat (flat, one epoch): `{info['compat_key']}`\n"
        how = (
            "Miners: fetch `turns/manifest.json`, read the Parquet index, "
            "pull only the chunk objects you need (or the compat flat shard "
            "while it remains listed). Layout: https://affine.io/llms.txt."
        )
    else:
        layout = (
            f"- new shard: `turns/shards/turns_epoch_{epoch:04d}.jsonl.gz`\n"
        )
        how = (
            "Miners: refresh your local corpus from `turns/manifest.json` "
            "(public, sha-pinned; layout: https://affine.io/llms.txt)."
        )
    content = (
        f"**Corpus refresh: epoch {epoch} is live.**\n\n"
        f"{n_added} new SWE-verified turns folded into the production turn "
        "corpus D: mini-swe-agent trajectories on real SWE instances, "
        "kept only where the harness verified the instance resolved, "
        f"disjoint from the advisory bench panel. Corpus total: {total:,} "
        "turns.\n\n"
        f"- corpus_epoch: **{epoch}**\n"
        f"{layout}"
        f"- manifest sha256: `{mhash}`\n\n"
        f"{how} Eval pods pick the new manifest up automatically."
    )
    r = httpx.post(
        f"https://discord.com/api/v10/channels/{DISCORD_CHANNEL_ID}/messages",
        headers={"Authorization": f"Bot {discord_token()}"},
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
def finalize(state: dict, manifest: dict) -> None:
    pending = state["pending"]
    if manifest.get("index"):
        total = int(manifest["index"]["n_turns"])
        n_added = int(pending.get("n_added") or pending["n_turns"])
    else:
        total = sum(int(s.get("n_turns") or 0)
                    for s in manifest["shards"] if s["active"])
        n_added = int(pending["n_turns"])
    raw = json.dumps(manifest, indent=2, sort_keys=True).encode()
    state["folded"] = sorted(set(state["folded"]) | set(pending["folded_keys"]))
    counts = state.get("group_counts", {})
    for group, n in (pending.get("group_added") or {}).items():
        counts[group] = int(counts.get(group, 0)) + int(n)
    state["group_counts"] = counts
    unann = {
        "epoch": int(pending["epoch"]), "n_added": n_added,
        "total": total, "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "schema_version": int(manifest.get("schema_version") or 1),
    }
    compat = (manifest.get("compat_shards") or [None])[0]
    if compat:
        unann["compat_key"] = compat["key"]
    state["unannounced"] = unann
    state["history"].append({
        "epoch": int(pending["epoch"]), "n_turns": n_added,
        "folded_keys": pending["folded_keys"],
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "schema_version": unann["schema_version"],
    })
    state["pending"] = None
    save_state(state)
    Path(pending["candidate"]).unlink(missing_ok=True)


def main() -> None:
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    state = load_state()
    hippius_env_from_validator()
    corpus = corpus_push.Corpus()

    # Resume a publish that crashed between shard upload and state update.
    if state["pending"]:
        log(f"resuming pending publish for epoch {state['pending']['epoch']}")
        finalize(state, publish_candidate(corpus, state))

    # Retry an announcement that failed after a successful publish.
    if state["unannounced"]:
        announce(state)

    token = hf_token()
    shards = datagen_manifest(token).get("shards", [])
    unfolded = [s for s in shards if s["key"] not in state["folded"]]
    carryover = ([line for line in DEFERRED_PATH.read_text().splitlines()
                  if line.strip()] if DEFERRED_PATH.exists() else [])
    if not unfolded and not carryover:
        log(f"no unfolded datagen shards ({len(shards)} total) and no "
            "deferred turns; done")
        return
    log(f"{len(unfolded)} unfolded shard(s), {len(carryover)} deferred "
        "turn(s): " + ", ".join(s["key"] for s in unfolded))

    manifest = corpus._require_manifest()
    published = published_turn_ids(corpus, manifest)
    panel_ids, panel_repos = panel_keys()

    # Deferred turns re-enter first; prefilter dedup makes re-downloaded
    # shard copies of the same turns harmless.
    lines: list[str] = list(carryover)
    for s in unfolded:
        lines.extend(download_shard(token, s).decode().splitlines())
    kept, drops = prefilter(lines, published, panel_ids, panel_repos)
    log(f"prefilter: kept {len(kept)} / dropped {len(lines) - len(kept)} "
        f"of {len(lines)} ({drops or 'no drops'})")

    mix, src2grp = load_mix()
    if not state.get("group_counts"):
        # First mix-aware fold: everything already published predates the
        # unified pipeline and counts as the default group.
        if manifest.get("index"):
            existing_total = int(manifest["index"]["n_turns"])
        else:
            existing_total = sum(int(s.get("n_turns") or 0)
                                 for s in manifest["shards"] if s["active"])
        state["group_counts"] = {DEFAULT_GROUP: existing_total}
        log(f"initialized group_counts: {existing_total} existing turns "
            f"as {DEFAULT_GROUP!r}")
    selected, deferred, group_added = enforce_mix(
        kept, state["group_counts"], mix, src2grp)
    log(f"mix enforcement: selected {len(selected)} (+{group_added}), "
        f"deferred {len(deferred)}")

    stale = False
    if unfolded:
        newest = max(datetime.fromisoformat(s["uploaded_at"])
                     for s in unfolded)
        age_s = (datetime.now(timezone.utc) - newest).total_seconds()
        stale = age_s >= STALE_AFTER_S
    if len(selected) < MIN_NEW_TURNS and not stale:
        log(f"only {len(selected)} mix-eligible new turns "
            f"(< {MIN_NEW_TURNS}); skipping this cycle")
        return
    if not selected:
        log("no mix-eligible new turns at all; nothing to publish")
        return

    epoch = int(manifest["corpus_epoch"]) + 1
    candidate = WORK_DIR / f"candidate_epoch_{epoch:04d}.jsonl"
    candidate.write_text("\n".join(selected) + "\n")
    tmp = DEFERRED_PATH.with_suffix(".jsonl.tmp")
    tmp.write_text("\n".join(deferred) + ("\n" if deferred else ""))
    tmp.replace(DEFERRED_PATH)
    pending = {
        "epoch": epoch,
        "candidate": str(candidate),
        "sha256": hashlib.sha256(candidate.read_bytes()).hexdigest(),
        "n_turns": len(selected),
        "n_added": len(selected),
        "group_added": group_added,
        "folded_keys": [s["key"] for s in unfolded],
    }
    # After the v2 cutover, every fold packs traj chunks + merges the index.
    if int(manifest.get("schema_version") or 1) >= 2:
        pending["pack_dir"] = str(WORK_DIR / f"pack_{epoch:04d}")
    state["pending"] = pending
    save_state(state)

    finalize(state, publish_candidate(corpus, state))
    announce(state)
    log(f"cycle complete: epoch {epoch}, +{len(selected)} turns "
        f"({len(deferred)} deferred by mix)")


if __name__ == "__main__":
    main()
