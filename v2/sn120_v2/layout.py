"""Object-store key layout. One place, so every process agrees on where things live.

Three zones:

  private/  never public: miner files, frozen batches, tickets.
  pending/  private staging for everything a box uploads. Nothing here is
            trusted until the controller has verified it.
  public/   verified artifacts under content-addressed keys that never change,
            plus a few small pointers. A pointer moves last, after every
            object it names has been verified.

A losing LoRA never leaves ``pending/``. The controller deletes a round's
pending prefix when the round closes. The fp32 master has its own prefix: it
is never stored beside the bf16 base files.

Production shares one bucket with v1, with no key prefix. Every key that v2
writes is named in this file: ``python -m sn120_v2.r2 --preflight`` reads the
public names of this module and lists each one in the bucket. So name a new
key here, as a constant or a function, and never as a literal elsewhere.
"""

from __future__ import annotations

import re

from .errors import ContractError

_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

# The current head: the bytes of HeadManifest.to_json(). Not ``public/current.json``: v1 has an object at that key.
CURRENT = "public/head.json"
LLMS_TXT = "public/llms.txt"
MINING_JSON = "public/mining.json"
WINNER_STATS = "public/winner-stats.json"  # crowned winners only; no waiting or losing batch facts
CONFIRMATION_STATS = "public/confirmation-stats.json"  # applied crowns' original z/delta and later paired test
# Public aliases at the bucket root; R2 preflight reads this separate namespace.
PUBLIC_ALIASES = {"winner-stats.json": WINNER_STATS}  # same bytes at the site root, with the canonical object
WINNER_STATS_ALIAS = next(iter(PUBLIC_ALIASES))
BENCHMARKS = "public/benchmarks.json"  # written by the benchmark service, read by the website
# The upload link files, only when the link job runs with ``--to-store`` (``links.py`` builds the same two keys).
LINK_FILES = "public/ff/links"
LINK_INDEX = "public/ff/links.json"
PENDING = "pending"  # the staging zone. The only zone in which v2 lists and deletes by prefix.


def _part(value: object) -> str:
    """One path component. Rejects separators and dot paths, so a key cannot escape its zone."""
    text = str(value)
    if not _PART.fullmatch(text):
        raise ContractError(f"unsafe object key component {text!r}")
    return text


def miner_object(hotkey: str) -> str:
    """The miner's own private file. Same key as v1, so existing miner kits keep working."""
    return f"private/ff/v1/miners/{_part(hotkey)}/batch.json"


def frozen_batch(round_id: str, hotkey: str) -> str:
    return f"private/rounds/{_part(round_id)}/batches/{_part(hotkey)}.json"


def ticket(round_id: str, lease_id: str) -> str:
    return f"private/rounds/{_part(round_id)}/{_part(lease_id)}/ticket.json"


def sealed_result(round_id: str, lease_id: str) -> str:
    """Where the controller keeps an accepted sealed result as private evidence."""
    return f"private/rounds/{_part(round_id)}/{_part(lease_id)}/result.json"


def pending_round_prefix(round_id: str) -> str:
    return f"{PENDING}/rounds/{_part(round_id)}"


def pending_lease_prefix(round_id: str, lease_id: str) -> str:
    return f"{pending_round_prefix(round_id)}/{_part(lease_id)}"


def pending_result(round_id: str, lease_id: str) -> str:
    return f"{pending_lease_prefix(round_id, lease_id)}/result.json"


def pending_lora_prefix(round_id: str, lease_id: str, queue_pos: int) -> str:
    return f"{pending_lease_prefix(round_id, lease_id)}/loras/{int(queue_pos):03d}"


def pending_fold_prefix(version_id: str) -> str:
    return f"{PENDING}/folds/{_part(version_id)}"


def pending_fold_ahead_prefix(round_id: str) -> str:
    """The staged files of a fold that starts before its head is in the journal (``afterking.py``). The delete of the round's prefix removes them."""
    return f"{pending_round_prefix(round_id)}/fold-ahead"


def public_lora_prefix(lora_hash: str) -> str:
    return f"public/loras/{_part(lora_hash)}"


def public_base_prefix(base_hash: str) -> str:
    return f"public/bases/{_part(base_hash)}"


def public_master_prefix(master_digest: str) -> str:
    return f"public/masters/{_part(master_digest)}"


def public_base_file(base_hash: str, name: str) -> str:
    """One file of a bf16 base. ``name`` is a key of ``HeadManifest.base_files``."""
    return f"{public_base_prefix(base_hash)}/{_part(name)}"


def public_master_file(master_digest: str, name: str) -> str:
    return f"{public_master_prefix(master_digest)}/{_part(name)}"


def public_lora_file(lora_hash: str, name: str) -> str:
    return f"{public_lora_prefix(lora_hash)}/{_part(name)}"


def head_manifest(version_id: str) -> str:
    return f"public/heads/{_part(version_id)}.json"


# The head chain (``headchain.py``): genesis, then one step for each fold, and which heads are checkpoints.
HEAD_CHAIN = "public/head_chain.json"


def head_checkpoint(version_id: str) -> str:
    """The marker of a checkpoint head. It is written last, after the full base and master are verified in the store."""
    return f"public/head_checkpoints/{_part(version_id)}.json"


def head_fold(version_id: str) -> str:
    """How the box folded the head ``version_id`` (the head with the stacked LoRAs) when it kept the weights local."""
    return f"public/head_folds/{_part(version_id)}.json"


def round_file(round_id: str) -> str:
    return f"public/rounds/{_part(round_id)}/round.json"


def round_scores(round_id: str) -> str:
    return f"public/rounds/{_part(round_id)}/scores.json"


def winner_batch(round_id: str) -> str:
    """The winning batch, byte for byte. Only a winner's batch is ever published."""
    return f"public/rounds/{_part(round_id)}/batch.json"


ROLLOUT_BENCH_PREFIX = "public/rollouts/bench/"
ROLLOUT_GATE_PREFIX = "public/rollouts/gate/"
ROLLOUT_INDEX = "public/rollouts/index.json"


def public_rollout_bench(model_id: str, benchmark_id: str) -> str:
    return f"{ROLLOUT_BENCH_PREFIX}{_part(model_id)}/{_part(benchmark_id)}.jsonl.gz"


def public_rollout_bench_meta(model_id: str, benchmark_id: str) -> str:
    return f"{ROLLOUT_BENCH_PREFIX}{_part(model_id)}/{_part(benchmark_id)}.meta.json"


def public_rollout_gate(round_id: str, role: str = "head") -> str:
    if role not in ("head", "winner"):
        raise ContractError("a public gate rollout is the head or the winner")
    return f"{ROLLOUT_GATE_PREFIX}{_part(round_id)}/{role}.jsonl.gz"


def public_rollout_gate_entry(round_id: str, queue_pos: int) -> str:
    """The merged-model eval rollouts of one scored queue entry, including the winner."""
    if type(queue_pos) is not int or queue_pos < 0:
        raise ContractError("a public gate entry rollout needs a nonnegative integer queue position")
    return f"{ROLLOUT_GATE_PREFIX}{_part(round_id)}/entries/{queue_pos}.jsonl.gz"


def public_rollout_gate_meta(round_id: str) -> str:
    return f"{ROLLOUT_GATE_PREFIX}{_part(round_id)}/meta.json"


def irt_snapshot(snapshot_id: str) -> str:
    return f"public/irt/{_part(snapshot_id)}.json"


def active_pool(sha256: str) -> str:
    return f"public/pool/active/{_part(sha256)}.json"


def pool_rotation(round_id: str) -> str:
    return f"public/pool/rotation/{_part(round_id)}.json"


def pool_rotation_state(sha256: str) -> str:
    return f"public/pool/state/{_part(sha256)}.json"


def pool_ranking(round_id: str) -> str:
    return f"public/pool/ranking/{_part(round_id)}.json"


POOL_SUMMARY = "public/pool/summary.json"

RETIRED_POOL = "public/pool/retired.json"
RESERVE_MANIFEST = "public/pool/reserve/MANIFEST.jsonl"
RESERVE_WITHDRAWN = "public/pool/reserve/withdrawn.jsonl"
RETIRED_TASKS = "public/pool/retired_tasks.json"


def reserve_shard(name: str, sha256: str) -> str:
    return f"public/pool/reserve/shards/{_part(name)}.{_part(sha256)}.jsonl"


DEDUP_FIRST_SEEN = "private/dedup/first_seen.json"


def dedup_rollout_hashes(batch_sha256: str) -> str:
    """Private cached normalized rollout-hash set; batch content is never stored here."""
    return f"private/dedup/rollout_hashes/{_part(batch_sha256)}.json"


JOURNAL = "private/journal"


def journal_segment(name: str) -> str:
    return f"{JOURNAL}/{_part(name)}"


def journal_record(seq: int) -> str:
    """One journal record. The store keeps the journal as one object per record, never rewritten."""
    return journal_segment(f"{seq:012d}.json")
