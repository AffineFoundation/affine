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


def round_file(round_id: str) -> str:
    return f"public/rounds/{_part(round_id)}/round.json"


def round_scores(round_id: str) -> str:
    return f"public/rounds/{_part(round_id)}/scores.json"


def winner_batch(round_id: str) -> str:
    """The winning batch, byte for byte. Only a winner's batch is ever published."""
    return f"public/rounds/{_part(round_id)}/batch.json"


def irt_snapshot(snapshot_id: str) -> str:
    return f"public/irt/{_part(snapshot_id)}.json"


JOURNAL = "private/journal"


def journal_segment(name: str) -> str:
    return f"{JOURNAL}/{_part(name)}"


def journal_record(seq: int) -> str:
    """One journal record. The store keeps the journal as one object per record, never rewritten."""
    return journal_segment(f"{seq:012d}.json")
