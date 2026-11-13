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
"""

from __future__ import annotations

import re

from .errors import ContractError

_PART = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")

CURRENT = "public/current.json"  # pointer to the current head manifest
LLMS_TXT = "public/llms.txt"
MINING_JSON = "public/mining.json"


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
    return f"pending/rounds/{_part(round_id)}"


def pending_lease_prefix(round_id: str, lease_id: str) -> str:
    return f"{pending_round_prefix(round_id)}/{_part(lease_id)}"


def pending_result(round_id: str, lease_id: str) -> str:
    return f"{pending_lease_prefix(round_id, lease_id)}/result.json"


def pending_lora_prefix(round_id: str, lease_id: str, queue_pos: int) -> str:
    return f"{pending_lease_prefix(round_id, lease_id)}/loras/{int(queue_pos):03d}"


def pending_fold_prefix(version_id: str) -> str:
    return f"pending/folds/{_part(version_id)}"


def public_lora_prefix(lora_hash: str) -> str:
    return f"public/loras/{_part(lora_hash)}"


def public_base_prefix(base_hash: str) -> str:
    return f"public/bases/{_part(base_hash)}"


def public_master_prefix(master_digest: str) -> str:
    return f"public/masters/{_part(master_digest)}"


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


def journal_segment(name: str) -> str:
    return f"private/journal/{_part(name)}"
