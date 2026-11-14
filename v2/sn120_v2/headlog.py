"""Head transitions. Pure functions: each one returns the next ``HeadVersion``.

The head is a base checkpoint plus an ordered list of applied LoRAs. Only
the controller appends a transition to the journal. Replaying the journal
from genesis gives every ``version_id`` again, because an id is a hash of
its content and of its parent's id.
"""

from __future__ import annotations

from .errors import HeadMismatchError
from .params import GenesisSpec, Params
from .types import Entry, FoldArtifact, HeadVersion, LoRAArtifact, LoRARef

GENESIS_ROUND = "genesis"


def genesis_head(spec: GenesisSpec, at: int) -> HeadVersion:
    return HeadVersion.make(
        parent_version_id=None,
        base_hash=spec.base_hash,
        base_fp32_master_digest=spec.base_fp32_master_digest,
        loras=(),
        round_id=GENESIS_ROUND,
        applied_at=at,
    )


def lora_ref(round_id: str, entry: Entry, artifact: LoRAArtifact) -> LoRARef:
    return LoRARef(
        lora_hash=artifact.lora_hash,
        round_id=round_id,
        miner_hotkey=entry.reg.hotkey,
        miner_uid=entry.reg.uid,
        batch_sha256=artifact.batch_sha256,
        recipe_id=artifact.recipe_id,
        rank=artifact.rank,
    )


def apply_lora(head: HeadVersion, artifact: LoRAArtifact, ref: LoRARef, at: int) -> HeadVersion:
    """Stack a winning LoRA on the head. The LoRA must have been trained on exactly this head."""
    if artifact.head_version_id != head.version_id:
        raise HeadMismatchError("the winning LoRA was trained on another head; it cannot be applied")
    if ref.lora_hash != artifact.lora_hash:
        raise HeadMismatchError("the LoRA reference names another LoRA")
    return HeadVersion.make(
        parent_version_id=head.version_id,
        base_hash=head.base_hash,
        base_fp32_master_digest=head.base_fp32_master_digest,
        loras=head.loras + (ref,),
        round_id=ref.round_id,
        applied_at=at,
    )


def fold_due(head: HeadVersion, params: Params) -> bool:
    return len(head.loras) >= params.fold_every_k


def fold_head(head: HeadVersion, fold: FoldArtifact, at: int) -> HeadVersion:
    """Replace base + stacked LoRAs by the folded base. The fold must start from exactly this head."""
    if fold.parent_version_id != head.version_id:
        raise HeadMismatchError("the fold was made from another head; it cannot replace this one")
    if not head.loras:
        raise HeadMismatchError("nothing to fold: the head has no stacked LoRA")
    return HeadVersion.make(
        parent_version_id=head.version_id,
        base_hash=fold.base_hash,
        base_fp32_master_digest=fold.base_fp32_master_digest,
        loras=(),
        round_id=head.round_id,
        applied_at=at,
    )
