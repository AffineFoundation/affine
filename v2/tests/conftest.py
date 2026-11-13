"""Shared builders for tests. They make small, valid records by hand."""

from __future__ import annotations

import dataclasses

import pytest

from sn120_v2.canonical import manifest_digest, sha256_hex
from sn120_v2.params import GenesisSpec, Params, Recipe
from sn120_v2.types import (
    NO_DECISION,
    SCORED,
    BoxResult,
    Draw,
    Entry,
    EntryResult,
    FrozenBatch,
    HeadVersion,
    LoRAArtifact,
    QueueEntry,
    Registration,
    RoundTicket,
    TrainReport,
    Verdict,
)

IMAGE = "sim-image"


def h(text: str) -> str:
    """A valid sha256 hex string made from any label."""
    return sha256_hex(text.encode())


class Build:
    """Builders with small defaults. Every method returns a valid record."""

    image = IMAGE

    @staticmethod
    def params(**changes) -> Params:
        base = dict(n_tasks=8, min_scored_tasks=2, looks=(1.0,), batch_cap=4, recipe=Recipe(image_digests=(IMAGE,)))
        return Params(**(base | changes))

    @staticmethod
    def genesis_spec() -> GenesisSpec:
        return GenesisSpec("org/model", "rev0", h("base"), h("master"), "5ValidatorHotkey")

    @staticmethod
    def head() -> HeadVersion:
        return HeadVersion.make(
            parent_version_id=None,
            base_hash=h("base"),
            base_fp32_master_digest=h("master"),
            loras=(),
            round_id="genesis",
            applied_at=1000,
        )

    @staticmethod
    def reg(uid: int, block: int | None = None, hotkey: str | None = None, registered_at: int = 1, coldkey: str = "") -> Registration:
        return Registration(uid, hotkey or f"hk{uid}", 100 + uid if block is None else block, registered_at, coldkey)

    @staticmethod
    def seat(uid: int, block: int | None = None, **flags) -> QueueEntry:
        reg_fields = {k: flags.pop(k) for k in ("hotkey", "registered_at", "coldkey") if k in flags}
        return QueueEntry(Build.reg(uid, block, **reg_fields), **flags)

    @staticmethod
    def queue(*seats: QueueEntry) -> dict[tuple[int, int], QueueEntry]:
        return {seat.reg.key: seat for seat in seats}

    @staticmethod
    def verdicts(*hotkeys: str) -> dict[str, Verdict]:
        return {hotkey: Verdict(hotkey, True, batch_sha256=h(hotkey)) for hotkey in hotkeys}

    @staticmethod
    def batch(reg: Registration, round_id: str = "r000001", train: dict | None = None) -> FrozenBatch:
        uri = f"private/rounds/{round_id}/batches/{reg.hotkey}.json"
        return FrozenBatch(reg.hotkey, h(f"batch-{reg.hotkey}"), uri, "affine-batch-v2", 5, 10, train or {})

    @staticmethod
    def entry(position: int, uid: int | None = None, train: dict | None = None) -> Entry:
        reg = Build.reg(10 + position if uid is None else uid)
        return Entry(position, reg, Build.batch(reg, train=train))

    @staticmethod
    def artifact(head: HeadVersion, entry: Entry, params: Params, tag: str = "") -> LoRAArtifact:
        files = {"adapter_config.json": h(f"cfg-{entry.reg.hotkey}{tag}"), "adapter_model.safetensors": h(f"w-{entry.reg.hotkey}{tag}")}
        recipe = params.recipe.for_miner(entry.batch.train)
        return LoRAArtifact(
            lora_hash=manifest_digest(files),
            files=files,
            sizes={name: 100 for name in files},
            prefix=f"pending/rounds/r000001/r000001-l0/loras/{entry.queue_pos:03d}",
            head_version_id=head.version_id,
            batch_sha256=entry.batch.batch_sha256,
            recipe_id=params.recipe_id,
            rank=recipe.rank,
            alpha=recipe.alpha,
            target_modules=recipe.target_modules,
            report=TrainReport(seed=1, steps=1, loss=0.5, step_norm=0.01, tokens=100, wall_s=1.0, gpu_class="cpu", image_digest=IMAGE),
        )

    @staticmethod
    def ticket(
        n_entries: int = 2,
        n_tasks: int = 8,
        params: Params | None = None,
        weights: tuple[float, ...] | None = None,
        lease: int = 0,
        first_pos: int = 0,
        train: dict[int, dict] | None = None,
    ) -> RoundTicket:
        params = params or Build.params(n_tasks=n_tasks)
        task_ids = tuple(f"t{i:03d}" for i in range(n_tasks))
        positions = range(first_pos, first_pos + n_entries)
        return RoundTicket(
            round_id="r000001",
            lease_id=f"r000001-l{lease}",
            box_id=f"box-{lease}",
            head=Build.head(),
            params=params,
            draw=Draw("snap-0", "seed-0", task_ids, weights or (1.0,) * n_tasks),
            entries=tuple(Build.entry(pos, train=(train or {}).get(pos)) for pos in positions),
            issued_at=1000,
            expires_at=5000,
        )

    @staticmethod
    def result(ticket: RoundTicket, null: str | tuple[str, ...], loras: dict[int, object], **changes) -> BoxResult:
        """``loras`` maps queue_pos to outcome string(s), or to a ready ``EntryResult``, or to a status name."""
        as_tuple = lambda value: (value,) if isinstance(value, str) else tuple(value)
        entries = []
        for seat in ticket.entries:
            value = loras.get(seat.queue_pos, NO_DECISION)
            if isinstance(value, EntryResult):
                entries.append(value)
            elif value in (NO_DECISION, "skip"):
                entries.append(EntryResult(seat.queue_pos, value, reason="test"))
            else:
                artifact = Build.artifact(ticket.head, seat, ticket.params)
                entries.append(EntryResult(seat.queue_pos, SCORED, lora=artifact, outcomes=as_tuple(value)))
        fields = dict(
            round_id=ticket.round_id,
            lease_id=ticket.lease_id,
            box_id=ticket.box_id,
            ticket_digest=ticket.digest(),
            head_version_id=ticket.head.version_id,
            image_digest=IMAGE,
            null_outcomes=as_tuple(null),
            entries=tuple(entries),
            started_at=1001,
            finished_at=1500,
        )
        return BoxResult(**(fields | changes))

    @staticmethod
    def dropped(ticket: RoundTicket, position: int, outcomes: str, look: int) -> EntryResult:
        seat = next(e for e in ticket.entries if e.queue_pos == position)
        artifact = Build.artifact(ticket.head, seat, ticket.params)
        return EntryResult(position, SCORED, lora=artifact, outcomes=(outcomes,), eliminated_at_look=look)

    replace = staticmethod(dataclasses.replace)


@pytest.fixture
def build() -> type[Build]:
    return Build
