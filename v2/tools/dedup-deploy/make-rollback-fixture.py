#!/usr/bin/env python3
"""Make an isolated synthetic dedup journal using the real new controller.

Never reads a production journal/store and never contacts a service. The output
is disposable evidence for replay and rollback qualification, not a live state.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import dataclasses
import json
from pathlib import Path
import struct

from sn120_v2 import layout
from sn120_v2.canonical import canonical_json, manifest_digest, sha256_hex
from sn120_v2.sim import SimAdmission, SimWorld, sim_params
from sn120_v2.sampling import train_seed
from sn120_v2.types import EntryDecision


def affine_batch(hotkey: str, rows: list[tuple[str, float]]) -> bytes:
    # Distinct outputs preserve rollout-overlap evidence while two reward
    # classes make every rollout train under the production signal floor.
    # Simulated quality belongs outside the affine batch schema.
    return canonical_json({"version": "affine-batch-v3", "hotkey": hotkey, "train": {},
                           "groups": [{"prompt": f"prompt-{i}", "rollouts": [{"output": out, "reward": reward}]}
                                      for i, (out, reward) in enumerate(rows)]})


def valid_affine_read(world, hotkey):
    """Admission sees valid affine bytes; quality never enters their schema."""
    if not world.store.exists(layout.miner_object(hotkey)):
        return None
    return world.store.get(layout.miner_object(hotkey)), {"valid": True, "quality": 0.0,
                                                           "cap_rate": 0.0, "train": {}}


def sim_read(raws, qualities):
    return lambda hotkey: (raws[hotkey], {"valid": True, "quality": qualities.get(hotkey, 0.0),
                                          "cap_rate": 0.0, "train": {}}) if hotkey in raws else None


@contextmanager
def affine_sim_batches(world):
    """Keep simulator metadata beside, not inside, validated affine batch bytes.

    SimTrainer's tiny quality side channel is a fixture concern. A real affine
    file cannot carry that metadata, so this seam maps it by the frozen hash;
    check_batch, Stage A, training seeds and all sealed-result checks stay real.
    """
    real_train = world.trainer.train
    from sn120_v2.types import TrainReport

    def train(request):
        raw = world.store.get(request.entry.batch.uri)
        body = json.loads(raw)
        assert body["version"] == "affine-batch-v3"
        quality = world.sim_qualities.get(request.entry.batch.batch_sha256, 0.0)
        sidecar = canonical_json({"quality": quality, "cap_rate": 0.0, "train": request.entry.batch.train})
        real_read = world.store.get
        world.store.get = lambda key: sidecar if key == request.entry.batch.uri else real_read(key)
        try:
            return real_train(request)
        finally:
            world.store.get = real_read

    world.trainer.train = train
    world.sim_qualities = {}
    try:
        yield
    finally:
        world.trainer.train = real_train


def upload_affine(world, hotkey, raw, quality):
    world.store.put(layout.miner_object(hotkey), raw)
    world.sim_qualities[sha256_hex(raw)] = quality
    world.admission._read = lambda key: ((world.store.get(layout.miner_object(key)),
                                         {"valid": True, "quality": 0.0, "cap_rate": 0.0, "train": {}})
                                        if world.store.exists(layout.miner_object(key)) else None)


def lora_bytes(value: float, rank: int) -> bytes:
    """A valid rank-by-1 / 1-by-rank update in the recipe's declared rank."""
    header = json.dumps({
        "layer.lora_A.weight": {"dtype": "F32", "shape": [rank, 1], "data_offsets": [0, 4 * rank]},
        "layer.lora_B.weight": {"dtype": "F32", "shape": [1, rank], "data_offsets": [4 * rank, 8 * rank]},
    }, separators=(",", ":")).encode()
    return struct.pack("<Q", len(header)) + header + struct.pack(f"<{2 * rank}f", *([value] * (2 * rank)))


def synthetic_lora(world, artifact, value: float, image: str):
    """Add comparable weights without changing the pinned recipe or artifacts."""
    raw = lora_bytes(value, artifact.rank)
    world.store.put(f"{artifact.prefix}/adapter_model.safetensors", raw)
    files = dict(artifact.files) | {"adapter_model.safetensors": sha256_hex(raw)}
    return dataclasses.replace(
        artifact, files=files, sizes=dict(artifact.sizes) | {"adapter_model.safetensors": len(raw)},
        lora_hash=manifest_digest(files),
        report=dataclasses.replace(artifact.report, image_digest=image),
    )


@contextmanager
def simulated_worker_identity():
    """Adapt only SimWorld's image label; every other result check stays real."""
    import sn120_v2.judging as judging_module
    original = judging_module._need

    def fixture_identity(condition, message):
        if "worker image is not in the pinned set" in message or "trainer image is not pinned" in message:
            return
        original(condition, message)

    judging_module._need = fixture_identity
    try:
        yield
    finally:
        judging_module._need = original


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    root = args.output
    if root.exists():
        raise ValueError("output must not exist")
    root.mkdir(parents=True)
    world = SimWorld(root, sim_params(n_min_start=2, fold_every_k=2))
    affine_metadata = affine_sim_batches(world)
    affine_metadata.__enter__()
    from sn120_v2.sim import SimTrainer
    sim_train = SimTrainer.train
    real_affine_train = world.trainer.train
    from sn120_v2.sim import SIM_IMAGE
    production = json.loads((Path(__file__).parent / "new-params.json").read_text())
    from sn120_v2.params import Params
    # The journal carries the exact reviewed production pin. The isolated CPU
    # fixture adapts worker evidence only; it never changes that historical pin.
    full = Params.from_dict(production)
    assert full.params_hash == "d5979c301c7acf8a56cb6a9989e5b212e5887ec1bf0117f9250202af6a95f35c"
    world.controller.set_params(full, world.clock())
    world.params = full
    assert world.controller.open_round(world.clock(), "backoff-probe") == []
    # SimWorld's synthetic clock advances once per call. Production's 2400 s
    # training budget otherwise expires before the simulated trainer starts.
    # Only that simulated clock is held during worker execution; the journal,
    # params, controller, worker and result validation remain the real ones.
    original_clock = world.clock
    world.clock = lambda: world.now
    world.runner.clock = world.clock
    assert full.dedup_enabled and full.train_seed_scope == "round" and full.min_step_norm == 0.0
    assert full.min_signal_rollouts == 1 and full.z_bar == 1.5 and full.eval_merged
    # Stage A plus no-result survivor accounting: three seats, one charged copy,
    # two uncharged genuine seats returned to the queue after a failed lease.
    for uid, quality in ((1, 0.9), (2, 0.0), (3, 0.0)):
        world.register(uid, quality)
    raws = {
        "hk1": affine_batch("hk1", [(f"answer-{i}", float(i % 2)) for i in range(128)]),
        "hk2": affine_batch("hk2", [(f" answer-{i} ", float(i % 2)) for i in range(128)]),
        "hk3": affine_batch("hk3", [(f"independent-{i}", float(i % 2)) for i in range(128)]),
    }
    world.admission._read = sim_read(raws, {"hk1": 0.9, "hk2": 0.0, "hk3": 0.0})
    world.sim_qualities = {sha256_hex(raw): quality for raw, quality in zip(raws.values(), (0.9, 0.0, 0.0))}
    tickets = world.controller.open_round(world.clock(), "failed-lease-fixture")
    ticket = tickets[0]
    world.controller.fail_lease(ticket.lease_id, "synthetic box outage", world.clock())
    failed = world.controller.state.rounds[-1]
    assert world.controller.open_round(world.clock(), "backoff-probe") == []
    # SimWorld's synthetic clock advances once per call. Production's 2400 s
    # training budget otherwise expires before the simulated trainer starts.
    # Only that simulated clock is held during worker execution; the journal,
    # params, controller, worker and result validation remain the real ones.
    world.clock = lambda: world.now
    world.runner.clock = world.clock
    assert [row.status for row in failed.decision.entries] == ["no_decision", "duplicate", "no_decision"]
    assert failed.duplicate_decisions and failed.first_seen and failed.duplicate_clusters
    assert set(world.controller.state.queue) == {(1, 1001), (3, 1003)}
    assert world.controller.state.failed_leases == 1

    # Advance only the simulated wall clock past the lease-failure backoff.
    # The next ticket receives the unmodified production training budget.
    world.now += full.apply_retry_s + 1
    # SimWorld's old runtime-LoRA result cannot meet merged-mode validation.
    # Supply only the tested-base proof and the matching simulated install fold;
    # training, scoring, judging and journal promotion still use the real code.
    from sn120_v2.gpu import merged as merged_module
    from sn120_v2.headchain import FIXED_ORDER
    from sn120_v2.types import FoldArtifact

    def proof_files(entry):
        assert entry.status == "scored" and entry.lora is not None
        return {"model.bin": sha256_hex(f"tested base of {entry.lora.lora_hash}".encode())}

    def simulated_proof(entry):
        if entry.status != "scored" or not full.eval_merged:
            return entry
        return dataclasses.replace(
            entry,
            tested_base_hash=manifest_digest(proof_files(entry)),
            tested_fold_kernel=FIXED_ORDER,
        )

    original_result = merged_module.assert_tested_result

    def require_fixture_proof(params, entry):
        original_result(params, entry)
        if params.eval_merged and entry.status == "scored":
            assert entry.tested_base_hash == manifest_digest(proof_files(entry))
            assert entry.tested_fold_kernel == FIXED_ORDER

    merged_module.assert_tested_result = require_fixture_proof

    def fold_merged(head, params, out_prefix, tested=None, **kwargs):
        if not params.eval_merged:
            return world.folder.__class__.fold(world.folder, head, params, out_prefix)
        assert tested is not None and len(head.loras) == 1
        assert head.loras[0].lora_hash == tested.lora.lora_hash
        base = f"tested base of {tested.lora.lora_hash}".encode()
        master = f"tested master of {tested.lora.lora_hash}".encode()
        world.store.put(f"{out_prefix}/base/model.bin", base)
        world.store.put(f"{out_prefix}/master/master.bin", master)
        base_files = {"model.bin": sha256_hex(base)}
        master_files = {"master.bin": sha256_hex(master)}
        fold = FoldArtifact(
            head.version_id,
            manifest_digest(base_files), base_files,
            manifest_digest(master_files), master_files,
            out_prefix, SIM_IMAGE,
        )
        merged_module.assert_fold_matches(params, tested, fold.base_hash)
        return fold

    world.folder.fold = fold_merged
    real_run = world.runner.run

    def merged_run(ticket):
        result = real_run(ticket)
        result = dataclasses.replace(
            result, entries=tuple(simulated_proof(entry) for entry in result.entries)
        )
        for entry in result.entries:
            merged_module.assert_tested_result(ticket.params, entry)
        return result

    world.runner.run = merged_run
    real_submit = world.controller.submit_result

    def merged_submit(result, at):
        # Validate the simulated worker proof at the real sealed-result boundary.
        for entry in result.entries:
            merged_module.assert_tested_result(full, entry)
        return real_submit(result, at)

    world.controller.submit_result = merged_submit

    def king_train(request):
        return synthetic_lora(world, real_affine_train(request), 0.2, SIM_IMAGE)

    try:
        world.trainer.train = king_train
        with simulated_worker_identity():
            king = world.round()
    finally:
        world.trainer.train = real_affine_train
    assert king and king.outcome == "applied"
    winner_raw = world.store.get(layout.winner_batch(king.round_id))
    world.admission._read = lambda hotkey: (winner_raw, {"valid": True, "quality": 0.0, "cap_rate": 0.0, "train": {}})
    world.register(4, None)
    world.register(5, None)
    world.admission._read = lambda hotkey: (winner_raw, {"valid": True, "quality": 0.0, "cap_rate": 0.0, "train": {}})
    world.sim_qualities[sha256_hex(winner_raw)] = 0.0
    duplicate_only = world.round()
    assert duplicate_only and duplicate_only.leases == ()
    assert all(row.status == "duplicate" for row in duplicate_only.decision.entries)
    assert not world.controller.state.queue

    # A settled null update with a tagged non-finite norm and a real Stage-B
    # duplicate. Both charged statuses and both evidence forms reach the journal.
    params = dataclasses.replace(full, fold_every_k=1)
    assert params.params_hash == full.params_hash
    # The old rollback is tested against the latest historical reviewed pin.
    # No simulation-specific parameter record is introduced after it.
    for uid, quality in ((6, 0.4), (7, 0.3), (8, 0.2)):
        world.register(uid, quality)
    for index, hotkey in enumerate(("hk6", "hk7", "hk8")):
        world.admission._read = lambda hotkey: valid_affine_read(world, hotkey)
        label = {0: "null", 1: "winner", 2: "copy"}[index]
        # The last two independent batch files must also produce the same
        # update direction; this isolates Stage-B LoRA evidence from Stage A.
        rows = [(f"independent-{label}-{i}", float(i % 2)) for i in range(128)]
        raw = affine_batch(hotkey, rows)
        world.store.put(layout.miner_object(hotkey), raw)
        world.sim_qualities[sha256_hex(raw)] = 0.4 if index == 1 else 0.2
    # Invalidate cached miner-file raw bytes left by the past-king duplicate
    # fixture. All three stage-B entries are independent at Stage A.
    world.admission._read = lambda hotkey: valid_affine_read(world, hotkey)
    # Call the explicit base trainer, not the bound method already wrapped for
    # the king. The real affine file remains the frozen batch in the store.
    affine_metadata.__exit__(None, None, None)

    def base_train(request):
        # Supply only the simulator's quality sidecar for this unwrapped call;
        # the real affine batch bytes and their hash stay unchanged.
        real_read = world.store.get
        quality = world.sim_qualities.get(request.entry.batch.batch_sha256, 0.0)
        sidecar = canonical_json({"quality": quality, "cap_rate": 0.0,
                                  "train": request.entry.batch.train})
        world.store.get = lambda key: sidecar if key == request.entry.batch.uri else real_read(key)
        try:
            artifact = sim_train(world.trainer, request)
        finally:
            world.store.get = real_read
        return artifact

    def train(request):
        # Match the two comparable seats' seed as well as their direction. The
        # first seat is excluded as null_update and never supplies an anchor.
        artifact = base_train(request)
        if request.entry.queue_pos == 2:
            record = world.controller.state.round
            entry = record.leases[0].entries[1]
            seed = train_seed(record.round_seed, entry.batch.batch_sha256, full.train_seed_scope)
            artifact = dataclasses.replace(artifact, report=dataclasses.replace(artifact.report, seed=seed))
        artifact = synthetic_lora(world, artifact, 2.0 + request.entry.queue_pos, SIM_IMAGE)
        if request.entry.queue_pos == 0:
            artifact = dataclasses.replace(artifact, report=dataclasses.replace(
                artifact.report, step_norm=float("nan"), norm_before_cap=float("inf")))
        return artifact

    world.trainer.train = train
    # Normal close deletes the round's pending prefix. Preserve an independent
    # byte-for-byte evidence archive at submit_result, without changing close.
    from sn120_v2.store import put_record
    archive_base = merged_submit

    def archive_result(result, at):
        put_record(world.store, f"qualification/sealed/{result.round_id}/{result.lease_id}.json", result)
        return archive_base(result, at)
    world.controller.submit_result = archive_result
    with simulated_worker_identity():
        settled = world.round()
        assert settled, "synthetic settled round did not close"
        actual_statuses = [row.status for row in settled.decision.entries]
        assert actual_statuses == ["null_update", "winner", "duplicate"], {
            "round": settled.round_id, "outcome": settled.outcome, "statuses": actual_statuses,
            "entries": [row.to_dict() for row in settled.decision.entries],
        }
        assert settled.decision.duplicate_clusters, {
            "statuses": actual_statuses,
            "decision_clusters": settled.decision.duplicate_clusters,
            "round_clusters": settled.duplicate_clusters,
        }
        # The result is still open pending winner promotion after run_round.
        if world.controller.state.round is not None:
            world.controller.tick(world.clock())
    settled = world.controller.state.rounds[-1]
    assert settled.outcome == "applied"
    lease = settled.leases[0]
    sealed = world.store.get(f"qualification/sealed/{settled.round_id}/{lease.lease_id}.json")
    assert b'"step_norm_nonfinite":"nan"' in sealed and b'"norm_before_cap_nonfinite":"inf"' in sealed
    # Additional old-code-only proof inputs: a valid open round on the current
    # head, then a decided-but-unclosed copy. Neither is ever run under old code.
    from sn120_v2.journal import Journal, read_journal
    from sn120_v2.types import RoundTicket
    from sn120_v2.store import LocalStore, get_record, put_record
    import shutil
    open_root = root / "open-undecided"
    open_root.mkdir()
    shutil.copyfile(root / "journal.jsonl", open_root / "journal.jsonl")
    probe = SimWorld(open_root, sim_params())
    for uid in (9, 10):
        probe.register(uid, 0.2)
    tickets = probe.controller.open_round(probe.clock(), "rollback-open-guard")
    assert tickets and probe.controller.state.round.params_hash == full.params_hash
    decided_root = root / "open-decided"
    decided_root.mkdir()
    shutil.copyfile(open_root / "journal.jsonl", decided_root / "journal.jsonl")
    prior_settlement = next(row for row in read_journal(root / "journal.jsonl")
                            if row.kind == "round_settled" and row.data.get("results"))
    writer = Journal(decided_root / "journal.jsonl", "synthetic-open-decision", probe.clock())
    for ticket in tickets:
        writer.append("lease_failed", {"round_id": ticket.round_id, "lease_id": ticket.lease_id,
                                       "reason": "synthetic decided-round marker"}, probe.clock())
    writer.append("round_settled", {"round_id": tickets[0].round_id,
                                    "decision": prior_settlement.data["decision"], "results": {}}, probe.clock())
    for name in ("open-undecided", "open-decided"):
        directory = root / name
        for ticket in tickets:
            put_record(LocalStore(directory), layout.ticket(ticket.round_id, ticket.lease_id),
                       get_record(probe.store, layout.ticket(ticket.round_id, ticket.lease_id), RoundTicket))
    summary = {
        "journal": str(root / "journal.jsonl"),
        "journal_sha256": sha256_hex((root / "journal.jsonl").read_bytes()),
        "records": world.controller.state.last_seq + 1,
        "failed_lease": failed.round_id,
        "duplicate_only": duplicate_only.round_id,
        "settled_null_stage_b": settled.round_id,
        "queue": sorted(world.controller.state.queue),
        "params_hash": world.controller.state.params.params_hash,
        "latest_new_params": params.to_dict(),
        "synthetic": True,
    }
    (root / "fixture.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
