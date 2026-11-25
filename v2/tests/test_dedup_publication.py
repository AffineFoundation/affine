"""The generated miner contract covers duplicate, null-update and training-seed rules."""

from __future__ import annotations

import dataclasses
import json

from sn120_v2 import public, publisher, sim
from sn120_v2.admission import RULES
from sn120_v2.dedup import STAGE_EXACT, STAGE_LORA
from sn120_v2.params import Params, Recipe

SITE = "https://files.example"
NOW = 1_800_000_000


def published_pair(tmp_path, **changes):
    world = sim.SimWorld(tmp_path / "world", sim.sim_params())
    defaults = dict(
        train_seed_scope="round",
        dedup_enabled=True,
        dedup_rollout_overlap=0.5,
        dedup_cos_shared_seed=0.135,
        dedup_cos_own_seed=0.002,
        min_signal_rollouts=64,
        min_step_norm=0.02,
        z_bar=1.5,
        eval_merged=True,
    )
    params = dataclasses.replace(
        Params.from_dict(world.params.to_dict() | defaults), **changes
    )
    snapshot = public.snapshot(world.controller.state, params, NOW)
    mining, llms = publisher.render(snapshot, SITE)
    return params, json.loads(mining), llms.decode(), mining


def flat(text: str) -> str:
    return " ".join(text.split())


def test_duplicate_and_null_publication_and_check_pair(tmp_path):
    params, doc, text, mining = published_pair(tmp_path)
    sentences = flat(text)

    assert text.count("### Duplicate and null batches\n") == 1
    section = text.split("### Duplicate and null batches\n", 1)[1].split("### How to upload", 1)[0]
    required = [
        "Overlap >= 0.5 (`dedup_rollout_overlap`)",
        "strip its ends, collapse each whitespace run to one space, and ignore rewards",
        "intersection size / the size of the smaller set",
        "greedy priority is (first_seen, queue_pos)",
        "past winner always keeps its position",
        "Reusing a past winner's rollouts counts as a duplicate even from the same hotkey: a king resubmitting >= dedup_rollout_overlap of its own already-applied rollouts is dropped as `past_king_rollout`, or `past_king_exact` for the same batch bytes.",
        "cosine >= 0.135 (`dedup_cos_shared_seed`)",
        "threshold is 0.002 (`dedup_cos_own_seed`)",
        "Regenerated or rewritten rollouts are new evidence, not an exemption",
        "than 64 rollouts with a nonzero advantage",
        "The configured minimum signal is 64; `minibatches` adds no admission floor.",
        "With fewer active rows than requested minibatches, training makes fewer nonempty minibatches rather than rejecting the file.",
        "after `no_signal`",
        "invalid and uncharged: the seat keeps its place",
        "stored `step_norm` below 0.02, or one that is not finite",
        "charged, cannot win or be paid",
        "before any training GPU work",
        "cannot beat M except by luck",
        "`queue_pos`, `uid`, `hotkey`, `batch_sha256`, `stage`, `similarity` and `first_seen_unix`",
        "All published times are UTC",
    ]
    assert all(phrase in flat(section) for phrase in required)
    assert "never includes batch content" in sentences.lower()
    assert "prompts, outputs, rewards or the normalized rollout hashes" in flat(section)
    assert RULES["min_rollouts"] in text and RULES["null_update"] in text
    assert "| `min_rollouts` | " in text and "| `null_update` | " in text
    assert "- `duplicate` and `null_update`: your frozen batch was a duplicate" in sentences
    assert "If Stage A already found duplicates, those copies remain charged `duplicate`" in sentences
    assert "every other seat stays uncharged and goes to the front, and the same retry delay applies" in sentences

    policy = doc["contract"]["duplicate_policy"]
    assert policy["stage_a"]["rollout_overlap_min"] == params.dedup_rollout_overlap
    assert policy["stage_b"]["shared_verified_train_seed_min"] == params.dedup_cos_shared_seed
    assert policy["stage_b"]["own_or_mixed_train_seed_min"] == params.dedup_cos_own_seed
    assert policy["public_cluster_member_fields"] == list(publisher.DUPLICATE_MEMBER_COLUMNS)
    assert policy["stages"] == publisher.DUPLICATE_STAGES
    assert doc["params_hash"] == params.params_hash == doc["contract"]["params_hash"]
    assert doc["params"]["z_bar"] == 1.5 and doc["params"]["eval_merged"] is True
    assert "Win rule: z is strictly above 1.5" in sentences
    assert "dropped when (z + c) / sqrt(t) <= 1.5" in sentences
    assert "Merged-candidate evaluation: Each candidate is tested as the merged model round_bf16(fp32 king master + your LoRA update)" in sentences
    assert doc["contract"]["candidate_evaluation"]["tested_equals_installed"] is True
    assert all(name in doc["params"] for name in (
        "train_seed_scope", "dedup_enabled", "dedup_rollout_overlap", "dedup_cos_shared_seed",
        "dedup_cos_own_seed", "min_signal_rollouts", "min_step_norm", "z_bar", "eval_merged",
    ))
    assert publisher.check_pair(mining, text.encode(), params) == []


def test_floor_free_publication_states_only_rules_that_are_on(tmp_path):
    params, doc, text, mining = published_pair(tmp_path, min_signal_rollouts=1, min_step_norm=0.0)
    section = flat(text.split("### Duplicate and null batches\n", 1)[1].split("### How to upload", 1)[0])
    signal = next(line for line in section.split("- ") if line.startswith("Minimum signal:")).strip()
    stored = next(line for line in section.split("- ") if line.startswith("Stored null update:")).strip()
    assert signal == (
        "Minimum signal: after `no_signal` rejects a file in which nothing trains, `min_rollouts` rejects a file with fewer "
        "than 1 rollouts with a nonzero advantage and at least one usable output token inside the training caps. The configured minimum "
        "signal is 1; `minibatches` adds no admission floor. With fewer active rows than requested minibatches, training makes fewer "
        "nonempty minibatches rather than rejecting the file. Without tokenizer checks no rollout is known to be dropped by the training "
        "caps. This admission rejection is invalid and uncharged: the seat keeps its place, and no training GPU is used. Upload a new "
        "file with enough signal rather than equal-reward filler, whose zero-advantage rows do not count."
    )
    assert stored == (
        "Stored null update: after training, a stored `step_norm` that is not finite is terminal `null_update`. It is charged, "
        "cannot win or be paid, and is not a candidate or keeper in Stage B. There is no minimum norm."
    )
    for phrase in ("64", "0.02", "sub-floor", "cannot beat M except by luck", "Static null-update admission:", "max("):
        assert phrase not in section
    assert "There is no minimum norm." in stored
    assert "There is no minimum norm, so every valid nonnegative cap passes this static floor check." in flat(text)
    assert "Values below 0 are invalid" not in text and "stored update fell below the floor" not in text
    assert "or its stored `step_norm` was not finite" in flat(text)
    assert "Reusing a past winner's rollouts counts as a duplicate even from the same hotkey" in section
    assert "Seed (`train_seed_scope` is `round`)" in flat(text)
    policy = doc["contract"]["duplicate_policy"]
    assert policy["min_signal_rollouts"] == "min_signal_rollouts; no implicit effective recipe.minibatches floor"
    assert policy["min_step_norm"] == 0.0
    assert policy["min_step_norm_rule"] == "no minimum norm; only a non-finite stored step_norm is null_update"
    assert publisher.check_pair(mining, text.encode(), params) == []


def test_floored_publication_retains_static_and_stored_floor_rules(tmp_path):
    params, doc, text, mining = published_pair(tmp_path)
    section = flat(text.split("### Duplicate and null batches\n", 1)[1].split("### How to upload", 1)[0])
    for phrase in (
        "The configured minimum signal is 64; `minibatches` adds no admission floor.",
        "Static null-update admission: `null_update` rejects `train.global_norm_cap` below 0.02 (`min_step_norm`)",
        "stored `step_norm` below 0.02, or one that is not finite",
        "The floor applies to the stored norm after any global cap",
        "Why a sub-floor update cannot beat M except by luck",
    ):
        assert phrase in section
    assert "Values below 0.02 are invalid (`null_update`, section 5)" in flat(text)
    assert doc["contract"]["duplicate_policy"]["min_step_norm_rule"] == (
        "static global_norm_cap and stored step_norm must be at least min_step_norm"
    )
    assert publisher.check_pair(mining, text.encode(), params) == []


def test_round_and_legacy_batch_seed_formulas(tmp_path):
    _, _, text, _ = published_pair(tmp_path)
    sentences = flat(text)
    assert 'Seed (`train_seed_scope` is `round`): the first 63 bits of sha256("<round_seed>|train"), as a whole number.' in sentences
    assert "every batch in the round shares one training seed" in sentences
    assert "every LoRA in the round starts from the same A" in sentences
    assert "different rows give different minibatch orders" in sentences
    assert "read entropy only after those files are frozen" in sentences
    assert "advance knowledge of the entropy, round seed or training seeds" in sentences

    _, _, legacy, _ = published_pair(tmp_path / "batch", train_seed_scope="batch")
    legacy = flat(legacy)
    assert 'Seed (`train_seed_scope` is `batch`): the first 63 bits of sha256("<round_seed>|train|<batch_sha256>"), as a whole number.' in legacy
    assert "same file with the same round seed gets the same A and the same minibatch orders" in legacy


def test_duplicate_cluster_evidence_is_in_mining_and_llms(tmp_path):
    params, doc, _, _ = published_pair(tmp_path)
    keeper = {
        "queue_pos": 2, "uid": 12, "hotkey": "hk12", "registration_block": 1012,
        "batch_sha256": "a" * 64, "first_seen_unix": NOW - 10, "stage": "", "similarity": 1.0,
    }
    dropped = {
        "queue_pos": 5, "uid": 15, "hotkey": "hk15", "registration_block": 1015,
        "batch_sha256": "b" * 64, "first_seen_unix": NOW, "stage": STAGE_EXACT,
        "similarity": 1.0, "keeper_queue_pos": 2,
    }
    lora_drop = dropped | {"queue_pos": 6, "uid": 16, "hotkey": "hk16", "batch_sha256": "c" * 64,
                           "stage": STAGE_LORA, "similarity": params.dedup_cos_shared_seed}
    cluster = {"keeper": keeper, "dropped": [dropped, lora_drop]}
    snapshot = public.snapshot(sim.SimWorld(tmp_path / "world", sim.sim_params()).controller.state, params, NOW)
    snapshot["round"] = {"round_id": "r000001", "duplicate_clusters": [cluster]}
    mining, llms = publisher.render(snapshot, SITE)
    rendered = llms.decode()
    assert json.loads(mining)["round"]["duplicate_clusters"] == [cluster]
    assert "Open round duplicate clusters (public evidence only; timestamps are UTC):" in rendered
    for member in (keeper, dropped, lora_drop):
        for key in publisher.DUPLICATE_MEMBER_COLUMNS:
            assert str(member[key]) in rendered
    assert "| 5 | 2 |" in rendered and "| 6 | 2 |" in rendered
    assert publisher.check_pair(mining, llms, params) == []


def test_publication_tracks_nondefault_dedup_params(tmp_path):
    changed = dict(dedup_rollout_overlap=0.75, dedup_cos_shared_seed=0.2, dedup_cos_own_seed=0.01,
                   min_signal_rollouts=80, min_step_norm=0.04)
    params, _, text, mining = published_pair(tmp_path, **changed)
    sentences = flat(text)
    assert "Overlap >= 0.75 (`dedup_rollout_overlap`)" in sentences
    assert "cosine >= 0.2 (`dedup_cos_shared_seed`)" in sentences
    assert "threshold is 0.01 (`dedup_cos_own_seed`)" in sentences
    assert "The configured minimum signal is 80; `minibatches` adds no admission floor." in sentences
    assert "stored `step_norm` below 0.04" in sentences
    assert "Win rule: z is strictly above 1.5" in sentences
    assert publisher.check_pair(mining, text.encode(), params) == []
