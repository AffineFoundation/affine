from __future__ import annotations

import pytest

from affine.core.rng import derive_challenge_id
from affine.core.types import Sample, Verdict
from affine.envs.mult8 import Mult8Env
from affine.validators import merge, vtrust, weights
from affine.validators.merge import VerifiedSample
from affine.validators.sampler import ChallengeCommitment, DuplicateDetector


def _make_sample(miner_id: int, *, role: str, ok: bool, reason: str) -> Sample:
    sample = Sample(
        version=1,
        env_id="mult8-v0",
        env_spec_version=1,
        challenge_id="abcd",
        validator="validator",
        miner_id=str(miner_id),
        role=role,
        request_id=None,
        prompt="prompt",
        response="response",
        info={"expected": 1, "challenge_id": "abcd"},
        ok=ok,
        reason=reason,
        timing_ms=10,
        bytes=5,
    )
    sample.compute_hash()
    return sample


def _make_mult8_sample(challenge_id: str | None = None) -> Sample:
    env = Mult8Env()
    obs, info = env.reset(options={"challenge_id": "feedface"})
    spec_hash = env.spec_hash()
    counter = 0
    cid = challenge_id or derive_challenge_id("validator", env.env_id(), counter, "anchor", spec_hash)
    meta = dict(info)
    meta.update(
        {
            "spec_hash": spec_hash,
            "counter": counter,
            "epoch": 0,
            "epoch_anchor": "anchor",
        }
    )
    sample = Sample(
        version=1,
        env_id=env.env_id(),
        env_spec_version=env.spec_version(),
        challenge_id=cid,
        validator="validator",
        miner_id="1",
        role="contender",
        request_id=None,
        prompt=str(obs),
        response=str(meta["expected"]),
        info=meta,
        ok=True,
        reason="",
        timing_ms=10,
        bytes=len(str(meta["expected"])),
    )
    sample.compute_hash()
    return sample


def test_scoreboard_winner_takes_all() -> None:
    contender_sample = _make_sample(31, role="contender", ok=True, reason="")
    champion_sample = _make_sample(7, role="champion", ok=False, reason="loss")
    buckets = {
        ("mult8-v0", "abcd"): {
            "validator": {
                "contender": VerifiedSample(contender_sample, Verdict(True, ""), True),
                "champion": VerifiedSample(champion_sample, Verdict(False, "loss"), True),
            }
        }
    }
    vtrust_scores = {"validator": 0.8}
    scores = weights.scoreboard(buckets, vtrust_scores)
    assert scores == {31: 0.8}
    assert weights.winner_takes_all(scores) == {31: 1.0}


def test_compute_vtrust_uses_wilson_lower_bound() -> None:
    stats = {"validator": (9, 10), "novice": (0, 0)}
    scores = vtrust.compute_vtrust(stats, confidence=0.95)
    assert scores["validator"] < 1.0
    assert scores["novice"] == 0.0


def test_duplicate_detector_marks_duplicates() -> None:
    detector = DuplicateDetector(window_size=2)
    assert detector.check_and_mark("env", "cid", 1) is False
    assert detector.check_and_mark("env", "cid", 1) is True
    assert detector.check_and_mark("env", "cid2", 1) is False


def test_challenge_commitment_roundtrip() -> None:
    commitment = ChallengeCommitment("validator", "anchor")
    seed = "00" * 32
    digest = commitment.commit(3, seed)
    commitment.reveal(3, seed)
    spec_hash = Mult8Env.spec_hash()
    cid = commitment.challenge_id(3, Mult8Env.env_id(), 0, spec_hash)
    assert isinstance(cid, str)
    assert commitment.commitment(3) == digest
    assert commitment.revealed_seed(3) == bytes.fromhex(seed)


def test_verify_samples_detects_mismatch() -> None:
    valid_sample = _make_mult8_sample()
    buckets, stats = merge.verify_samples([valid_sample])
    assert stats["validator"] == (1, 1)
    verified = buckets[(valid_sample.env_id, valid_sample.challenge_id)]["validator"]["contender"]
    assert verified.valid
    tampered = _make_mult8_sample()
    tampered.challenge_id = "b" * len(tampered.challenge_id)
    tampered.compute_hash()
    buckets_bad, stats_bad = merge.verify_samples([tampered])
    assert stats_bad["validator"] == (0, 1)
    bad_verified = buckets_bad[(tampered.env_id, tampered.challenge_id)]["validator"]["contender"]
    assert not bad_verified.valid
    assert "challenge-mismatch" in bad_verified.errors
