from nacl.signing import SigningKey

from affine import Sample, build_block, verify_block, verify_chain


def sample_fixture() -> Sample:
    sample = Sample(
        version=1,
        env_id="mult8-v0",
        env_spec_version=1,
        challenge_id="abcd",
        validator="validator",
        miner_id="1",
        role="contender",
        request_id=None,
        prompt="Compute 11 × 11.",
        response="121",
        info={"expected": 121},
        ok=True,
        reason="win",
        timing_ms=10,
        bytes=3,
    )
    sample.compute_hash()
    return sample


def test_build_and_verify_block() -> None:
    signing_key = SigningKey.generate()
    sample = sample_fixture()
    block, digest = build_block(
        [sample],
        validator="validator",
        block_index=0,
        prev_hash="0" * 64,
        env_spec_versions={"mult8-v0": 1},
        signing_key=signing_key,
    )
    assert block.header.sample_count == 1
    assert verify_block(block, signing_key.verify_key)


def test_verify_block_detects_tampering() -> None:
    signing_key = SigningKey.generate()
    sample = sample_fixture()
    block, _ = build_block(
        [sample],
        validator="validator",
        block_index=0,
        prev_hash="0" * 64,
        env_spec_versions={"mult8-v0": 1},
        signing_key=signing_key,
    )
    block.samples[0].response = "999"  # mutate payload
    assert not verify_block(block, signing_key.verify_key)


def test_verify_chain_detects_breaks() -> None:
    signing_key = SigningKey.generate()
    sample = sample_fixture()
    block1, digest1 = build_block(
        [sample],
        validator="validator",
        block_index=0,
        prev_hash="0" * 64,
        env_spec_versions={"mult8-v0": 1},
        signing_key=signing_key,
    )
    sample_new = sample_fixture()
    sample_new.challenge_id = "dcba"
    sample_new.compute_hash()
    block2, _ = build_block(
        [sample_new],
        validator="validator",
        block_index=1,
        prev_hash=digest1,
        env_spec_versions={"mult8-v0": 1},
        signing_key=signing_key,
    )
    assert verify_chain(
        [block1, block2],
        {"validator": signing_key.verify_key},
        genesis_hash="0" * 64,
    )
    block2.header.prev_hash = "deadbeef"
    assert not verify_chain(
        [block1, block2],
        {"validator": signing_key.verify_key},
        genesis_hash="0" * 64,
    )
