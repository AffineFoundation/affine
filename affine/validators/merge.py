from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, Optional, Tuple

from ..core.hashing import hash_bytes
from ..core.rng import derive_challenge_id
from ..core.types import Sample, Verdict
from ..envs import registry as env_registry


@dataclass
class VerifiedSample:
    sample: Sample
    verdict: Verdict
    valid: bool
    errors: Tuple[str, ...] = ()


def _load_response(sample: Sample):
    try:
        return json.loads(sample.response)
    except Exception:
        return sample.response


def _expected_challenge_id(sample: Sample, spec_hash: str) -> Optional[str]:
    counter = sample.info.get("counter")
    epoch_anchor = sample.info.get("epoch_anchor")
    if counter is None or epoch_anchor is None:
        return None
    try:
        counter_int = int(counter)
    except (TypeError, ValueError):
        return None
    anchor = str(epoch_anchor)
    seed_hex = sample.info.get("seed")
    if seed_hex:
        try:
            seed_bytes = bytes.fromhex(str(seed_hex))
        except ValueError:
            return None
        payload = b":".join(
            [
                seed_bytes,
                sample.validator.encode("utf-8"),
                sample.env_id.encode("utf-8"),
                str(counter_int).encode("utf-8"),
                anchor.encode("utf-8"),
                spec_hash.encode("utf-8"),
            ]
        )
        digest = hash_bytes(payload)
        return digest[:16].hex()
    return derive_challenge_id(sample.validator, sample.env_id, counter_int, anchor, spec_hash)


def verify_samples(samples: Iterable[Sample]) -> Tuple[Dict[Tuple[str, str], Dict[str, Dict[str, VerifiedSample]]], Dict[str, Tuple[int, int]]]:
    """Verify samples and bucket them by (env_id, challenge_id, validator)."""
    buckets: Dict[Tuple[str, str], Dict[str, Dict[str, VerifiedSample]]] = defaultdict(dict)
    validator_stats: Dict[str, Tuple[int, int]] = defaultdict(lambda: (0, 0))

    for sample in samples:
        env_cls = env_registry.get(sample.env_id)
        errors: list[str] = []

        declared_hash = sample.sample_hash
        computed_hash = sample.compute_hash()
        if declared_hash and declared_hash != computed_hash:
            errors.append("sample-hash-mismatch")

        if sample.env_spec_version != env_cls.spec_version():
            errors.append("spec-version-mismatch")

        expected_spec_hash = env_cls.spec_hash()
        info_spec_hash = str(sample.info.get("spec_hash", expected_spec_hash))
        if info_spec_hash != expected_spec_hash:
            errors.append("spec-hash-mismatch")

        expected_challenge = _expected_challenge_id(sample, info_spec_hash)
        if expected_challenge is not None and sample.challenge_id.lower() != expected_challenge.lower():
            errors.append("challenge-mismatch")

        response = _load_response(sample)
        verdict = env_cls().verify(response, sample.info)
        if verdict.ok != sample.ok or verdict.reason != sample.reason:
            errors.append("verdict-mismatch")

        if sample.role not in ("contender", "champion"):
            errors.append("invalid-role")

        valid = not errors
        wins, total = validator_stats[sample.validator]
        validator_stats[sample.validator] = (wins + (1 if valid else 0), total + 1)
        bucket = buckets[(sample.env_id, sample.challenge_id)].setdefault(sample.validator, {})
        bucket[sample.role] = VerifiedSample(sample=sample, verdict=verdict, valid=valid, errors=tuple(errors))

    return buckets, validator_stats


def decide_group(group: Dict[str, VerifiedSample]) -> Optional[str]:
    if "contender" not in group or "champion" not in group:
        return None
    contender = group["contender"].verdict
    champion = group["champion"].verdict
    score_map = {"win": 2, "draw": 1, "loss": 0}

    def score(verdict: Verdict) -> int:
        if verdict.ok and not verdict.reason:
            return 2
        return score_map.get(verdict.reason, 1 if verdict.ok else 0)

    cont_score = score(contender)
    champ_score = score(champion)
    if cont_score > champ_score:
        return "contender"
    if cont_score < champ_score:
        return "champion"
    return None


__all__ = ["VerifiedSample", "decide_group", "verify_samples"]
