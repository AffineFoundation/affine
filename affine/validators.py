from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from nacl import signing
from nacl.exceptions import BadSignatureError

from .core import (
    Block,
    BlockHeader,
    Sample,
    Verdict,
    canonical_bytes,
    canonical_timestamp,
    derive_challenge_id,
    hash_bytes,
    hash_hex,
    merkle_root,
)
from .duel import wilson_interval
from .envs import AffineEnv, env_names, get_env
from .network import ChutesClient


def _jsonify(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(value)
    if isinstance(value, (np.floating, np.float64, np.float32, np.float16)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


class ChallengeCommitment:
    def __init__(self, validator_hotkey: str, epoch_anchor: str) -> None:
        self.validator_hotkey = validator_hotkey
        self.epoch_anchor = epoch_anchor
        self._commits: Dict[int, str] = {}
        self._seeds: Dict[int, bytes] = {}

    def _normalize_seed(self, seed: bytes | str) -> bytes:
        if isinstance(seed, bytes):
            return seed
        try:
            return bytes.fromhex(seed)
        except ValueError:
            return seed.encode("utf-8")

    def commit(self, epoch: int, seed: bytes | str) -> str:
        material = self._normalize_seed(seed)
        digest = hash_hex(material)
        self._commits[epoch] = digest
        return digest

    def reveal(self, epoch: int, seed: bytes | str) -> None:
        material = self._normalize_seed(seed)
        digest = hash_hex(material)
        expected = self._commits.get(epoch)
        if expected is not None and expected != digest:
            raise ValueError("revealed seed does not match prior commitment.")
        self._commits[epoch] = digest
        self._seeds[epoch] = material

    def revealed_seed(self, epoch: int) -> Optional[bytes]:
        return self._seeds.get(epoch)

    def commitment(self, epoch: int) -> Optional[str]:
        return self._commits.get(epoch)

    def challenge_id(
        self, epoch: int, env_id: str, counter: int, spec_hash: str, *, size: int = 16
    ) -> str:
        try:
            seed = self._seeds[epoch]
        except KeyError as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"seed for epoch {epoch} has not been revealed") from exc
        payload = b":".join(
            [
                seed,
                self.validator_hotkey.encode("utf-8"),
                env_id.encode("utf-8"),
                str(counter).encode("utf-8"),
                self.epoch_anchor.encode("utf-8"),
                spec_hash.encode("utf-8"),
            ]
        )
        digest = hash_bytes(payload)
        size = max(8, min(size, len(digest)))
        return digest[:size].hex()


class DuplicateDetector:
    def __init__(self, window_size: int = 10_000) -> None:
        self._window = max(1, window_size)
        self._entries: Dict[tuple[str, str, int], float] = {}

    def check_and_mark(self, env_id: str, challenge_id: str, miner_uid: int) -> bool:
        key = (env_id, challenge_id, int(miner_uid))
        now = time.time()
        if key in self._entries:
            self._entries[key] = now
            return True
        self._entries[key] = now
        if len(self._entries) > self._window:
            self._prune()
        return False

    def _prune(self) -> None:
        ordered = sorted(self._entries.items(), key=lambda item: item[1])
        self._entries = dict(ordered[-self._window :])


def _load_response(sample: Sample):
    try:
        return json.loads(sample.response)
    except Exception:
        return sample.response


def _expected_challenge_id(sample: Sample, spec_hash_value: str) -> Optional[str]:
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
                spec_hash_value.encode("utf-8"),
            ]
        )
        digest = hash_bytes(payload)
        return digest[:16].hex()
    return derive_challenge_id(
        sample.validator, sample.env_id, counter_int, anchor, spec_hash_value
    )


@dataclass
class MinerOutcome:
    uid: int
    prompt: str
    response: str
    info: Mapping[str, Any]
    verdict: Verdict
    latency_ms: int
    bytes: int
    request_id: Optional[str]
    transcript: List[Dict[str, Any]]


@dataclass
class ChallengeOutcome:
    env_id: str
    env_spec_version: int
    env_spec_hash: str
    challenge_id: str
    counter: int
    info: Mapping[str, Any]
    contender: MinerOutcome
    champion: MinerOutcome
    winner: Optional[str]

    def as_samples(self, validator: str) -> List[Sample]:
        return [
            self._to_sample(self.champion, validator, "champion"),
            self._to_sample(self.contender, validator, "contender"),
        ]

    def _to_sample(self, outcome: MinerOutcome, validator: str, role: str) -> Sample:
        merged_info: Dict[str, Any] = dict(self.info)
        merged_info.update({"transcript": outcome.transcript})
        if isinstance(outcome.info, Mapping):
            merged_info.update(outcome.info)
        sample = Sample(
            version=1,
            env_id=self.env_id,
            env_spec_version=self.env_spec_version,
            challenge_id=self.challenge_id,
            validator=validator,
            miner_id=str(outcome.uid),
            role=role,
            request_id=outcome.request_id,
            prompt=outcome.prompt,
            response=outcome.response,
            info=merged_info,
            ok=outcome.verdict.ok,
            reason=outcome.verdict.reason,
            timing_ms=outcome.latency_ms,
            bytes=outcome.bytes,
        )
        sample.compute_hash()
        return sample


class ValidatorSampler:
    def __init__(
        self,
        validator_hotkey: str,
        chutes: ChutesClient,
        *,
        epoch_anchor: str,
        env_ids: Sequence[str] | None = None,
        epoch: int = 0,
        commitment: ChallengeCommitment | None = None,
        duplicate_detector: DuplicateDetector | None = None,
    ) -> None:
        self.validator_hotkey = validator_hotkey
        self.chutes = chutes
        self.epoch_anchor = epoch_anchor
        self.env_ids = list(env_ids or env_names())
        self.epoch = int(epoch)
        self._commitment = commitment
        self._commit_digest = commitment.commitment(self.epoch) if commitment else None
        self._duplicates = duplicate_detector or DuplicateDetector()
        self._counters: Dict[str, int] = {env_id: 0 for env_id in self.env_ids}
        self._env_spec_versions: Dict[str, int] = {}
        self._env_spec_hashes: Dict[str, str] = {}
        for env_id in self.env_ids:
            env_cls = get_env(env_id)
            self._env_spec_versions[env_id] = env_cls.spec_version()
            self._env_spec_hashes[env_id] = env_cls.spec_hash()

    def env_spec_versions(self) -> Mapping[str, int]:
        return dict(self._env_spec_versions)

    def sample(
        self,
        env_id: str,
        *,
        champion_uid: int,
        contender_uid: int,
        timeout: float = 30.0,
    ) -> ChallengeOutcome:
        counter, challenge_id = self._next_challenge(env_id)
        env_cls = get_env(env_id)
        champion = self._run_episode(
            env_cls, env_id, champion_uid, challenge_id, timeout
        )
        contender = self._run_episode(
            env_cls, env_id, contender_uid, challenge_id, timeout
        )
        winner = self._decide(contender.verdict, champion.verdict)
        info: Dict[str, Any] = {
            "env_id": env_id,
            "challenge_id": challenge_id,
            "spec_version": self._env_spec_versions[env_id],
            "spec_hash": self._env_spec_hashes[env_id],
            "counter": counter,
            "epoch": self.epoch,
            "epoch_anchor": self.epoch_anchor,
            "commitment": self._commit_digest,
        }
        if self._commitment is not None:
            seed_bytes = self._commitment.revealed_seed(self.epoch)
            if seed_bytes is not None:
                info["seed"] = seed_bytes.hex()
        return ChallengeOutcome(
            env_id=env_id,
            env_spec_version=self._env_spec_versions[env_id],
            env_spec_hash=self._env_spec_hashes[env_id],
            challenge_id=challenge_id,
            counter=counter,
            info=info,
            contender=contender,
            champion=champion,
            winner=winner,
        )

    def _next_challenge(self, env_id: str) -> tuple[int, str]:
        counter = self._counters[env_id]
        self._counters[env_id] = counter + 1
        spec_hash_value = self._env_spec_hashes[env_id]
        if self._commitment is not None:
            challenge_id = self._commitment.challenge_id(
                self.epoch, env_id, counter, spec_hash_value
            )
        else:
            challenge_id = derive_challenge_id(
                self.validator_hotkey,
                env_id,
                counter,
                self.epoch_anchor,
                spec_hash_value,
            )
        return counter, challenge_id

    def _run_episode(
        self,
        env_cls: type[AffineEnv],
        env_id: str,
        uid: int,
        challenge_id: str,
        timeout: float,
    ) -> MinerOutcome:
        env = env_cls()
        env_name = env_cls.env_id()
        if self._duplicates.check_and_mark(env_name, challenge_id, uid):
            raise RuntimeError(
                f"duplicate sample for miner {uid} on challenge {challenge_id}"
            )
        observation, info = env.reset(options={"challenge_id": challenge_id})
        prompt = (
            observation
            if isinstance(observation, str)
            else (
                observation.decode()
                if isinstance(observation, (bytes, bytearray))
                else json.dumps(_jsonify(observation), ensure_ascii=False)
            )
        )
        transcript: List[Dict[str, Any]] = []
        miner_moves: List[int] = []
        total_latency = 0
        total_bytes = 0
        last_request_id: Optional[str] = None
        final_info: Mapping[str, Any] = info

        try:
            while True:
                payload = {
                    "env_id": env_name,
                    "challenge_id": challenge_id,
                    "observation": _jsonify(observation),
                    "info": _jsonify(info),
                }
                response = self.chutes.invoke(uid, payload, timeout=timeout)
                total_latency += response.latency_ms
                encoded = response.text.encode("utf-8")
                total_bytes += len(encoded)
                last_request_id = response.request_id
                transcript.append(
                    {
                        "role": "validator",
                        "observation": _jsonify(observation),
                        "info": _jsonify(info),
                        "response": response.text,
                        "latency_ms": response.latency_ms,
                        "request_id": response.request_id,
                    }
                )
                action = env.decode_action(response.text)
                if isinstance(action, int):
                    miner_moves.append(action)
                observation, reward, terminated, truncated, info = env.step(action)
                final_info = info
                transcript.append(
                    {
                        "role": "env",
                        "observation": _jsonify(observation),
                        "info": _jsonify(info),
                        "reward": reward,
                        "terminated": terminated,
                        "truncated": truncated,
                    }
                )
                if terminated or truncated:
                    break
        finally:
            env.close()

        verification_payload: Any = miner_moves if miner_moves else transcript
        verdict = env.verify(verification_payload, final_info)
        response_blob = json.dumps(
            {"moves": miner_moves, "transcript": transcript}, ensure_ascii=False
        )
        return MinerOutcome(
            uid=uid,
            prompt=prompt,
            response=response_blob,
            info=final_info,
            verdict=verdict,
            latency_ms=total_latency,
            bytes=total_bytes,
            request_id=last_request_id,
            transcript=transcript,
        )

    @staticmethod
    def _decide(contender: Verdict, champion: Verdict) -> Optional[str]:
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


@dataclass
class VerifiedSample:
    sample: Sample
    verdict: Verdict
    valid: bool
    errors: Tuple[str, ...] = ()


def verify_samples(
    samples: Iterable[Sample],
) -> Tuple[
    Dict[Tuple[str, str], Dict[str, Dict[str, VerifiedSample]]],
    Dict[str, Tuple[int, int]],
]:
    buckets: Dict[Tuple[str, str], Dict[str, Dict[str, VerifiedSample]]] = {}
    validator_stats: Dict[str, Tuple[int, int]] = {}

    for sample in samples:
        env_cls = get_env(sample.env_id)
        expected_spec_version = env_cls.spec_version()
        expected_spec_hash = env_cls.spec_hash()
        info_spec_version = int(sample.info.get("spec_version", expected_spec_version))
        info_spec_hash = str(sample.info.get("spec_hash", expected_spec_hash))
        errors: List[str] = []

        if info_spec_version != expected_spec_version:
            errors.append("spec-version-mismatch")
        if info_spec_hash != expected_spec_hash:
            errors.append("spec-hash-mismatch")

        expected_challenge = _expected_challenge_id(sample, info_spec_hash)
        if (
            expected_challenge is not None
            and sample.challenge_id.lower() != expected_challenge.lower()
        ):
            errors.append("challenge-mismatch")

        response = _load_response(sample)
        verdict = env_cls().verify(response, sample.info)
        if verdict.ok != sample.ok or verdict.reason != sample.reason:
            errors.append("verdict-mismatch")

        if sample.role not in ("contender", "champion"):
            errors.append("invalid-role")

        valid = not errors
        wins, total = validator_stats.get(sample.validator, (0, 0))
        validator_stats[sample.validator] = (wins + (1 if valid else 0), total + 1)
        bucket = buckets.setdefault(
            (sample.env_id, sample.challenge_id), {}
        ).setdefault(sample.validator, {})
        bucket[sample.role] = VerifiedSample(
            sample=sample, verdict=verdict, valid=valid, errors=tuple(errors)
        )
    return buckets, validator_stats


def decide_group(group: Mapping[str, VerifiedSample]) -> Optional[str]:
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


def compute_vtrust(
    stats: Mapping[str, Tuple[int, int]], *, confidence: float = 0.95
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    for validator, (wins, total) in stats.items():
        if total == 0:
            scores[validator] = 0.0
            continue
        lower, _ = wilson_interval(wins, total, confidence)
        scores[validator] = lower
    return scores


def scoreboard(
    buckets: Mapping[tuple[str, str], Mapping[str, Mapping[str, VerifiedSample]]],
    vtrust: Mapping[str, float],
) -> Dict[int, float]:
    scores: Dict[int, float] = {}
    for (_env_id, _challenge_id), validators in buckets.items():
        for validator, roles in validators.items():
            if "contender" not in roles or "champion" not in roles:
                continue
            if not roles["contender"].valid or not roles["champion"].valid:
                continue
            winner = decide_group(roles)
            if winner is None:
                continue
            weight = vtrust.get(validator, 0.0)
            if weight <= 0.0:
                continue
            sample = roles[winner].sample
            try:
                uid = int(sample.miner_id)
            except ValueError:
                continue
            scores[uid] = scores.get(uid, 0.0) + weight
    return scores


def winner_takes_all(scores: Mapping[int, float]) -> Dict[int, float]:
    if not scores:
        return {}
    winner = max(scores.items(), key=lambda item: item[1])[0]
    return {winner: 1.0}


def set_weights(
    netuid: int,
    scores: Mapping[int, float],
    *,
    dry_run: bool = False,
    wallet=None,
    wait_for_inclusion: bool = True,
    prompt: bool = False,
):
    weights_map = winner_takes_all(scores)
    if not weights_map:
        return {}
    uids = list(weights_map.keys())
    values = [weights_map[uid] for uid in uids]
    if dry_run:
        return {"uids": uids, "weights": values}
    import bittensor as bt  # type: ignore

    return bt.subtensor().set_weights(
        netuid=netuid,
        uids=uids,
        weights=values,
        wait_for_finalization=wait_for_inclusion,
        prompt=prompt,
        wallet=wallet,
    )


def _leaf_bytes(sample: Sample) -> bytes:
    return canonical_bytes(sample.canonical_dict())


def block_hash(block: Block) -> str:
    payload = {
        "header": block.header.canonical_dict(),
        "samples": list(block.sample_hashes()),
    }
    return hash_hex(canonical_bytes(payload))


def build_block(
    samples: Sequence[Sample],
    *,
    validator: str,
    block_index: int,
    prev_hash: str,
    env_spec_versions: Mapping[str, int],
    signing_key: signing.SigningKey,
    version: int = 1,
) -> Tuple[Block, str]:
    if not samples:
        raise ValueError("Cannot build block without samples.")
    for sample in samples:
        sample.compute_hash()
    root = merkle_root([_leaf_bytes(sample) for sample in samples])
    header = BlockHeader(
        version=version,
        prev_hash=prev_hash,
        block_index=block_index,
        timestamp_ms=canonical_timestamp(),
        validator=validator,
        env_spec_versions=dict(env_spec_versions),
        sample_count=len(samples),
        merkle_root=root,
    )
    signature = signing_key.sign(header.signing_bytes()).signature
    header.signature = f"ed25519:{signature.hex()}"
    block = Block(header=header, samples=list(samples))
    digest = block_hash(block)
    return block, digest


def serialize_block(block: Block) -> str:
    return json.dumps(
        block.canonical_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


def load_block(data: Mapping[str, Any]) -> Block:
    header_data = data["header"]  # type: ignore[index]
    header = BlockHeader(
        version=int(header_data["version"]),
        prev_hash=str(header_data["prev_hash"]),
        block_index=int(header_data["block_index"]),
        timestamp_ms=int(header_data["timestamp"]),
        validator=str(header_data["validator"]),
        env_spec_versions={
            k: int(v) for k, v in header_data["env_spec_versions"].items()
        },
        sample_count=int(header_data["sample_count"]),
        merkle_root=str(header_data["merkle_root"]),
        signature=str(header_data.get("signature", "")),
    )
    samples: List[Sample] = []
    embedded = data.get("embedded", [])  # type: ignore[assignment]
    for sample_data in embedded:
        sample = Sample(
            version=int(sample_data["version"]),
            env_id=str(sample_data["env_id"]),
            env_spec_version=int(sample_data["env_spec_version"]),
            challenge_id=str(sample_data["challenge_id"]),
            validator=str(sample_data["validator"]),
            miner_id=str(sample_data["miner_id"]),
            role=str(sample_data["role"]),
            request_id=sample_data.get("request_id"),
            prompt=str(sample_data["prompt"]),
            response=sample_data["response"],
            info=sample_data.get("info", {}),
            ok=bool(sample_data["ok"]),
            reason=str(sample_data.get("reason", "")),
            timing_ms=int(sample_data.get("timing_ms", 0)),
            bytes=int(sample_data.get("bytes", 0)),
            sample_hash=sample_data.get("sample_hash"),
        )
        samples.append(sample)
    return Block(header=header, samples=samples)


def verify_block(block: Block, public_key: signing.VerifyKey) -> bool:
    signature_prefixed = block.header.signature
    if not signature_prefixed.startswith("ed25519:"):
        return False
    signature = bytes.fromhex(signature_prefixed.split(":", 1)[1])
    try:
        public_key.verify(block.header.signing_bytes(), signature)
    except BadSignatureError:
        return False
    expected_root = merkle_root(
        [_leaf_bytes(sample) for sample in block.samples if isinstance(sample, Sample)]
    )
    return expected_root == block.header.merkle_root


def verify_chain(
    blocks: Sequence[Block],
    keys: Mapping[str, signing.VerifyKey],
    *,
    genesis_hash: str | None = None,
) -> bool:
    if not blocks:
        return True
    previous_digest = genesis_hash
    for block in blocks:
        validator = block.header.validator
        public_key = keys.get(validator)
        if public_key is None:
            return False
        if not verify_block(block, public_key):
            return False
        digest = block_hash(block)
        if previous_digest is not None and block.header.prev_hash != previous_digest:
            return False
        previous_digest = digest
    return True


__all__ = [
    "ChallengeCommitment",
    "ChallengeOutcome",
    "DuplicateDetector",
    "MinerOutcome",
    "ValidatorSampler",
    "VerifiedSample",
    "block_hash",
    "build_block",
    "compute_vtrust",
    "decide_group",
    "load_block",
    "scoreboard",
    "serialize_block",
    "set_weights",
    "verify_block",
    "verify_chain",
    "verify_samples",
    "winner_takes_all",
]
