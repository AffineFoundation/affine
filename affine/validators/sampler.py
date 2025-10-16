from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence

from ..core.hashing import hash_bytes, hash_hex
from ..core.rng import derive_challenge_id
from ..core.types import Sample, Verdict
from ..envs import registry as env_registry
from ..miners.chutes_client import ChutesClient, ChutesResponse


def _stringify_observation(observation: Any) -> str:
    if isinstance(observation, (str, bytes, bytearray)):
        return observation.decode() if isinstance(observation, (bytes, bytearray)) else observation
    return json.dumps(observation, ensure_ascii=False)


def _as_bytes(value: bytes | str) -> bytes:
    if isinstance(value, bytes):
        return value
    try:
        return bytes.fromhex(value)
    except ValueError:
        return value.encode("utf-8")


class ChallengeCommitment:
    """Commit-reveal schedule for deterministic challenge generation."""

    def __init__(self, validator_hotkey: str, epoch_anchor: str) -> None:
        self.validator_hotkey = validator_hotkey
        self.epoch_anchor = epoch_anchor
        self._commits: Dict[int, str] = {}
        self._seeds: Dict[int, bytes] = {}

    def commit(self, epoch: int, seed: bytes | str) -> str:
        material = _as_bytes(seed)
        digest = hash_hex(material)
        self._commits[epoch] = digest
        return digest

    def reveal(self, epoch: int, seed: bytes | str) -> None:
        material = _as_bytes(seed)
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

    def challenge_id(self, epoch: int, env_id: str, counter: int, spec_hash: str, *, size: int = 16) -> str:
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
    """Track recent challenge assignments to avoid duplicates."""

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
        merged_info = dict(self.info)
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
    """Generate deterministic challenges and capture miner transcripts."""

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
        self.env_ids = list(env_ids or env_registry.names())
        self.epoch = int(epoch)
        self._commitment = commitment
        self._commit_digest = commitment.commitment(self.epoch) if commitment else None
        self._duplicates = duplicate_detector or DuplicateDetector()
        self._counters: Dict[str, int] = {env_id: 0 for env_id in self.env_ids}
        self._env_spec_versions: Dict[str, int] = {}
        self._env_spec_hashes: Dict[str, str] = {}
        for env_id in self.env_ids:
            env_cls = env_registry.get(env_id)
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
        env_cls = env_registry.get(env_id)
        champion = self._run_episode(env_cls, env_id, champion_uid, challenge_id, timeout)
        contender = self._run_episode(env_cls, env_id, contender_uid, challenge_id, timeout)
        winner = self._decide(contender.verdict, champion.verdict)
        info = {
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

    # -- Internal helpers ----------------------------------------------
    def _next_challenge(self, env_id: str) -> tuple[int, str]:
        counter = self._counters[env_id]
        self._counters[env_id] = counter + 1
        spec_hash = self._env_spec_hashes[env_id]
        if self._commitment is not None:
            challenge_id = self._commitment.challenge_id(self.epoch, env_id, counter, spec_hash)
        else:
            challenge_id = derive_challenge_id(
                self.validator_hotkey,
                env_id,
                counter,
                self.epoch_anchor,
                spec_hash,
            )
        return counter, challenge_id

    def _run_episode(self, env_cls, env_id: str, uid: int, challenge_id: str, timeout: float) -> MinerOutcome:
        env = env_cls()
        env_name = env_cls.env_id()
        if self._duplicates.check_and_mark(env_name, challenge_id, uid):
            raise RuntimeError(f"duplicate sample for miner {uid} on challenge {challenge_id}")
        observation, info = env.reset(options={"challenge_id": challenge_id})
        prompt = _stringify_observation(observation)
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
                    "observation": observation,
                    "info": info,
                }
                response = self.chutes.invoke(uid, payload, timeout=timeout)
                total_latency += response.latency_ms
                total_bytes += len(response.text.encode("utf-8"))
                last_request_id = response.request_id
                transcript.append(
                    {
                        "observation": observation,
                        "info": info,
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
                        "observation": observation,
                        "info": info,
                        "reward": reward,
                        "terminated": terminated,
                        "truncated": truncated,
                    }
                )
                if terminated or truncated:
                    break
        finally:
            env.close()

        verification_payload: Any
        if miner_moves:
            verification_payload = miner_moves
        else:
            verification_payload = transcript
        verdict = env.verify(verification_payload, final_info)
        response_blob = json.dumps({"moves": miner_moves, "transcript": transcript}, ensure_ascii=False)
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


__all__ = ["ChallengeCommitment", "ChallengeOutcome", "DuplicateDetector", "ValidatorSampler"]
