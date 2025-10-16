from __future__ import annotations

import inspect
import json
import math
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from statistics import NormalDist
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import requests
from gymnasium import Env, spaces
from nacl import signing
from nacl.exceptions import BadSignatureError

try:  # pragma: no cover - optional dependency
    from blake3 import blake3  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    blake3 = None  # type: ignore

JsonPrimitive = Union[str, int, float, bool, None]
JsonValue = Union[JsonPrimitive, Sequence["JsonValue"], Mapping[str, "JsonValue"]]  # type: ignore
_NORMAL = NormalDist()


# --------------------------------------------------------------------------- #
# Hashing & canonicalisation helpers                                         #
# --------------------------------------------------------------------------- #


def canonical_bytes(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def hash_bytes(data: bytes) -> bytes:
    if blake3 is not None:
        return blake3(data).digest()
    import hashlib

    return hashlib.sha256(data).digest()


def hash_hex(data: bytes) -> str:
    tag = "b3" if blake3 is not None else "sha256"
    return f"{tag}:{hash_bytes(data).hex()}"


def merkle_root(leaves: Sequence[bytes]) -> str:
    if not leaves:
        return hash_hex(hash_bytes(b""))
    layer = [hash_bytes(b"\x00" + leaf) for leaf in leaves]
    while len(layer) > 1:
        if len(layer) % 2 == 1:
            layer.append(layer[-1])
        layer = [hash_bytes(b"\x01" + layer[i] + layer[i + 1]) for i in range(0, len(layer), 2)]
    return hash_hex(layer[0])


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _freeze(v) for k, v in sorted(value.items())}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze(v) for v in value)
    return value


def _ensure_int(value: Any) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    raise TypeError(f"Expected integer-compatible value, got {type(value)!r}")


def canonical_timestamp(dt: datetime | None = None) -> int:
    target = dt or datetime.now(timezone.utc)
    return int(target.timestamp() * 1000)


# --------------------------------------------------------------------------- #
# Core dataclasses                                                            #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Challenge:
    env_id: str
    challenge_id: str
    info: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"env_id": self.env_id, "challenge_id": self.challenge_id, "info": _freeze(self.info)}


@dataclass(frozen=True)
class Verdict:
    ok: bool
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"ok": bool(self.ok), "reason": self.reason}


@dataclass
class Sample:
    version: int
    env_id: str
    env_spec_version: int
    challenge_id: str
    validator: str
    miner_id: str
    role: str
    request_id: Optional[str]
    prompt: str
    response: str
    info: Mapping[str, Any]
    ok: bool
    reason: str
    timing_ms: int
    bytes: int
    sample_hash: Optional[str] = None

    def canonical_dict(self) -> Dict[str, Any]:
        return {
            "version": _ensure_int(self.version),
            "env_id": self.env_id,
            "env_spec_version": _ensure_int(self.env_spec_version),
            "challenge_id": self.challenge_id,
            "validator": self.validator,
            "miner_id": self.miner_id,
            "role": self.role,
            "request_id": self.request_id,
            "prompt": self.prompt,
            "response": self.response,
            "info": _freeze(self.info),
            "ok": bool(self.ok),
            "reason": self.reason,
            "timing_ms": _ensure_int(self.timing_ms),
            "bytes": _ensure_int(self.bytes),
        }

    def compute_hash(self) -> str:
        digest = hash_bytes(canonical_bytes(self.canonical_dict()))
        self.sample_hash = digest.hex()
        return self.sample_hash

    def to_dict(self) -> Dict[str, Any]:
        payload = self.canonical_dict()
        payload["sample_hash"] = self.sample_hash or self.compute_hash()
        return payload


@dataclass
class BlockHeader:
    version: int
    prev_hash: str
    block_index: int
    timestamp_ms: int
    validator: str
    env_spec_versions: Mapping[str, int]
    sample_count: int
    merkle_root: str
    signature: str = ""

    def canonical_dict(self, include_signature: bool = True) -> Dict[str, Any]:
        payload = {
            "version": _ensure_int(self.version),
            "prev_hash": self.prev_hash,
            "block_index": _ensure_int(self.block_index),
            "timestamp": _ensure_int(self.timestamp_ms),
            "validator": self.validator,
            "env_spec_versions": _freeze(self.env_spec_versions),
            "sample_count": _ensure_int(self.sample_count),
            "merkle_root": self.merkle_root,
        }
        if include_signature:
            payload["signature"] = self.signature
        return payload

    def signing_bytes(self) -> bytes:
        return canonical_bytes(self.canonical_dict(include_signature=False))


@dataclass
class Block:
    header: BlockHeader
    samples: Sequence[Sample | str]

    def sample_hashes(self) -> Tuple[str, ...]:
        hashes: List[str] = []
        for sample in self.samples:
            if isinstance(sample, Sample):
                hashes.append(sample.sample_hash or sample.compute_hash())
            else:
                hashes.append(sample)
        return tuple(hashes)

    def canonical_dict(self) -> Dict[str, Any]:
        return {
            "version": self.header.version,
            "header": self.header.canonical_dict(),
            "samples": list(self.sample_hashes()),
            "embedded": [s.to_dict() for s in self.samples if isinstance(s, Sample)],
        }


@dataclass(frozen=True)
class DuelResult:
    env_id: str
    outcome: str
    wins: int
    losses: int
    ties: int
    trials: int
    ci: Tuple[float, float]


# --------------------------------------------------------------------------- #
# RNG utilities                                                               #
# --------------------------------------------------------------------------- #


def challenge_seed(env_id: str, spec_version: int, challenge_id: str) -> int:
    challenge_clean = challenge_id.lower().removeprefix("0x")
    payload = f"{env_id}:{spec_version}:{challenge_clean}".encode()
    return int.from_bytes(hash_bytes(payload)[:8], "little")


def make_rng(env_id: str, spec_version: int, challenge_id: str) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(challenge_seed(env_id, spec_version, challenge_id)))


def derive_challenge_id(
    validator_hotkey: str,
    env_id: str,
    counter: int,
    epoch_anchor: str,
    spec_hash: str,
    size: int = 16,
) -> str:
    payload = f"{validator_hotkey}:{env_id}:{counter}:{epoch_anchor}:{spec_hash}".encode()
    digest = hash_bytes(payload)
    size = max(8, min(size, len(digest)))
    return digest[:size].hex()


# --------------------------------------------------------------------------- #
# Environment helpers and implementations                                    #
# --------------------------------------------------------------------------- #


def last_integer(text: str) -> int | None:
    matches = re.findall(r"[-+]?\d+", text)
    if not matches:
        return None
    return int(matches[-1])


def ensure_last_integer(text: str, expected: int) -> Verdict:
    value = last_integer(text)
    if value is None:
        return Verdict(False, "no-integer-found")
    if value == expected:
        return Verdict(True, "")
    return Verdict(False, f"mismatch:{value}")


class AffineEnv(Env):
    metadata: Dict[str, Any] = {"env_id": "affine-env", "spec_version": 0}

    def __init__(self) -> None:
        super().__init__()
        self._challenge: Optional[Challenge] = None
        self._rng: Optional[np.random.Generator] = None

    @classmethod
    def env_id(cls) -> str:
        env = cls.metadata.get("env_id")
        if not env:
            raise ValueError(f"{cls.__name__} must define metadata['env_id'].")
        return str(env)

    @classmethod
    def spec_version(cls) -> int:
        return int(cls.metadata.get("spec_version", 0))

    @classmethod
    def spec_hash(cls) -> str:
        try:
            source = inspect.getsource(cls)
        except (OSError, TypeError):
            source = f"{cls.__module__}.{cls.__qualname__}"
        payload = {
            "module": cls.__module__,
            "class": cls.__qualname__,
            "spec_version": cls.spec_version(),
            "source": source,
        }
        return hash_hex(canonical_bytes(payload))

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[Any, Dict[str, Any]]:
        options = dict(options or {})
        challenge_id = options.get("challenge_id")
        if challenge_id is None:
            if seed is None:
                seed = int(np.random.SeedSequence().entropy)
            challenge_id = f"{seed:032x}"
        challenge_id = str(challenge_id).lower().removeprefix("0x")
        rng = make_rng(self.env_id(), self.spec_version(), challenge_id)
        self._rng = rng
        info: Dict[str, Any] = {
            "challenge_id": challenge_id,
            "env_id": self.env_id(),
            "spec_version": self.spec_version(),
            "spec_hash": self.spec_hash(),
        }
        observation, extra = self._reset(rng=rng, info=info, options=options)
        merged = dict(info)
        merged.update(extra or {})
        self._challenge = Challenge(env_id=self.env_id(), challenge_id=challenge_id, info=merged)
        return observation, merged

    def _reset(
        self,
        *,
        rng: np.random.Generator,
        info: MutableMapping[str, Any],
        options: Mapping[str, Any],
    ) -> Tuple[Any, Mapping[str, Any]]:
        raise NotImplementedError

    def step(self, action: Any):  # pragma: no cover - interface documentation
        raise NotImplementedError

    def verify(self, response: Any, info: Mapping[str, Any]) -> Verdict:
        raise NotImplementedError

    def decode_action(self, response: Any) -> Any:
        return response

    @property
    def rng(self) -> np.random.Generator:
        if self._rng is None:
            raise RuntimeError("Environment accessed before reset().")
        return self._rng

    @property
    def challenge(self) -> Challenge:
        if self._challenge is None:
            raise RuntimeError("Challenge metadata unavailable before reset().")
        return self._challenge


class Mult8Env(AffineEnv):
    metadata = {"env_id": "mult8-v0", "spec_version": 1}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Text(max_length=128)
        self.action_space = spaces.Text(max_length=128)
        self._factors: tuple[int, int] | None = None

    def _reset(
        self,
        *,
        rng: np.random.Generator,
        info: MutableMapping[str, Any],
        options: Mapping[str, Any],
    ) -> Tuple[str, Mapping[str, Any]]:
        a = int(rng.integers(10_000_000, 100_000_000))
        b = int(rng.integers(10_000_000, 100_000_000))
        self._factors = (a, b)
        info.update({"factors": {"a": a, "b": b}, "expected": a * b, "difficulty": 0})
        return f"Compute {a} × {b}. Return only the integer result.", {}

    def step(self, action: Any):
        if not self._factors:
            raise RuntimeError("Environment used before reset().")
        a, b = self._factors
        expected = a * b
        verdict = ensure_last_integer(str(action), expected)
        info = {
            "challenge_id": self.challenge.challenge_id,
            "env_id": self.env_id(),
            "factors": {"a": a, "b": b},
            "expected": expected,
            "reason": verdict.reason or ("win" if verdict.ok else "loss"),
        }
        return f"Compute {a} × {b}. Return only the integer result.", 1.0 if verdict.ok else -1.0, True, False, info

    def verify(self, response: Any, info: Mapping[str, Any]) -> Verdict:
        factors = info.get("factors")
        if isinstance(factors, Mapping):
            expected = int(factors["a"]) * int(factors["b"])
            if "expected" in info and int(info["expected"]) != expected:
                return Verdict(False, "expected-mismatch")
        elif "challenge_id" in info:
            rng = make_rng(self.env_id(), self.spec_version(), str(info["challenge_id"]))
            a = int(rng.integers(10_000_000, 100_000_000))
            b = int(rng.integers(10_000_000, 100_000_000))
            expected = a * b
            if "expected" in info and int(info["expected"]) != expected:
                return Verdict(False, "expected-mismatch")
        elif "expected" in info:
            expected = int(info["expected"])
        else:
            return Verdict(False, "missing-expected")
        return ensure_last_integer(str(response), expected)


_WIN_LINES = np.array(
    [
        (0, 1, 2),
        (3, 4, 5),
        (6, 7, 8),
        (0, 3, 6),
        (1, 4, 7),
        (2, 5, 8),
        (0, 4, 8),
        (2, 4, 6),
    ],
    dtype=np.int8,
)


def _ttt_score(flat: np.ndarray) -> int | None:
    sums = flat[_WIN_LINES].sum(axis=1)
    if 3 in sums:
        return 1
    if -3 in sums:
        return -1
    if np.any(flat == 0):
        return None
    return 0


def _ttt_minimax(flat: np.ndarray, player: int) -> Tuple[int, int | None]:
    outcome = _ttt_score(flat)
    if outcome is not None:
        return {-1: 1, 0: 0, 1: -1}[outcome], None
    empties = np.flatnonzero(flat == 0)
    best = -2 if player == -1 else 2
    choice = int(empties[0])
    for move in map(int, empties):
        flat[move] = player
        score, _ = _ttt_minimax(flat, -player)
        flat[move] = 0
        if player == -1:
            if score > best:
                best, choice = score, move
            if score == 1:
                break
        else:
            if score < best:
                best, choice = score, move
            if score == -1:
                break
    return best, choice


def _ttt_opening(rng: np.random.Generator) -> Tuple[np.ndarray, List[int]]:
    flat = np.zeros(9, dtype=np.int8)
    history: List[int] = []
    for _ in range(int(rng.integers(0, 4))):
        candidates = [int(m) for m in rng.permutation(9) if flat[m] == 0]
        if not candidates:
            break
        move = candidates[0]
        flat[move] = 1
        if _ttt_score(flat) is not None:
            flat[move] = 0
            continue
        history.append(move)
        _, reply = _ttt_minimax(flat, -1)
        if reply is None or flat[reply] != 0:
            flat[move] = 0
            history.pop()
            break
        flat[reply] = -1
        if _ttt_score(flat) is not None:
            flat[reply] = 0
            flat[move] = 0
            history.pop()
            break
        history.append(int(reply))
    return flat, history


def _parse_moves(payload: Any) -> List[int]:
    if isinstance(payload, int):
        return [int(payload)]
    if isinstance(payload, str):
        return [int(x) for x in re.findall(r"-?\d+", payload)]
    if isinstance(payload, Mapping):
        for key in ("moves", "miner_moves", "sequence", "actions", "move", "action", "index"):
            if key in payload:
                return _parse_moves(payload[key])
        raise TypeError("no move field in mapping")
    if isinstance(payload, Sequence):
        moves: List[int] = []
        for item in payload:
            moves.extend(_parse_moves(item))
        return moves
    raise TypeError(f"Cannot parse moves from {type(payload)}")


def _replay(flat: np.ndarray, moves: Sequence[int]) -> str:
    for move in moves:
        if move < 0 or move >= 9:
            return f"illegal:{move}"
        if flat[move] != 0:
            return f"occupied:{move}"
        flat[move] = 1
        outcome = _ttt_score(flat)
        if outcome == 1:
            return "win"
        if outcome == 0:
            return "draw"
        _, reply = _ttt_minimax(flat, -1)
        if reply is None:
            return "draw"
        flat[reply] = -1
        outcome = _ttt_score(flat)
        if outcome == -1:
            return "loss"
        if outcome == 0:
            return "draw"
    return "incomplete"


class TicTacToeEnv(AffineEnv):
    metadata = {"env_id": "tictactoe-v0", "spec_version": 1}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3, 3), dtype=np.int8)
        self.action_space = spaces.Discrete(9)
        self._board: np.ndarray | None = None
        self._opening: List[int] = []
        self._transcript: List[Tuple[str, int]] = []

    def _reset(
        self,
        *,
        rng: np.random.Generator,
        info: MutableMapping[str, Any],
        options: Mapping[str, Any],
    ) -> Tuple[np.ndarray, Mapping[str, Any]]:
        flat, opening = _ttt_opening(rng)
        self._board = flat
        self._opening = opening
        self._transcript = []
        info.update(
            {
                "miner_first": True,
                "difficulty": len(opening) // 2,
                "opening_moves": list(opening),
            }
        )
        return flat.reshape(3, 3).copy(), {}

    def _info(self, reason: str) -> Mapping[str, Any]:
        return {
            "challenge_id": self.challenge.challenge_id,
            "env_id": self.env_id(),
            "spec_hash": self.spec_hash(),
            "spec_version": self.spec_version(),
            "miner_first": True,
            "difficulty": len(self._opening) // 2,
            "opening_moves": list(self._opening),
            "transcript": [{"role": role, "move": move} for role, move in self._transcript],
            "reason": reason,
        }

    def step(self, action: Any):
        if self._board is None:
            raise RuntimeError("Environment not reset.")
        move = int(action)
        board = self._board
        if not self.action_space.contains(move):
            self._transcript.append(("miner", move))
            info = dict(self._info("illegal"))
            info["transcript"][-1]["illegal"] = True
            return board.reshape(3, 3).copy(), -1.0, True, False, info
        if board[move] != 0:
            self._transcript.append(("miner", move))
            info = dict(self._info("occupied"))
            info["transcript"][-1]["illegal"] = True
            return board.reshape(3, 3).copy(), -1.0, True, False, info

        board[move] = 1
        self._transcript.append(("miner", move))
        outcome = _ttt_score(board)
        if outcome == 1:
            return board.reshape(3, 3).copy(), 1.0, True, False, self._info("win")
        if outcome == 0:
            return board.reshape(3, 3).copy(), 0.0, True, False, self._info("draw")

        _, reply = _ttt_minimax(board, -1)
        if reply is None:
            return board.reshape(3, 3).copy(), 0.0, True, False, self._info("draw")

        board[reply] = -1
        self._transcript.append(("env", int(reply)))
        outcome = _ttt_score(board)
        if outcome == -1:
            return board.reshape(3, 3).copy(), -1.0, True, False, self._info("loss")
        if outcome == 0:
            return board.reshape(3, 3).copy(), 0.0, True, False, self._info("draw")
        return board.reshape(3, 3).copy(), 0.0, False, False, self._info("continue")

    def verify(self, response: Any, info: Mapping[str, Any]) -> Verdict:
        challenge_id = info.get("challenge_id")
        if not challenge_id:
            return Verdict(False, "missing-challenge")
        try:
            moves = _parse_moves(response)
        except Exception as exc:
            return Verdict(False, f"parse-error:{exc}")
        rng = make_rng(self.env_id(), self.spec_version(), str(challenge_id))
        flat, opening = _ttt_opening(rng)
        declared = info.get("opening_moves")
        if declared is not None and list(declared) != opening:
            return Verdict(False, "opening-mismatch")
        result = _replay(flat.copy(), moves)
        if result == "win":
            return Verdict(True, "win")
        return Verdict(False, result)

    def decode_action(self, response: Any) -> int:
        moves = _parse_moves(response)
        if not moves:
            raise ValueError("No moves to decode.")
        return moves[0]


ENV_REGISTRY: Dict[str, type[AffineEnv]] = {
    Mult8Env.env_id(): Mult8Env,
    TicTacToeEnv.env_id(): TicTacToeEnv,
}


def get_env(name: str) -> type[AffineEnv]:
    try:
        return ENV_REGISTRY[name]
    except KeyError as exc:  # pragma: no cover - defensive
        raise KeyError(f"Unknown environment '{name}'.") from exc


def env_names() -> Iterable[str]:
    return ENV_REGISTRY.keys()


# --------------------------------------------------------------------------- #
# Wilson interval & duel mechanics                                           #
# --------------------------------------------------------------------------- #


def wilson_interval(wins: int, trials: int, confidence: float = 0.95) -> Tuple[float, float]:
    if trials <= 0:
        return 0.0, 1.0
    if wins < 0 or wins > trials:
        raise ValueError("wins must be between 0 and trials (inclusive).")
    z = _NORMAL.inv_cdf(0.5 + confidence / 2.0)
    phat = wins / trials
    denom = 1.0 + z * z / trials
    center = phat + z * z / (2 * trials)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * trials)) / trials)
    lower = (center - margin) / denom
    upper = (center + margin) / denom
    return max(0.0, lower), min(1.0, upper)


def beats_target(wins: int, trials: int, target: float, confidence: float) -> bool:
    lower, _ = wilson_interval(wins, trials, confidence)
    return lower > target


def loses_to_target(wins: int, trials: int, target: float, confidence: float) -> bool:
    _, upper = wilson_interval(wins, trials, confidence)
    return upper < target


def _z_to_confidence(z: float) -> float:
    return max(0.0, min(1.0, 2 * (_NORMAL.cdf(z) - 0.5)))


def _iter_stream(stream: Iterable[Optional[bool]]) -> Iterator[Optional[bool]]:
    if callable(stream):  # type: ignore
        return iter(stream())  # type: ignore[arg-type]
    return iter(stream)


def duel_env(
    stream_results: Iterable[Optional[bool]],
    *,
    env_id: str = "",
    ratio_to_beat_env: float = 0.51,
    z: float = 1.96,
    max_budget: int = 200,
) -> DuelResult:
    wins = losses = ties = 0
    trials = 0
    confidence = _z_to_confidence(z)
    iterator = _iter_stream(stream_results)
    for outcome in iterator:
        if outcome is None:
            ties += 1
            continue
        if outcome:
            wins += 1
        else:
            losses += 1
        trials = wins + losses
        if trials == 0:
            continue
        lower, upper = wilson_interval(wins, trials, confidence)
        if lower > ratio_to_beat_env:
            return DuelResult(env_id, "contender", wins, losses, ties, trials, (lower, upper))
        if upper < ratio_to_beat_env:
            return DuelResult(env_id, "champion", wins, losses, ties, trials, (lower, upper))
        if trials >= max_budget:
            break
    lower, upper = wilson_interval(wins, max(trials, 1), confidence)
    return DuelResult(env_id, "inconclusive", wins, losses, ties, trials, (lower, upper))


class RatioSchedule:
    def __init__(
        self,
        initial: float = 0.51,
        *,
        baseline: float = 0.5,
        half_life_seconds: float = 7 * 24 * 3600,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._baseline = baseline
        self._value = max(initial, baseline)
        self._half_life = max(half_life_seconds, 1.0)
        self._clock = clock or time.monotonic
        self._updated = self._clock()

    def current(self) -> float:
        now = self._clock()
        elapsed = max(0.0, now - self._updated)
        decay = math.exp(-elapsed * math.log(2) / self._half_life)
        return self._baseline + (self._value - self._baseline) * decay

    def update(self, ratio: float) -> None:
        self._value = max(self._baseline, min(0.95, ratio))
        self._updated = self._clock()


def _geometric_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return math.exp(sum(math.log(max(v, 1e-9)) for v in values) / len(values))


def duel_many_envs(
    contender_uid: int,
    champion_uid: int,
    env_ids: Sequence[str],
    *,
    stream_factory: Callable[[str], Iterable[Optional[bool]]],
    ratio_to_beat_global: float = 0.51,
    per_env_ratio: float = 0.51,
    z: float = 1.96,
    max_budget: int = 200,
    ratio_schedule: RatioSchedule | None = None,
) -> Dict[str, object]:
    if not env_ids:
        raise ValueError("At least one environment required.")

    ratio_before = ratio_to_beat_global
    if ratio_schedule is not None:
        ratio_current = ratio_schedule.current()
        ratio_to_beat_global = max(ratio_to_beat_global, ratio_current)
        ratio_before = ratio_to_beat_global

    per_env: Dict[str, DuelResult] = {}
    env_wins_cont = 0
    env_wins_champ = 0
    total_envs = len(env_ids)
    wins_required = max(1, math.ceil(ratio_to_beat_global * total_envs))
    stopped_early = False
    samples_used = 0

    for env_id in env_ids:
        stream = stream_factory(env_id)
        result = duel_env(stream, env_id=env_id, ratio_to_beat_env=per_env_ratio, z=z, max_budget=max_budget)
        per_env[env_id] = result
        samples_used += result.wins + result.losses + result.ties
        if result.outcome == "contender":
            env_wins_cont += 1
        elif result.outcome == "champion":
            env_wins_champ += 1

        remaining = total_envs - len(per_env)
        if env_wins_cont >= wins_required:
            stopped_early = remaining > 0
            break
        if env_wins_cont + remaining < wins_required:
            stopped_early = remaining > 0
            break

    if env_wins_cont >= wins_required:
        winner_uid = contender_uid
        winning_ratios = [
            result.wins / max(1, result.wins + result.losses)
            for result in per_env.values()
            if result.outcome == "contender"
        ]
        ratio_snapshot = min(0.95, _geometric_mean(winning_ratios)) if winning_ratios else ratio_to_beat_global
        dethroned = True
        if ratio_schedule:
            ratio_schedule.update(ratio_snapshot)
    else:
        winner_uid = champion_uid
        ratio_snapshot = ratio_to_beat_global
        dethroned = False

    if ratio_schedule:
        ratio_after = ratio_schedule.current()
    else:
        ratio_after = ratio_snapshot if dethroned else ratio_to_beat_global

    return {
        "winner_uid": winner_uid,
        "contender_uid": contender_uid,
        "champion_uid": champion_uid,
        "dethroned": dethroned,
        "ratio_to_beat": ratio_before,
        "ratio_snapshot": ratio_snapshot,
        "ratio_next": ratio_after,
        "per_env": per_env,
        "samples_used": samples_used,
        "stopped_early": stopped_early,
        "wins_required": wins_required,
        "env_wins": {"contender": env_wins_cont, "champion": env_wins_champ},
    }


# --------------------------------------------------------------------------- #
# Validator tooling                                                           #
# --------------------------------------------------------------------------- #


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
    return derive_challenge_id(sample.validator, sample.env_id, counter_int, anchor, spec_hash_value)


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

    def _next_challenge(self, env_id: str) -> tuple[int, str]:
        counter = self._counters[env_id]
        self._counters[env_id] = counter + 1
        spec_hash_value = self._env_spec_hashes[env_id]
        if self._commitment is not None:
            challenge_id = self._commitment.challenge_id(self.epoch, env_id, counter, spec_hash_value)
        else:
            challenge_id = derive_challenge_id(
                self.validator_hotkey,
                env_id,
                counter,
                self.epoch_anchor,
                spec_hash_value,
            )
        return counter, challenge_id

    def _run_episode(self, env_cls: type[AffineEnv], env_id: str, uid: int, challenge_id: str, timeout: float) -> MinerOutcome:
        env = env_cls()
        env_name = env_cls.env_id()
        if self._duplicates.check_and_mark(env_name, challenge_id, uid):
            raise RuntimeError(f"duplicate sample for miner {uid} on challenge {challenge_id}")
        observation, info = env.reset(options={"challenge_id": challenge_id})
        if isinstance(observation, str):
            prompt = observation
        elif isinstance(observation, (bytes, bytearray)):
            prompt = observation.decode()
        else:
            prompt = json.dumps(observation, ensure_ascii=False)
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


@dataclass
class VerifiedSample:
    sample: Sample
    verdict: Verdict
    valid: bool
    errors: Tuple[str, ...] = ()


def verify_samples(samples: Iterable[Sample]) -> Tuple[Dict[Tuple[str, str], Dict[str, Dict[str, VerifiedSample]]], Dict[str, Tuple[int, int]]]:
    buckets: Dict[Tuple[str, str], Dict[str, Dict[str, VerifiedSample]]] = {}
    validator_stats: Dict[str, Tuple[int, int]] = {}
    for sample in samples:
        env_cls = get_env(sample.env_id)
        errors: List[str] = []

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
        wins, total = validator_stats.get(sample.validator, (0, 0))
        validator_stats[sample.validator] = (wins + (1 if valid else 0), total + 1)
        bucket = buckets.setdefault((sample.env_id, sample.challenge_id), {}).setdefault(sample.validator, {})
        bucket[sample.role] = VerifiedSample(sample=sample, verdict=verdict, valid=valid, errors=tuple(errors))
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


def compute_vtrust(stats: Mapping[str, Tuple[int, int]], *, confidence: float = 0.95) -> Dict[str, float]:
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


# --------------------------------------------------------------------------- #
# Blocks & signatures                                                         #
# --------------------------------------------------------------------------- #


def _leaf_bytes(sample: Sample) -> bytes:
    return canonical_bytes(sample.canonical_dict())


def block_hash(block: Block) -> str:
    payload = {"header": block.header.canonical_dict(), "samples": list(block.sample_hashes())}
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
    return json.dumps(block.canonical_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def load_block(data: Mapping[str, Any]) -> Block:
    header_data = data["header"]  # type: ignore[index]
    header = BlockHeader(
        version=int(header_data["version"]),
        prev_hash=str(header_data["prev_hash"]),
        block_index=int(header_data["block_index"]),
        timestamp_ms=int(header_data["timestamp"]),
        validator=str(header_data["validator"]),
        env_spec_versions={k: int(v) for k, v in header_data["env_spec_versions"].items()},
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
    expected_root = merkle_root([_leaf_bytes(sample) for sample in block.samples if isinstance(sample, Sample)])
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


# --------------------------------------------------------------------------- #
# Network glue                                                                #
# --------------------------------------------------------------------------- #


@dataclass
class ChutesResponse:
    text: str
    request_id: Optional[str]
    latency_ms: int
    tokens_in: Optional[int] = None
    tokens_out: Optional[int] = None


class ChutesClient:
    def __init__(
        self,
        base_url: str,
        *,
        api_key: str | None = None,
        default_timeout: float = 30.0,
        session: Optional[requests.Session] = None,
    ) -> None:
        if base_url.endswith("/"):
            base_url = base_url[:-1]
        self._base_url = base_url
        self._timeout = default_timeout
        self._session = session or requests.Session()
        self._api_key = api_key

    @property
    def base_url(self) -> str:
        return self._base_url

    def invoke(
        self,
        uid: int,
        payload: Mapping[str, Any],
        *,
        timeout: Optional[float] = None,
    ) -> ChutesResponse:
        url = f"{self._base_url}/invoke/{uid}"
        headers: Dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        start = time.perf_counter()
        response = self._session.post(url, json=dict(payload), headers=headers, timeout=timeout or self._timeout)
        response.raise_for_status()
        elapsed_ms = int((time.perf_counter() - start) * 1000)
        data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
        text = data.get("text") or data.get("response") or response.text
        return ChutesResponse(
            text=text,
            request_id=data.get("request_id"),
            latency_ms=elapsed_ms,
            tokens_in=data.get("tokens_in"),
            tokens_out=data.get("tokens_out"),
        )

    def close(self) -> None:
        self._session.close()


__all__ = [
    "AffineEnv",
    "Block",
    "BlockHeader",
    "Challenge",
    "ChallengeCommitment",
    "ChallengeOutcome",
    "ChutesClient",
    "ChutesResponse",
    "DuplicateDetector",
    "DuelResult",
    "ENV_REGISTRY",
    "Mult8Env",
    "RatioSchedule",
    "Sample",
    "TicTacToeEnv",
    "Verdict",
    "beats_target",
    "block_hash",
    "build_block",
    "canonical_bytes",
    "canonical_timestamp",
    "compute_vtrust",
    "decide_group",
    "derive_challenge_id",
    "duel_env",
    "duel_many_envs",
    "env_names",
    "ensure_last_integer",
    "get_env",
    "hash_bytes",
    "hash_hex",
    "loses_to_target",
    "last_integer",
    "load_block",
    "make_rng",
    "MinerOutcome",
    "scoreboard",
    "serialize_block",
    "set_weights",
    "ValidatorSampler",
    "verify_block",
    "verify_chain",
    "verify_samples",
    "VerifiedSample",
    "winner_takes_all",
    "wilson_interval",
]
