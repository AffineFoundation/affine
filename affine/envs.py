from __future__ import annotations

import inspect
import json
import re
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from gymnasium import Env, spaces

from .core import Challenge, Verdict, canonical_bytes, hash_hex, make_rng


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

    def __init__(self) -> None:
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


__all__ = [
    "AffineEnv",
    "ENV_REGISTRY",
    "Mult8Env",
    "TicTacToeEnv",
    "ensure_last_integer",
    "env_names",
    "get_env",
    "last_integer",
]
