from __future__ import annotations

import re
from typing import Any, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
from gymnasium import spaces

from ..core.rng import make_rng
from ..core.types import Verdict
from .base import AffineEnv

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


def _score(flat: np.ndarray) -> int | None:
    sums = flat[_WIN_LINES].sum(axis=1)
    if 3 in sums:
        return 1
    if -3 in sums:
        return -1
    if np.any(flat == 0):
        return None
    return 0


def _minimax(flat: np.ndarray, player: int) -> Tuple[int, int | None]:
    outcome = _score(flat)
    if outcome is not None:
        return {-1: 1, 0: 0, 1: -1}[outcome], None
    empties = np.flatnonzero(flat == 0)
    best = -2 if player == -1 else 2
    choice = int(empties[0])
    for move in map(int, empties):
        flat[move] = player
        score, _ = _minimax(flat, -player)
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


def _opening(rng: np.random.Generator) -> Tuple[np.ndarray, List[int]]:
    flat = np.zeros(9, dtype=np.int8)
    history: List[int] = []
    for _ in range(int(rng.integers(0, 4))):
        candidates = [int(m) for m in rng.permutation(9) if flat[m] == 0]
        if not candidates:
            break
        move = candidates[0]
        flat[move] = 1
        if _score(flat) is not None:
            flat[move] = 0
            continue
        history.append(move)
        _, reply = _minimax(flat, -1)
        if reply is None or flat[reply] != 0:
            flat[move] = 0
            history.pop()
            break
        flat[reply] = -1
        if _score(flat) is not None:
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
        outcome = _score(flat)
        if outcome == 1:
            return "win"
        if outcome == 0:
            return "draw"
        _, reply = _minimax(flat, -1)
        if reply is None:
            return "draw"
        flat[reply] = -1
        outcome = _score(flat)
        if outcome == -1:
            return "loss"
        if outcome == 0:
            return "draw"
    return "incomplete"


class TicTacToeEnv(AffineEnv):
    """Deterministic tic-tac-toe with a perfect opponent."""

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
        flat, opening = _opening(rng)
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
        outcome = _score(board)
        if outcome == 1:
            return board.reshape(3, 3).copy(), 1.0, True, False, self._info("win")
        if outcome == 0:
            return board.reshape(3, 3).copy(), 0.0, True, False, self._info("draw")

        _, reply = _minimax(board, -1)
        if reply is None:
            return board.reshape(3, 3).copy(), 0.0, True, False, self._info("draw")

        board[reply] = -1
        self._transcript.append(("env", int(reply)))
        outcome = _score(board)
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
        flat, opening = _opening(rng)
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


__all__ = ["TicTacToeEnv"]
