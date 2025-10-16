from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
from gymnasium import spaces

from ..core.types import Verdict
from ..core.rng import make_rng
from .base import AffineEnv


Board = np.ndarray


def _winning_lines(board: Board) -> List[np.ndarray]:
    return [
        board[0, :],
        board[1, :],
        board[2, :],
        board[:, 0],
        board[:, 1],
        board[:, 2],
        board.diagonal(),
        np.fliplr(board).diagonal(),
    ]


def _winner(board: Board) -> int | None:
    for line in _winning_lines(board):
        if np.all(line == 1):
            return 1
        if np.all(line == -1):
            return -1
    return None


def _available_moves(board: Board) -> List[int]:
    return [int(i) for i, value in enumerate(board.ravel()) if value == 0]


def _board_full(board: Board) -> bool:
    return not (board == 0).any()


def _minimax(board: Board, player: int) -> Tuple[int, Optional[int]]:
    winner = _winner(board)
    if winner is not None:
        return (1 if winner == -1 else -1, None)
    if _board_full(board):
        return 0, None

    moves = _available_moves(board)
    if player == -1:  # environment turn, maximise result
        best_score = -2
        best_move: Optional[int] = None
        for move in moves:
            r, c = divmod(move, 3)
            board[r, c] = -1
            score, _ = _minimax(board, 1)
            board[r, c] = 0
            if score > best_score:
                best_score = score
                best_move = move
            if best_score == 1:
                break
        return best_score, best_move

    best_score = 2
    best_move = moves[0]
    for move in moves:
        r, c = divmod(move, 3)
        board[r, c] = 1
        score, _ = _minimax(board, -1)
        board[r, c] = 0
        if score < best_score:
            best_score = score
            best_move = move
        if best_score == -1:
            break
    return best_score, best_move


def _coerce_moves(data: Any) -> List[int]:
    if isinstance(data, Mapping):
        moves = data.get("moves", data.get("miner_moves"))
        if moves is None:
            raise ValueError("Transcript dict missing 'moves' key.")
        return _coerce_moves(moves)
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
        result: List[int] = []
        for item in data:
            if isinstance(item, Mapping):
                if "miner" in item:
                    item = item["miner"]
                elif "action" in item:
                    item = item["action"]
                elif "move" in item:
                    item = item["move"]
                else:
                    raise ValueError("Transcript entry missing move/action.")
            result.append(int(item))
        return result
    if isinstance(data, str):
        cleaned = [part for part in data.replace(",", " ").split() if part]
        return [int(part) for part in cleaned]
    if isinstance(data, int):
        return [data]
    raise TypeError(f"Unsupported transcript payload: {type(data)!r}")


def _opening_for_challenge(challenge_id: str) -> Tuple[Board, int, List[int]]:
    """Generate deterministic opening positions from the challenge id."""
    rng = make_rng(TicTacToeEnv.env_id(), TicTacToeEnv.spec_version(), challenge_id)
    board = np.zeros((3, 3), dtype=np.int8)
    ordering = list(rng.permutation(9))
    cursor = 0
    transcript: List[int] = []

    pairs = int(rng.integers(0, 4))  # up to three opening pairs
    for _ in range(pairs):
        while cursor < len(ordering) and board.flat[ordering[cursor]] != 0:
            cursor += 1
        if cursor >= len(ordering):
            break
        miner_move = int(ordering[cursor])
        cursor += 1
        row, col = divmod(miner_move, 3)
        if board[row, col] != 0:
            continue
        board[row, col] = 1
        transcript.append(miner_move)
        if _winner(board) == 1 or _board_full(board):
            board[row, col] = 0
            transcript.pop()
            break

        _score, env_move = _minimax(board.copy(), -1)
        if env_move is None:
            board[row, col] = 0
            transcript.pop()
            break

        erow, ecol = divmod(env_move, 3)
        if board[erow, ecol] != 0:
            board[row, col] = 0
            transcript.pop()
            break
        board[erow, ecol] = -1
        transcript.append(env_move)

        if _winner(board) == -1 or _board_full(board):
            board[erow, ecol] = 0
            transcript.pop()
            break

    difficulty = len(transcript) // 2
    return board, difficulty, transcript


@dataclass
class TicTacToeState:
    board: Board
    miner_turn: bool
    transcript: List[Dict[str, Any]]


class TicTacToeEnv(AffineEnv):
    """Deterministic TicTacToe environment with perfect opponent."""

    metadata = {"env_id": "tictactoe-v0", "spec_version": 1}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3, 3), dtype=np.int8)
        self.action_space = spaces.Discrete(9)
        self._state: Optional[TicTacToeState] = None

    # -- Gym methods ---------------------------------------------------
    def _reset(
        self,
        *,
        rng: np.random.Generator,
        info: MutableMapping[str, Any],
        options: Mapping[str, Any],
    ) -> Tuple[Board, Mapping[str, Any]]:
        board, difficulty, opening = _opening_for_challenge(info["challenge_id"])
        self._state = TicTacToeState(board=board.copy(), miner_turn=True, transcript=[])
        info.update(
            {
                "miner_first": True,
                "difficulty": int(difficulty),
                "opening_moves": list(opening),
            }
        )
        return board.copy(), {"transcript": []}

    def step(self, action: Any):
        if self._state is None:
            raise RuntimeError("Environment used before reset().")
        state = self._state
        if not state.miner_turn:
            raise RuntimeError("Miner acted out of turn.")
        try:
            move = int(action)
        except Exception as exc:  # pragma: no cover - type safety
            raise ValueError("Action must be convertible to int.") from exc

        info = {"challenge_id": self.challenge.challenge_id, "env_id": self.env_id()}

        if not self.action_space.contains(move):
            state.transcript.append({"role": "miner", "move": move, "illegal": True})
            return state.board.copy(), -1.0, True, False, {**info, "reason": "illegal"}

        row, col = divmod(move, 3)
        if state.board[row, col] != 0:
            state.transcript.append({"role": "miner", "move": move, "illegal": True})
            return state.board.copy(), -1.0, True, False, {**info, "reason": "occupied"}

        state.board[row, col] = 1
        state.transcript.append({"role": "miner", "move": move})

        winner = _winner(state.board)
        if winner == 1:
            reward = 1.0
            terminated = True
            reason = "win"
        elif _board_full(state.board):
            reward = 0.0
            terminated = True
            reason = "draw"
        else:
            state.miner_turn = False
            reward, terminated, reason = self._env_move()

        info.update(
            {
                "miner_first": True,
                "spec_hash": self.spec_hash(),
                "spec_version": self.spec_version(),
                "difficulty": int(self.challenge.info.get("difficulty", 0)),
                "opening_moves": list(self.challenge.info.get("opening_moves", [])),
                "transcript": list(state.transcript),
                "reason": reason,
            }
        )
        truncated = False
        return state.board.copy(), reward, terminated, truncated, info

    def _env_move(self) -> Tuple[float, bool, str]:
        if self._state is None:
            raise RuntimeError("Environment used before reset().")
        board = self._state.board
        score, move = _minimax(board.copy(), -1)
        if move is None:
            # draw
            self._state.miner_turn = True
            return 0.0, True, "draw"
        row, col = divmod(move, 3)
        board[row, col] = -1
        self._state.transcript.append({"role": "env", "move": move})
        winner = _winner(board)
        if winner == -1:
            return -1.0, True, "loss"
        if _board_full(board):
            return 0.0, True, "draw"
        self._state.miner_turn = True
        return 0.0, False, "continue"

    # -- Verification --------------------------------------------------
    def verify(self, response: Any, info: Mapping[str, Any]) -> Verdict:
        challenge_id = str(info.get("challenge_id", "")).lower()
        if not challenge_id:
            return Verdict(False, "missing-challenge")

        try:
            moves = _coerce_moves(response)
        except Exception as exc:
            return Verdict(False, f"parse-error:{exc}")

        board, _difficulty, _opening = _opening_for_challenge(challenge_id)
        board = board.copy()

        for move in moves:
            if move < 0 or move >= 9:
                return Verdict(False, f"illegal:{move}")
            row, col = divmod(move, 3)
            if board[row, col] != 0:
                return Verdict(False, f"occupied:{move}")
            board[row, col] = 1
            if _winner(board) == 1:
                return Verdict(True, "win")
            if _board_full(board):
                return Verdict(False, "draw")
            _score, env_move = _minimax(board.copy(), -1)
            if env_move is None:
                return Verdict(False, "draw")
            erow, ecol = divmod(env_move, 3)
            if board[erow, ecol] != 0:
                return Verdict(False, "judge-error:occupied")
            board[erow, ecol] = -1
            if _winner(board) == -1:
                return Verdict(False, "loss")
            if _board_full(board):
                return Verdict(False, "draw")

        # If miner transcript ended prematurely, assume environment completes optimal play
        while True:
            _score, env_move = _minimax(board.copy(), -1)
            if env_move is None:
                break
            erow, ecol = divmod(env_move, 3)
            board[erow, ecol] = -1
            if _winner(board) == -1:
                return Verdict(False, "loss")
            if _board_full(board):
                return Verdict(False, "draw")
            # Miner failed to move when required -> treat as loss
            return Verdict(False, "incomplete")

        return Verdict(False, "incomplete")

    def decode_action(self, response: Any) -> int:
        if isinstance(response, (int, np.integer)):
            return int(response)
        if isinstance(response, str):
            parts = [part for part in response.replace(",", " ").split() if part]
            for part in parts:
                try:
                    return int(part)
                except ValueError:
                    continue
            raise ValueError("response does not contain an integer move")
        if isinstance(response, Mapping):
            for key in ("move", "action", "index"):
                if key in response:
                    return int(response[key])
        raise TypeError(f"Cannot decode action from {type(response)!r}")


__all__ = ["TicTacToeEnv"]
