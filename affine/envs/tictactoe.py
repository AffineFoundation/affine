from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Mapping, MutableMapping, Tuple

import numpy as np
from gymnasium import spaces

from ..core.rng import make_rng
from ..core.types import Verdict
from .base import AffineEnv

Board = np.ndarray


def _winner(board: Board) -> int | None:
    """Return 1 if X wins, -1 if O wins, None otherwise."""
    for i in range(3):
        if np.all(board[i, :] == 1) or np.all(board[:, i] == 1):
            return 1
        if np.all(board[i, :] == -1) or np.all(board[:, i] == -1):
            return -1
    if np.all(board.diagonal() == 1) or np.all(np.fliplr(board).diagonal() == 1):
        return 1
    if np.all(board.diagonal() == -1) or np.all(np.fliplr(board).diagonal() == -1):
        return -1
    return None


def _minimax(board: Board, player: int) -> Tuple[int, int | None]:
    """Return (score, best_move) for player. Score from env perspective (-1)."""
    winner = _winner(board)
    if winner:
        return (1 if winner == -1 else -1, None)
    moves = [i for i in range(9) if board.flat[i] == 0]
    if not moves:
        return (0, None)

    best_score = -2 if player == -1 else 2
    best_move = moves[0]
    for move in moves:
        r, c = divmod(move, 3)
        board[r, c] = player
        score, _ = _minimax(board, -player)
        board[r, c] = 0
        if (player == -1 and score > best_score) or (player == 1 and score < best_score):
            best_score, best_move = score, move
            if (player == -1 and score == 1) or (player == 1 and score == -1):
                break
    return best_score, best_move


def _parse_moves(data: Any) -> List[int]:
    """Extract list of integers from various formats."""
    if isinstance(data, int):
        return [data]
    if isinstance(data, str):
        return [int(x) for x in data.replace(",", " ").split() if x.isdigit()]
    if isinstance(data, dict):
        if "moves" in data:
            return _parse_moves(data["moves"])
        if "miner_moves" in data:
            return _parse_moves(data["miner_moves"])
    if isinstance(data, (list, tuple)):
        result = []
        for item in data:
            if isinstance(item, int):
                result.append(item)
            elif isinstance(item, dict) and "move" in item:
                result.append(int(item["move"]))
            elif isinstance(item, dict) and "action" in item:
                result.append(int(item["action"]))
            elif isinstance(item, dict) and "miner" in item:
                result.append(int(item["miner"]))
        return result
    raise TypeError(f"Cannot parse moves from {type(data)}")


@dataclass
class TicTacToeState:
    board: Board
    miner_turn: bool
    transcript: List[dict]


class TicTacToeEnv(AffineEnv):
    """Deterministic TicTacToe with perfect opponent."""

    metadata = {"env_id": "tictactoe-v0", "spec_version": 1}

    def __init__(self):
        super().__init__()
        self.observation_space = spaces.Box(low=-1, high=1, shape=(3, 3), dtype=np.int8)
        self.action_space = spaces.Discrete(9)
        self._state: TicTacToeState | None = None

    def _reset(self, *, rng: np.random.Generator, info: MutableMapping[str, Any], options: Mapping[str, Any]) -> Tuple[Board, dict]:
        board = np.zeros((3, 3), dtype=np.int8)
        transcript = []
        # Generate opening position
        pairs = int(rng.integers(0, 4))
        moves = list(rng.permutation(9))
        for i in range(pairs):
            if i * 2 >= len(moves):
                break
            m = int(moves[i * 2])
            r, c = divmod(m, 3)
            if board[r, c] != 0:
                continue
            board[r, c] = 1
            transcript.append(m)
            if _winner(board) or not (board == 0).any():
                board[r, c] = 0
                transcript.pop()
                break
            _, em = _minimax(board.copy(), -1)
            if em is None:
                board[r, c] = 0
                transcript.pop()
                break
            er, ec = divmod(em, 3)
            board[er, ec] = -1
            transcript.append(em)
            if _winner(board) or not (board == 0).any():
                board[er, ec] = 0
                transcript.pop()
                break

        self._state = TicTacToeState(board.copy(), True, [])
        info.update({"miner_first": True, "difficulty": len(transcript) // 2, "opening_moves": transcript})
        return board.copy(), {}

    def step(self, action: Any):
        state = self._state
        if not state or not state.miner_turn:
            raise RuntimeError("Invalid state")

        move = int(action)
        info = {"challenge_id": self.challenge.challenge_id, "env_id": self.env_id()}

        if not self.action_space.contains(move):
            state.transcript.append({"role": "miner", "move": move, "illegal": True})
            return state.board.copy(), -1.0, True, False, {**info, "reason": "illegal"}

        r, c = divmod(move, 3)
        if state.board[r, c] != 0:
            state.transcript.append({"role": "miner", "move": move, "illegal": True})
            return state.board.copy(), -1.0, True, False, {**info, "reason": "occupied"}

        state.board[r, c] = 1
        state.transcript.append({"role": "miner", "move": move})

        winner = _winner(state.board)
        if winner == 1:
            reward, terminated, reason = 1.0, True, "win"
        elif not (state.board == 0).any():
            reward, terminated, reason = 0.0, True, "draw"
        else:
            _, em = _minimax(state.board.copy(), -1)
            if em is None:
                reward, terminated, reason = 0.0, True, "draw"
            else:
                er, ec = divmod(em, 3)
                state.board[er, ec] = -1
                state.transcript.append({"role": "env", "move": em})
                winner = _winner(state.board)
                if winner == -1:
                    reward, terminated, reason = -1.0, True, "loss"
                elif not (state.board == 0).any():
                    reward, terminated, reason = 0.0, True, "draw"
                else:
                    reward, terminated, reason = 0.0, False, "continue"

        info.update({
            "miner_first": True,
            "spec_hash": self.spec_hash(),
            "spec_version": self.spec_version(),
            "difficulty": self.challenge.info.get("difficulty", 0),
            "opening_moves": self.challenge.info.get("opening_moves", []),
            "transcript": list(state.transcript),
            "reason": reason,
        })
        return state.board.copy(), reward, terminated, False, info

    def verify(self, response: Any, info: Mapping[str, Any]) -> Verdict:
        challenge_id = info.get("challenge_id")
        if not challenge_id:
            return Verdict(False, "missing-challenge")

        try:
            moves = _parse_moves(response)
        except Exception as e:
            return Verdict(False, f"parse-error:{e}")

        rng = make_rng(self.env_id(), self.spec_version(), str(challenge_id))
        board = np.zeros((3, 3), dtype=np.int8)

        # Apply opening moves (regenerate)
        pairs = int(rng.integers(0, 4))
        perm = list(rng.permutation(9))
        for i in range(pairs):
            if i * 2 >= len(perm):
                break
            m = int(perm[i * 2])
            r, c = divmod(m, 3)
            if board[r, c] != 0:
                continue
            board[r, c] = 1
            if _winner(board) or not (board == 0).any():
                board[r, c] = 0
                break
            _, em = _minimax(board.copy(), -1)
            if em is None:
                board[r, c] = 0
                break
            er, ec = divmod(em, 3)
            board[er, ec] = -1
            if _winner(board) or not (board == 0).any():
                board[er, ec] = 0
                break

        # Replay miner moves
        for move in moves:
            if move < 0 or move >= 9:
                return Verdict(False, f"illegal:{move}")
            r, c = divmod(move, 3)
            if board[r, c] != 0:
                return Verdict(False, f"occupied:{move}")
            board[r, c] = 1
            if _winner(board) == 1:
                return Verdict(True, "win")
            if not (board == 0).any():
                return Verdict(False, "draw")
            _, em = _minimax(board.copy(), -1)
            if em is None:
                return Verdict(False, "draw")
            er, ec = divmod(em, 3)
            board[er, ec] = -1
            if _winner(board) == -1:
                return Verdict(False, "loss")
            if not (board == 0).any():
                return Verdict(False, "draw")

        return Verdict(False, "incomplete")

    def decode_action(self, response: Any) -> int:
        if isinstance(response, (int, np.integer)):
            return int(response)
        if isinstance(response, str):
            parts = [x for x in response.replace(",", " ").split() if x.isdigit()]
            if parts:
                return int(parts[0])
        if isinstance(response, dict):
            for key in ("move", "action", "index"):
                if key in response:
                    return int(response[key])
        raise ValueError(f"Cannot decode action from {type(response)}")


__all__ = ["TicTacToeEnv"]
