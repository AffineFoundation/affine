"""The private facts of a probe round, read from the files of the live gate.

Terms:

* epoch: one turn of one miner in the live queue. Batch ``e<N>`` is epoch N.
* round: the tasks that the live gate drew for its probes, with a secret
  seed. The gate writes ``SAMPLE.private.json`` for an epoch when it gives
  the epoch its round, and ``ROUND.private.json`` for the round.
* plan seed: the seed of one probe run. The candidate of an epoch has one.
  Each HEAD block of the round has one.
* HEAD block: one probe run of the HEAD on every task of the round. The live
  gate makes its baseline from ``head_blocks`` blocks of
  ``head_rollouts // head_blocks`` rollouts for each task.

This module only reads. Everything that it returns is private to the
validator host: it goes to a box over ssh, into a directory that the box job
removes, and never into an object store or the journal.

The two seed formulas are the formulas of ``baseline_v1`` (``cand_seed`` and
``plan_seed``) and of ``pe_round_seed``. A test compares them.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Protocol, Tuple

MAX_BYTES = 64 << 20
ROUND_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9._-]{0,127}')
HEX64 = re.compile(r'[0-9a-f]{64}')
SAME_RUN = 'same-run'


class RoundNotReady(RuntimeError):
    """The live gate has not given this epoch its round yet. Ask again later."""


class RoundError(RuntimeError):
    """The round files are not usable. Nothing is probed and nothing is judged."""


@dataclass(frozen=True)
class Round:
    round_id: str
    env_id: str
    indices: Tuple[int, ...]
    plan_seed: str
    block_seeds: Tuple[str, ...]
    k_cand: int
    k_block: int
    max_tokens: int
    min_cover: float

    @property
    def blocks(self) -> int:
        return len(self.block_seeds)

    def __repr__(self) -> str:
        # The seeds and the task ids stay out of every log.
        return 'Round(%s, %d tasks, %d blocks)' % (self.round_id, len(self.indices), self.blocks)


class Rounds(Protocol):
    def of(self, epoch: int) -> Round:
        """The round of this epoch. ``RoundNotReady``: not drawn yet."""


def cand_seed(secret: str, epoch: int, attempt: int = 1) -> str:
    text = '%s:prior-baseline:cand:%d' % (secret, epoch)
    if attempt > 1:
        text += ':attempt:%d' % attempt
    return hashlib.sha256(text.encode()).hexdigest()


def block_seed(secret: str, block: int) -> str:
    return hashlib.sha256(('%s:prior-baseline:%d' % (secret, block)).encode()).hexdigest()


def _read(path: Path) -> Any:
    try:
        if path.stat().st_size > MAX_BYTES:
            return None
        return json.loads(path.read_bytes())
    except (OSError, ValueError, RecursionError):
        return None


def _count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


class HostRounds:
    """``Rounds`` on the state directory of the live gate (the parent of
    ``epochs/`` and ``heldout/``)."""

    def __init__(self, gate_state: Path, *, max_tokens: Optional[int] = None):
        """``max_tokens`` is the token cap of a round whose settings name none
        (the live gate then reads its caps file). The scorer refuses a
        baseline that was made with another cap than the cap of the round."""
        self.epochs = Path(gate_state) / 'epochs'
        self.heldout = Path(gate_state) / 'heldout'
        self.max_tokens = max_tokens

    def of(self, epoch: int) -> Round:
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise RoundError('an epoch is a number from zero up')
        sample = _read(self.epochs / str(epoch) / 'SAMPLE.private.json')
        if not isinstance(sample, Mapping) or not sample.get('round_id'):
            raise RoundNotReady('epoch %d has no round yet' % epoch)
        round_id = sample['round_id']
        if not isinstance(round_id, str) or ROUND_ID.fullmatch(round_id) is None or '..' in round_id:
            raise RoundError('epoch %d names a round id that is not valid' % epoch)
        doc = _read(self.heldout / round_id / 'ROUND.private.json')
        if not isinstance(doc, Mapping) or not doc.get('seed') or not doc.get('indices'):
            raise RoundNotReady('round %s is not drawn yet' % round_id)
        return self._round(epoch, round_id, sample, doc)

    def _round(self, epoch: int, round_id: str, sample: Mapping[str, Any], doc: Mapping[str, Any]) -> Round:
        secret, indices, cfg = doc['seed'], doc['indices'], doc.get('cfg')
        if not isinstance(secret, str) or not isinstance(indices, list) or not isinstance(cfg, Mapping):
            raise RoundError('round %s is not valid' % round_id)
        if not all(isinstance(i, int) and not isinstance(i, bool) and i >= 0 for i in indices) or len(
                set(indices)) != len(indices):
            raise RoundError('round %s has task ids that are not valid' % round_id)
        if sample.get('indices') is not None and list(sample['indices']) != indices:
            raise RoundError('the sample of epoch %d is not the draw of round %s' % (epoch, round_id))
        if cfg.get('scoring_mode') == SAME_RUN:
            # Such a round scores each turn against a HEAD run of its own turn. The baseline here is one for a round.
            raise RoundError('round %s scores each turn against its own HEAD run' % round_id)
        env_id, min_cover = cfg.get('probe_env'), cfg.get('min_cover')
        numbers = [cfg.get(key) for key in ('k_cand', 'head_rollouts', 'head_blocks')]
        numbers.append(cfg.get('max_tokens') or self.max_tokens)
        if (not isinstance(env_id, str) or not env_id or not all(_count(n) for n in numbers)
                or not isinstance(min_cover, (int, float)) or isinstance(min_cover, bool) or not 0 < min_cover <= 1):
            raise RoundError('round %s has settings that are not valid' % round_id)
        k_cand, head_rollouts, head_blocks, max_tokens = numbers
        if head_rollouts // head_blocks < 1:
            raise RoundError('round %s has no rollouts for a HEAD block' % round_id)
        attempt = sample.get('attempt') or 1
        if not _count(attempt):
            raise RoundError('the sample of epoch %d has an attempt that is not valid' % epoch)
        named = sample.get('plan_seed')
        if attempt >= 2 and isinstance(named, str) and HEX64.fullmatch(named) is not None:
            plan_seed = named
        else:
            plan_seed = cand_seed(secret, epoch, attempt)
        return Round(
            round_id=round_id, env_id=env_id, indices=tuple(indices), plan_seed=plan_seed,
            block_seeds=tuple(block_seed(secret, block) for block in range(head_blocks)),
            k_cand=k_cand, k_block=head_rollouts // head_blocks, max_tokens=max_tokens, min_cover=float(min_cover),
        )


__all__ = ['HostRounds', 'Round', 'RoundError', 'RoundNotReady', 'Rounds', 'block_seed', 'cand_seed']
