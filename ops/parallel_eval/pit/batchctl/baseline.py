"""The HEAD baseline of a round, made of blocks that the boxes of this run probe.

A candidate is judged against rollouts of the HEAD that it was trained from
(the baseline). The baseline must come from the same hardware class and the
same serving layout as the probe of the candidate. So the boxes of this run
make it, and the baseline of the serial gate is not used.

One baseline belongs to one round and one HEAD checkpoint. It has
``Round.blocks`` blocks. A block is one probe run of the HEAD on every task of
the round, with ``Round.k_block`` rollouts for each task. A box runs a block
after the probe of its own batch, so the blocks of one baseline come from
different boxes. When HEAD changes, the next boxes start from the new HEAD,
and they make its baseline in the same way.

This store is a directory on the validator host. It is private: a block names
the tasks of the round. Two kinds of files:

``block-<n>.json``    one stored block. Written once.
``claim-f<fence>.json``  the blocks that the box of one lease runs. Written
                      once, so a launch that is sent again names the same
                      blocks.
``released-f<fence>`` the result of that lease was read. What it claimed and
                      did not bring is free for another box.
``ANCHOR.json``       the fingerprint (hardware class, layout, sampling) of
                      the first block. Each later block must have the same.

A block that a lost box held is free again when its lease is not live any more.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, Collection, Dict, List, Mapping, Optional, Tuple

from .layout import is_hex64
from .rounds import Round

BLOCK_SCHEMA = 'affine-batchctl-head-block-v1'
CLAIM_SCHEMA = 'affine-batchctl-head-claim-v1'
ANCHOR_SCHEMA = 'affine-batchctl-head-anchor-v1'
BLOCK_FILE = re.compile(r'block-(0|[1-9][0-9]{0,3})\.json')
CLAIM_FILE = re.compile(r'claim-f(0|[1-9][0-9]{0,17})\.json')
RELEASE_FILE = re.compile(r'released-f(0|[1-9][0-9]{0,17})')
MAX_BYTES = 256 << 20


class BaselineError(RuntimeError):
    """The store cannot be read or written. Nothing is judged."""


def _read(path: Path) -> Any:
    try:
        if path.stat().st_size > MAX_BYTES:
            return None
        return json.loads(path.read_bytes())
    except (OSError, ValueError, RecursionError):
        return None


def _write_once(path: Path, doc: Mapping[str, Any]) -> bool:
    """Write a file whole. False: the file was there before, and it stays."""
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    tmp = path.with_name('.%s.%d.%d.tmp' % (path.name, os.getpid(), threading.get_ident()))
    with os.fdopen(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), 'w') as handle:
        handle.write(json.dumps(doc, sort_keys=True, allow_nan=False))
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(tmp, path)
    except FileExistsError:
        return False
    finally:
        os.unlink(tmp)
    return True


def graded(cells: Any) -> Dict[str, Dict[str, Any]]:
    """The cells of tasks that have a rollout with a grade."""
    if not isinstance(cells, Mapping):
        return {}
    return {str(index): dict(cell) for index, cell in cells.items() if isinstance(cell, Mapping) and cell.get('n')}


class Baselines:
    """Thread-safe. Every method reads the directory again, so a restart needs no load step."""

    def __init__(self, root: Path, *, clock: Callable[[], float] = time.time):
        self.root = Path(root)
        self.clock = clock
        self._lock = threading.Lock()

    def _dir(self, rnd: Round, checkpoint: str) -> Path:
        if not is_hex64(checkpoint):
            raise BaselineError('a baseline belongs to one checkpoint')
        return self.root / rnd.round_id / checkpoint

    # -- reading -------------------------------------------------------------

    def _blocks(self, rnd: Round, checkpoint: str) -> Dict[int, Dict[str, Any]]:
        out: Dict[int, Dict[str, Any]] = {}
        folder = self._dir(rnd, checkpoint)
        for number in range(rnd.blocks):
            doc = _read(folder / ('block-%d.json' % number))
            if (isinstance(doc, dict) and doc.get('schema') == BLOCK_SCHEMA and doc.get('block') == number
                    and doc.get('round_id') == rnd.round_id and doc.get('checkpoint_id') == checkpoint
                    and doc.get('k') == rnd.k_block and doc.get('max_tokens') == rnd.max_tokens
                    and isinstance(doc.get('cells'), dict)):
                out[number] = doc
        return out

    def _claims(self, rnd: Round, checkpoint: str) -> Dict[int, Tuple[int, ...]]:
        out: Dict[int, Tuple[int, ...]] = {}
        folder = self._dir(rnd, checkpoint)
        try:
            names = sorted(os.listdir(folder))
        except OSError:
            return out
        for name in names:
            found = CLAIM_FILE.fullmatch(name)
            doc = _read(folder / name) if found is not None else None
            if isinstance(doc, dict) and doc.get('schema') == CLAIM_SCHEMA and isinstance(doc.get('blocks'), list):
                out[int(found.group(1))] = tuple(b for b in doc['blocks'] if isinstance(b, int))
        return out

    def missing(self, rnd: Round, checkpoint: str) -> List[int]:
        stored = self._blocks(rnd, checkpoint)
        return [number for number in range(rnd.blocks) if number not in stored]

    def _released(self, rnd: Round, checkpoint: str) -> List[int]:
        try:
            names = os.listdir(self._dir(rnd, checkpoint))
        except OSError:
            return []
        return [int(found.group(1)) for found in map(RELEASE_FILE.fullmatch, names) if found is not None]

    def free(self, rnd: Round, checkpoint: str, live: Collection[int]) -> List[int]:
        """The blocks that are not stored and that no live lease can still bring."""
        gone = set(self._released(rnd, checkpoint))
        held = {block for fence, blocks in self._claims(rnd, checkpoint).items()
                if fence in live and fence not in gone for block in blocks}
        return [number for number in self.missing(rnd, checkpoint) if number not in held]

    def anchor(self, rnd: Round, checkpoint: str) -> Optional[Dict[str, Any]]:
        doc = _read(self._dir(rnd, checkpoint) / 'ANCHOR.json')
        if isinstance(doc, dict) and doc.get('schema') == ANCHOR_SCHEMA and isinstance(doc.get('fingerprint'), dict):
            return doc['fingerprint']
        return None

    def complete(self, rnd: Round, checkpoint: str) -> Optional[Dict[str, Any]]:
        """The blocks of a complete baseline, or None while a block is missing."""
        stored = self._blocks(rnd, checkpoint)
        anchor = self.anchor(rnd, checkpoint)
        if anchor is None or len(stored) != rnd.blocks:
            return None
        return {
            'blocks': [stored[number]['cells'] for number in range(rnd.blocks)],
            'fingerprint': anchor, 'rollouts': rnd.k_block * rnd.blocks, 'max_tokens': rnd.max_tokens,
            'done_at': max(float(stored[number]['stored_at']) for number in stored),
            'sources': [stored[number].get('source') for number in range(rnd.blocks)],
        }

    # -- writing -------------------------------------------------------------

    def claim(self, rnd: Round, checkpoint: str, fence: int, *, live: Collection[int], others: int) -> Tuple[int, ...]:
        """The blocks that the box of this lease runs after its own probe.

        ``live`` are the leases that still run or wait for a verdict.
        ``others`` is the number of boxes that will start from this checkpoint
        soon. With enough of them, a box takes one block and the blocks are
        split across boxes. With none, the box takes each free block, so the
        baseline does not wait for a box that never comes.
        """
        with self._lock:
            path = self._dir(rnd, checkpoint) / ('claim-f%d.json' % fence)
            known = self._claims(rnd, checkpoint).get(fence)
            if known is not None:
                return known
            free = self.free(rnd, checkpoint, [f for f in live if f != fence])
            take = tuple(free[:max(1, len(free) - max(0, others))])
            _write_once(path, {'schema': CLAIM_SCHEMA, 'fence': fence, 'blocks': list(take), 'at': self.clock()})
            return self._claims(rnd, checkpoint).get(fence, ())

    def release(self, rnd: Round, checkpoint: str, fence: int) -> None:
        """The result of this lease was read. A block that it claimed and did not bring is free again."""
        with self._lock:
            _write_once(self._dir(rnd, checkpoint) / ('released-f%d' % fence), {'fence': fence, 'at': self.clock()})

    def add(self, rnd: Round, checkpoint: str, block: int, *, k: Any, max_tokens: Any, cells: Any,
            fingerprint: Any, source: Mapping[str, Any]) -> Optional[str]:
        """Store one block. -> None when it is stored (now or before), else why it is refused."""
        if not isinstance(block, int) or isinstance(block, bool) or not 0 <= block < rnd.blocks:
            return 'no block of the round'
        if k != rnd.k_block or max_tokens != rnd.max_tokens:
            return 'another number of rollouts or another token cap than the round has'
        if not isinstance(fingerprint, Mapping) or not is_hex64(fingerprint.get('sha256')):
            return 'no hardware and layout fingerprint'
        wanted = {str(index) for index in rnd.indices}
        kept = {index: cell for index, cell in graded(cells).items() if index in wanted}
        if len(kept) < rnd.min_cover * len(wanted):
            return 'graded tasks %d of %d: under the cover of the round' % (len(kept), len(wanted))
        with self._lock:
            folder = self._dir(rnd, checkpoint)
            _write_once(folder / 'ANCHOR.json', {'schema': ANCHOR_SCHEMA, 'fingerprint': dict(fingerprint)})
            anchor = self.anchor(rnd, checkpoint)
            if anchor is None:
                raise BaselineError('the fingerprint of the baseline cannot be read')
            if anchor.get('sha256') != fingerprint['sha256']:
                return 'another hardware and layout fingerprint than the first block has'
            _write_once(folder / ('block-%d.json' % block), {
                'schema': BLOCK_SCHEMA, 'round_id': rnd.round_id, 'checkpoint_id': checkpoint, 'block': block,
                'k': rnd.k_block, 'max_tokens': rnd.max_tokens, 'cells': kept, 'stored_at': self.clock(),
                'source': dict(source),
            })
            if block not in self._blocks(rnd, checkpoint):
                raise BaselineError('block %d cannot be read after it was written' % block)
        return None


__all__ = ['BaselineError', 'Baselines', 'graded']
