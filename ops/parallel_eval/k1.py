"""Atomic native-schema K1 joins; infrastructure failures do not spend a turn.

The pre-training lifecycle writes only a frozen pending capture. Only this join
owns the staged committed native ledger. Existing FIFO/registration/grant queue
logic can read these files unchanged; it must not append independently.
"""
import json
import os
import tempfile
from pathlib import Path
from .durable import Stage, atomic, canonical, digest, file_sha, fsync_dir, read


class K1Join:
    def __init__(self, root):
        self.stage = Stage(root)
        self.root = self.stage.root / 'native-k1'
        self.root.mkdir(exist_ok=True)
        if self.root.resolve() != self.root:
            raise ValueError('symlink native K1 root')
        self.turns = self.root / 'K1-TURNS.jsonl'
        self.grants = self.root / 'K1-GRANTS.json'
        self.pending = self.root / 'pending'
        self.pending.mkdir(exist_ok=True)
        self.joins = self.root / 'joins'
        self.joins.mkdir(exist_ok=True)
        if any(p.resolve() != p for p in (self.pending, self.joins)):
            raise ValueError('symlink K1 capture/join directory')

    def _rows(self):
        if self.turns.is_symlink():
            raise ValueError('symlink K1 ledger')
        return [json.loads(line) for line in self.turns.read_bytes().splitlines()] if self.turns.exists() else []

    def _save_rows(self, rows):
        fd, name = tempfile.mkstemp(prefix='.K1-', dir=self.root)
        try:
            with os.fdopen(fd, 'wb') as f:
                for row in rows:
                    f.write(canonical(row) + b'\n')
                f.flush(); os.fsync(f.fileno())
            os.replace(name, self.turns)
            fsync_dir(self.root)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def initialize(self, rows, grants):
        with self.stage.locked():
            if self.turns.exists() or self.grants.exists():
                raise ValueError('native K1 already initialized')
            if len({r['epoch_index'] for r in rows}) != len(rows) or not isinstance(grants.get('grants'), list):
                raise ValueError('unique native K1 epochs and native grants required')
            self._save_rows(rows)
            atomic(self.grants, grants)

    @staticmethod
    def _epoch(epoch):
        if type(epoch) is not int or epoch < 0:
            raise ValueError('nonnegative native epoch')
        return str(epoch)

    def capture(self, epoch, public, native_row):
        key = self._epoch(epoch)
        if public['status'] != 'turn' or public['epoch_index'] != epoch or native_row['epoch_index'] != epoch:
            raise ValueError('actual selected native turn required')
        if any(native_row.get(k) != public.get(k) for k in ('hotkey', 'file_sha256', 'head', 'optimizer_step') if k in native_row):
            raise ValueError('native row/source binding')
        value = dict(public=public, row=native_row)
        path = self.pending / (key + '.json')
        with self.stage.locked():
            if path.exists():
                if read(path) != value:
                    raise ValueError('frozen native turn changed')
            else:
                atomic(path, value)
        # Claim digest is the actual immutable public turn file, not a timestamp
        # regenerated on every lifecycle retry.
        turn_path = self.pending / (key + '-TURN.json')
        with self.stage.locked():
            if turn_path.exists():
                if read(turn_path) != public:
                    raise ValueError('frozen public turn changed')
            else:
                atomic(turn_path, public)
        return file_sha(turn_path)

    def queue_rows(self, before_epoch=None):
        """Unchanged native FIFO counts include unresolved reservations ONCE.

        Terminal no-decision joins remove only their pending reservation; their
        retry grant uses the existing native priority rule. Physical committed
        rows never need editing/reversal to undo an infrastructure attempt.
        """
        with self.stage.locked():
            rows = self._rows(); epochs = {r['epoch_index'] for r in rows}
            for p in sorted(self.pending.glob('*.json')):
                if p.name.endswith('-TURN.json'):
                    continue
                capture = read(p); row = capture['row']; epoch = row['epoch_index']
                joined = self.joins / (str(epoch) + '.json')
                if epoch in epochs or joined.exists():
                    continue
                rows.append(row); epochs.add(epoch)
            return sorted((r for r in rows if before_epoch is None or r['epoch_index'] < before_epoch),
                          key=lambda r: r['epoch_index'])

    def join(self, payload, *, fault=lambda phase: None):
        epoch = payload['epoch']; key = self._epoch(epoch)
        if payload['verdict'] not in {'green', 'red', 'no_decision'}:
            raise ValueError('typed K1 verdict required')
        consume = payload['verdict'] != 'no_decision'
        if payload['consume_turn'] is not consume or payload['retry_grant'] is not (not consume):
            raise ValueError('infrastructure cannot consume a native turn')
        with self.stage.locked():
            capture = read(self.pending / (key + '.json'))
            if file_sha(self.pending / (key + '-TURN.json')) != payload['turn_sha256']:
                raise ValueError('K1 claim/source digest')
            path = self.joins / (key + '.json')
            if path.exists():
                receipt = read(path)
                if receipt['request_sha256'] != digest(payload):
                    raise ValueError('K1 join replay conflict')
                return receipt
            rows = self._rows(); matches = [r for r in rows if r['epoch_index'] == epoch]
            if consume:
                if matches and matches != [capture['row']]:
                    raise ValueError('native K1 ledger conflict')
                if not matches:
                    rows.append(capture['row']); self._save_rows(rows)
            elif matches:
                raise ValueError('infrastructure turn was prematurely committed')
            fault('ledger-durable')
            grants = read(self.grants)
            if not consume:
                grant = dict(hotkey=capture['row']['hotkey'], red_epoch=epoch,
                             reason='pe-no-decision', source_turn_sha256=payload['turn_sha256'])
                old = [g for g in grants['grants'] if g['hotkey'] == grant['hotkey'] and g['red_epoch'] == epoch]
                if old and old != [grant]:
                    raise ValueError('native retry grant conflict')
                if not old:
                    grants['grants'].append(grant); atomic(self.grants, grants)
            fault('grant-durable')
            receipt = dict(version='pe-native-k1-join-v1', request_sha256=digest(payload),
                           epoch=epoch, turn_sha256=payload['turn_sha256'], consume_turn=consume,
                           retry_grant=not consume, native_row_sha256=digest(capture['row']) if consume else None,
                           ledger_sha256=file_sha(self.turns), grants_sha256=file_sha(self.grants))
            atomic(path, receipt)
            return receipt
