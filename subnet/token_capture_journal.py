"""Small fsynced per-epoch receipts, written after immutable R2 publication.

The gateway snapshot remains the terminal projection. An interrupted capture
replays this private authenticated journal before touching mutable staging.
"""
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import stat

from .storage import canonical


class CaptureJournal:
    def __init__(self, gateway, epoch, state):
        self.gateway, self.epoch, self.state = gateway, epoch, state
        self.path = None
        if getattr(gateway, 'state_path', None):
            base = Path(gateway.state_path)
            self.path = base.parent / (base.name + '.token-capture-' +
                hashlib.sha256(epoch.encode()).hexdigest() + '.jsonl')

    def validate(self, miner, receipt):
        pending = self.state['commitment_pending'].get(miner)
        if not pending or type(receipt) is not dict or set(receipt) != {
            'slot', 'sha256', 'size', 'frozen_key', 'captured_at', 'assurance'}:
            raise ValueError('exact token capture receipt')
        slot = receipt['slot']
        entries = [b for b in pending['document']['payload']['batches']
                   if type(slot) is int and b['slot'] == slot]
        if len(entries) != 1:
            raise ValueError('original token capture slot')
        child = entries[0]
        at = receipt['captured_at']
        cutoff = self.state['commitment_binding'].get('freeze_until')
        if (receipt['sha256'] != child['training_sha256'] or
            type(receipt['size']) is not int or receipt['size'] != child['training_size'] or
            receipt['frozen_key'] != pending['root'] + '/training/' + str(slot) + '.json' or
            receipt['assurance'] != 'unaudited' or type(at) not in (int, float) or
            not math.isfinite(at) or at < self.state['start'] or
            cutoff is not None and at >= cutoff):
            raise ValueError('original immutable token receipt binding')

    def open(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or
            info.st_nlink != 1 or info.st_mode & 0o077):
            os.close(fd)
            raise ValueError('private owned token journal')
        return fd

    def body(self, miner, receipt):
        return dict(version='durable-token-capture-receipt-v1', epoch=self.epoch,
                    checkpoint=self.state['commitment_binding']['checkpoint'],
                    miner=miner, receipt=receipt)

    def replay(self):
        if self.path is None:
            return
        fd = self.open()
        try:
            with os.fdopen(os.dup(fd), 'rb') as stream:
                offset = 0
                for line in stream:
                    if not line.endswith(b'\n'):
                        # A process interrupted during append never admitted this
                        # unfinished receipt. Preserve all complete records.
                        os.ftruncate(fd, offset)
                        os.fsync(fd)
                        break
                    value = json.loads(line)
                    if (type(value) is not dict or set(value) != {'body', 'mac'} or
                        type(value['body']) is not dict or type(value['mac']) is not str or
                        canonical(value) + b'\n' != line):
                        raise ValueError('canonical token capture journal framing')
                    body = value['body']
                    expected = hmac.new(self.gateway.secret, canonical(body), 'sha256').hexdigest()
                    if not hmac.compare_digest(value['mac'], expected):
                        raise ValueError('authenticated token capture journal')
                    miner, receipt = body['miner'], body['receipt']
                    if body != self.body(miner, receipt):
                        raise ValueError('exact token journal epoch/checkpoint')
                    self.validate(miner, receipt)
                    rows = self.state.setdefault('training_document_snapshots', {}).setdefault(miner, {})
                    previous = rows.get(str(receipt['slot']))
                    if previous is not None and previous != receipt:
                        raise ValueError('immutable token journal receipt conflict')
                    rows[str(receipt['slot'])] = receipt
                    offset += len(line)
        finally:
            os.close(fd)

    def append(self, miner, receipt):
        self.validate(miner, receipt)
        if self.path is None:
            return False
        body = self.body(miner, receipt)
        line = canonical(dict(body=body, mac=hmac.new(
            self.gateway.secret, canonical(body), 'sha256').hexdigest())) + b'\n'
        fd = self.open()
        try:
            os.lseek(fd, 0, os.SEEK_END)
            with os.fdopen(os.dup(fd), 'wb') as stream:
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())
            # Also persist the directory entry when this epoch first creates it.
            directory = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            os.close(fd)
        return True
