"""Idempotent consumption of isolated, signed lifecycle/controller outboxes."""
from pathlib import Path
from .durable import Stage, atomic, digest, read


class Inbox:
    def __init__(self, root, protocol, authority):
        self.stage, self.protocol, self.authority = Stage(root), protocol, authority
        self.pending = self.stage.root / 'inbox'
        self.pending.mkdir(exist_ok=True)
        (self.stage.root / 'inbox-acks').mkdir(exist_ok=True)

    def consume(self, path):
        from subnet.backend_jobs import signed
        path = Path(path)
        if path.parent != self.pending or path.is_symlink():
            raise ValueError('isolated inbox input required')
        envelope = read(path)
        event = signed(envelope, self.authority)
        eid = digest(envelope)
        ack = self.stage.root / 'inbox-acks' / (eid + '.json')
        if ack.exists():
            return read(ack)
        kind, body = event['kind'], event['body']
        if kind == 'claim':
            self.protocol.claim(**body)
        elif kind == 'trained':
            self.protocol.trained(**body)
        elif kind == 'restore':
            self.protocol.request_restore(**body)
        elif kind == 'abandon':
            self.protocol.abandon(**body)
        elif kind == 'outcome':
            self.protocol.outcome(**body)
        else:
            raise ValueError('unknown lifecycle event')
        # Crash before ACK replays the idempotent protocol operation, not a new
        # train/eval. Never delete the immutable incoming source record.
        value = dict(event_sha256=eid, consumed=True, kind=kind)
        atomic(ack, value)
        return value

    def tick(self):
        errors = []
        for p in sorted(self.pending.glob('*.json')):
            try:
                self.consume(p)
            except Exception as ex:
                # One incomplete event cannot block other restore/keep/eval lanes.
                errors.append(dict(file=p.name, error_type=type(ex).__name__))
        self.protocol.reconcile()
        return errors
