"""Real filesystem producer and restart-safe effect worker, isolated roots only.

No production lifecycle is installed implicitly. Callers supply admitted effects;
receipts still pass Protocol's verifier. A process lock prevents duplicate physical
execution even after a protocol lease expires. Effects must reconcile their own
remote command id before retrying an interrupted call: a local lock is NOT a
remote quiescence proof.
"""
import fcntl
from .durable import Stage, atomic, digest, read


class Producer:
    def __init__(self, root, identity, inbox):
        self.stage, self.identity, self.inbox = Stage(root), identity, inbox
        if inbox.stage.root != self.stage.root or identity.id != inbox.authority:
            raise ValueError('producer root/authority mismatch')

    def emit(self, kind, body):
        from subnet.mimo_release_policy import sign
        if kind not in {'claim', 'trained', 'restore', 'abandon', 'outcome'}:
            raise ValueError('unknown producer event')
        event = sign(self.identity, dict(kind=kind, body=body))
        eid = digest(event)
        path = self.inbox.pending / (eid + '.json')
        with self.stage.locked():
            if path.exists():
                if read(path) != event:
                    raise ValueError('immutable producer event conflict')
            else:
                atomic(path, event)
        return eid

    def acknowledged(self, eid):
        if len(eid) != 64 or any(c not in '0123456789abcdef' for c in eid):
            raise ValueError('event digest')
        from subnet.backend_jobs import signed
        p = self.stage.root / 'inbox-acks' / (eid + '.json')
        if not p.exists():
            return False
        envelope = read(self.inbox.pending / (eid + '.json'))
        if digest(envelope) != eid:
            raise ValueError('producer event changed after emission')
        event = signed(envelope, self.identity.id)
        return read(p) == dict(event_sha256=eid, consumed=True, kind=event['kind'])


class EffectWorker:
    def __init__(self, root, protocol, name, effects):
        self.stage, self.protocol = Stage(root), protocol
        if protocol.stage.root != self.stage.root or not name or not effects:
            raise ValueError('same-root named effect worker required')
        self.name, self.effects = name, effects
        self.journal = self.stage.root / 'effects'
        self.journal.mkdir(exist_ok=True)
        if self.journal.resolve() != self.journal:
            raise ValueError('symlink effect journal')

    def tick(self, *, now=None, ttl=120, after_receipt=lambda command: None):
        self.protocol.reconcile()
        command = self.protocol.lease(set(self.effects), self.name, now=now, ttl=ttl)
        if command is None:
            return None
        key = command['id']
        path, lock_path = self.journal / (key + '.json'), self.journal / (key + '.lock')
        if lock_path.is_symlink():
            raise ValueError('symlink effect lock')
        with lock_path.open('a') as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return dict(command=key, state='physical-effect-still-running')
            if path.exists():
                row = read(path)
                if row['command'] != key or row['request_sha256'] != digest(command['payload']):
                    raise ValueError('cached physical receipt binding')
                receipt = row['receipt']
            else:
                def heartbeat():
                    return self.protocol.renew(command, ttl=ttl)
                receipt = self.effects[command['kind']](command, heartbeat)
                if not self.protocol.verify(command['kind'], command['payload'], receipt):
                    raise ValueError('physical effect returned unauthenticated receipt')
                atomic(path, dict(command=key, request_sha256=digest(command['payload']), receipt=receipt))
            # Death here must reuse the signed receipt, not rerun a publication.
            after_receipt(command)
            self.protocol.ack(command, receipt)
            self.protocol.reconcile()
            return dict(command=key, state='acked')
