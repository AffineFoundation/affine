"""Staged lifecycle consumers: claim-before-job and pending/discard handoff.

These are callable adapters, not an installer. The caller must compose them
before legacy early-gate/job creation and freeze the native K1 capture. GREEN
handoff deliberately fails closed until native promotion authorization and its
CURRENT write seam have been qualified; a candidate publication is not an apply.
"""
from pathlib import Path
from .durable import Stage, atomic, digest, file_sha, read


class Pending:
    def __init__(self, reason):
        self.reason = reason


class LifecycleConsumers:
    def __init__(self, root, protocol, producer):
        self.stage, self.protocol, self.producer = Stage(root), protocol, producer
        if protocol.stage.root != self.stage.root or producer.stage.root != self.stage.root:
            raise ValueError('same isolated lifecycle/controller root required')

    @staticmethod
    def head(current):
        h = current['pe_head']
        if (h.get('version') != 'pe-head-v1' or digest(h) != current['pe_head_id']
                or h['checkpoint'] != current['checkpoint']['id']
                or any(not isinstance(h[k], str) or len(h[k]) != 64
                       or any(c not in '0123456789abcdef' for c in h[k])
                       for k in ('master_digest', 'generation_id'))):
            raise ValueError('generation-aware native current required; no bf16 fallback')
        return current['pe_head_id']

    def prepare_job(self, epoch, current, turn_path):
        """Nonblocking. Native job creation may run ONLY after this returns a claim."""
        path = Path(turn_path).absolute()
        if path.resolve() != path or not path.is_relative_to(self.stage.root):
            raise ValueError('frozen native K1 turn must remain inside stage')
        if type(epoch) is not int or epoch < 0 or current['epoch_index'] != epoch:
            raise ValueError('exact native epoch required')
        head = self.head(current); s = self.stage.load()
        if s['apply'] is not None or any(j['state'] == 'green' for j in s['jobs'].values()):
            return Pending('promotion-must-reconcile-before-next-job')
        if head != s['head'] or current['optimizer_step'] != s['optimizer_step']:
            raise ValueError('native CURRENT and PE head/step disagree')
        if s['trainer_at'] != head or any(j['state'] == 'pinning' for j in s['jobs'].values()):
            if not any(j['state'] == 'pinning' for j in s['jobs'].values()):
                self.producer.emit('restore', dict(epoch=epoch))
            return Pending('restore-or-durable-keep-not-ready')
        claim = dict(epoch=epoch, training_head=head, turn_sha256=file_sha(path))
        event = self.producer.emit('claim', claim)
        if not self.producer.acknowledged(event):
            return Pending('controller-claim-not-durable')
        if self.stage.load()['claims'].get(str(epoch)) != claim:
            raise ValueError('claim ACK does not match actual controller state')
        return claim

    def trained(self, epoch, candidate, artifact):
        claim = self.stage.load()['claims'][str(epoch)]
        binding = dict(epoch=epoch, candidate=candidate, training_head=claim['training_head'])
        if not self.protocol.verify('trained', binding, artifact):
            raise ValueError('real authenticated native artifact required')
        return self.producer.emit('trained', dict(binding, artifact=artifact))

    def result(self, epoch, current):
        """Advance the native epoch after keep/pins, NOT after the slow eval.

        A pending no-update preserves the head/optimizer and is NOT a miner RED.
        K1 selection must include pending reservations via K1Join.queue_rows().
        Promotion of a later GREEN is a separate qualified CURRENT transaction;
        it must never spend an unrelated current miner's epoch.
        """
        self.head(current)
        if type(epoch) is not int or current['epoch_index'] != epoch:
            raise ValueError('cannot finalize an unrelated native epoch')
        s = self.stage.load(); j = s['jobs'].get(str(epoch))
        if not j or not all(self.protocol._acked(s, j[k]) for k in ('keep', 'pin')):
            return Pending('durable-candidate-keep-and-pins-pending')
        if current['pe_head_id'] != s['head'] or current['optimizer_step'] != s['optimizer_step']:
            raise ValueError('CURRENT moved before native epoch handoff')
        # This result finalizes ONLY the training slot, not a queued promotion.
        # Even if its eval has already gone GREEN, adopt nothing here.
        state = j['outcome'] if j['outcome'] in {'red', 'no_decision'} else 'pending'
        return dict(status='no-update', checkpoint=current['checkpoint'],
                    optimizer_step=current['optimizer_step'],
                    checkpoint_authorization=current.get('checkpoint_authorization'),
                    pe_origin_epoch=epoch, pe_candidate=j['candidate'], pe_state=state)

    def promotion_result(self, current):
        self.head(current)
        if self.stage.load()['apply'] is not None:
            raise ValueError('native GREEN/CURRENT handoff not qualified; never consume another miner epoch')
        return Pending('no-queued-promotion')

    def observe_epoch_current(self, epoch, before, current_path, finalized_path, authority):
        """Join actual signed native FINALIZED and disk CURRENT before K1/ledger.

        Called at the native CURRENT write seam, before speculative pre-open.
        A guess about next epoch or a differently-scored future result is rejected.
        """
        from subnet.backend_jobs import signed
        for p in (Path(current_path).absolute(), Path(finalized_path).absolute()):
            if p.resolve() != p or not p.is_relative_to(self.stage.root):
                raise ValueError('isolated durable lifecycle records required')
        s = self.stage.load(); j = s['jobs'][str(epoch)]
        if before['epoch_index'] != epoch or self.head(before) != s['claims'][str(epoch)]['training_head']:
            raise ValueError('native epoch/input-head claim mismatch')
        if not all(self.protocol._acked(s, j[k]) for k in ('keep', 'pin')):
            raise ValueError('cannot finalize before durable candidate retention')
        current = read(current_path); envelope = read(finalized_path); final = signed(envelope, authority)
        result = final['result']
        if result.get('pe_state') != 'pending' and result.get('pe_state') != j['outcome']:
            raise ValueError('native finalization invented a gate decision')
        if (result.get('status') != 'no-update' or result.get('pe_origin_epoch') != epoch
                or result.get('pe_candidate') != j['candidate']
                or result.get('pe_state') not in {'pending', 'red', 'no_decision'}
                or result['checkpoint'] != before['checkpoint']
                or result['optimizer_step'] != before['optimizer_step']
                or result.get('checkpoint_authorization') != before.get('checkpoint_authorization')
                or final['input_checkpoint'] != before['checkpoint']['id']
                or current['epoch_index'] != before['epoch_index'] + 1
                or any(current[k] != before[k] for k in ('checkpoint', 'optimizer_step', 'pe_head', 'pe_head_id'))):
            raise ValueError('signed native FINALIZED/durable epoch CURRENT mismatch')
        self.head(current)
        value = dict(version='pe-native-pending-current-v1', epoch=epoch,
                     finalized_sha256=digest(envelope), current_sha256=file_sha(current_path),
                     pe_state=result['pe_state'], head=current['pe_head_id'],
                     candidate=j['candidate'], turn_sha256=s['claims'][str(epoch)]['turn_sha256'])
        atomic(self.stage.root / 'epoch-current' / (str(epoch) + '.json'), value)
        return value
