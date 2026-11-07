"""Durable PE domain/outbox, staging only. No SSH, rental or service operations.

Network side effects are leased to independent workers; reconciliation never
waits for a restore, model publication or eval. External adapters MUST enforce
command-id idempotence. An ACK is accepted only through an explicit verifier.
"""
import copy
import time
import uuid
from .durable import Stage, digest


class Protocol:
    def __init__(self, root, verifier):
        self.stage = Stage(root)
        self.verify = verifier
        if not callable(verifier):
            raise TypeError('explicit authenticated receipt verifier required')

    def initialize(self, head, optimizer_step, round_spec):
        self._validate_round(round_spec)
        with self.stage.locked():
            if (self.stage.root / 'STATE.json').exists():
                s = self.stage.load()
                if s['genesis'] != head:
                    raise ValueError('genesis conflict')
                return
            s = dict(version='pe-protocol-v1', genesis=head, head=head,
                     optimizer_step=optimizer_step, chain=[], rounds={}, jobs={},
                     claims={}, commands={}, trainer_at=None, apply=None, epoch_failures={}, publications={}, retirements=[])
            self._round(s, head, round_spec)
            s['head_pin'] = self._command(s, 'pin-head', head, dict(head=head))
            self.stage.save(s)

    @staticmethod
    def _command(s, kind, key, payload):
        cid = digest(dict(kind=kind, key=key, payload=payload))
        row = dict(id=cid, kind=kind, key=key, payload=payload, state='pending', attempts=0)
        old = s['commands'].get(cid)
        if old and (old['kind'] != kind or old['payload'] != payload):
            raise ValueError('command conflict')
        s['commands'].setdefault(cid, row)
        return cid

    @staticmethod
    def _validate_round(spec):
        exact = dict(n_drawn=500, k_cand=8, head_rollouts=12, max_output_tokens=65536, max_num_seqs=512, dtype='bfloat16')
        if any(spec.get(k) != v for k,v in exact.items()):
            raise ValueError('exact live probe sample/serving contract required')
        for key in ('draw_sha256', 'settings_sha256'):
            value = spec.get(key)
            if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('pinned draw/settings digests required')

    def _round(self, s, head, spec):
        self._validate_round(spec)
        # One immutable draw/baseline per head; no turn-count refresh path.
        if head in s['rounds']:
            if s['rounds'][head]['spec'] != spec:
                raise ValueError('second draw for same head')
            return
        s['rounds'][head] = dict(head=head, spec=copy.deepcopy(spec), state='pinning', closed=False)
        s['rounds'][head]['pin'] = self._command(s, 'pin', 'round:'+head, dict(owner='round:'+head, heads=[head]))

    def claim(self, epoch, training_head, turn_sha256):
        """Must precede old-gate early-candidate creation, not follow training."""
        with self.stage.locked():
            s = self.stage.load()
            key = str(epoch)
            if not isinstance(turn_sha256, str) or len(turn_sha256) != 64 or any(c not in '0123456789abcdef' for c in turn_sha256):
                raise ValueError('immutable K1 turn digest required')
            claim = dict(epoch=epoch, training_head=training_head, turn_sha256=turn_sha256)
            if key in s['claims']:
                if s['claims'][key] != claim:
                    raise ValueError('epoch already claimed differently')
                return
            if not self._acked(s, s['head_pin']) or s['trainer_at'] != s['head'] or any(j['state'] == 'pinning' for j in s['jobs'].values()):
                raise ValueError('restore and durable candidate keep/pins required before training')
            if training_head != s['head']:
                raise ValueError('training must start at current head')
            s['claims'][key] = claim
            self.stage.save(s)

    def trained(self, epoch, candidate, training_head, artifact):
        if not self.verify('trained', dict(epoch=epoch, candidate=candidate, training_head=training_head), artifact):
            raise ValueError('unauthenticated trained candidate')
        with self.stage.locked():
            s = self.stage.load()
            key = str(epoch)
            claim = s['claims'].get(key)
            if not claim or claim['epoch'] != epoch or claim['training_head'] != training_head or key in s['epoch_failures']:
                raise ValueError('claim-before-training required')
            incoming = dict(epoch=epoch, candidate=candidate, training_head=training_head, artifact=artifact)
            if key in s['jobs']:
                if any(s['jobs'][key][k] != v for k, v in incoming.items()):
                    raise ValueError('candidate replay conflict')
                return
            dispatch_head = s['head']
            job = dict(incoming, state='pinning', dispatch_head=dispatch_head,
                       round_head=dispatch_head, outcome=None)
            job['keep'] = self._command(s, 'keep', key, incoming)
            job['pin'] = self._command(s, 'pin', key, dict(owner=key, heads=[training_head, candidate]))
            s['trainer_at'] = candidate
            s['jobs'][key] = job
            self.stage.save(s)

    @staticmethod
    def _acked(s, cid):
        return cid is not None and s['commands'][cid]['state'] == 'acked'

    def request_restore(self, epoch):
        with self.stage.locked():
            s = self.stage.load()
            if any(j['state'] == 'pinning' for j in s['jobs'].values()):
                raise ValueError('candidate keep and master pins must precede next training')
            cid = self._command(s, 'restore', str(epoch), dict(epoch=epoch, head=s['head']))
            self.stage.save(s)
            return cid

    def lease(self, kinds, worker, now=None, ttl=120):
        now = time.time() if now is None else now
        if ttl <= 0:
            raise ValueError('positive lease required')
        with self.stage.locked():
            s = self.stage.load()
            for c in s['commands'].values():
                if c['kind'] not in kinds or c['state'] in ('acked', 'cancelled'):
                    continue
                if c['state'] == 'leased' and c['lease_until'] > now:
                    continue
                c.update(state='leased', worker=worker, lease=uuid.uuid4().hex,
                         lease_until=now + ttl, attempts=c['attempts'] + 1)
                self.stage.save(s)
                return copy.deepcopy(c)
        return None

    def renew(self, command, now=None, ttl=120):
        """Renew only an unexpired physical-effect lease, never steal one back."""
        now = time.time() if now is None else now
        if ttl <= 0:
            raise ValueError('positive lease required')
        with self.stage.locked():
            s = self.stage.load()
            c = s['commands'][command['id']]
            if c['state'] != 'leased' or c['lease'] != command['lease'] or c['lease_until'] <= now:
                raise ValueError('lost or expired lease cannot renew')
            c['lease_until'] = now + ttl
            self.stage.save(s)
            return c['lease_until']

    def ack(self, command, receipt):
        # Verify outside state lock; an external readback must not stall lane watchdogs.
        if not self.verify(command['kind'], command['payload'], receipt):
            raise ValueError('unauthenticated or mismatched side-effect receipt')
        with self.stage.locked():
            s = self.stage.load()
            c = s['commands'][command['id']]
            if c['state'] == 'acked':
                if c['receipt'] != receipt:
                    raise ValueError('ACK replay conflict')
                return
            if c['state'] != 'leased' or c['lease'] != command['lease']:
                raise ValueError('lost lease cannot ACK')
            c.update(state='acked', receipt=copy.deepcopy(receipt))
            self.stage.save(s)

    @staticmethod
    def _score_binding(s, epoch, verdict, green_at):
        j = s['jobs'][str(epoch)]
        rd = s['rounds'][j['round_head']]
        if 'baseline' not in rd:
            raise ValueError('dispatch head baseline not ready')
        return dict(epoch=epoch, candidate=j['candidate'], round_head=j['round_head'], verdict=verdict,
                    spec=rd['spec'], baseline=rd['baseline'], green_at=green_at if verdict == 'green' else None)

    def outcome(self, epoch, verdict, score_receipt, green_at=None):
        if verdict not in ('green', 'red', 'no_decision'):
            raise ValueError('unknown outcome')
        binding = self._score_binding(self.stage.load(), epoch, verdict, green_at)
        if not self.verify('score', binding, score_receipt):
            raise ValueError('score not authenticated/bound to dispatch round')
        with self.stage.locked():
            s = self.stage.load()
            j = s['jobs'][str(epoch)]
            if self._score_binding(s, epoch, verdict, green_at) != binding:
                raise ValueError('score binding changed during verification')
            if j['outcome'] is not None:
                if j['outcome'] != verdict or j['score'] != score_receipt:
                    raise ValueError('decision replay conflict')
                return
            if j['state'] != 'evaluating':
                raise ValueError('candidate not dispatched')
            j.update(outcome=verdict, score=score_receipt)
            if verdict == 'green':
                if green_at is None or not isinstance(green_at, (int, float)) or not (0 <= green_at < float('inf')):
                    raise ValueError('durable GREEN completion time required')
                j.update(state='green', green_at=green_at)
            else:
                self._final(s, j, verdict, 'gate' if verdict == 'red' else 'infrastructure')
            self.stage.save(s)

    def abandon(self, epoch, reason, quiescence_receipt):
        """Exhausted infra/restore/missing-master failure, not a miner RED.

        The trusted adapter MUST attest that all leased effects for this epoch
        are stopped/fenced before issuing this receipt. Revoking a lease alone
        cannot prove a remote restore/pin process has ceased writing.
        """
        if not isinstance(reason, str) or not reason or len(reason) > 200:
            raise ValueError('bounded infrastructure reason')
        claim = self.stage.load()['claims'][str(epoch)]
        binding = dict(epoch=epoch, claim=claim, reason=reason, quiesced_required=True)
        if not self.verify('abandon', binding, quiescence_receipt):
            raise ValueError('authenticated worker quiescence proof required')
        with self.stage.locked():
            s = self.stage.load()
            key = str(epoch)
            if s['claims'][key] != claim:
                raise ValueError('claim changed during verification')
            j = s['jobs'].get(key)
            if key in s['epoch_failures']:
                if s['epoch_failures'][key]['reason'] != reason:
                    raise ValueError('epoch failure replay conflict')
                return
            if j and j['state'] in ('applying', 'recording', 'releasing', 'complete'):
                raise ValueError('already applying/final; reconcile durable CURRENT instead')
            for c in s['commands'].values():
                if c['key'] == key and c['kind'] in ('restore', 'keep', 'pin', 'dispatch') and c['state'] != 'acked':
                    c['state'] = 'cancelled'
            if j:
                j['outcome'] = 'no_decision'
                self._final(s, j, 'no_decision', reason)
                s['epoch_failures'][key] = dict(reason=reason, candidate=True, quiescence_receipt=quiescence_receipt)
            else:
                p = dict(epoch=epoch, candidate=None, verdict='no_decision', reason=reason,
                         turn_sha256=claim['turn_sha256'])
                ledger = self._command(s, 'ledger', key, p)
                k1 = self._command(s, 'k1', key, dict(p, consume_turn=False, retry_grant=True))
                s['epoch_failures'][key] = dict(reason=reason, candidate=False, ledger=ledger, k1=k1, state='recording', quiescence_receipt=quiescence_receipt)
            self.stage.save(s)

    def _final(self, s, j, verdict, reason):
        key = str(j['epoch'])
        j.update(state='recording', final=verdict, reason=reason)
        binding = dict(epoch=j['epoch'], candidate=j['candidate'], training_head=j['training_head'],
                       dispatch_head=j['dispatch_head'], verdict=verdict, reason=reason,
                       turn_sha256=s['claims'][key]['turn_sha256'])
        j['ledger'] = self._command(s, 'ledger', key, binding)
        j['k1'] = self._command(s, 'k1', key, dict(binding, consume_turn=verdict != 'no_decision',
                                                 retry_grant=verdict == 'no_decision'))

    def reconcile(self):
        with self.stage.locked():
            s = self.stage.load()
            for rd in s['rounds'].values():
                if rd['state'] == 'pinning' and self._acked(s, rd['pin']):
                    rd['command'] = self._command(s, 'baseline', rd['head'], dict(head=rd['head'], spec=rd['spec']))
                    rd['state'] = 'preparing'
                if rd['state'] == 'preparing' and self._acked(s, rd['command']):
                    rd['state'] = 'ready'
                    rd['baseline'] = s['commands'][rd['command']]['receipt']
                    self._command(s, 'unpin', 'round:'+rd['head'], dict(owner='round:'+rd['head'], heads=[rd['head']]))
            for c in s['commands'].values():
                if c['kind'] == 'restore' and c['state'] == 'acked' and not c.get('reconciled'):
                    if c['payload']['head'] == s['head']:
                        s['trainer_at'] = s['head']
                    c['reconciled'] = True
            for key, j in s['jobs'].items():
                if j['state'] == 'pinning' and all(self._acked(s, j[k]) for k in ('keep', 'pin')):
                    j['dispatch_head'] = j['round_head'] = s['head']
                    j['dispatch'] = self._command(s, 'dispatch', key, dict(epoch=j['epoch'], candidate=j['candidate'],
                        training_head=j['training_head'], round_head=j['round_head'], spec=s['rounds'][j['round_head']]['spec']))
                    j['state'] = 'dispatching'
                if j['state'] == 'dispatching' and self._acked(s, j['dispatch']):
                    j['state'] = 'evaluating'
                if j['state'] == 'recording' and all(self._acked(s, j[k]) for k in ('ledger', 'k1')):
                    j['release'] = self._command(s, 'unpin', key, dict(owner=key, heads=[j['training_head'], j['candidate']]))
                    j['state'] = 'releasing'
                if j['state'] == 'releasing' and self._acked(s, j['release']):
                    j['state'] = 'complete'
            if self._acked(s, s['head_pin']):
                for retired in s['retirements']:
                    if retired['state'] == 'waiting':
                        self._command(s, 'unpin-head', retired['head'], dict(head=retired['head'], publication=retired['publication']))
                        retired['state'] = 'requested'
            for failure in s['epoch_failures'].values():
                if failure.get('state') == 'recording' and all(self._acked(s, failure[k]) for k in ('ledger', 'k1')):
                    failure['state'] = 'complete'
            if s['apply'] is None:
                greens = sorted((j for j in s['jobs'].values() if j['state'] == 'green'), key=lambda j: (j['green_at'], j['epoch']))
                if greens:
                    j = greens[0]
                    heads = [s['genesis']] + [a['head'] for a in s['chain']]
                    indices = [i for i, h in enumerate(heads) if h == j['training_head']]
                    stale = len(heads) - 1 - indices[-1] if indices else None
                    if stale is None or stale > 2:
                        self._final(s, j, 'no_decision', 'stale-overflow')
                    else:
                        p = dict(epoch=j['epoch'], candidate=j['candidate'], training_head=j['training_head'],
                                 base_head=s['head'], staleness=stale, optimizer_step=s['optimizer_step'] + 1,
                                 artifact=j['artifact'])
                        cid = self._command(s, 'publish', str(j['epoch']), p)
                        s['apply'] = dict(job=str(j['epoch']), base_head=s['head'], publish=cid, state='publishing')
                        j['state'] = 'applying'
            a = s['apply']
            if a and a['state'] == 'publishing' and self._acked(s, a['publish']):
                receipt = s['commands'][a['publish']]['receipt']
                a['offer'] = self._command(s, 'offer', a['job'], dict(base_head=a['base_head'], publication=receipt))
                a['state'] = 'offered'
            self.stage.save(s)

    def confirm_current(self, finalized, current, next_round_spec):
        """Only actual durable CURRENT + matching signed FINALIZED can close a round."""
        preview = self.stage.load()
        a = preview['apply']
        if a is None:
            return False
        if a['state'] != 'offered' or not self._acked(preview, a['offer']):
            raise ValueError('apply not offered/acknowledged')
        binding = dict(base_head=a['base_head'], publication=preview['commands'][a['publish']]['receipt'])
        if not self.verify('durable-current', binding, dict(finalized=finalized, current=current)):
            raise ValueError('FINALIZED and actual CURRENT binding required')
        with self.stage.locked():
            s = self.stage.load()
            if s['apply'] != a:
                raise ValueError('apply changed during verification')
            new_head = current['pe_head_id']
            if current['pe_head'].get('version') != 'pe-head-v1' or digest(current['pe_head']) != new_head or current['pe_head']['checkpoint'] != current['checkpoint']['id']:
                raise ValueError('generation-aware CURRENT head identity')
            if new_head in [s['genesis']] + [row['head'] for row in s['chain']]:
                raise ValueError('duplicate head generation; CURRENT adapter must bind master digest and generation id')
            if current['optimizer_step'] != s['optimizer_step'] + 1 or s['head'] != a['base_head']:
                raise ValueError('head or optimizer changed under apply')
            j = s['jobs'][a['job']]
            s['chain'].append(dict(epoch=j['epoch'], candidate=j['candidate'], head=new_head,
                                   base_head=a['base_head'], green_at=j['green_at']))
            s['rounds'][a['base_head']]['closed'] = True
            s['retirements'].append(dict(head=a['base_head'], publication=s['publications'].get(a['base_head']), state='waiting'))
            s['publications'][new_head] = binding['publication']
            s['head_pin'] = self._command(s, 'pin-head', new_head, dict(head=new_head))
            s['head'] = new_head
            s['optimizer_step'] = current['optimizer_step']
            self._round(s, new_head, next_round_spec)
            self._final(s, j, 'green', 'durable-apply')
            s['apply'] = None
            self.stage.save(s)
            return True
