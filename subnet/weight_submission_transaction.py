"""Durable boundary for one hourly weight transaction.

The caller holds the existing global writer lock; ChainAdapter additionally
holds weights.lock. All planning, preparation and signing are pre-broadcast.
Only submit_prepared() can send, exactly once, after fsynced intent + signed
identity + cursor. No old transaction is ever rebroadcast during recovery.

SDK execute() has no signed-call hook and replans. This adapter deliberately
uses its documented prepare/attach/submit-signed transport flow instead; the
locally inspected SDK seams are pinned by the operator's writer module policy.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from subnet.weight_submission_reconciliation import (
    validate_era,
    BittensorReadonlyReader, collect_evidence, journal_intent, make_intent,
    reconcile, validate_intent,
)


def sdk_seam_paths():
    # Candidate uses documented transport methods through private Client members;
    # these precise Python seams must be independently pinned before activation.
    import importlib.util
    modules = ('bittensor.intents.weights', 'bittensor.executor', 'bittensor._substrate',
               'bittensor._transport.interface', 'bittensor._transport.codec',
               'bittensor._transport.contract', 'bittensor.signing',
               'bittensor_core', 'bittensor_core.bittensor_core')
    return {str(Path(importlib.util.find_spec(name).origin).resolve()) for name in modules}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def durable_atomic(path, value):
    """Unique temporary, fsync file, rename, fsync directory."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        fd = os.open(temp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'wb') as stream:
            stream.write(canonical(value) + b'\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temp, path)
        fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    finally:
        if temp.exists():
            temp.unlink()


def _unsigned(value, bits, label):
    if type(value) is not int or not 0 <= value < 1 << bits:
        raise ValueError('invalid '+label)
    return value


def _compact(value):
    # SCALE compact length. Our bounded vectors/ciphertext never require mode3.
    _unsigned(value, 30, 'SCALE compact length')
    if value < 64: return bytes([value << 2])
    if value < 16384: return ((value << 2) | 1).to_bytes(2, 'little')
    return ((value << 2) | 2).to_bytes(4, 'little')


def _weights_payload(hotkey, uids, values, version_key):
    """Exact Rust WeightsTlockPayload: Vec<u8>, Vec<u16>, Vec<u16>, u64."""
    if type(hotkey) is not bytes or len(hotkey) != 32 or not 0 < len(uids) == len(values) <= 65536:
        raise ValueError('bounded exact owner/normalized weights payload')
    if len(set(uids)) != len(uids): raise ValueError('duplicate normalized UID')
    encoded = _compact(32) + hotkey
    for vector in (uids, values):
        encoded += _compact(len(vector)) + b''.join(_unsigned(x, 16, 'u16 weight vector').to_bytes(2, 'little') for x in vector)
    return encoded + _unsigned(version_key, 64, 'version key').to_bytes(8, 'little')


def _inner_ciphertext(envelope, expected_round):
    # Installed encrypt_at_round returns SCALE UserData(Vec<u8>, u64), whereas
    # the weights extrinsic expects only its inner compressed TLE ciphertext.
    if type(envelope) is not bytes or not 9 < len(envelope) <= 2*1024**2:
        raise ValueError('bounded native timelock envelope')
    mode=envelope[0] & 3
    if mode == 3: raise ValueError('unbounded native ciphertext length')
    width=(1,2,4)[mode];length=int.from_bytes(envelope[:width], 'little') >> 2
    if envelope[:width] != _compact(length) or len(envelope) != width+length+8 or length == 0:
        raise ValueError('canonical exact native timelock envelope')
    if int.from_bytes(envelope[-8:], 'little') != expected_round:
        raise ValueError('ciphertext encryption round differs from call round')
    return envelope[width:width+length]


async def _own_pending_rounds(substrate, owner, netuid, mechid):
    if mechid != 0: raise ValueError('only qualified mechanism zero')
    number=await substrate.block_number();block_hash=await substrate.block_hash(number)
    own=[];seen=0
    for name,width in (('TimelockedWeightCommits',4),('CRV3WeightCommits',3),('CRV3WeightCommitsV2',4)):
        rows=await substrate.query_map('SubtensorModule',name,[netuid],block_hash=block_hash)
        for epoch,queue in rows:
            _unsigned(epoch,64,'pending epoch')
            if not isinstance(queue,(list,tuple)):raise ValueError('unknown pending queue shape')
            for entry in queue:
                seen+=1
                if seen>65536 or not isinstance(entry,(list,tuple)) or len(entry)!=width or not isinstance(entry[0],str):
                    raise ValueError('bounded known pending commit shape')
                if entry[0]==owner:
                    round_number=_unsigned(entry[-1],64,'own pending reveal round')
                    if round_number==0:raise ValueError('zero own reveal round')
                    own.append(dict(storage=name,epoch=epoch,reveal_round=round_number))
    return dict(block=number,block_hash=block_hash,netuid=netuid,mecid=mechid,owner=owner,pending=own)


async def _epoch_snapshot(substrate, owner, netuid, mechid):
    snapshot=await _own_pending_rounds(substrate,owner,netuid,mechid)
    schedule={}
    for name in ('Tempo','RevealPeriodEpochs','LastEpochBlock','PendingEpochAt','SubnetEpochIndex','BlocksSinceLastStep'):
        value=await substrate.query('SubtensorModule',name,[netuid],block_hash=snapshot['block_hash'])
        schedule[name]=_unsigned(value,16 if name=='Tempo' else 64,'epoch schedule '+name)
    if schedule['Tempo']==0 or schedule['RevealPeriodEpochs']==0 or schedule['LastEpochBlock']>snapshot['block']:
        raise ValueError('active known timelock epoch schedule required')
    snapshot['schedule']=schedule
    available=[]
    for number in sorted({r['reveal_round'] for r in snapshot['pending'] if r['epoch']==schedule['SubnetEpochIndex']}):
        if await substrate.query('Drand','Pulses',[number],block_hash=snapshot['block_hash']) is not None:
            available.append(number)
    snapshot['already_available_same_epoch_rounds']=available
    return snapshot


def _epoch_round(snapshot, computed, *, era_death, now):
    """Reuse one exact round only within a stable current commit epoch.

    Cover the entire existing mortal era, not just SDK's next-block guess.
    This conservative predicate covers both installed SDK and inspected runtime
    safety-net variants; a scheduled transition defers before signing.
    """
    schedule=snapshot['schedule'];block=snapshot['block'];epoch=schedule['SubnetEpochIndex']
    same=[r for r in snapshot['pending'] if r['epoch']==epoch]
    future=[r for r in snapshot['pending'] if r['epoch']>epoch]
    if future:raise ValueError('pending future epoch requires a fresh settled snapshot')
    if not same:return computed,'SDK_no_same_epoch_pending'
    # Legacy stores have a separate processing order. Do not infer FIFO across
    # distinct queues, even when their pulse number happens to match.
    if any(r['storage']!='TimelockedWeightCommits' for r in same):
        raise ValueError('same-epoch legacy queue cannot establish current FIFO')
    rounds={r['reveal_round'] for r in same}
    if len(rounds)!=1:raise ValueError('mixed same-epoch pending encryption rounds')
    last_valid=era_death-1
    delta=last_valid-block
    if delta<0 or delta>128:raise ValueError('bounded live mortal inclusion window')
    next_auto=schedule['LastEpochBlock']+schedule['Tempo']
    pending=schedule['PendingEpochAt']
    if (last_valid>=next_auto or (pending and last_valid>=pending)
            or schedule['BlocksSinceLastStep']+delta>=schedule['Tempo']):
        raise ValueError('epoch transition inside mortal inclusion window; retry after boundary')
    chosen=next(iter(rounds))
    if chosen in snapshot['already_available_same_epoch_rounds']:
        raise ValueError('same-epoch pulse is already available; no public-round encryption')
    # Quicknet genesis/period are pinned by the native SDK. Never intentionally
    # encrypt a fresh submission to an already public/past round. This is a
    # minimum secrecy margin, not a claim that wall time predicts chain progress.
    if 1692803367+(chosen-1)*3 <= now+60:
        raise ValueError('same-epoch round is already public or too near; no past-round encryption')
    return chosen,'same_epoch_exact_pending_round'


async def _build_epoch_timelocked(substrate, owner, hotkey, netuid, mechid, uids, values, version_key):
    from bittensor.intents.weights import _build_timelocked, _core, calls
    from bittensor.intents.base import BuiltCall
    pending=await _epoch_snapshot(substrate,owner,netuid,mechid)
    class PinnedBlock:
        def __getattr__(self,name):return getattr(substrate,name)
        async def block_number(self):return pending['block']
        async def block_hash(self,number):
            if number!=pending['block']:raise ValueError('SDK changed pinned schedule block')
            return pending['block_hash']
    # Validate before native encryption as well as immediately before signing.
    _epoch_round(pending,1,era_death=pending['block']+64,now=time.time())
    built=await _build_timelocked(PinnedBlock(),hotkey,netuid,mechid,uids,values,version_key,4)
    computed=_unsigned(built.extras['reveal_round'],64,'SDK reveal round')
    chosen,reason=_epoch_round(pending,computed,era_death=pending['block']+64,now=time.time())
    if chosen==0:raise ValueError('zero chosen reveal round')
    if chosen!=computed:
        envelope,actual=_core.encrypt_at_round(_weights_payload(hotkey,uids,values,version_key),chosen)
        if actual!=chosen:raise ValueError('native encryption returned another round')
        ciphertext=_inner_ciphertext(envelope,chosen)
        call=await substrate.compose(calls.SubtensorModule.commit_timelocked_mechanism_weights(
            netuid=netuid,mecid=mechid,commit=ciphertext,reveal_round=chosen,commit_reveal_version=4))
        built=BuiltCall(call,dict(built.extras,reveal_round=chosen))
    guard=dict(version='own-pending-same-epoch-encryption-round-v1',**pending,
        SDK_computed_round=computed,chosen_encryption_round=chosen,selection_reason=reason,
        ciphertext_reencrypted=chosen!=computed,call_sha256=hashlib.sha256(built.call.data).hexdigest())
    return built,guard


async def _check_epoch_before_sign(substrate, guard, era):
    fresh=await _epoch_snapshot(substrate,guard['owner'],guard['netuid'],guard['mecid'])
    if await substrate.block_hash(guard['block']) != guard['block_hash']:
        raise ValueError('planning block changed before signing')
    # No same-owner mutation may be hidden between planning and signing. A new
    # block is normal, but a queue/epoch/schedule change requires a new plan.
    stable=('Tempo','RevealPeriodEpochs','LastEpochBlock','PendingEpochAt','SubnetEpochIndex')
    if fresh['pending']!=guard['pending'] or any(fresh['schedule'][k]!=guard['schedule'][k] for k in stable):
        raise ValueError('own pending queue or epoch schedule changed before signing')
    # The pinned SDK anchors period-only mortal eras at the finalized head,
    # which may precede the planning best head. Require canonical ancestry and
    # a still-live exact64 era instead of incorrectly requiring a newer birth.
    era=validate_era(era)
    if not (0<=era['birth']<=fresh['block']<era['death'] and guard['block']<=fresh['block']):
        raise ValueError('actual era is not anchored to fresh planning state')
    if await substrate.block_hash(era['birth']) != era['block_hash']:
        raise ValueError('actual era birth block is not canonical before signing')
    if await substrate.block_hash(fresh['block']) != fresh['block_hash']:
        raise ValueError('fresh planning block changed before signing')
    chosen,reason=_epoch_round(fresh,guard['SDK_computed_round'],era_death=era['death'],now=time.time())
    if chosen!=guard['chosen_encryption_round'] or reason!=guard['selection_reason']:
        raise ValueError('encryption epoch/round changed before signing')
    return dict(snapshot=fresh,era=era,checked_at=time.time())

def make_recorded_intent(bt, *, netuid, uids, weights, version_key):
    """Same SDK SetWeights recipe, exposing the vector used by THIS plan.

    No global monkeypatch. The supported Intent subclass seam delegates to the
    installed weight helpers; those private helper files must be policy-pinned.
    Only direct timelocked CR4/mecid0 is qualified here.
    """
    from bittensor.intents.weights import _preflight, _conform

    class RecordedSetWeights(bt.SetWeights):
        async def build(self, substrate, wallet):
            preflight = await _preflight(substrate, self.hotkey_address(wallet), self.netuid, self.mechid)
            norm_uids, values = _conform(self.uids, self.weights, preflight, self.netuid)
            if self.mechid != 0 or not preflight.commit_reveal:
                raise ValueError('only current direct timelocked mechanism is qualified')
            if preflight.uid in norm_uids:
                raise ValueError('owner allocation is forbidden')
            built, guard = await _build_epoch_timelocked(substrate, self.hotkey_address(wallet), self.hotkey_public_key(wallet),
                self.netuid, self.mechid, norm_uids, values, self.version_key)
            self.journal_reveal_guard = guard
            self.journal_vector = sorted([int(uid), int(value)] for uid, value in zip(norm_uids, values))
            self.journal_owner_uid = int(preflight.uid)
            self.journal_call_sha256 = hashlib.sha256(built.call.data).hexdigest()
            return built

    return RecordedSetWeights(netuid=netuid, uids=uids, weights=weights, version_key=version_key)


def prepare_exact_plan(chain, plan, intent, wallet):
    """Read, sign and assemble but do NOT transmit. Returns exact SDK object."""
    if not plan.ok or plan.signer != 'hotkey' or plan.signer_address != wallet.hotkey.ss58_address:
        raise ValueError('actual direct owner plan required')
    if hashlib.sha256(plan.call.data).hexdigest() != intent.journal_call_sha256:
        raise ValueError('normalized vector is not bound to this plan')
    unsigned = chain.prepare_call(plan.call, address=plan.signer_address,
                                  crypto_type=wallet.hotkey.crypto_type, period=64)
    if unsigned.address != plan.signer_address or unsigned.call_data != plan.call.data:
        raise ValueError('prepared transaction changed owner or call')
    # The actual prepared mortal era is signed into this transaction. Period64
    # has phase granularity1, so the SDK's normalized current is its birth.
    if not isinstance(unsigned.era, dict) or set(unsigned.era) != {'period', 'current'}:
        raise ValueError('actual normalized mortal era required')
    intent.journal_era = validate_era(dict(period=unsigned.era['period'],
        birth=unsigned.era['current'], death=unsigned.era['current'] + 64,
        block_hash=unsigned.era_block_hash))
    guard=getattr(intent,'journal_reveal_guard',None)
    if guard is not None:
        guard['pre_sign']=chain._call(_check_epoch_before_sign(chain._client._substrate,guard,intent.journal_era))
    signature = wallet.hotkey.sign(unsigned.payload)
    # Documented SDK transport assembly method, no network broadcast. Pin SDK.
    signed = chain._call(chain._client._substrate.raw.attach_signature(unsigned, signature))
    expected_hash = '0x' + hashlib.blake2b(signed.data, digest_size=32).hexdigest()
    if signed.extrinsic_hash != expected_hash:
        raise ValueError('SDK signed extrinsic hash mismatch')
    return signed, int(unsigned.nonce)


def submit_prepared(chain, signed, wallet):
    """Only broadcast path. No planning, re-signing, retry or second ciphertext."""
    return chain._call(chain._client._substrate.submit_signed(
        signed, wallet.hotkey, wait_for_inclusion=True, wait_for_finalization=True))


class SubmissionJournal:
    def __init__(self, cursor_path, chain_state, *, assessment_sha256=None,
                 policy_sha256=None, authority=None):
        self.cursor_path, self.chain_state = Path(cursor_path), Path(chain_state)
        self.assessment_sha256, self.policy_sha256 = assessment_sha256, policy_sha256
        self.authority = authority

    def begin(self, *, owner, owner_uid, netuid, window_end, vector, registrations,
              attempt_start_block, signed, nonce, era=None, reveal_guard=None):
        """Fails before broadcast. Orphan journals are never treated as sent."""
        if self.cursor_path.exists():
            cursor = json.loads(self.cursor_path.read_text())
            if cursor.get('status') == 'submitting':
                raise RuntimeError('uncertain chain outcome must be reconciled')
        document = make_intent(owner=owner, owner_uid=owner_uid, netuid=netuid,
            window_end=window_end, expected_weights=vector,
            assessment_sha256=self.assessment_sha256, policy_sha256=self.policy_sha256,
            registrations_sha256=digest(registrations), attempt_start_block=attempt_start_block,
            nonce_hint=nonce)
        directory = self.cursor_path.parent / 'attempts' / (str(window_end) + '-' + uuid.uuid4().hex)
        directory.mkdir(parents=True, mode=0o700)
        for parent in (directory.parent, directory.parent.parent):
            fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        journal_intent(directory / 'intent.json', document)
        observation = dict(intent_sha256=document['sha256'], extrinsic_hash=signed.extrinsic_hash,
                           nonce=nonce, signed_bytes_sha256=hashlib.sha256(signed.data).hexdigest())
        if era is not None: observation['era'] = validate_era(era)
        if reveal_guard is not None:
            if (reveal_guard['version'] != 'own-pending-same-epoch-encryption-round-v1' or reveal_guard['owner'] != owner
                    or reveal_guard['netuid'] != netuid or reveal_guard['mecid'] != 0 or reveal_guard['chosen_encryption_round'] <= 0
                    or reveal_guard['pre_sign']['era'] != era):
                raise ValueError('exact same-epoch encryption guard journal')
            observation['reveal_guard'] = reveal_guard
        durable_atomic(directory / 'signed-observation.json', observation)
        cursor = dict(status='submitting', window_end=window_end,
            assessment_sha256=self.assessment_sha256, policy_sha256=self.policy_sha256,
            intent_sha256=document['sha256'], attempt_directory=str(directory),
            signed_observation_sha256=digest(observation))
        # This fsynced fence is the last local operation before the sole send.
        durable_atomic(self.cursor_path, cursor)
        return cursor

    def recover(self, chain, *, reader=None):
        """Caller holds both existing writer locks. Proven terminal states only."""
        cursor = json.loads(self.cursor_path.read_text())
        if cursor.get('status') != 'submitting':
            return dict(status='no_recovery_required')
        # An old cursor has no actual signed identity. Never guess/clear it.
        if not cursor.get('attempt_directory'):
            raise RuntimeError('legacy uncertain chain outcome requires actual-chain reconciliation')
        directory = Path(cursor['attempt_directory'])
        if directory.parent.resolve() != (self.cursor_path.parent / 'attempts').resolve() or directory.is_symlink():
            raise ValueError('attempt journal outside local cursor namespace')
        document = json.loads((directory / 'intent.json').read_text())
        intent = validate_intent(document)
        observation = json.loads((directory / 'signed-observation.json').read_text())
        if (document['sha256'] != cursor['intent_sha256'] or
            digest(observation) != cursor['signed_observation_sha256'] or
            intent['window_end'] != cursor['window_end'] or
            intent['assessment_sha256'] != cursor['assessment_sha256'] or
            intent['policy_sha256'] != cursor['policy_sha256'] or
            observation['intent_sha256'] != document['sha256']):
            raise ValueError('attempt journal integrity mismatch')
        evidence = collect_evidence(reader or BittensorReadonlyReader(chain), document,
                                    max_blocks=128, budget_seconds=90, actual_signed_observation=observation)
        decision = reconcile(document, evidence, actual_signed_observation=observation)
        receipt = dict(original_cursor=cursor, evidence=evidence, decision=decision)
        durable_atomic(directory / ('reconciliation-' + digest(receipt) + '.json'), receipt)
        if decision['preserve_fence']:
            return decision
        if decision['status'] in ('failed_finalized', 'expired_unincluded'):
            # A proven terminal non-submission does NOT count as paid. Preserve
            # the old chain cursor; the writer may now attempt the current hour.
            terminal = dict(cursor, status=decision['status'], recovered=True,
                            reconciliation_sha256=digest(receipt))
            durable_atomic(directory / 'terminal-cursor.json', terminal)
            durable_atomic(self.cursor_path, terminal)
            return decision
        status_path = self.chain_state / 'weights.json'
        state = json.loads(status_path.read_text()) if status_path.exists() else {}
        resolved = intent['window_end']
        if state.get('last_submitted_window', -1) <= resolved:
            state['last_submitted_window'] = max(state.get('last_submitted_window', -1), resolved)
            state['latest'] = dict(status='submitted', window_end=resolved,
                recovered=True, assessment_sha256=intent['assessment_sha256'],
                reconciliation_sha256=digest(receipt), block_hash=decision['commit']['block_hash'],
                extrinsic_hash=decision['commit']['extrinsic_hash'])
            durable_atomic(status_path, state)
        # State first: a crash between writes only causes another read-only recovery.
        durable_atomic(self.cursor_path, dict(cursor, status='submitted', recovered=True,
                        reconciliation_sha256=digest(receipt)))
        return decision
