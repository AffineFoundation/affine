"""Prospective CPU capture ordering; public miner and grading rules are unchanged."""
import hashlib
import importlib.util
import json
from pathlib import Path
import secrets
import time

VERSION='postcommit-token-capture-order-v1'
canonical=lambda value:json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
sha=lambda value:hashlib.sha256(value).hexdigest()


def validate_config(previous,proposed):
    expected=json.loads(json.dumps(previous))
    old=expected['learner_capture_policy']
    if old!={'version':'bounded-parallel-token-capture-v2','workers':8,'max_document_bytes':2000000,'max_inflight_bytes':16000000,'completion_order':'first-completed','journal_version':'fsynced-per-epoch-capture-v1','state_checkpoint_documents':16}:
        raise ValueError('exact original eight-worker bounded capture policy')
    expected['learner_capture_policy']=dict(old,workers=16,max_inflight_bytes=32000000,state_checkpoint_documents=128)
    if proposed!=expected:raise ValueError('only prospective sixteen-worker capture I/O bound changes')
    return {'learner_capture_policy'}


def prepare_order(gateway,epoch):
    state=gateway.epochs[epoch]
    if state.get('commitment_capture_complete')is not True or time.time()<state['deadline']:
        raise ValueError('capture randomness is issued only after immutable commitments close')
    commitment_hash=sha(canonical([[miner,row['sha256']]for miner,row in sorted(state['commitment_pending'].items())]))
    old=state.get('training_document_capture_order')
    if old is not None:
        if (type(old)is not dict or set(old)!={'version','epoch','seed','commitments_sha256'}
                or old['version']!=VERSION or old['epoch']!=epoch
                or old['commitments_sha256']!=commitment_hash
                or type(old['seed'])is not str or len(old['seed'])!=64
                or any(c not in '0123456789abcdef'for c in old['seed'])):
            raise ValueError('exact original postcommit capture order')
        # A previous persist may have raised after replacing the durable file.
        # Keep the same choice and confirm durability before any network read.
        gateway.persist()
        return dict(old)
    result=dict(version=VERSION,epoch=epoch,seed=secrets.token_hex(32),commitments_sha256=commitment_hash)
    state['training_document_capture_order']=result
    gateway.persist()
    return dict(result)


def install(module,path,expected_sha256,earliest_round,*,contract_module=None,previous_policy=None):
    """Called only by a ROOT-pinned CPU entry; scientific worker files stay sealed."""
    path=Path(path)
    if type(earliest_round)is not int or earliest_round<1 or path.is_symlink() or sha(path.read_bytes())!=expected_sha256:
        raise ValueError('approved prospective capture implementation')
    spec=importlib.util.spec_from_file_location('subnet._approved_fair_token_capture',path)
    candidate=importlib.util.module_from_spec(spec);spec.loader.exec_module(candidate)
    original=module.capture
    original_freeze=module.freeze_receipts if hasattr(module,'freeze_receipts')else None
    prospective=None
    if contract_module is not None:
        before=dict(previous_policy or {})
        prospective=dict(before,workers=16,max_inflight_bytes=32000000,state_checkpoint_documents=128)
        validate_config({'learner_capture_policy':before},{'learner_capture_policy':prospective})
        original_contract=contract_module.contract
        def contract(config,round_number):
            result=original_contract(config,round_number)
            if round_number>=earliest_round:
                if result.get('learner_capture_policy')!=before:
                    raise ValueError('prospective capture original opening policy')
                result=dict(result,learner_capture_policy=dict(prospective))
            return result
        # Preserve Controller.open's exact code/globals. RemoteController uses
        # a per-call save_manifest projection for its first signed opening.
        contract_module.contract=contract
    def capture(gateway,epoch):
        if int(epoch.rsplit('-',1)[-1])<earliest_round:return original(gateway,epoch)
        if prospective is not None and gateway.epochs[epoch]['commitment_binding'].get('learner_capture_policy')!=prospective:
            raise ValueError('fair capture requires its exact prospective signed opening policy')
        return candidate.capture(gateway,epoch,ordering=prepare_order(gateway,epoch))
    module.capture=capture
    module.capture_policy=candidate.capture_policy
    if original_freeze is not None:
        def freeze_receipts(gateway,epoch):
            if int(epoch.rsplit('-',1)[-1])<earliest_round:return original_freeze(gateway,epoch)
            if prospective is not None and gateway.epochs[epoch]['commitment_binding'].get('learner_capture_policy')!=prospective:
                raise ValueError('parent publication exact prospective signed capture policy')
            return candidate.freeze_receipts(gateway,epoch,publication_workers=16)
        module.freeze_receipts=freeze_receipts
    return capture
