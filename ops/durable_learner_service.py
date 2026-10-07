"""Resume the approved learner under a durable ROOT-pinned CPU policy.

Startup changes no epoch, original job, checkpoint, or optimizer evidence.
The scientific controller retains responsibility for original-job observation.
"""
import argparse
import base64
import re
import hashlib
import json
import os
import stat
from pathlib import Path
import sys
from ops import durable_audit_services as guards

VERSION = 'durable-pinned-learner-service-v1'
OVERLAY_VERSION = 'coordinator-capture-overlay-v1'
CPU_OVERRIDES = {'subnet/capture_journal.py', 'subnet/training_documents.py',
                 'subnet/commitment_transport.py', 'subnet/controller.py',
                 'subnet/gpu_service.py', 'subnet/storage.py',
                 'subnet/remote_backend.py', 'subnet/checkpoint_upload_recovery.py'}


PEER_POLICY_FIELD='learner_selection_cpu_peer_admission'
PEER_OVERRIDES={'subnet/committed_training_inputs.py','subnet/training_receipts.py',
                'subnet/learner_blacklist_selection.py','subnet/learner_selection_operator_bridge.py'}

def validate_cpu_selection_peer(policy,source,cfg,authority):
    row=policy[PEER_POLICY_FIELD]
    fields={'version','authorization','remote_admission','peer_root','peer_files',
            'entry','entry_sha256','runner','runner_sha256'}
    if type(row)is not dict or set(row)!=fields or row['version']!='durable-CPU-selection-peer-v1':
        raise ValueError('exact default-off CPU selection peer policy')
    grant=guards.verify_document(row['authorization'],authority)
    grant_fields={'version','source_sha256','scientific_source_files','operator_files','minimum_round','epoch_prefix','peer_entry_sha256','peer_runner_sha256','backend_execution_allowed'}
    if (set(grant)!=grant_fields or grant['version']!=('cpu-selection-peer-miner-bound-authorization-v2' if policy['version'] in ('durable-pinned-k2l2-learner-service-v2','durable-pinned-k2l2-composite-learner-service-v3') else 'cpu-selection-peer-authorization-v1') or
        grant['source_sha256']!=policy['source_sha256']or grant['scientific_source_files']!=source['runtime_source_files']or
        set(grant['operator_files'])!=PEER_OVERRIDES or grant['operator_files']!=row['peer_files']or
        type(grant['minimum_round'])is not int or grant['minimum_round']<0 or
        type(grant['epoch_prefix'])is not str or not grant['epoch_prefix']or
        grant['backend_execution_allowed']is not True or
        grant['peer_entry_sha256']!=row['entry_sha256']or grant['peer_runner_sha256']!=row['runner_sha256']):
        raise ValueError('exact ROOT peer source177/override/entry/runner execution grant')
    overlay=policy.get('operator_overlay',{})
    if any(overlay.get('overrides',{}).get(name)!=h for name,h in row['peer_files'].items()):
        raise ValueError('coordinator and remote CPU metadata must match')
    root=Path(row['peer_root'])
    if not root.is_absolute()or root.resolve()!=root or root.is_symlink()or root.stat().st_uid!=os.getuid():raise ValueError('owned exact peer root')
    if {str(f.relative_to(root))for f in root.rglob('*')if f.is_file()}!=set(row['peer_files']):raise ValueError('exact peer membership')
    guards.pinned_files(root,row['peer_files'])
    for name in ('entry','runner'):
        path=guards.private_file(row[name])
        if guards.file_hash(path)!=row[name+'_sha256']:raise ValueError('actual separately admitted peer launcher SHA')
    role=cfg.get('remote',{}).get('roles',{}).get('train',{})
    peer=role.get('learner_selection_cpu_peer')
    if type(peer)is not dict or set(peer)!={'operator_root','entry','runner','bytecode_prefix_root','authorization_document'}or peer['authorization_document']!=guards.read(row['authorization']['path']):
        raise ValueError('exact train-only remote peer config')
    auto=guards.signed(cfg.get('learner_blacklist_selection_authorization'),authority)
    auto_fields={'version','source_sha256','writer_policy_sha256','audit_policy','maximum_age_seconds','assessment_path'}
    if (set(auto)!=auto_fields or auto['version']!='automatic-confirmed-blacklist-training-selection-v1'or
        auto['source_sha256']!=policy['source_sha256']or type(auto['maximum_age_seconds'])is not int or not 1<=auto['maximum_age_seconds']<=7200):
        raise ValueError('exact authenticated automatic opening mechanism')
    if type(auto['writer_policy_sha256'])is not str or len(auto['writer_policy_sha256'])!=64:
        raise ValueError('writer policy pin')
    try:bytes.fromhex(auto['writer_policy_sha256'])
    except ValueError:raise ValueError('writer policy pin')
    assessment=Path(auto['assessment_path'])
    if not assessment.is_absolute()or assessment.resolve()!=assessment or assessment.is_symlink():raise ValueError('ordinary assessment source path')
    # Execute only the separately pinned scalar parser function, not science v1/v2 policy.
    import ast,math
    text=(root/'subnet/learner_blacklist_selection.py').read_bytes();tree=ast.parse(text)
    functions=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='validate_audit_policy']
    if len(functions)!=1:raise ValueError('separately pinned scalar policy parser')
    scope={'math':math};exec(compile(ast.Module(body=functions,type_ignores=[]),str(root/'subnet/learner_blacklist_selection.py'),'exec'),scope)
    scope['validate_audit_policy'](auto['audit_policy'])
    admission=guards.verify_document(row['remote_admission'],authority)
    expected=dict(version='CPU-selection-remote-peer-admission-v1',source_sha256=policy['source_sha256'],
        scientific_source_files_sha256=guards.digest(source['runtime_source_files']),authorization_payload_sha256=guards.digest(grant),
        host=role.get('host'),port=role.get('port'),user=role.get('user','root'),python=role.get('python'),
        scientific_root=role.get('code'),operator_root=peer['operator_root'],entry=peer['entry'],runner=peer['runner'],
        bytecode_prefix_root=peer['bytecode_prefix_root'],operator_files=row['peer_files'],
        entry_sha256=row['entry_sha256'],runner_sha256=row['runner_sha256'],
        CPU_only=True,model_loaded=False,proof_reverification=False,admitted=True)
    if admission!=expected:raise ValueError('ROOT-authenticated exact remote CPU peer readiness')
    for key in ('operator_root','entry','runner','bytecode_prefix_root'):
        if not Path(peer[key]).is_absolute()or '..'in Path(peer[key]).parts:raise ValueError('explicit remote peer path')
    if any('learner_selection_cpu_peer'in v for k,v in cfg['remote']['roles'].items()if k!='train'and isinstance(v,dict)):
        raise ValueError('no mining/evaluation/verifier CPU peer override')
    return grant


def validate_operator_overlay(overlay, source, policy, cfg):
    fields = {'version', 'root', 'full_source_files', 'overrides',
              'baseline_source_sha256', 'baseline_inventory_sha256', 'learner_capture_policy'}
    if (type(overlay) is not dict or set(overlay) != fields or
        overlay['version'] != OVERLAY_VERSION or
        overlay['baseline_source_sha256'] != policy['source_sha256'] or
        overlay['baseline_inventory_sha256'] != guards.digest(source['full_source_files'])):
        raise ValueError('exact ROOT-bound CPU overlay baseline')
    root = Path(overlay['root'])
    if (not root.is_absolute() or root.resolve() != root or root == Path(policy['source_root']) or
        not stat.S_ISDIR(root.lstat().st_mode) or root.lstat().st_uid != os.getuid()):
        raise ValueError('distinct canonical owned CPU overlay root')
    changes = overlay['overrides']
    if type(changes) is not dict or not changes or not set(changes) <= (CPU_OVERRIDES | ({'subnet/late_capture_recovery.py'} if 'capture_recovery' in policy else set()) | ({'subnet/persistent_training_controller.py'} if 'native_training_eligibility' in policy else set()) | (PEER_OVERRIDES if PEER_POLICY_FIELD in policy else set())):
        raise ValueError('only explicit coordinator transport modules may differ')
    original = source['full_source_files']
    if any(k not in original and k not in ({'subnet/capture_journal.py', 'subnet/checkpoint_upload_recovery.py'} | ({'subnet/late_capture_recovery.py'} if 'capture_recovery'in policy else set()) | ({'subnet/learner_blacklist_selection.py','subnet/learner_selection_operator_bridge.py'} if PEER_POLICY_FIELD in policy else set())) for k in changes):
        raise ValueError('only explicit CPU sidecars may extend original membership')
    expected = dict(original, **changes)
    if overlay['full_source_files'] != expected:
        raise ValueError('complete CPU overlay inventory and exact overrides')
    if any(original.get(k) == value for k, value in changes.items()):
        raise ValueError('redundant CPU overlay override declaration')
    # Even the source admission's historical cache exception does not permit
    # unlisted files in this fresh overlay or its baseline tree.
    for directory, files in ((Path(policy['source_root']), original), (root, expected)):
        observed = {str(f.relative_to(directory)) for f in directory.rglob('*') if f.is_file()}
        if observed != set(files) or any(f.is_symlink() for f in directory.rglob('*')):
            raise ValueError('exact ordinary source/overlay membership')
        guards.pinned_files(directory, files)
    capture = overlay['learner_capture_policy']
    keys = {'version', 'workers', 'max_document_bytes', 'max_inflight_bytes',
            'completion_order', 'journal_version', 'state_checkpoint_documents'}
    if (type(capture) is not dict or set(capture) != keys or
        capture['version'] != 'bounded-parallel-token-capture-v2' or
        type(capture['workers']) is not int or capture['workers'] not in (4, 8, 16) or
        type(capture['max_document_bytes']) is not int or capture['max_document_bytes'] != 2_000_000 or
        type(capture['max_inflight_bytes']) is not int or capture['max_inflight_bytes'] != capture['workers'] * 2_000_000 or
        capture['completion_order'] != 'first-completed' or
        capture['journal_version'] != 'fsynced-per-epoch-capture-v1' or
        type(capture['state_checkpoint_documents']) is not int or not 1 <= capture['state_checkpoint_documents'] <= 16 or
        cfg.get('learner_capture_policy') != capture):
        raise ValueError('exact prospective signed capture policy')
    if (cfg.get('training_input_policy') != 'committed-unaudited-training-v1' or
        cfg.get('submission_transport_policy') != 'small-commitment-pairs-v2' or
        cfg.get('hourly_execution_policy') is None):
        raise ValueError('CPU overlay cannot change scientific transport')
    return root


def verify_admission(row, authority):
    # Historical learner approvals predate BASE64 deployment policies. Bind
    # their exact bytes and declared encoding; never rewrite signed evidence.
    if guards.file_hash(row['path']) != row['file_sha256']:
        raise ValueError('historical admission bytes changed')
    document = guards.read(row['path'])
    encoding = row.get('signature_encoding', 'base64')
    if encoding == 'hex':
        if not re.fullmatch('[0-9a-f]{128}', document.get('signature', '')):
            raise ValueError('exact historical HEX signature required')
        document = dict(document, signature=base64.b64encode(bytes.fromhex(document['signature'])).decode())
    elif encoding != 'base64':
        raise ValueError('explicit admission signature encoding required')
    payload = guards.signed(document, authority)
    if guards.digest(payload) != row['payload_sha256']:
        raise ValueError('historical admission payload changed')
    return payload


def validate_stop_boundary(p,cfg,authority):
    row=p.get('stop_after_current_round')
    if row is None:
        if 'stop_after_current_round'in p:raise ValueError('explicit null stop boundary')
        return None
    fields={'version','round','epoch','original_opening'}
    if type(row)is not dict or set(row)!=fields or row['version']!='signed-current-round-CPU-stop-v1'or type(row['round'])is not int or row['round']<0 or type(row['epoch'])is not str:
        raise ValueError('exact current-round CPU boundary')
    opening=guards.verify_document(row['original_opening'],authority)
    if opening.get('epoch')!=row['epoch']or opening.get('source_bundle',{}).get('sha256')!=p['source_sha256']:
        raise ValueError('original published opening context')
    state=guards.read(guards.private_file(Path(cfg['state'])/'controller.json'))
    active=state.get('active')
    if (state.get('round')!=row['round']or not isinstance(active,dict)or active.get('epoch')!=row['epoch']or
        active.get('phase')in (None,'opening')or state.get('checkpoint',{}).get('id')!=opening.get('checkpoint',{}).get('id')):
        raise ValueError('resume only SAME published current round; never open another')
    return row


def run_original_boundary(service,p,cfg,authority=guards.AUTHORITY):
    row=validate_stop_boundary(p,cfg,authority)
    if row is None:return service.run(cfg)
    # Exact current epoch resumes through original durable paths; --once stops
    # naturally after it closes. No GPU signals, new policy, or live hooks.
    service.run(cfg,once=True)
    status=guards.read(guards.private_file(Path(cfg['state'])/'controller.json'))
    if status.get('round')!=row['round']+1 or status.get('active')is not None:
        raise ValueError('original current round did not reach natural CPU boundary')
    print(json.dumps(dict(version='original-round-CPU-boundary-terminal-v1',round=status['round'],epoch=row['epoch'],active=None,closure_authentication_still_required=True)))


def validate_policy(document, authority=guards.AUTHORITY):
    p = guards.signed(document, authority)
    fields = {'version', 'execute_allowed', 'authority', 'identity', 'config',
              'source_root', 'source_sha256', 'source_approval', 'qualification_approval',
              'reward_activation', 'authority_seed', 'singleton_lock', 'excluded_units',
              'runner_file_sha256', 'guards_file_sha256'}
    if 'operator_overlay' in p:
        fields.add('operator_overlay')
    if 'native_training_eligibility' in p:fields.add('native_training_eligibility')
    if p.get('version') in ('durable-pinned-k2l2-learner-service-v2','durable-pinned-k2l2-composite-learner-service-v3'):
        fields.add('scientific_admission_file_sha256')
    if p.get('version') == 'durable-pinned-k2l2-composite-learner-service-v3':
        fields.add('core_admission_file_sha256')
    if PEER_POLICY_FIELD in p:fields.add(PEER_POLICY_FIELD)
    if 'stop_after_current_round'in p:fields.add('stop_after_current_round')
    if 'capture_recovery'in p:fields.add('capture_recovery')
    if set(p) != fields or p['version'] not in (VERSION, 'durable-pinned-k2l2-learner-service-v2','durable-pinned-k2l2-composite-learner-service-v3') or p['execute_allowed'] is not True or p['authority'] != authority:
        raise ValueError('exact durable learner policy required')
    for path, expected in ((Path(__file__).resolve(), p['runner_file_sha256']),
                           (Path(guards.__file__).resolve(), p['guards_file_sha256'])):
        if guards.file_hash(path) != expected:
            raise ValueError('durable learner operator drift')
    identity = p['identity']
    if set(identity) != {'uid', 'machine_id_sha256'} or identity['uid'] != os.getuid() or identity['machine_id_sha256'] != hashlib.sha256(Path('/etc/machine-id').read_bytes()).hexdigest():
        raise ValueError('approved physical CPU identity required')
    if not isinstance(p['excluded_units'], list) or not p['excluded_units'] or any(not isinstance(v, str) or not v.endswith('.service') for v in p['excluded_units']):
        raise ValueError('explicit predecessor exclusion required')
    if guards.file_hash(p['config']['path']) != p['config']['file_sha256']:
        raise ValueError('approved config drift')
    cfg = guards.read(p['config']['path'])
    source = verify_admission(p['source_approval'], authority)
    qualification = verify_admission(p['qualification_approval'], authority)
    reward = guards.verify_document(p['reward_activation'], authority)
    scientific_successor = p['version'] in ('durable-pinned-k2l2-learner-service-v2','durable-pinned-k2l2-composite-learner-service-v3')
    composite = p['version'] == 'durable-pinned-k2l2-composite-learner-service-v3'
    if composite:
        helper_path=Path(__file__).resolve().with_name('k2l2_composite_admission.py')
        core_path=Path(__file__).resolve().with_name('k2l2_scientific_admission.py')
        if guards.file_hash(helper_path)!=p['scientific_admission_file_sha256'] or guards.file_hash(core_path)!=p['core_admission_file_sha256']:
            raise ValueError('exact new composite and unchanged strict core admission')
        from ops import k2l2_composite_admission, k2l2_scientific_admission
        if Path(k2l2_composite_admission.__file__).resolve()!=helper_path or Path(k2l2_scientific_admission.__file__).resolve()!=core_path:
            raise ValueError('composite admission module origin')
        k2l2_composite_admission.validate(source,qualification,cfg,p['source_sha256'],authority,verify_admission,guards=guards,strict_admission=k2l2_scientific_admission,source_root=p['source_root'])
    elif scientific_successor:
        helper_path = Path(__file__).resolve().with_name('k2l2_scientific_admission.py')
        if guards.file_hash(helper_path) != p['scientific_admission_file_sha256']:
            raise ValueError('new scientific admission helper drift')
        from ops import k2l2_scientific_admission as scientific_admission
        if Path(scientific_admission.__file__).resolve() != helper_path:
            raise ValueError('new scientific admission import shadowing')
        scientific_admission.validate(source, qualification, cfg, p['source_sha256'], authority, verify_admission)
    elif source['version'] != 'ordinary-orchestration-only-source-approval-v1' or source['approved'] is not True or source['source_sha256'] != p['source_sha256']:
        raise ValueError('approved scientific source required')
    if source['optimizer_reset'] is not False or source['historical_relabel'] is not False:
        raise ValueError('optimizer lineage and historical evidence must remain unchanged')
    guards.pinned_files(p['source_root'], source['full_source_files'])
    root = Path(p['source_root'])
    inventory = {str(f.relative_to(root)) for f in root.rglob('*') if f.is_file() and '__pycache__' not in f.parts}
    if inventory != set(source['full_source_files']):
        raise ValueError('complete scientific source inventory required')
    if (not scientific_successor and len(source['runtime_source_files']) != 177) or any(source['full_source_files'].get(k) != v for k, v in source['runtime_source_files'].items()):
        raise ValueError('approved 177-file runtime closure required')
    for row in source['evidence'].values():
        if guards.file_hash(row['path']) != row['file_sha256']:
            raise ValueError('source qualification evidence drift')
    if (not scientific_successor and qualification['version'] != 'ordinary-orchestration-only-training-qualification-approval-v1') or qualification['approved'] is not True or qualification['candidate_source_sha256'] != p['source_sha256']:
        raise ValueError('approved training qualification required')
    translation = cfg['persistent_training_qualification_translation']
    if (translation['path'] != qualification['translation_path'] or
        translation['sha256'] != qualification['translation_file_sha256'] or
        guards.file_hash(translation['path']) != translation['sha256'] or
        translation['approval_path'] != p['qualification_approval']['path'] or
        translation['approval_sha256'] != p['qualification_approval']['file_sha256']):
        raise ValueError('exact qualification translation required')
    if cfg['source_bundle']['sha256'] != p['source_sha256'] or cfg['persistent_training_admission']['source_sha256'] != p['source_sha256'] or cfg['persistent_training_admission']['gpu_qualification_sha256'] != translation['sha256']:
        raise ValueError('persistent training source binding changed')
    if cfg.get('preparation_only') is not False or cfg.get('activation_allowed') is not True or cfg.get('activation_approved') is not True or cfg['deployment_gate']['execution_allowed'] is not True:
        raise ValueError('actual activation approval required')
    if cfg.get('token_artifact_policy') is not None or cfg.get('native_source_validation_policy') is not None:
        raise ValueError('scientific proof contract cannot change at CPU restart')
    if cfg.get('training_input_policy') != 'committed-unaudited-training-v1':
        raise ValueError('existing unaudited training policy required')
    if guards.read(p['reward_activation']['path']) != cfg['continuous_reward_activation_document'] or p['source_sha256'] not in reward['approved_sources']:
        raise ValueError('exact approved reward activation required')
    seed = p['authority_seed']
    if guards.file_hash(seed['path']) != seed['file_sha256'] or guards.private_file(seed['path']).stat().st_mode & 0o077:
        raise ValueError('private original authority seed required')
    from nacl.signing import SigningKey
    if SigningKey(bytes.fromhex(Path(seed['path']).read_text().strip())).verify_key.encode().hex() != authority:
        raise ValueError('original authority identity changed')
    state = Path(cfg['state'])
    if state.resolve() / 'authority.seed' != Path(seed['path']).resolve():
        raise ValueError('original learner state path required')
    # Never initialize a fresh controller during recovery. Its authenticated
    # remote controller handles existing jobs and durable optimizer commits.
    status = guards.read(guards.private_file(state / 'controller.json'))
    if status.get('initial_published') is not True or status.get('persistent_state_committed') is not True or not status.get('trainer_state'):
        raise ValueError('existing committed persistent learner required')
    if PEER_POLICY_FIELD in p:validate_cpu_selection_peer(p,source,cfg,authority)
    elif 'learner_blacklist_selection_authorization'in cfg or any('learner_selection_cpu_peer'in v for v in cfg.get('remote',{}).get('roles',{}).values()if isinstance(v,dict)):
        raise ValueError('explicit selection requires separately admitted CPU peer policy')
    if 'stop_after_current_round'in p:validate_stop_boundary(p,cfg,authority)
    if 'operator_overlay' in p:
        validate_operator_overlay(p['operator_overlay'], source, p, cfg)
    if 'native_training_eligibility' in p:validate_native_operator(p,authority,source)
    if 'capture_recovery'in p:validate_capture_recovery(p,cfg,authority)
    return p


def prepare_runtime(p):
    for name in list(sys.modules):
        if name == 'subnet' or name.startswith('subnet.'):
            del sys.modules[name]
    root = p['operator_overlay']['root'] if 'operator_overlay' in p else p['source_root']
    sys.path.insert(0, root)
    from subnet import gpu_service
    if Path(gpu_service.__file__).resolve() != Path(root) / 'subnet/gpu_service.py':
        raise ValueError('unexpected pinned learner import')
    if 'native_training_eligibility' in p:
        install_native_constructor(gpu_service,p)
    if 'capture_recovery'in p:install_capture_recovery(gpu_service,p)
    return gpu_service



def validate_native_operator(p,authority,source):
    row=p['native_training_eligibility']
    fields={'version','root','files','authorization','tokenizer_root','interpreter','boundary','lifecycle_policy'}
    if type(row) is not dict or set(row)!=fields or row['version']!='pinned-native-eligibility-operator-v1':
        raise ValueError('exact native operator opt-in')
    root=Path(row['root'])
    if not root.is_absolute() or root.resolve()!=root or root.is_symlink() or root.stat().st_uid!=os.getuid():
        raise ValueError('owned native operator root')
    if set(row['files'])!={'native_training_outcome_filter.py','native_training_eligibility.py','native_training_lifecycle.py'}:
        raise ValueError('exact native operator import closure')
    guards.pinned_files(root,row['files'])
    if {str(f.relative_to(root)) for f in root.rglob('*') if f.is_file()}!=set(row['files']):
        raise ValueError('native operator complete membership')
    auth=guards.verify_document(row['authorization'],authority)
    if 'benchmark_scope' in auth:raise ValueError('readonly benchmark cannot authorize production operator')
    if (auth.get('source_root')!=p['source_root'] or auth.get('source_sha256')!=p['source_sha256'] or
        auth.get('source_files')!=source['runtime_source_files'] or
        auth.get('execution_root')!=p.get('operator_overlay',{}).get('root') or
        'subnet/persistent_training_controller.py' not in p.get('operator_overlay',{}).get('overrides',{})):
        raise ValueError('native scientific baseline and explicit CPU controller exception')
    native=load_native_operator(row)
    native.validate_future_boundary(row['boundary'])
    lifecycle=sys.modules[native.__package__+'.native_training_lifecycle']
    if row['lifecycle_policy']!=lifecycle.LIFECYCLE_POLICY:raise ValueError('exact bounded native lifecycle policy')
    return auth


def load_native_operator(row):
    # Private relative-import package: no global ops module can shadow approved bytes.
    import importlib.util
    import types
    root=Path(row['root'])
    guards.pinned_files(root,row['files'])
    name='_root_pinned_native_eligibility'
    package=types.ModuleType(name);package.__path__=[str(root)]
    sys.modules[name]=package
    for leaf in ('native_training_outcome_filter','native_training_eligibility','native_training_lifecycle'):
        fullname=name+'.'+leaf
        spec=importlib.util.spec_from_file_location(fullname,root/(leaf+'.py'))
        module=importlib.util.module_from_spec(spec);sys.modules[fullname]=module
        spec.loader.exec_module(module)
    return sys.modules[name+'.native_training_eligibility']


def install_native_constructor(service,p):
    row=p['native_training_eligibility']
    selector=load_native_operator(row).FutureNativeEligibilitySelector
    lifecycle=sys.modules[selector.__module__.rsplit('.',1)[0]+'.native_training_lifecycle']
    authorization=guards.read(row['authorization']['path'])
    original=service.RemoteController
    class NativeController(original):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.native_training_eligibility_selector=selector(self,authorization,row['tokenizer_root'],row['interpreter'],row['boundary'])
            self.handle_native_no_update=lambda error,manifest,status:lifecycle.close_no_update(self,error,manifest,status)
            self.native_completion_fields=lambda epoch:lifecycle.completion_fields(self,epoch)
            self.native_retire_completed=lambda epoch:lifecycle.retire_completed(self,epoch)
            # --check and prepare_runtime do not create threads or perform GETs.
            if getattr(service,'_native_lifecycle_execute',False):
                lifecycle.start_retirement_observer(self)
    service.RemoteController=NativeController

def validate_capture_recovery(p,cfg,authority):
    row=p['capture_recovery']
    if type(row)is not dict or set(row)!={'authorization','first_signed_manifest','epoch'}:
        raise ValueError('explicit signed capture recovery policy')
    authorization=guards.verify_document(row['authorization'],authority)
    first=guards.read(row['first_signed_manifest']['path'])
    if guards.file_hash(row['first_signed_manifest']['path'])!=row['first_signed_manifest']['file_sha256']:
        raise ValueError('original first signed manifest drift')
    m=guards.signed(first,authority)
    if guards.digest(first)!=authorization['first_signed_manifest_sha256']or m['epoch']!=row['epoch']:
        raise ValueError('exact original recovery manifest')
    state=guards.read(Path(cfg['state'])/'controller.json')
    gateway=guards.read(Path(cfg['state'])/'gateway.json')['epochs'][row['epoch']]
    installed=guards.read(row['authorization']['path']) in gateway.get('capture_recovery_authorizations',[])
    if not installed and (state.get('active',{}).get('epoch')!=row['epoch']or state.get('active',{}).get('phase')!='collect'):
        raise ValueError('initial recovery only original collect phase')
    # Validate a separately pinned pure CPU module without importing model runtimes.
    # prepare_runtime performs full import; here signature and baseline digests suffice.
    if (authorization['epoch']!=row['epoch']or authorization['source']!=p['source_sha256']or
        authorization['checkpoint']!=m['checkpoint']['id']or authorization['start']!=gateway['start']or
        authorization['deadline']!=gateway['deadline']or authorization['original_freeze_until']!=gateway['commitment_binding']['freeze_until']or
        authorization['binding_sha256']!=guards.digest(gateway['commitment_binding'])or
        authorization['miners_sha256']!=guards.digest(sorted(gateway['miners']))):
        raise ValueError('same original recovery gateway scope')
    if 'subnet/late_capture_recovery.py'not in p['operator_overlay']['overrides']:
        raise ValueError('explicit pinned recovery CPU sidecar')
    return authorization


def install_capture_recovery(service,p):
    row=p['capture_recovery'];original=service.Gateway
    from subnet.late_capture_recovery import attach
    from types import SimpleNamespace
    document=guards.read(row['authorization']['path']);first=guards.read(row['first_signed_manifest']['path'])
    cfg=guards.read(p['config']['path'])
    saved=guards.read(Path(cfg['state'])/'gateway.json')
    attach(SimpleNamespace(epochs=saved['epochs'],persist=lambda:None),row['epoch'],document,first)
    class RecoveryGateway(original):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            attach(self,row['epoch'],document,first)
    service.Gateway=RecoveryGateway


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--policy', required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if sys.flags.optimize:
        raise ValueError('scientific guards require unoptimized Python')
    p = validate_policy(guards.read(args.policy))
    with guards.singleton(p['singleton_lock']):
        guards.no_predecessors(p)
        service = prepare_runtime(p)
        if args.check:
            print(json.dumps({'checked': True, 'source': p['source_sha256'],
                              'operator_overlay': 'operator_overlay' in p,
                              'existing_state_preserved': True, 'jobs_dispatched': False}))
            return
        if 'native_training_eligibility'in p:service._native_lifecycle_execute=True
        run_original_boundary(service,p,guards.read(p['config']['path']))


if __name__ == '__main__':
    main()
