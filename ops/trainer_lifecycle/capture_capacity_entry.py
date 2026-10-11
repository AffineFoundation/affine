"""Prospective coordinator capacity; check-only drafts cannot run services."""
import argparse
import ast
import contextlib
import copy
import hashlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import sys
import threading
import types

HERE=Path(__file__).resolve().parent
VERSION='prospective-capture64-H11-successor-v1'
AUTHORITY='3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd'

def need(ok,message):
    if not ok:raise ValueError(message)

def pin(path):
    path=Path(path)
    need(path.is_absolute() and path.resolve()==path and path.is_file() and not path.is_symlink(),'ordinary exact file')
    return dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def policies():
    old=dict(version='bounded-parallel-token-capture-v2',workers=16,max_document_bytes=2000000,
        max_inflight_bytes=32000000,completion_order='first-completed',
        journal_version='fsynced-per-epoch-capture-v1',state_checkpoint_documents=128)
    return old,dict(old,version='bounded-parallel-token-capture-v3',workers=64,max_inflight_bytes=128000000)

def function_source(path,name):
    source=Path(path).read_text();tree=ast.parse(source)
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name==name]
    need(len(nodes)==1,'one exact source function')
    node=nodes[0];return '\n'.join(source.splitlines()[node.lineno-1:node.end_lineno])+'\n'

def validate(policy,guards):
    fields={'version','authority','execute_allowed','first_round','parent_policy','parent_entry','config','kind','baseline_runtime','source_sha256','runner_sha256',
        'ancestor_entry','ancestor_learner','capture_wrapper','capture_implementation',
        'storage_implementation','original_storage','commitment_implementation','original_commitment','runner','old_policy','new_policy',
        'historical_rules_unchanged','trainer_lineage_unchanged','hourly_writer_unchanged',
        'public_source_unchanged','freeze_budget_seconds'}
    need(type(policy)is dict and set(policy)==fields and policy['version']==VERSION,'exact capture operator declaration')
    need(policy['authority']==AUTHORITY==guards.AUTHORITY,'known ROOT authority')
    need(type(policy['execute_allowed'])is bool,'explicit execution state')
    need(type(policy['first_round'])is int and policy['first_round']>=151,'explicit future capture boundary')
    for name in ('parent_policy','parent_entry','ancestor_entry','ancestor_learner','capture_wrapper',
                 'capture_implementation','storage_implementation','original_storage','commitment_implementation','original_commitment','runner'):
        need(pin(policy[name]['path'])==policy[name],'exact operator file '+name)
    need(policy['runner']==pin(__file__),'exact operator entry')
    need(policy['runner_sha256']==policy['runner']['sha256'],'public observer same physical runner')
    old,new=policies()
    need(policy['old_policy']==old and policy['new_policy']==new,'exact capacity-only signed policy')
    need(policy['freeze_budget_seconds']==60,'unchanged total freeze budget')
    for name in ('historical_rules_unchanged','trainer_lineage_unchanged','hourly_writer_unchanged','public_source_unchanged'):
        need(policy[name]is True,'explicit unchanged scope '+name)
    parent=guards.signed(json.loads(Path(policy['parent_policy']['path']).read_bytes()))
    need(parent['kind']=='learner' and parent['runner_sha256']==policy['parent_entry']['sha256'],'actual signed H11 learner')
    for name in ('config','kind','baseline_runtime','source_sha256'):
        need(policy[name]==parent[name],'exact inherited compatibility field '+name)
    need(parent['cpu_overlay']['files']['subnet/storage.py']==policy['original_storage']['sha256'],'actual pinned coordinator storage')
    need(parent['cpu_overlay']['files']['subnet/commitment_transport.py']==policy['original_commitment']['sha256'],'actual pinned commitment transport')
    original_commitment=Path(policy['original_commitment']['path']).read_text()
    needle_commitment="   if policy['version']==DURABLE_CAPTURE_VERSION:\n"
    need(original_commitment.count(needle_commitment)==1,'one existing durable tiny-commitment gate')
    need(Path(policy['commitment_implementation']['path']).read_text()==original_commitment.replace(needle_commitment,
        "   if policy['version']in(DURABLE_CAPTURE_VERSION,'bounded-parallel-token-capture-v3'):\n"),'only prospective commitment WAL eligibility changes')
    old_source=function_source(policy['ancestor_entry']['path'],'learner')
    needle="            expected=dict(capture['previous_policy'],workers=16,max_inflight_bytes=32000000,state_checkpoint_documents=128)\n"
    need(old_source.count(needle)==1,'exact inherited existing-manifest guard')
    replacement=old_source.replace(needle,needle+"            if current['round']>=_capture_capacity_first_round:\n                expected=dict(_capture_capacity_new_policy)\n")
    need(Path(policy['ancestor_learner']['path']).read_text()==replacement,'only inherited capture reentry guard changes')
    module=load(policy['parent_entry']['path'],'capture64_H11_validation')
    config=module.validate(parent,guards)
    return parent,config

@contextlib.contextmanager
def adapted_ancestor(policy):
    """Startup-only amendment of one exact ancestor function, never file writes.

    No manifest is projected or rewritten. The original constructor checks the
    actual v2 policy before the boundary and the actual v3 policy afterwards.
    All other statements and all historical validation are the original bytes.
    """
    need(threading.current_thread()is threading.main_thread(),'single-thread startup adaptation')
    original=importlib.machinery.SourceFileLoader.exec_module
    seen={'imports':[],'executions':[]};target=policy['ancestor_entry'];replacement=Path(policy['ancestor_learner']['path']).read_text()
    def execute(loader,module):
        original(loader,module)
        if Path(loader.path).resolve()==Path(target['path']):
            need(pin(loader.path)==target,'inherited constructor original bytes')
            namespace=dict(vars(module),_capture_capacity_first_round=policy['first_round'],
                _capture_capacity_new_policy=copy.deepcopy(policy['new_policy']))
            # Unchanged inner closures are source-pinned by later inherited
            # operators to this original filename. Preserve that provenance;
            # the amended outer function and its exact replacement bytes are
            # separately declared in our signed policy and diagnostic receipt.
            original_source=Path(target['path']).read_text()
            old_function=function_source(target['path'],'learner')
            need(original_source.count(old_function)==1,'one inherited constructor source span')
            # Compile in the original module context: Python uses top-level import
            # knowledge when compiling nested LOAD_GLOBAL/LOAD_ATTR instructions.
            # Only take the function code; do not execute the module a second time.
            combined=original_source.replace(old_function,replacement)
            compiled=compile(combined,target['path'],'exec')
            code=next(c for c in compiled.co_consts if isinstance(c,types.CodeType)and c.co_name=='learner')
            amended=types.FunctionType(code,namespace,'learner',module.learner.__defaults__,module.learner.__closure__)
            old_inner=[c for c in module.learner.__code__.co_consts if isinstance(c,types.CodeType)]
            new_inner=[c for c in amended.__code__.co_consts if isinstance(c,types.CodeType)]
            need(old_inner==new_inner,'every inherited nested closure retains exact code and source coordinates')
            def counted(*args,**kwargs):
                seen['executions'].append(module.__name__)
                return amended(*args,**kwargs)
            module.learner=counted;seen['imports'].append(module.__name__)
    importlib.machinery.SourceFileLoader.exec_module=execute
    try:yield seen
    finally:importlib.machinery.SourceFileLoader.exec_module=original

def install_storage_method(policy,storage):
    need(pin(storage.__file__)==policy['original_storage'],'same actual coordinator storage class')
    need(not hasattr(storage.Bucket,'capture_io_client'),'no conflicting prospective capture transport')
    source=Path(policy['storage_implementation']['path']).read_text();tree=ast.parse(source)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef)and n.name=='Bucket')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef)and n.name=='capture_io_client')
    module=ast.Module(body=[method],type_ignores=[]);ast.fix_missing_locations(module)
    namespace=dict(vars(storage));exec(compile(module,policy['storage_implementation']['path'],'exec'),namespace)
    storage.Bucket.capture_io_client=namespace['capture_io_client']

def learner(policy,parent,guards,baseline):
    with adapted_ancestor(policy)as seen:
        service=load(policy['parent_entry']['path'],'capture64_H11_constructor').learner(parent,guards,baseline)
    need(len(seen['executions'])==1,'one actual inherited constructor amended: '+repr(seen))
    from subnet import storage,training_documents,commitment_transport
    install_storage_method(policy,storage)
    wrapper=load(policy['capture_wrapper']['path'],'capture64_prospective_wrapper')
    wrapper.install(training_documents,policy['capture_implementation']['path'],policy['capture_implementation']['sha256'],
        policy['first_round'],contract_module=service,previous_policy=policy['old_policy'])
    need(pin(commitment_transport.__file__)==policy['original_commitment'],'exact active commitment module')
    candidate=load(policy['commitment_implementation']['path'],'subnet._capture64_commitment_transport')
    original_freeze=commitment_transport.freeze
    def freeze(gateway,epoch):
        if int(epoch.rsplit('-',1)[-1])<policy['first_round']:return original_freeze(gateway,epoch)
        need(gateway.epochs[epoch]['commitment_binding'].get('learner_capture_policy')==policy['new_policy'],'exact future commitment capture policy')
        return candidate.freeze(gateway,epoch)
    commitment_transport.freeze=freeze
    service._capture64=dict(first_round=policy['first_round'],policy=copy.deepcopy(policy['new_policy']),
        ancestor_adapted_count=len(seen['executions']),ancestor_import_count=len(seen['imports']),
        nested_closure_code_unchanged=True,trainer_lineage_unchanged=True,hourly_writer_unchanged=True)
    return service

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--policy',required=True)
    parser.add_argument('--check-only',action='store_true');parser.add_argument('--unsigned-draft',action='store_true')
    args=parser.parse_args();raw=json.loads(Path(args.policy).read_bytes())
    need(not args.unsigned_draft or args.check_only,'unsigned drafts are check-only')
    proposed=raw if args.unsigned_draft else raw['payload']
    parent_document=json.loads(Path(proposed['parent_policy']['path']).read_bytes())
    sys.path.insert(0,parent_document['payload']['baseline_runtime'])
    from ops import durable_audit_services as guards,durable_learner_service as baseline
    policy=proposed if args.unsigned_draft else guards.signed(raw)
    if args.unsigned_draft:need(policy['execute_allowed']is False,'unsigned proposal cannot authorize execution')
    if not args.check_only:need(policy['execute_allowed']is True,'signed explicit execution authorization required')
    if args.check_only:
        # No singleton, service.run, gateway construction, cloud I/O or subprocess.
        # As in the existing H11 constructor check, import library bindings before
        # denying subprocesses: CUDA's import-only libdl discovery calls ldconfig.
        # No Runtime, model or GPU is constructed.
        import socket,sqlite3,subprocess,requests,torch,transformers
        from unittest.mock import patch
        def denied(*a,**k):raise AssertionError('check-only denies network, jobs, database, singleton and subprocess')
        with patch.object(socket.socket,'connect',side_effect=denied),patch.object(socket,'create_connection',side_effect=denied), \
             patch.object(sqlite3,'connect',side_effect=denied),patch.object(subprocess,'run',side_effect=denied), \
             patch.object(subprocess,'Popen',side_effect=denied),patch.object(requests.sessions.Session,'request',side_effect=denied), \
             patch.object(guards,'singleton',side_effect=denied):
            parent,config=validate(policy,guards);service=learner(policy,parent,guards,baseline)
            old=service.contract(config,policy['first_round']-1);new=service.contract(config,policy['first_round'])
            need(old['learner_capture_policy']==policy['old_policy'] and new['learner_capture_policy']==policy['new_policy'],'actual prospective contract projection')
            left=copy.deepcopy(old);right=copy.deepcopy(new)
            left.pop('learner_capture_policy');right.pop('learner_capture_policy')
            need(left==right,'same scientific training/mining/opening contract')
            need(service._independent_training_dispatch['historical_routes_preserved']is True,'historical trainer dispatch preserved')
        print(json.dumps(dict(passed=True,unsigned_draft=args.unsigned_draft,actual_parent_signature=True,
            actual_constructor=True,prospective_contract=service._capture64,jobs_dispatched=False,network_calls=False,
            live_service_mutations=False,check_only=True),sort_keys=True))
        return
    parent,config=validate(policy,guards)
    with guards.singleton(parent['singleton_lock']):
        guards.no_predecessors(parent)
        service=learner(policy,parent,guards,baseline);service.run(config)

if __name__=='__main__':main()
