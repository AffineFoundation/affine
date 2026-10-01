"""Real original Calendar controls over actor-only RPC; no model assertions."""
import copy
import json
import urllib.error
from pathlib import Path
from subnet.native_eog_isolation import sha
from subnet.native_eog_split import OperatorBroker, PublicActor, request, REVISION
from ops.probe_native_eog import public_relocation_policy

def owned(root,task,runtime,label,negative=False,expected=None):
    broker=OperatorBroker(task,runtime)
    actor=PublicActor(broker.endpoint,broker.actor_capability,sha(broker.public))
    try:
        public=actor.reset()
        if expected is None:public_relocation_policy(actor,negative)
        else:
            for event in expected['events']:
                if actor.call(event['name'],event['arguments'])!=event['observation']:
                    raise ValueError('split original native observation mismatch')
        actor.close()
        private=request(broker.endpoint,broker.operator_capability,'operator',{'operation':'grade'})
        if expected and (private['grade']['reward']!=expected['reward'] or private['logical_database_sha256']!=expected['logical_database_sha256']):
            raise ValueError('split original reward/state mismatch')
        artifact={'public_descriptor':public,'events':copy.deepcopy(broker.session.events),
                  'reward':private['grade']['reward'],'logical_database_sha256':private['logical_database_sha256']}
        p=root/(label+'.public.json');p.write_text(json.dumps(artifact,indent=2))
        p=root/(label+'.operator.private.json');p.write_text(json.dumps({'grader_descriptor':broker.private_grader,'result':private},indent=2));p.chmod(0o600)
        return artifact
    finally:broker.close()

def main():
    original=Path('state/native-eog-isolation');root=Path('state/native-eog-split');root.mkdir(exist_ok=True)
    task=json.loads((original/'original-calendar-relocation.private.json').read_text())
    runtime=json.loads((original/'runtime-descriptor.json').read_text())
    positive=owned(root,task,runtime,'positive')
    negative=owned(root,task,runtime,'negative',negative=True)
    assert positive['reward']==1 and negative['reward']==0
    replay=owned(root,task,runtime,'positive-replay',expected=positive)
    assert positive==replay
    broker=OperatorBroker(task,runtime);actor=PublicActor(broker.endpoint,broker.actor_capability,sha(broker.public))
    denied=[]
    try:
        public=actor.reset()
        assert not {'verifiers','seed_file','sql_content','context','database_state','operator_capability'}&set(public)
        assert set(vars(actor))=={'endpoint','capability','expected'}
        for path,role,operation in [('/operator','operator','grade'),('/actor','actor','grade'),
              ('/actor','actor','state'),('/actor','actor','seed'),('/actor','actor','read_file')]:
            try:request(broker.endpoint,broker.actor_capability,role,{'operation':operation})
            except urllib.error.HTTPError as error:
                assert error.code in (400,403);denied.append(role+':'+operation)
            else:raise AssertionError('privileged actor request allowed')
        from mcp.server.fastmcp.exceptions import ToolError
        for tool in ('sql-runner','seed-database','/api/sql-runner','get_verifiers'):
            try:actor.call(tool,{'query':'SELECT 1','path':'/etc/passwd'})
            except ToolError as error:
                assert str(error)=='Unknown tool: '+tool;denied.append('unselected:'+tool)
            else:raise AssertionError('unselected tool accepted')
        initial=request(broker.endpoint,broker.operator_capability,'operator',{'operation':'grade'})
        assert initial['grade']['reward']==0 and initial['calls']==0
        actor.close()
        try:actor.call('get_calendar_list',{})
        except urllib.error.HTTPError as error:assert error.code==400;denied.append('sealed-write')
        else:raise AssertionError('sealed actor accepted tool')
        # Keep only operator/public content hashes, never capabilities.
        descriptor_sha=sha(broker.public);grader_sha=sha(broker.private_grader)
    finally:broker.close()
    falsifications={}
    changed=copy.deepcopy(positive);changed['events'][0]['observation']+=' forged'
    mutations={'observation':changed}
    changed=copy.deepcopy(positive);changed['reward']=0;mutations['reward']=changed
    changed=copy.deepcopy(positive);changed['logical_database_sha256']='0'*64;mutations['logical_state']=changed
    for name,value in mutations.items():
        try:owned(root,task,runtime,'mutated-'+name,expected=value)
        except ValueError as error:falsifications[name]={'rejected':True,'reason':str(error)}
        else:raise AssertionError('forged split native claim accepted')
    report={'passed':True,'revision':REVISION,'positive_reward':positive['reward'],'negative_reward':negative['reward'],
        'native_tool_calls_each':len(positive['events']),'exact_positive_replay':positive==replay,
        'public_descriptor_sha256':descriptor_sha,'private_grader_descriptor_sha256':grader_sha,
        'denied_actor_operations':denied,'falsifications':falsifications,
        'actor_exposes_private_seed_or_grader':False,'scope':'controlled co-located broker RPC, not host-root isolation',
        'model_proofs_verified':False,'shared_epoch_verified':False,'chain_submission':False}
    (root/'split-controls.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'passed':True,'positive':1,'negative':0,'exact_replay':True,'denied_operations':len(denied),'falsifications':len(falsifications)}))

if __name__=='__main__':main()
