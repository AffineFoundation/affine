import copy
import pytest
from ops.publication_local_projection import project,FIELD,enforce_size,SUFFIX

def signed(doc,authority):
    if doc['signer']!=authority or doc.get('signature')!='valid':raise ValueError('signature')
    return copy.deepcopy(doc['payload'])
def trainer_project(original,sign):
    result=copy.deepcopy(original);result[FIELD]=sign(original)
    result['optimizer_state_export_policy']='trainer-local-only-v1';result.pop('independent_state_readback_budget',None)
    return result
@pytest.fixture
def m():
    old={'epoch':'e88','checkpoint':{'id':'old','files':{'m':'h'}},'source_bundle':{'sha256':'science'},'inputs':['original'], 'optimizer_state_export_policy':'durable','independent_state_readback_budget':123}
    m=trainer_project(old,lambda x:{'payload':x,'signer':'ROOT','signature':'valid'})
    m['checkpoint']={'id':'new','files':{'m':'newhash'}};return m

def test_only_wrapper_removed(m):
    p=project(m,'ROOT',signed,trainer_project);assert p=={k:v for k,v in m.items()if k!=FIELD}
    assert m[FIELD]['payload']['checkpoint']['id']=='old'
@pytest.mark.parametrize('field,value',[('epoch','other'),('inputs',['fake']),('source_bundle',{'sha256':'changed'}),('optimizer_state_export_policy','durable')])
def test_no_other_change(m,field,value):
    m[field]=value
    with pytest.raises(ValueError):project(m,'ROOT',signed,trainer_project)
@pytest.mark.parametrize('field,value',[('signer','evil'),('signature','bad')])
def test_authentication(m,field,value):
    m[FIELD][field]=value
    with pytest.raises(ValueError):project(m,'ROOT',signed,trainer_project)
def test_nested_rejected(m):
    m[FIELD]['payload'][FIELD]={}
    with pytest.raises(ValueError):project(m,'ROOT',signed,trainer_project)
def test_no_wrapper_passes_original(m):
    del m[FIELD];assert project(m,'ROOT',signed,trainer_project)is None
def test_recovery_uses_existing_gate(m):
    m['training_startup_recovery']={};assert project(m,'ROOT',signed,trainer_project)is None
def test_actual_large_shape(m):
    m[FIELD]['payload']['capabilities']='a'*1326025;m[FIELD]['payload']['learner_blacklist_selection_policy']='b'*1526480
    m['capabilities']='a'*1326025;m['learner_blacklist_selection_policy']='b'*1526480
    with pytest.raises(ValueError):enforce_size(m)
    p=project(m,'ROOT',signed,trainer_project);assert enforce_size(p)<4_000_000
    assert SUFFIX=='-local-state-v1'
def test_size_limit_not_raised():
    with pytest.raises(ValueError):enforce_size({'payload':'x'*4_000_000})
from ops.publication_local_projection import validate_recovery,sha,VERSION
@pytest.fixture
def recovery(m,tmp_path):
    job={'role':'upload','job_id':'oldjob','created_at':1,'manifest':{'payload':m,'signer':'ROOT','signature':'valid'}}
    failure={'job_id':'oldjob','phase':'failed','exit_code':1,'started_at':2,'finished_at':3}
    report={'role':'train','success':True,'epoch':'e88','new_checkpoint':dict(m['checkpoint'],path='/remote/model')}
    store={'job':{'payload':job,'signer':'ROOT','signature':'valid'},'failure':failure,'training':report}
    d={'version':VERSION,'created_at':4,'original_job':{'path':'job','payload_sha256':sha(job)},'original_failure':{'path':'failure','sha256':sha(failure)},'original_report_path':str(tmp_path/'absent'), 'successful_training_report':{'path':'training','sha256':sha(report)},'projected_manifest_sha256':sha(project(m,'ROOT',signed,trainer_project)), 'original_label':'e88-publish-new','replacement_label':'e88-publish-new'+SUFFIX,'checkpoint_remote_path':'/remote/model'}
    return d,store

def test_recovery_valid(recovery):
    d,store=recovery;assert validate_recovery(d,'ROOT',signed,trainer_project,store.__getitem__)['checkpoint']=='new'
@pytest.mark.parametrize('field',['role','success','epoch','new_checkpoint'])
def test_wrong_successful_report(recovery,field):
    d,store=recovery;store['training'][field]={'id':'other'}if field=='new_checkpoint'else'wrong';d['successful_training_report']['sha256']=sha(store['training'])
    with pytest.raises(ValueError):validate_recovery(d,'ROOT',signed,trainer_project,store.__getitem__)
def test_not_terminal(recovery):
    d,store=recovery;store['failure']['phase']='running';d['original_failure']['sha256']=sha(store['failure'])
    with pytest.raises(ValueError):validate_recovery(d,'ROOT',signed,trainer_project,store.__getitem__)
def test_keep_old_label_forbidden(recovery):
    d,store=recovery;d['replacement_label']=d['original_label']
    with pytest.raises(ValueError):validate_recovery(d,'ROOT',signed,trainer_project,store.__getitem__)
def test_old_report_exists(recovery):
    from pathlib import Path
    d,store=recovery;Path(d['original_report_path']).write_text('{}')
    with pytest.raises(ValueError):validate_recovery(d,'ROOT',signed,trainer_project,store.__getitem__)
