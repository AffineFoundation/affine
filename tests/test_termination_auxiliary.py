"""CPU-only real tiny-model gradients; synthetic signed fixture, no GPU claim."""
import copy
import base64
import hashlib
from pathlib import Path
import pytest
import torch
from nacl.signing import SigningKey
from subnet.storage import canonical
from subnet.epoch_optimizer import preference_loss
from subnet import termination_auxiliary as aux


def sign(key,payload):return dict(payload=copy.deepcopy(payload),signer=key.verify_key.encode().hex(),signature=base64.b64encode(key.sign(canonical(payload)).signature).decode())


def fixture(lam=.001,output=None,reward=1.):
    root=SigningKey.generate();worker=SigningKey.generate();authority=root.verify_key.encode().hex();eos=2
    pos=dict(env_id='affine_math',index=7,task_hash='a'*64,classification='positive',reward=reward,turns=[dict(prompt=[0,1],output=[1,3,2]if output is None else output,done=True,classification='positive',reward=reward)]);batch=dict(env_id='affine_math',index=7,rollouts=[pos])
    manifest=dict(epoch='original',checkpoint={'id':'b'*64},environments=[dict(env_id='affine_math',indices=[7],spec={'max_turns':1})]);job=dict(role='verify',job_id='verify-original',created_at=10.,expires_at=20.,source_files={'subnet/gpu_runtime.py':'e'*64},manifest=sign(root,manifest),submissions=[dict(sha256='c'*64,commitment_ref=dict(batch_sha256=aux.sha(batch)))])
    report=dict(success=True,completed_at=15.,source_files=job['source_files'],role='verify',job_id=job['job_id'],job_sha256=aux.sha(job),epoch='original',checkpoint='b'*64,audits=[dict(submission_sha256='c'*64,accepted=[batch],outcomes=[dict(index=7,env_id='affine_math',valid=True,fully_audited=True)])]);req=sign(worker,dict(report=report))
    policy=dict(version=aux.VERSION,**{'lambda':lam},eos_token_id=eos,max_output_tokens=10)
    experiment=sign(root,dict(epoch='isolated-experiment',original_epoch='original',checkpoint='b'*64,experiment_only=True,payable=False,verified_workers=[worker.verify_key.encode().hex()],termination_auxiliary=policy,source_files={aux.BASE_MODULE:hashlib.sha256(Path(aux.__file__).with_name('epoch_optimizer.py').read_bytes()).hexdigest(),aux.MODULE:hashlib.sha256(Path(aux.__file__).read_bytes()).hexdigest(),aux.GRADER:'d'*64}))
    grade=sign(root,dict(version=aux.NATIVE_VERSION,original_epoch='original',checkpoint='b'*64,batch_sha256=aux.sha(batch),positive_sha256=aux.sha(pos),native_correct=True,grader_source_sha256='d'*64,original_report_sha256=aux.sha(report)))
    return root,worker,dict(experiment=experiment,authority=authority,original_job=sign(root,job),worker_request=req,native_receipt=grade,batch=batch,positive=pos)


def mutate_signed(values,key,field,path,value):
    payload=copy.deepcopy(values[field]['payload']);v=payload
    for part in path[:-1]:v=v[part]
    v[path[-1]]=value;values[field]=sign(key,payload)


def test_default_loss_and_real_tiny_model_gradients_identical():
    torch.manual_seed(5);model=torch.nn.Linear(4,5,bias=False);inputs=torch.eye(4)[:3];out=model(inputs);margin=torch.log_softmax(out,-1)[range(3),[1,3,2]].mean();expected=preference_loss(torch,margin,float(margin.detach()),.1);expected.backward();before=model.weight.grad.clone();model.zero_grad();out=model(inputs);margin=torch.log_softmax(out,-1)[range(3),[1,3,2]].mean();actual=aux.loss(torch,margin,float(margin.detach()));actual.backward();assert torch.equal(before,model.weight.grad);assert actual.item()==expected.item()


def test_genuine_tiny_model_eos_gradient_only_and_zero_lambda_identical():
    root,worker,values=fixture();torch.manual_seed(7);model=torch.nn.Linear(4,5,bias=False);inputs=torch.eye(4)[:3];logits=model(inputs);margin=torch.log_softmax(logits,-1)[range(3),[1,3,2]].mean()+1.;reference=float(margin.detach());baseline=preference_loss(torch,margin,reference,.1);baseline.backward();base=model.weight.grad.clone();model.zero_grad();logits=model(inputs);actual=aux.loss(torch,torch.log_softmax(logits,-1)[range(3),[1,3,2]].mean()+1.,reference,positive_logits=logits,**values);actual.backward();grad=model.weight.grad;assert torch.equal(grad[:,0],base[:,0]);assert torch.equal(grad[:,1],base[:,1]);assert grad[2,2]<0;assert torch.any(grad[:,2]!=base[:,2]);assert grad[:,3].eq(0).all()
    _,_,zero=fixture(lam=0.);model.zero_grad();logits=model(inputs);aux.loss(torch,torch.log_softmax(logits,-1)[range(3),[1,3,2]].mean()+1.,reference,positive_logits=logits,**zero).backward();assert torch.equal(model.weight.grad,base)


@pytest.mark.parametrize('output',[[1,2,3,2],[1,3,4],[1,3,4,1,3,4,1,3,4,2]])
def test_early_eos_missing_eos_and_capped_output_refused(output):
    _,_,v=fixture(output=output)
    with pytest.raises(ValueError,match='already-present'):aux.admit(**{k:x for k,x in v.items()if k!='positive_logits'})


@pytest.mark.parametrize('field,path,value',[
 ('experiment',['payable'],True),('experiment',['experiment_only'],False),
 ('experiment',['termination_auxiliary','lambda'],.006),('experiment',['termination_auxiliary','lambda'],True),
 ('experiment',['source_files',aux.MODULE],'f'*64),('experiment',['source_files',aux.BASE_MODULE],'f'*64),
 ('native_receipt',['native_correct'],False),('native_receipt',['positive_sha256'],'f'*64),
 ('native_receipt',['grader_source_sha256'],'f'*64)])
def test_signed_but_invalid_policy_and_native_evidence_refused(field,path,value):
    root,worker,v=fixture();mutate_signed(v,root,field,path,value)
    with pytest.raises(ValueError):aux.admit(**v)


@pytest.mark.parametrize('field,value',[('fully_audited',False),('fully_audited',1),('valid',False)])
def test_missing_proof_or_unaudited_or_invalid_refused(field,value):
    root,worker,v=fixture();mutate_signed(v,worker,'worker_request',['report','audits',0,'outcomes',0,field],value)
    with pytest.raises(ValueError):aux.admit(**v)


def test_forged_label_signature_and_mutated_original_tokens_refused():
    _,_,v=fixture();v['positive']['classification']='negative'
    with pytest.raises(ValueError):aux.admit(**v)
    _,_,v=fixture();v['native_receipt']['signature']=base64.b64encode(bytes(64)).decode()
    with pytest.raises(ValueError):aux.admit(**v)


def test_missing_evidence_and_unapproved_worker_refused():
    root,worker,v=fixture();v['worker_request']=sign(SigningKey.generate(),v['worker_request']['payload'])
    with pytest.raises(ValueError,match='approved'):aux.admit(**v)
    _,_,v=fixture();v['original_job']['payload']['submissions']=[]
    with pytest.raises(ValueError):aux.admit(**v)


def test_no_untrusted_admission_dictionary_or_silent_optin():
    with pytest.raises(ValueError):aux.loss(torch,torch.tensor(0.),0.,positive={})

@pytest.mark.parametrize("path,value",[(["completed_at"],21.),(["source_files","subnet/gpu_runtime.py"],"f"*64)])
def test_expired_original_report_or_changed_scientific_source_refused(path,value):
    root,worker,v=fixture();mutate_signed(v,worker,"worker_request",["report"]+path,value)
    with pytest.raises(ValueError):aux.admit(**v)


def test_missing_terminal_logit_or_nonfinite_logit_refused():
    _,_,v=fixture()
    with pytest.raises(ValueError):aux.loss(torch,torch.tensor(0.),0.,positive_logits=torch.zeros(2,5),**v)
    logits=torch.zeros(3,5);logits[-1,0]=float("nan")
    with pytest.raises(ValueError,match="finite"):aux.loss(torch,torch.tensor(0.),0.,positive_logits=logits,**v)


def test_foreign_environment_context_refused():
    root,worker,v=fixture();job=copy.deepcopy(v["original_job"]["payload"]);original=copy.deepcopy(job["manifest"]["payload"]);original["environments"][0]["evaluation_only"]=True;job["manifest"]=sign(root,original);v["original_job"]=sign(root,job);report=copy.deepcopy(v["worker_request"]["payload"]);report["report"]["job_sha256"]=aux.sha(job);v["worker_request"]=sign(worker,report)
    with pytest.raises(ValueError,match="mining-only"):aux.admit(**v)


def test_boolean_positive_reward_cannot_replace_numeric_native_score():
    _,_,v=fixture(reward=True)
    with pytest.raises(ValueError,match="correct-positive"):aux.admit(**v)
