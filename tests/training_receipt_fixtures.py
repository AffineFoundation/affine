"""Synthetic signed receipts for CPU tests; never operational admission evidence."""
import base64
import copy
import hashlib
import numpy as np
from nacl.signing import SigningKey
from subnet.storage import canonical
from subnet import forced_sampling as sampling, harness
from subnet.training_policy import coverage_manifest
from subnet.training_receipts import receipt_payload, VERSION, sha


def sign(key, payload):
    return dict(payload=copy.deepcopy(payload),signer=key.verify_key.encode().hex(),
        signature=base64.b64encode(key.sign(canonical(payload)).signature).decode())


def signed_receipt(key,manifest,miner,frozen,batch,*,worker=None,job_id='synthetic-verify'):
    """Build a genuine signature chain with synthetic claimed audit results."""
    worker=worker or SigningKey.generate();identity=worker.verify_key.encode().hex()
    audit=dict(epoch=manifest['epoch'],submission_sha256=frozen['sha256'],accepted=[copy.deepcopy(batch)],
        outcomes=[dict(batch=0,fully_audited=True,valid=True,env_id=batch['env_id'],index=batch['index'])],
        training_eligibility='fully-audited-only',sampling_assurance=sampling.assurance(manifest))
    job=dict(schema=1,job_id=job_id,role='verify',created_at=22,expires_at=100,
        manifest=sign(key,manifest),source_files={'subnet/model.py':'a'*64},
        runtime_versions={'torch':'synthetic','transformers':'synthetic','toploc':'synthetic'},
        submissions=[dict(sha256=frozen['sha256'],url='synthetic-exact-get')])
    report=dict(job_id=job_id,job_sha256=sha(job),operator=key.verify_key.encode().hex(),
        role='verify',epoch=manifest['epoch'],checkpoint=manifest['checkpoint']['id'],
        source_files=job['source_files'],runtime_versions=job['runtime_versions'],
        backend_profile=manifest['backend_profile'],numerical_policy=manifest['numerical_policy'],
        success=True,chain_transactions=False,completed_at=40,audits=[audit])
    request=sign(worker,dict(action='report',job_id=job_id,at=41,nonce='synthetic-report-nonce',
        token='synthetic-lease-token',report=report))
    envelope=sign(key,job)
    payload=receipt_payload(envelope,request,key.verify_key.encode().hex(),{identity:['verify']},
        manifest,miner,frozen,audit)
    return sign(key,payload),audit,envelope,request


def transport_fixture(key, *, policy='bf16-full-adamw-covered-fixed-reference-v3'):
    from subnet.backend_jobs import BACKEND_PROFILE,NUMERICAL_POLICY,REVISION,file_map
    from subnet.batches import pack
    miner='e'*64;files={'config.json':'1'*64,'model.safetensors':'2'*64}
    manifest=dict(epoch='nonpayable-synthetic-receipt',deadline=20,payable=False,training_policy=policy,
        checkpoint=dict(id=file_map(files),files=files),source_bundle={'sha256':'7'*64},K=1,L=1,max_batches=3,
        model_runtime_revision=REVISION,backend_profile=BACKEND_PROFILE,numerical_policy=NUMERICAL_POLICY,
        audit_policy={'mode':'full'},harness_source_hash=harness.source_hash(),
        environments=[dict(env_id='math',spec=dict(id='math',version='synthetic-v1',num_samples=10),
            indices=[0,1],harness=dict(harness.DEFAULT))],
        sampling_contract=sampling.new_contract({'version':sampling.VERSION,'max_attempts':16}),
        sampling_source_hash=sampling.source_hash())
    rollouts=[dict(seed=i,sampling=sampling.receipt(sampling.binding(manifest),i),
        classification='positive'if i==0 else'negative',reward=1 if i==0 else 0,
        env_id='math',index=0,task_hash='6'*64,turns=[dict(prompt=[1,2],output=[3+i])])for i in range(2)]
    batch=dict(schema=2,epoch=manifest['epoch'],checkpoint=manifest['checkpoint']['id'],
        env_id='math',environment_version='synthetic-v1',index=0,sample_index=0,rollouts=rollouts)
    data=pack([(batch,[[np.full((1,4),.25,np.float32)]for _ in range(2)])])
    frozen=dict(sha256=hashlib.sha256(data).hexdigest(),size=len(data),frozen_key='private/synthetic/frozen')
    receipts={miner:frozen};challenge=dict(seed='a'*64,receipts=receipts,generated_after_freeze_at=21)
    manifest=coverage_manifest(manifest,receipts,challenge)
    receipt,audit,job,request=signed_receipt(key,manifest,miner,frozen,batch)
    manifest['training_input_policy']=VERSION
    obj=dict(sha256=frozen['sha256'],size=frozen['size'],url='https://synthetic.r2.cloudflarestorage.com/private/frozen?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Signature=synthetic',
        accepted_batch_sha256=[sha(batch)],verifier_receipt=receipt)
    return dict(manifest=manifest,miner=miner,batch=batch,data=data,frozen=frozen,receipts=receipts,
        challenge=challenge,receipt=receipt,audit=audit,verify_job=job,worker_request=request,submission=obj)
