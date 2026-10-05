"""Synchronous epoch controller, artifact authority and trainer barrier."""
import base64
import json
import os
import secrets
import subprocess
import sys
import time
import tempfile
from pathlib import Path
from .model import ENV, Runtime, model_files,NUMERICAL_RUNTIME_REVISION
from .storage import canonical, sha, Identity
from .scoring import score
from .protocol import classification
from . import harness as harness_policy
from .environments import EnvironmentSpec, legacy_spec, legacy_harness

class CheckpointCapacityError(RuntimeError):
    pass


def save_manifest(path, manifest):
    """Publish local JSON only after its complete bytes have reached disk."""
    path = Path(path)
    data = canonical(manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + '.', suffix='.tmp', dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def require_checkpoint_space(checkpoint_path,destination,min_free_bytes=2*1024**3):
    import shutil
    if type(min_free_bytes) is not int or min_free_bytes<0:raise ValueError('checkpoint capacity budget')
    source=Path(checkpoint_path)
    model_bytes=sum((source/name).stat().st_size for name in model_files(source))
    required=max(min_free_bytes,2*model_bytes+512*1024**2)
    ancestor=Path(destination).parent
    while not ancestor.exists():ancestor=ancestor.parent
    free=shutil.disk_usage(ancestor).free
    if free<required:raise CheckpointCapacityError(f'insufficient checkpoint disk headroom: {free} bytes free, {required} required')
    return dict(free_bytes=free,required_bytes=required,model_bytes=model_bytes)


class Controller:
    def __init__(self,bucket,gateway,state):
        self.bucket,self.gateway,self.state=bucket,gateway,Path(state)
        self.state.mkdir(parents=True,exist_ok=True)
        key=self.state/'authority.seed'
        if key.exists():self.authority=Identity(bytes.fromhex(key.read_text()))
        else:
            self.authority=Identity();key.write_text(self.authority.key.encode().hex());key.chmod(0o600)

    def signed(self,value):
        payload=canonical(value)
        return dict(payload=value,signer=self.authority.id,signature=base64.b64encode(self.authority.key.sign(payload).signature).decode())

    def publish_checkpoint(self,path):
        files=model_files(path)
        identifier=sha(canonical(files))
        for name in files:
            self.bucket.upload(f'public/checkpoints/{identifier}/{name}',Path(path)/name)
            if sha(self.bucket.get(f'public/checkpoints/{identifier}/{name}'))!=files[name]:
                raise ValueError('checkpoint upload corruption')
        descriptor_key=f'public/checkpoints/{identifier}/authorities/{self.authority.id}/checkpoint.json'
        checkpoint=dict(id=identifier,files=files,base_url=f'{self.gateway.url}/public/checkpoints/{identifier}',descriptor_key=descriptor_key)
        descriptor=dict(id=identifier,files=files)
        from botocore.exceptions import ClientError
        from nacl.signing import VerifyKey
        def existing(key):
            try:return self.bucket.get(key)
            except ClientError as exc:
                if str(exc.response.get('Error',{}).get('Code')) not in ('NoSuchKey','404','NotFound'):raise
            except KeyError:pass
            return None
        old=existing(descriptor_key)
        if old is None:self.bucket.json(descriptor_key,self.signed(descriptor))
        else:
            signed=json.loads(old)
            if signed['signer']!=self.authority.id:raise ValueError('checkpoint descriptor authority collision')
            VerifyKey(bytes.fromhex(self.authority.id)).verify(canonical(signed['payload']),base64.b64decode(signed['signature'],validate=True))
            if any(signed['payload'].get(k)!=v for k,v in descriptor.items()):raise ValueError('checkpoint descriptor content collision')
        # Content may be shared by independently authorized controllers. Their
        # signatures must never replace the original historical trust anchor.
        legacy_key=f'public/checkpoints/{identifier}/checkpoint.json'
        if existing(legacy_key) is None:self.bucket.json(legacy_key,self.signed(descriptor))
        return checkpoint

    def open(self,epoch,checkpoint,miners,duration=600,environment=None,runtime_profile=None,harness=None,environments=None,audit_policy=None,evaluation=None,source_bundle=None,model_runtime_revision=None,numerical_policy=None,backend_profile=None,model_id=None,sample_harness_registry=None,training_policy=None,artifact_policy=None,task_assets=None,live_reward_anchor_document=None,live_reward_registration_snapshot=None,sampling_policy=None,trainer_state_binding=None,submission_transport_policy=None,commitment_max_batches=3,hourly_execution_policy=None,optimizer_state_transport=None,persistent_publication_policy=None,reward_publication_policy=None,optimizer_state_export_policy=None,artifact_compression_policy=None,proof_copy_policy=None):
        if artifact_compression_policy is not None:
            from .batches import compression_policy
            artifact_compression_policy=compression_policy(artifact_compression_policy)
            if submission_transport_policy is None:raise ValueError('prospective compression requires per-pair commitments')
        if persistent_publication_policy is not None:
            from .persistent_publication import validate_policy
            persistent_publication_policy=validate_policy(persistent_publication_policy)
        if optimizer_state_export_policy is not None:
            from .persistent_publication import export_policy
            export_policy(dict(optimizer_state_export_policy=optimizer_state_export_policy,persistent_publication_policy=persistent_publication_policy))
        if optimizer_state_transport is not None:
            from .persistent_training_state import transport_concurrency
            transport_concurrency({'optimizer_state_transport':optimizer_state_transport})
        from .live_reward_bridge import prevalidate_opening_arguments
        prevalidate_opening_arguments(epoch,checkpoint,miners,duration,audit_policy,source_bundle,live_reward_anchor_document,live_reward_registration_snapshot,self.authority.id)
        sampling_contract=None
        if sampling_policy is not None:
            from .forced_sampling import new_contract,validate_harness
            sampling_contract=new_contract(sampling_policy)
            for definition in environments or [dict(harness=harness)]:
                raw=definition.get('harness')
                if isinstance(raw,dict) and raw.get('version')=='indexed-harness-v1':
                    for row in raw['by_index'].values():validate_harness(row)
                else:validate_harness(raw)
        for definition in environments or []:
            if 'evaluation_only' in definition:
                if type(definition['evaluation_only'])is not bool or (definition['evaluation_only'] and definition.get('indices')!=[]):
                    raise ValueError('evaluation-only mining indices')
        if training_policy is not None:
            from .training_policy import epoch_policy
            from .backend_profiles import profile
            profile(model_runtime_revision)
            epoch_policy({'training_policy':training_policy})
        from .persistent_cpu_adamw import POLICY as PERSISTENT_POLICY
        if training_policy==PERSISTENT_POLICY or trainer_state_binding is not None:
            from .persistent_training_protocol import validate_binding
            validate_binding(trainer_state_binding,dict(epoch=epoch,checkpoint=checkpoint,
                training_policy=training_policy,source_bundle=source_bundle or {}))
            if sampling_contract is None:raise ValueError('persistent training opening requires forced sampling')
        deadline=int(time.time())+duration
        if getattr(self.gateway,'direct_r2',False):
            # Immutable public model reads must survive freeze, queueing, audits
            # and training. Upload capabilities still end at the epoch deadline.
            checkpoint=dict(checkpoint,read_urls={name:self.bucket.presign(f"public/checkpoints/{checkpoint['id']}/{name}",expires=604800) for name in checkpoint['files']})
        if proof_copy_policy is not None:
            from .selected_proof_copy import validate_policy
            proof_copy_policy=validate_policy(proof_copy_policy)
            if submission_transport_policy is None or audit_policy is None or audit_policy.get("version")!="bounded-random-v1" or hourly_execution_policy is None:raise ValueError("selected proof copy requires bounded hourly commitments")
        commitment_binding=None
        if submission_transport_policy is not None:
            from .commitment_transport import VERSION
            if submission_transport_policy!=VERSION or type(commitment_max_batches)is not int or not 1<=commitment_max_batches<=256:raise ValueError('commitment transport policy/cap')
            commitment_binding=dict(version=VERSION,checkpoint=checkpoint['id'],source=source_bundle['sha256'],max_batches=commitment_max_batches)
        if reward_publication_policy is not None:
            from .reward_publication import validate_policy
            validate_policy(reward_publication_policy)
        if hourly_execution_policy is not None:
            from .hourly_policy import validate
            hourly_execution_policy=validate(hourly_execution_policy,duration)
            if commitment_binding is None:raise ValueError('hourly policy requires small commitments')
            commitment_binding['freeze_until']=deadline+hourly_execution_policy['freeze_seconds']
        if proof_copy_policy is not None:commitment_binding['proof_copy_policy']=proof_copy_policy
        from .artifact_budget import for_manifest
        budget=for_manifest(dict(artifact_policy=artifact_policy,model_runtime_revision=model_runtime_revision,backend_profile=backend_profile,numerical_policy=numerical_policy)) if artifact_policy is not None else {'compressed_bytes':100_000_000}
        caps=self.gateway.open(epoch,miners,deadline,upload_limit=budget['compressed_bytes'],**({'commitment_binding':commitment_binding}if commitment_binding else {}))
        env=dict(environment or ENV)
        definitions=[]
        for definition in environments or [dict(spec=env,harness=harness)]:
            raw=definition['spec']
            spec=legacy_spec(raw) if 'id' not in raw else EnvironmentSpec.from_dict(raw)
            from .sample_harness import validate as validate_sample_harness
            chosen=validate_sample_harness(definition.get('harness') or (legacy_harness(raw) if 'id' not in raw else None),definition.get('indices',list(range(spec.num_samples))))
            row=dict(env_id=spec.id,spec=spec.to_dict(),harness=chosen,indices=definition.get('indices',list(range(spec.num_samples))))
            if definition.get('evaluation_only',False):row['evaluation_only']=True
            definitions.append(row)
        manifest=dict(payable=not epoch.startswith(('nonpayable-', 'test-', 'mock-')),epoch=epoch,checkpoint=checkpoint,environment=env,indices=definitions[0]['indices'],environments=definitions,harness_source_hash=harness_policy.source_hash(),
                      tokenizer_binding={name:digest for name,digest in checkpoint['files'].items() if 'token' in name or 'template' in name},
                      K=1,L=1,max_batches=4,start=self.gateway.epochs[epoch].get('start',int(time.time())),deadline=deadline,capabilities=caps,audit_policy=dict(audit_policy or {'mode':'full','version':1}),
                      numerical_policy='cpu-float32-eager-exact-toploc-logprob-atol1e-5',model_runtime_revision=NUMERICAL_RUNTIME_REVISION,runtime_profile=dict(runtime_profile or {}),
                      environment_revision='trusted-adapter-registry-v1')
        if sample_harness_registry is not None:manifest['sample_harness_registry']=sample_harness_registry
        if submission_transport_policy is not None:manifest['submission_transport_policy']=submission_transport_policy
        if artifact_compression_policy is not None:manifest['artifact_compression_policy']=artifact_compression_policy
        if proof_copy_policy is not None:manifest['proof_copy_policy']=proof_copy_policy
        if hourly_execution_policy is not None:manifest['hourly_execution_policy']=hourly_execution_policy
        if reward_publication_policy is not None:manifest['reward_publication_policy']=reward_publication_policy
        manifest['transport_policy']='direct-r2-v1' if getattr(self.gateway,'direct_r2',False) else 'gateway-v1'
        if model_runtime_revision is not None:manifest['model_runtime_revision']=model_runtime_revision
        if numerical_policy is not None:manifest['numerical_policy']=numerical_policy
        if backend_profile is not None:manifest['backend_profile']=backend_profile
        if model_id is not None:manifest['model_id']=model_id
        if training_policy is not None:manifest['training_policy']=training_policy
        if trainer_state_binding is not None:manifest['trainer_state_binding']=trainer_state_binding
        if optimizer_state_export_policy is not None:manifest['optimizer_state_export_policy']=optimizer_state_export_policy
        if optimizer_state_transport is not None:manifest['optimizer_state_transport']=dict(optimizer_state_transport)
        if persistent_publication_policy is not None:manifest['persistent_publication_policy']=persistent_publication_policy
        if artifact_policy is not None:manifest['artifact_policy']=artifact_policy
        if task_assets is not None:manifest['task_assets']=task_assets
        if sampling_contract is not None:
            from .forced_sampling import source_hash
            manifest['sampling_contract']=sampling_contract
            manifest['sampling_source_hash']=source_hash()
        from .runtime_factory import validate_backend
        validate_backend(manifest)
        if source_bundle is not None:manifest['source_bundle']=dict(source_bundle)
        if evaluation is not None:
            manifest['evaluation']=dict(evaluation,harness=harness_policy.normalize(evaluation.get('harness')))
        if live_reward_anchor_document is not None:
            from .live_reward_bridge import inject_opening_manifest
            manifest=inject_opening_manifest(manifest,live_reward_anchor_document,self.authority.id,live_reward_registration_snapshot)
        elif live_reward_registration_snapshot is not None:
            raise ValueError('reward registration snapshot requires signed forward-live contract')
        from .protocol import entries as validate_entries
        validate_entries(manifest)
        save_manifest(self.state/f'{epoch}-manifest.json', manifest)
        self.bucket.json(f'public/{epoch}/manifest.json',self.signed(manifest))
        pointer='public/current.json' if manifest['payable'] else f'public/{epoch}/current.json'
        self.bucket.json(pointer,self.signed(dict(epoch=epoch,manifest=f'public/{epoch}/manifest.json')))
        return manifest

    def finalize(self,manifest,checkpoint_path):
        epoch=manifest['epoch']
        saved=self.state/f'{epoch}-scores.json'
        if saved.exists():
            result=json.loads(saved.read_text())
            reports={miner:json.loads((self.state/f'{epoch}-{miner}-report.json').read_text()) for miner in result['receipts']}
            self.bucket.json(f'public/{epoch}/scores.json',self.signed(result))
            return result,reports
        receipts=self.gateway.freeze(epoch);reports={}
        seed_path=self.state/f'{epoch}-audit-challenge.json'
        if seed_path.exists(): challenge=json.loads(seed_path.read_text())
        else:
            challenge=dict(seed=secrets.token_hex(32),generated_after_freeze_at=time.time(),receipts=receipts)
            seed_path.write_bytes(canonical(challenge));seed_path.chmod(0o600)
        if challenge['receipts']!=receipts:raise ValueError('audit challenge receipt binding')
        audit_manifest=dict(manifest,audit_seed=challenge['seed'],audit_frozen_receipts=receipts)
        audit_manifest_path=self.state/f'{epoch}-audit-manifest.json'
        audit_manifest_path.write_bytes(canonical(audit_manifest))
        self.bucket.json(f'public/{epoch}/audit-challenge.json',self.signed(challenge))
        for miner,receipt in receipts.items():
            artifact=self.state/f'{epoch}-{miner}.zip';self.bucket.download(receipt['frozen_key'],artifact)
            reportpath=self.state/f'{epoch}-{miner}-report.json'
            subprocess.run([sys.executable,'-m','subnet.verifier',str(artifact),str(audit_manifest_path),str(checkpoint_path),str(reportpath)],check=True,timeout=600)
            report=json.loads(reportpath.read_text())
            if report.get('submission_sha256') != receipt['sha256']:
                report=dict(accepted=[],valid=False,reason='missing or wrong report binding',submission_sha256=receipt['sha256'])
            reports[miner]=report
            self.bucket.json(f'public/{epoch}/audits/{miner}.json',self.signed(report))
        result=score(reports)
        result.update(payable=manifest.get('payable',False),epoch_id=epoch,finalized_at=time.time(),receipts=receipts,checkpoint=manifest['checkpoint']['id'])
        (self.state/f'{epoch}-scores.json').write_bytes(canonical(result))
        self.bucket.json(f'public/{epoch}/scores.json',self.signed(result))
        return result,reports

    def train(self,manifest,reports,checkpoint_path,destination,steps=1,min_free_bytes=2*1024**3):
        capacity=require_checkpoint_space(checkpoint_path,destination,min_free_bytes)
        pairs=[]
        for report in reports.values():
            for batch in report['accepted']:
                positive=[r for r in batch['rollouts'] if classification(r)=='positive']
                negative=[r for r in batch['rollouts'] if classification(r)=='negative']
                pairs.extend(zip(positive,negative))
        if not pairs:raise ValueError('no independently verified training pairs')
        from .protocol import entry,harness_for
        first=entry(manifest,pairs[0][0].get('env_id'))
        initial_harness=harness_for(first,pairs[0][0]['index'])
        job=self.state/f"{manifest['epoch']}-training-job.json"
        metricspath=self.state/f"{manifest['epoch']}-training-metrics.json"
        job.write_bytes(canonical(dict(checkpoint=str(checkpoint_path),files=manifest['checkpoint']['files'],pairs=pairs,destination=str(destination),steps=steps,environment=first['spec'],harness=initial_harness,runtime_profile=manifest.get('runtime_profile',{}))))
        job.chmod(0o600)
        subprocess.run([sys.executable,'-m','subnet.trainer',str(job),str(metricspath)],check=True,timeout=1200)
        metrics=json.loads(metricspath.read_text())
        checkpoint=self.publish_checkpoint(destination)
        if checkpoint['id']==manifest['checkpoint']['id']:raise ValueError('unchanged checkpoint')
        metrics.update(source_epoch=manifest['epoch'],input_pairs=len(pairs),checkpoint=checkpoint['id'],capacity_preflight=capacity)
        metricspath.write_bytes(canonical(metrics))
        self.bucket.json(f"public/{manifest['epoch']}/training.json",self.signed(metrics))
        return checkpoint,metrics
