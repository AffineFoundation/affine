"""Prospective root-attested durable publication eligibility, never GPU work."""
import json,time
from pathlib import Path
from .storage import canonical
from .live_reward_bridge import signed,sha
VERSION='durable-training-before-reward-v1'
class PublicationPending(RuntimeError):
    def __init__(self,epoch):self.epoch=epoch;super().__init__('durable reward publication pending: '+epoch)
def validate_policy(value):
    if value!=VERSION or type(value)is not str:raise ValueError('reward publication policy')
    return value
def read(state,epoch,label):return json.loads((Path(state)/(epoch+'-'+label+'.json')).read_text())
def checked_evidence(state,manifest,authority,publication=None,checkpoint_document=None,*,enforce_latest=False):
    epoch=manifest['epoch'];validate_policy(manifest['reward_publication_policy'])
    scores=read(state,epoch,'scores');score_document=read(state,epoch,'signed-compute-scores')
    if signed(score_document,authority)!=scores or scores['epoch_id']!=epoch:raise ValueError('original signed reward scores')
    if type(scores.get('points'))is not dict or any(type(v)is not int or v<0 for v in scores['points'].values()):raise ValueError('typed original reward points')
    base=dict(epoch=epoch,manifest_sha256=sha(manifest),source_sha256=manifest['source_bundle']['sha256'],score_sha256=sha(scores),input_checkpoint=manifest['checkpoint']['id'])
    metrics_path=Path(state)/(epoch+'-training-metrics.json')
    if not metrics_path.exists():
        if any(v!=0 for v in scores['points'].values()):raise PublicationPending(epoch)
        path=Path(state)/(epoch+'-empty-closed.json')
        if not path.exists():raise PublicationPending(epoch)
        empty=json.loads(path.read_text())
        if empty!={'epoch':epoch,'status':'closed_no_accepted_batches','payable':False,'checkpoint':manifest['checkpoint']['id']}:raise ValueError('explicit no-update reward closure')
        if publication is not None or checkpoint_document is not None:raise ValueError('empty epoch cannot invent publication')
        return dict(base,status='closed_no_update',training_steps=0,output_checkpoint=base['input_checkpoint'],empty_sha256=sha(empty))
    from .persistent_training_protocol import validate_pointer,validate_parent,validate_report
    from .backend_jobs import signed as decode
    metrics=json.loads(metrics_path.read_text());binding=manifest['trainer_state_binding']
    pointer=validate_pointer(metrics['trainer_state']);steps=metrics['steps']
    if (type(steps)is not int or steps<=0 or pointer['optimizer_steps']!=binding['global_step_before']+steps or metrics['source_epoch']!=epoch or metrics['input_checkpoint']!=base['input_checkpoint'] or metrics['new_checkpoint']['id']!=pointer['inference_checkpoint']):raise ValueError('actual advanced persistent training')
    from .training_startup_recovery import local_request
    record,job,recovery_evidence=local_request(state,epoch,authority)
    report=read(Path(state)/'roles',record['job_id'],'report');training_manifest=decode(job['manifest'],authority)
    if (job['job_id']!=record['job_id'] or sha(job)!=record['job_sha256'] or metrics['original_job_sha256']!=sha(job) or metrics['trainer_binding_sha256']!=sha(binding) or training_manifest['trainer_state_binding']!=binding or training_manifest.get('reward_publication_policy')!=VERSION or job['steps']!=steps):raise ValueError('original trained reward execution binding')
    validate_report(report,job,training_manifest)
    if enforce_latest and json.loads((Path(state)/'latest-trainer-state.json').read_text())!=pointer:raise ValueError('actual latest committed reward state')
    if publication is None or checkpoint_document is None:raise PublicationPending(epoch)
    output_binding=dict(binding,input_checkpoint=pointer['inference_checkpoint'],genesis=None,parent=pointer,global_step_before=pointer['optimizer_steps'])
    validate_parent(publication,output_binding,authority)
    body=signed(publication,authority)
    if body['job_id']!=job['job_id'] or body['job_sha256']!=sha(job):raise ValueError('durable publication original training job')
    cp=metrics['new_checkpoint'];cpbody=signed(checkpoint_document,authority)
    if cpbody!={'id':cp['id'],'files':cp['files']}:raise ValueError('authority durable inference checkpoint')
    receipt=read(state,epoch,'checkpoint-publication')
    if receipt['checkpoint']!=cp['id'] or receipt.get('operator_independent_hashes')is not True or {n:r['sha256']for n,r in receipt['objects'].items()}!=cp['files']:raise ValueError('actual inference publication receipt')
    if recovery_evidence is not None:base['training_startup_recovery']=recovery_evidence
    return dict(base,status='durably_trained',training_steps=steps,output_checkpoint=cp['id'],optimizer_steps=pointer['optimizer_steps'],parent_optimizer_steps=binding['global_step_before'],input_parent=binding['parent'],trainer_binding_sha256=sha(binding),original_job_id=job['job_id'],original_job_sha256=sha(job),source_files_sha256=sha(job['source_files']),runtime_versions_sha256=sha(job['runtime_versions']),trainer_state=pointer,metrics_sha256=sha(metrics),checkpoint_receipt_sha256=sha(receipt),publication=publication,checkpoint_document=checkpoint_document)
def checked_ready(document,state,manifest,authority):
    ready=signed(document,authority)
    if set(ready)!={'version','ready_at','evidence'} or ready['version']!=VERSION or type(ready['ready_at'])not in(int,float)or not __import__('math').isfinite(ready['ready_at']):raise ValueError('signed reward publication readiness')
    evidence=ready['evidence'];expected=checked_evidence(state,manifest,authority,evidence.get('publication'),evidence.get('checkpoint_document'))
    if evidence!=expected:raise ValueError('immutable reward publication evidence')
    if ready['ready_at']<read(state,manifest['epoch'],'scores')['finalized_at']:raise ValueError('readiness cannot predate audited scores')
    return ready
def require(state,manifest,authority):
    if manifest.get('reward_publication_policy')is None:return None
    validate_policy(manifest['reward_publication_policy']);epoch=manifest['epoch'];p=Path(state)/(epoch+'-reward-publication-ready.json')
    if not p.exists():raise PublicationPending(epoch)
    return checked_ready(json.loads(p.read_text()),state,manifest,authority)
def emit(controller,manifest):
    if manifest.get('reward_publication_policy')is None:return None
    state=controller.state;epoch=manifest['epoch'];path=state/(epoch+'-reward-publication-ready.json')
    if path.exists():return require(state,manifest,controller.authority.id)
    # Recover actual immutable bucket attestation after public PUT/local-save crash.
    from botocore.exceptions import ClientError
    from .remote_backend import save
    key='public/'+epoch+'/reward-publication-ready.json'
    try:existing=json.loads(controller.bucket.get(key))
    except ClientError as error:
        if str(error.response.get('Error',{}).get('Code'))not in('NoSuchKey','404','NotFound'):raise
    else:
        ready=checked_ready(existing,state,manifest,controller.authority.id);save(path,existing);return ready
    publication=checkpoint_document=None;metrics=state/(epoch+'-training-metrics.json')
    if metrics.exists():
        m=json.loads(metrics.read_text());publication=json.loads(controller.bucket.get(m['trainer_state']['descriptor_key']));checkpoint_document=json.loads(controller.bucket.get(m['new_checkpoint']['descriptor_key']))
    evidence=checked_evidence(state,manifest,controller.authority.id,publication,checkpoint_document,enforce_latest=True)
    ready=dict(version=VERSION,ready_at=time.time(),evidence=evidence);document=controller.signed(ready)
    controller.bucket.json('public/'+epoch+'/reward-publication-ready.json',document)
    if signed(json.loads(controller.bucket.get('public/'+epoch+'/reward-publication-ready.json')),controller.authority.id)!=ready:raise ValueError('durable reward readiness readback')
    from .remote_backend import save
    save(path,document);return ready
