"""Default-off terminal-EOS auxiliary for isolated, nonpayable experiments only.

No production training caller selects this module. Admission authenticates
existing GPU verification and a separate native-grading receipt. No trajectory
or label is generated, appended, shortened or reclassified here.
"""
import hashlib
import math
from pathlib import Path
from .storage import canonical
from .training_receipts import authenticate, sha
from .epoch_optimizer import preference_loss

VERSION='terminal-eos-auxiliary-experiment-v1'
NATIVE_VERSION='experiment-native-positive-grade-v1'
MAX_LAMBDA=.005
GRADER='subnet/vendor/legacy/rollouts/envs/affine_math_v1/affine_math_v1/verify.py'
MODULE='subnet/termination_auxiliary.py'
BASE_MODULE='subnet/epoch_optimizer.py'


def policy(manifest):
    value=manifest.get('termination_auxiliary')
    if value is None:return None
    if (not isinstance(value,dict)or set(value)!={'version','lambda','eos_token_id','max_output_tokens'} or
        value['version']!=VERSION or type(value['lambda'])not in(int,float) or
        not math.isfinite(value['lambda']) or not 0<=value['lambda']<=MAX_LAMBDA or
        type(value['eos_token_id'])is not int or value['eos_token_id']<0 or
        type(value['max_output_tokens'])is not int or not 2<=value['max_output_tokens']<=2048 or
        manifest.get('experiment_only')is not True or manifest.get('payable')is not False):
        raise ValueError('explicit bounded nonpayable experimental objective')
    if (manifest.get('source_files',{}).get(MODULE)!=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()or
        manifest.get('source_files',{}).get(BASE_MODULE)!=hashlib.sha256(Path(__file__).with_name('epoch_optimizer.py').read_bytes()).hexdigest()):
        raise ValueError('exact experimental objective source pin')
    return dict(value)


def admit(experiment,authority,original_job,worker_request,native_receipt,batch,positive):
    """Return a bound admission only after immutable original proof/grade checks."""
    manifest=authenticate(experiment,authority);p=policy(manifest)
    if p is None:raise ValueError('explicit experiment policy required')
    job=authenticate(original_job,authority);original=authenticate(job['manifest'],authority)
    if (original.get('epoch')!=manifest.get('original_epoch') or
        original.get('checkpoint',{}).get('id')!=manifest.get('checkpoint') or
        job.get('role')!='verify' or job.get('job_id')is None or
        type(job.get('created_at'))not in(int,float)or type(job.get('expires_at'))not in(int,float)or
        not math.isfinite(job['created_at'])or not math.isfinite(job['expires_at'])or
        job['created_at']>=job['expires_at']):
        raise ValueError('original verified checkpoint/epoch/job required')
    worker=worker_request.get('signer')
    if worker not in manifest.get('verified_workers',[]):raise ValueError('approved original verifier required')
    request=authenticate(worker_request,worker);report=request.get('report',{})
    if (report.get('success')is not True or report.get('role')!='verify' or
        report.get('job_id')!=job['job_id'] or report.get('job_sha256')!=sha(job) or
        report.get('epoch')!=original['epoch'] or report.get('checkpoint')!=manifest['checkpoint']or
        report.get('source_files')!=job.get('source_files')or not isinstance(job.get('source_files'),dict)or not job['source_files']or
        type(report.get('completed_at'))not in(int,float)or not math.isfinite(report['completed_at'])or
        not job['created_at']<=report['completed_at']<=job['expires_at']):
        raise ValueError('original signed scientific verifier report binding')
    definitions=[row for row in original.get('environments',[])if row.get('env_id')=='affine_math']
    if (len(definitions)!=1 or batch.get('env_id')!='affine_math'or
        definitions[0].get('evaluation_only',False)or batch.get('index')not in definitions[0].get('indices',[])or
        definitions[0].get('spec',{}).get('max_turns')!=1):
        raise ValueError('original mining-only one-turn native MATH context required')
    batch_sha=sha(batch)
    objects=[obj for obj in job.get('submissions',[])if obj.get('commitment_ref',{}).get('batch_sha256')==batch_sha]
    if len(objects)!=1:raise ValueError('original signed submission exact batch binding')
    audits=[a for a in report.get('audits',[])if a.get('submission_sha256')==objects[0]['sha256']]
    if len(audits)!=1:raise ValueError('original exact submission audit required')
    audit=audits[0]
    accepted=[b for b in audit.get('accepted',[])if sha(b)==batch_sha]
    outcomes=[o for o in audit.get('outcomes',[])if o.get('index')==batch.get('index') and o.get('env_id')==batch.get('env_id')]
    if (len(accepted)!=1 or len(outcomes)!=1 or outcomes[0].get('valid')is not True or
        outcomes[0].get('fully_audited')is not True):
        raise ValueError('fully proof-audited accepted original batch required')
    positive_sha=sha(positive)
    if (len([v for v in batch.get('rollouts',[])if sha(v)==positive_sha])!=1 or
        positive.get('classification')!='positive' or type(positive.get('reward'))not in(int,float)or positive.get('reward')!=1. or
        positive.get('env_id')!=batch.get('env_id') or positive.get('index')!=batch.get('index') or
        positive.get('task_hash')is None):
        raise ValueError('original correct-positive membership required')
    turns=positive.get('turns')
    if not isinstance(turns,list)or len(turns)!=1:raise ValueError('one-turn native MATH positive required')
    turn=turns[0];output=turn.get('output');eos=p['eos_token_id']
    if (not isinstance(output,list)or not 2<=len(output)<p['max_output_tokens'] or
        any(type(t)is not int or t<0 for t in output)or output[-1]!=eos or output.count(eos)!=1 or
        turn.get('done')is not True or turn.get('classification')!='positive'or
        type(turn.get('reward'))not in(int,float)or turn.get('reward')!=1.):
        raise ValueError('already-present final EOS without early EOS or capped output required')
    native=authenticate(native_receipt,authority)
    fields={'version','original_epoch','checkpoint','batch_sha256','positive_sha256','native_correct','grader_source_sha256','original_report_sha256'}
    if (set(native)!=fields or native['version']!=NATIVE_VERSION or native['native_correct']is not True or
        native['original_epoch']!=original['epoch'] or native['checkpoint']!=manifest['checkpoint'] or
        native['batch_sha256']!=batch_sha or native['positive_sha256']!=positive_sha or
        native['original_report_sha256']!=sha(report)or
        native['grader_source_sha256']!=manifest.get('source_files',{}).get(GRADER)):
        raise ValueError('separate exact authenticated native correctness receipt required')
    return dict(policy=p,experiment_sha256=sha(experiment),positive_sha256=positive_sha,
        output_sha256=sha(output),output_tokens=len(output),original_report_sha256=sha(report),
        native_receipt_sha256=sha(native_receipt))


def loss(torch,margin,reference,*,experiment=None,authority=None,original_job=None,
         worker_request=None,native_receipt=None,batch=None,positive=None,positive_logits=None):
    """Historical loss unchanged by default; opt-in adds only original EOS CE.

    ``positive_logits`` are the same teacher-forced logits for the complete
    positive output; the caller must retain its graph and apply the existing
    task/pair weight to this total loss. No new optimizer or update occurs here.
    """
    original=preference_loss(torch,margin,reference,.1)
    if experiment is None:
        if any(v is not None for v in (authority,original_job,worker_request,native_receipt,batch,positive,positive_logits)):
            raise ValueError("experimental inputs require explicit authenticated policy")
        return original
    admission=admit(experiment,authority,original_job,worker_request,native_receipt,batch,positive)
    if positive is None or sha(positive)!=admission['positive_sha256']:
        raise ValueError('original admitted positive unchanged')
    output=positive['turns'][0]['output'];p=admission['policy']
    if (sha(output)!=admission['output_sha256']or len(output)!=admission['output_tokens'] or
        positive_logits is None or positive_logits.ndim!=2 or positive_logits.shape[0]!=len(output) or
        p['eos_token_id']>=positive_logits.shape[1]):raise ValueError('same complete positive teacher-forced logits required')
    if not torch.isfinite(positive_logits[-1]).all():raise ValueError('finite EOS logits required')
    if p['lambda']==0:return original
    eos_logprob=torch.log_softmax(positive_logits[-1].float(),-1)[p['eos_token_id']]
    return original-p['lambda']*eos_logprob
