"""Default-off verifier tightening. Original miner/sampler/native source is unchanged.

The new signed opening opts in; old openings execute their exact old verifier.
This only establishes sampler compatibility under empirical numeric assumptions.
"""
import hashlib, math, time
from pathlib import Path
VERSION='symmetric-prefill-source-preserving-v1'
FIELD='sampling_verification_policy'
FILES=('subnet/symmetric_verifier_policy.py','subnet/fast_prefill_audit.py',
       'subnet/backend_jobs.py','subnet/source_sampling_admission.py',
       'subnet/forced_sampling.py','subnet/environments.py')

def digest(x):
    from .forced_sampling import canonical
    return hashlib.sha256(canonical(x)).hexdigest()

def current_policy():
    from .forced_sampling import MINER_VERSION
    root=Path(__file__).resolve().parent.parent
    return dict(version=VERSION,sampling_contract=MINER_VERSION,
                uncertainty_adjudication='exact-cached-replay-v1',
                source_files={n:hashlib.sha256((root/n).read_bytes()).hexdigest()for n in FILES})

def for_manifest(manifest):
    if FIELD not in manifest:return None
    p=manifest[FIELD]
    if type(p)is not dict or p!=current_policy():raise ValueError('exact prospective verifier source policy')
    from .forced_sampling import validate,MINER_VERSION,source_hash
    c=validate(manifest.get('sampling_contract'))
    if c['version']!=MINER_VERSION or manifest.get('sampling_source_hash')!=source_hash():raise ValueError('unchanged miner-bound v5 sampler required')
    source=manifest.get('source_bundle',{}).get('sha256')
    if type(source)is not str or len(source)!=64 or any(x not in '0123456789abcdef'for x in source):raise ValueError('signed source bundle required')
    return p

def intervals(logprobs,tokens,draws,temperature,top_p,error):
    """Same distribution and exterior bound; both interior boundary bands escalate."""
    import torch
    from . import fast_prefill_audit as fast
    from .audit_policy import InvalidSample
    if type(error)not in(int,float)or not math.isfinite(error)or not 0<=error<=1e-3:raise ValueError('bounded admitted CDF error')
    p=fast.distribution(logprobs,temperature,top_p);n,v=p.shape
    if len(tokens)!=n or len(draws)!=n or any(type(t)is not int or not 0<=t<v for t in tokens):raise InvalidSample('token framing')
    if any(type(u)not in(int,float)or not math.isfinite(u)or not 0<=u<1 for u in draws):raise ValueError('trusted draw framing')
    idx=torch.tensor(tokens,dtype=torch.int64,device=p.device);pos=torch.arange(n,device=p.device);cdf=p.cumsum(-1);cdf[:,-1]=1.
    high=cdf[pos,idx];mass=p[pos,idx];low=high-mass;u=torch.tensor(draws,dtype=torch.float64,device=p.device)
    if bool((mass<=0).any()):raise fast.NumericalAmbiguity('support requires exact cached reference')
    if bool(((u<low-error)|(u>=high+error)).any()):raise InvalidSample('forced CDF interval outside calibrated region')
    if bool(((u<low+error)|(u>=high-error)).any()):raise fast.NumericalAmbiguity('symmetric CDF uncertainty')
    return dict(positions=n,all_intervals_verified=True,cdf_abs_error=error)

def verify_sampling(runtime,rollout,turn_index,prompt,output,logprobs):
    from . import fast_prefill_audit as fast,forced_sampling as sampler
    from .audit_policy import InvalidSample
    manifest=runtime.fast_sampling_manifest;policy=for_manifest(manifest)
    if policy is None:raise ValueError('explicit prospective verifier policy required')
    context=runtime.sampling_context
    if sampler.binding(manifest,context.get('miner'))!=context:raise ValueError('current miner/checkpoint/draw context')
    if fast.bind(manifest,runtime.harness)!=runtime.fast_sampling_calibration:raise ValueError('current checkpoint calibration')
    config=sampler.validate_harness(runtime.harness,context['contract']);sampler.validate_attempt(context,rollout['seed'])
    events=getattr(runtime,'sampling_verification_events',None)
    if events is not None and type(events)is not list:raise ValueError('operator telemetry list')
    start=time.monotonic();event=dict(rollout_sha256=digest(rollout),turn=turn_index,positions=len(output),adjudication='execution-error')
    try:
        eos=runtime.tokenizer.eos_token_id
        if type(output)is not list or not output or len(output)>config['max_output_tokens'] or any(type(t)is not int or t<0 for t in output)or eos in output[:-1]or(len(output)<config['max_output_tokens']and output[-1]!=eos):
            event['adjudication']='invalid-stop-framing';raise InvalidSample('forced generation stop or token framing')
        draws=[sampler.uniform(context,runtime.spec.id,rollout['task_hash'],rollout['index'],rollout['seed'],turn_index,k)for k in range(len(output))]
        try:
            result=intervals(logprobs,output,draws,config['temperature'],config['top_p'],runtime.fast_sampling_calibration['cdf_abs_error']);event['adjudication']='calibrated-interior'
        except fast.NumericalAmbiguity:
            try:result=fast.verify_cached_reference(runtime,prompt,output,rollout['seed'],turn_index,rollout['index'],rollout['task_hash'])
            except InvalidSample:
                event['adjudication']='exact-cached-mismatch';raise
            except Exception as exc:
                event['adjudication']='exact-cached-unavailable';raise fast.NumericalAmbiguity('cached reference unavailable; neutral audit outcome')from exc
            event['adjudication']='exact-cached-match'
        except InvalidSample:
            event['adjudication']='outside-calibrated-region';raise
        return dict(result,sampling_adjudication=event['adjudication'],historical_execution_proven=False,numerical_envelope_is_empirical=True)
    finally:
        event['seconds']=time.monotonic()-start
        if events is not None:events.append(event)

def require_report(manifest,report,committed_child=None):
    """Extra worker-assertion admission, not a new cryptographic inference proof."""
    policy=for_manifest(manifest)
    if policy is None:
        if FIELD in report:raise ValueError('historical audit must not assert prospective policy')
        return
    outcomes=report.get('outcomes',[]);accepted=report.get('accepted',[])
    fully_valid=any(o.get('valid')is True and o.get('fully_audited')is True for o in outcomes)
    if report.get(FIELD)!=policy:
        # An early transport/structural rejection has no model-level verdict.
        if not accepted and not fully_valid and all(o.get('valid')is False and o.get('fully_audited')is not True and (o.get('failure_kind')=='structural_invalid' or o.get('rejection_stage')=='transport') for o in outcomes):return
        raise ValueError('required verifier policy missing from report')
    if type(outcomes)is not list or type(accepted)is not list:raise ValueError('bounded report lists')
    batches={};positions=set();valid_outcomes={}
    for b in accepted:
        if type(b)is not dict or type(b.get('env_id'))is not str or type(b.get('index'))is not int:raise ValueError('accepted task identity')
        key=(b['env_id'],b['index'])
        if key in batches:raise ValueError('duplicate accepted child')
        batches[key]=b
    for o in outcomes:
        if o.get('failure_kind')=='numerical_ambiguous' and (o.get('valid')is not None or o.get('fully_audited')is not False):raise ValueError('numerical ambiguity must remain neutral')
        if o.get('valid')is True and (o.get('failure_kind')is not None or o.get('sampling_verification_complete')is False or o.get('environment_verification_complete')is False):raise ValueError('valid outcome cannot carry failed or incomplete verification')
        number=o.get('batch')
        if number is not None:
            if type(number)is not int or number<0 or number in positions:raise ValueError('unique outcome position')
            positions.add(number)
        if o.get('valid')is True:
            if o.get('fully_audited')is not True or type(o.get('env_id'))is not str or type(o.get('index'))is not int:raise ValueError('fully valid outcome identity')
            key=(o['env_id'],o['index'])
            if key in valid_outcomes:raise ValueError('duplicate fully valid child')
            valid_outcomes[key]=o
    if set(valid_outcomes)!=set(batches):raise ValueError('fully valid outcome requires exact accepted child')
    if manifest.get('submission_transport_policy'):
        if type(committed_child)is not dict:raise ValueError('authenticated committed child required')
        ref=committed_child.get('commitment_ref',{})
        if report.get('submission_sha256')!=committed_child.get('sha256'):raise ValueError('committed submission digest')
        if len(accepted)>1 or any(o.get('batch') not in (None,0) for o in outcomes):raise ValueError('single committed child outcome')
        for b in accepted:
            if digest(b)!=ref.get('batch_sha256') or b['env_id']!=ref.get('env_id') or b['index']!=ref.get('index'):raise ValueError('accepted exact committed batch digest')
    passed={}
    for o in outcomes:
        events=o.get('sampling_verification_events',[])
        if type(events)is not list:raise ValueError('bounded verification telemetry')
        own_passed={}
        for e in events:
            if type(e)is not dict or set(e)!={'rollout_sha256','turn','positions','adjudication','seconds'}:raise ValueError('verification event fields')
            if type(e['turn'])is not int or e['turn']<0 or type(e['positions'])is not int or not 0<e['positions']<=65536 or type(e['seconds'])not in(int,float)or not math.isfinite(e['seconds'])or e['seconds']<0:raise ValueError('verification event bounds')
            kind=e['adjudication']
            if kind not in {'calibrated-interior','exact-cached-match','exact-cached-mismatch','exact-cached-unavailable','outside-calibrated-region','execution-error','invalid-stop-framing'}:raise ValueError('verification adjudication')
            if kind=='exact-cached-unavailable' and(o.get('valid')is not None or o.get('fully_audited')is not False or o.get('failure_kind')!='numerical_ambiguous'):raise ValueError('unavailable reference cannot be valid or fraud')
            if o.get('valid')is True and o.get('fully_audited')is True:
                if kind not in {'calibrated-interior','exact-cached-match'}:raise ValueError('nonpass in accepted audit')
                key=(e['rollout_sha256'],e['turn'])
                if key in passed:raise ValueError('duplicate sampling credit')
                passed[key]=e['positions'];own_passed[key]=e['positions']
        if o.get('valid')is True and o.get('fully_audited')is True:
            b=batches[(o['env_id'],o['index'])]
            own_expected={(digest(r),i):len(t['output'])for r in b['rollouts']for i,t in enumerate(r['turns'])}
            if not own_expected or own_passed!=own_expected:raise ValueError('each valid child requires its own complete events')
    expected={(digest(r),i):len(t['output'])for b in accepted for r in b['rollouts']for i,t in enumerate(r['turns'])}
    if passed!=expected:raise ValueError('every accepted rollout turn requires exactly one passing verification event')
