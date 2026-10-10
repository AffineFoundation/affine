"""Default-off operator proposal: native label eligibility, never proof validity.

Call only on previously authenticated committed pairs. No mining code is run;
answers are decoded from original tokens and graded by the approved MATH script.
There is intentionally no controller hook or production enablement in this file.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

VERSION = 'bounded-native-training-label-filter-v1'
K2L2_VERSION = 'bounded-native-training-label-filter-k2l2-v2'
MULTI_VERSION = 'bounded-native-training-label-filter-multi-v3'

def document_pair_quota(manifest):
    from subnet.batch_quotas import configured_quotas
    configured_quotas(manifest)
    K,L=manifest.get('K'),manifest.get('L')
    if type(K)is not int or type(L)is not int or K!=L or not 2<=K<=64:
        raise ValueError('balanced signed multi-rollout quotas required')
    return K



def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _stamp(value):
    return tuple(getattr(value, name) for name in ('st_dev','st_ino','st_size','st_mode','st_uid','st_nlink','st_mtime_ns','st_ctime_ns'))


def _read(path, expected, maximum):
    path = Path(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or not 0 < before.st_size <= maximum:
        raise ValueError('bounded regular single-link trusted input')
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, 'rb') as stream:
        if _stamp(os.fstat(stream.fileno())) != _stamp(before):
            raise ValueError('trusted input identity changed')
        data = stream.read(maximum + 1)
        if _stamp(os.fstat(stream.fileno())) != _stamp(before):
            raise ValueError('trusted input changed while read')
    if _stamp(path.lstat()) != _stamp(before) or hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('trusted input SHA/identity')
    return data, _stamp(before)


def _model_token_bounds(policy, manifest, tokenizer_size):
    """Read authenticated small model metadata; tokenizer length is not logits width."""
    binding = policy.get('model_config_binding')
    if (type(binding) is not dict or set(binding) != {'path','sha256','vocab_size'} or
        type(binding['path']) is not str or not Path(binding['path']).is_absolute() or
        type(binding['vocab_size']) is not int or not 1 <= binding['vocab_size'] <= 1_000_000 or
        manifest.get('checkpoint',{}).get('files',{}).get('config.json') != binding['sha256']):
        raise ValueError('original model config binding')
    data, _ = _read(binding['path'], binding['sha256'], 1_000_000)
    config = json.loads(data)
    vocab = config.get('vocab_size')
    if (type(vocab) is not int or vocab != binding['vocab_size'] or
        type(tokenizer_size) is not int or not 1 <= tokenizer_size <= vocab):
        raise ValueError('authenticated model/tokenizer vocabulary')
    return vocab


class PinnedGrader:
    """Original isolated subprocess. Runtime mismatches remain indeterminate.

    Every invocation runs the original verify.py runtime-lock check. No float
    coercion or alternative parser can turn an unavailable grader into a zero.
    """
    def __init__(self, interpreter, script, script_sha256):
        from subnet.native_math_grader import isolated_argv, runtime_lock
        self.executable = Path(interpreter).resolve(strict=True)
        executable_sha = hashlib.sha256(self.executable.read_bytes()).hexdigest()
        if executable_sha not in {p['python_executable_sha256'] for p in runtime_lock()['profiles']}:
            raise ValueError('approved native interpreter executable SHA')
        self.executable_stamp = _read(self.executable, executable_sha, 64*1024**2)[1]
        self.argv = isolated_argv(interpreter, script, [])
        self.script = Path(script)
        self.sha = script_sha256
        self.stamp = _read(self.script, self.sha, 4*1024**2)[1]

    def __call__(self, gold, reply, timeout):
        if (Path(self.argv[0]).resolve(strict=True)!=self.executable or
            _stamp(self.executable.lstat())!=self.executable_stamp or
            _stamp(self.script.lstat()) != self.stamp):
            raise ValueError('grader source drift or interpreter identity drift')
        args = ['--json-arguments', json.dumps([gold, reply], ensure_ascii=True)]
        # Linux MAX_ARG_STRLEN is usually 128 KiB. JSON escaping can inflate
        # non-ASCII text sixfold; bound the actual serialized argument, not
        # only reply UTF-8 size. Never reinterpret oversize as a negative.
        if len(args[1].encode('utf-8')) > 96*1024:
            return None, 'native_argument_limit'
        try:
            argv = list(self.argv)
            argv[4] = ('import resource;resource.setrlimit(resource.RLIMIT_AS,(1073741824,1073741824));'
                       'resource.setrlimit(resource.RLIMIT_CPU,(' + str(max(1, math.ceil(timeout))) + ','
                       + str(max(1, math.ceil(timeout)) + 1) + '));' + argv[4])
            result = subprocess.run(argv + args, stdin=subprocess.DEVNULL,
                                    capture_output=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None, 'native_timeout'
        except OSError:
            return None, 'native_spawn_unavailable'
        if (Path(self.argv[0]).resolve(strict=True)!=self.executable or
            _stamp(self.executable.lstat())!=self.executable_stamp or
            _stamp(self.script.lstat()) != self.stamp):
            raise ValueError('grader source drift or interpreter identity drift')
        if result.returncode:
            return None, 'native_exit_' + str(result.returncode)
        if result.stdout.strip() == b'1.0':
            return 1, None
        if result.stdout.strip() == b'0.0':
            return 0, None
        return None, 'native_invalid_output'


def validate_limits(policy, *, manifest=None):
    fields = {'version','workers','max_pairs','per_grade_seconds','wall_seconds','max_reply_bytes'}
    if isinstance(policy,dict) and 'terminal_rule' in policy:
        fields.add('terminal_rule')
        if policy['terminal_rule'] != 'max-or-eos-v1':
            raise ValueError('explicit native terminal framing policy')
    if not isinstance(policy, dict) or set(policy) != fields or policy['version'] not in (VERSION,K2L2_VERSION,MULTI_VERSION):
        raise ValueError('exact opt-in native filter policy')
    policy=dict(policy)
    if policy['max_pairs']=='manifest':
        if policy['version']!=MULTI_VERSION or manifest is None:
            raise ValueError('manifest pair budget requires signed multi-rollout context')
        from subnet.committed_training_inputs import training_document_cap
        policy['max_pairs']=training_document_cap(manifest)*document_pair_quota(manifest)
    for key, low, high in (('workers',1,16),('max_pairs',1,16384 if policy['version']==MULTI_VERSION else 512 if policy['version']==K2L2_VERSION else 256),('per_grade_seconds',1,60),
                           ('wall_seconds',1,600),('max_reply_bytes',1,262144)):
        if type(policy[key]) is not int or not low <= policy[key] <= high:
            raise ValueError('bounded native filter ' + key)
    return dict(policy)


def _filter_admitted_pairs(pairs, policy, resolve, decode, grader, *, clock=time.monotonic):
    """Private CPU core; resolve/decode/grader are prepared by trusted operator.

    Dependency injection is for CPU tests, not a signed policy bypass. Public
    production integration must use the authenticated context builder below.
    """
    policy = validate_limits(policy)
    if len(pairs) > policy['max_pairs']:
        raise ValueError('native filter pair count')
    started = clock(); deadline = started + policy['wall_seconds']
    # Prevalidate every original before any grader side effect. Never accept a
    # miner-provided answer, grader path, task resolver, or rendered reply.
    prepared = []
    seen = set()
    for definition, positive, negative in pairs:
        identity = digest([definition, positive, negative])
        if identity in seen:
            raise ValueError('duplicate original pair')
        seen.add(identity)
        gold, task_hash, cap, eos, vocab = resolve(definition, positive, negative)
        if not isinstance(gold,str) or len(gold.encode()) > policy['max_reply_bytes']:
            raise ValueError('trusted task answer bounds')
        if type(cap) is not int or not 1 <= cap <= 2048 or type(vocab) is not int or vocab < 1:
            raise ValueError('trusted token profile')
        terminal_required=policy.get('terminal_rule')=='max-or-eos-v1'
        if terminal_required and (not isinstance(eos,set) or len(eos)!=1 or
                any(type(t)is not int or not 0<=t<vocab for t in eos)):
            raise ValueError('authenticated single tokenizer EOS')
        rollouts = []
        for rollout, claim in ((positive,'positive'), (negative,'negative')):
            turns = rollout.get('turns')
            if rollout.get('classification') != claim or rollout.get('task_hash') != task_hash or not isinstance(turns,list) or len(turns) != 1:
                raise ValueError('authenticated native task/class/turn binding')
            output = turns[0].get('output')
            if not isinstance(output,list) or not 1 <= len(output) <= cap or any(type(t) is not int or not 0 <= t < vocab for t in output):
                raise ValueError('native output token bounds')
            reply = decode(output)
            if not isinstance(reply,str) or len(reply.encode()) > policy['max_reply_bytes']:
                raise ValueError('decoded native reply bound')
            from subnet.math_completion import enabled as completed_math, final_box
            complete_required = completed_math(definition.get('spec', {}))
            rollouts.append((gold, reply, {'claim':claim, 'output_sha256':digest(output),
                             'decoded_reply_sha256':hashlib.sha256(reply.encode()).hexdigest(),
                             'output_tokens':len(output), 'non_eos_cap':len(output)==cap and output[-1] not in eos,
                             'submitted_text_matches_decoded':turns[0].get('text')==reply}))
            if complete_required:
                rollouts[-1][2]['complete_answer'] = final_box(reply) is not None
                rollouts[-1][2]['outcome_policy'] = definition['spec']['config']['math_outcome_policy']
            if terminal_required:
                rollouts[-1][2]['terminal_framing_valid']=(not any(t in eos for t in output[:-1]) and
                    (len(output)==cap or output[-1] in eos))
                rollouts[-1][2]['approved_output_cap']=cap
        if terminal_required:
            pair_valid=all(r[2]['terminal_framing_valid'] for r in rollouts)
            for r in rollouts:r[2]['pair_terminal_framing_valid']=pair_valid
        prepared.append((identity, rollouts))

    prepared_at = clock()
    def grade(item):
        gold, reply, receipt = item
        if receipt.get('pair_terminal_framing_valid') is False:
            return dict(receipt,native_score=None,reason='terminal_framing_exclusion',label_matches=None)
        if receipt.get('complete_answer') is False:
            return dict(receipt, native_score=None, reason='unresolved_math_answer', label_matches=None)
        remaining = deadline - clock()
        if remaining <= 0:
            return dict(receipt, native_score=None, reason='filter_deadline', label_matches=None)
        grade_start = clock()
        score, reason = grader(gold, reply, min(policy['per_grade_seconds'],remaining))
        receipt = dict(receipt, native_elapsed_seconds=clock()-grade_start)
        if type(score) is not int or score not in (0,1):
            if score is not None:
                raise ValueError('exact native binary outcome')
            return dict(receipt,native_score=None,reason=reason or 'native_indeterminate',label_matches=None)
        return dict(receipt,native_score=score,reason=None,label_matches=(score==1)==(receipt['claim']=='positive'))

    # map processes a bounded, finite population. The original per-process
    # timeout limits each running child; total deadline suppresses pending work.
    with ThreadPoolExecutor(max_workers=policy['workers']) as pool:
        grades = list(pool.map(grade, (r for _,rollouts in prepared for r in rollouts)))
    rows=[];accepted=[]
    for position, ((identity,_), pair) in enumerate(zip(prepared,pairs)):
        result=grades[2*position:2*position+2]
        status=('excluded_terminal_rule' if any(r.get('terminal_framing_valid') is False for r in result)
                else 'excluded_indeterminate' if any(r['label_matches'] is None for r in result)
                else 'accepted_native_labels' if all(r['label_matches'] for r in result)
                else 'excluded_label_mismatch')
        rows.append({'pair_sha256':identity,'status':status,'grades':result})
        if status=='accepted_native_labels':accepted.append(pair)
    return accepted, {'version':policy['version'],'policy_sha256':digest(policy),'elapsed_seconds':clock()-started,'prepare_seconds':prepared_at-started,
                      'grader_child_memory_limit_bytes':1073741824,
                      'grading_parallel_wall_seconds':clock()-prepared_at,
                      'sampling_assurance':'unaudited','proof_verification_performed':False,
                      'cheating_penalties':False,'claims_rewritten':False,'rows':rows,
                      **({'terminal_rule':policy['terminal_rule']} if 'terminal_rule' in policy else {})}


def _complete_document_pairs(pairs, decisions, required_pairs=2):
    """A quota-bound document is atomic: never train on a surviving partial batch."""
    if type(required_pairs)is not int or not 2<=required_pairs<=64:raise ValueError('bounded explicit document pair quota')
    identities={digest(list(pair)):pair for pair in pairs}
    if len(identities)!=len(pairs):raise ValueError('K2L2 duplicate original pair')
    used=set();accepted_ids=set()
    for decision in decisions:
        rows=decision['pair_sha256']
        if len(rows)!=required_pairs or len(set(rows))!=required_pairs or any(row not in identities or row in used for row in rows):
            raise ValueError('K2L2 complete disjoint document pair inventory')
        if type(decision['accepted'])is not bool:raise ValueError('K2L2 binary document admission')
        used.update(rows)
        if decision['accepted']:accepted_ids.update(rows)
    if used!=set(identities):raise ValueError('K2L2 full document pair coverage')
    return [pair for pair in pairs if digest(list(pair)) in accepted_ids]


def filter_authenticated_documents(document_paths, policy_envelope, job_envelope, authority, source_root,
                               tokenizer_root, interpreter):
    """Prospective entry: strict ROOT context and local approved source only.

    Authenticate original committed documents inside this boundary. Callers
    cannot inject prevalidated pairs. Prompt eligibility remains mandatory
    in the unchanged trainer; native label eligibility adds no proof claim.
    """
    from subnet.distributed_roles import authenticate
    policy=authenticate(policy_envelope,authority); job=authenticate(job_envelope,authority)
    manifest=authenticate(job['manifest'],authority)
    required={'version','limits','job_sha256','epoch','checkpoint','source_sha256','source_root',
              'source_files','snapshot_sha256','tokenizer_binding','grader_sha256','model_config_binding'}
    if set(policy)!=required or policy['version']!=VERSION or job['role']!='train':
        raise ValueError('exact ROOT native filter context')
    source_root=Path(source_root)
    if (policy['job_sha256']!=digest(job_envelope) or policy['epoch']!=manifest['epoch'] or
        policy['checkpoint']!=manifest['checkpoint']['id'] or policy['source_sha256']!=manifest['source_bundle']['sha256'] or
        policy['source_root']!=str(source_root) or policy['source_files']!=job['source_files'] or
        policy['tokenizer_binding']!=manifest['tokenizer_binding']):
        raise ValueError('native filter original job/source/checkpoint/tokenizer binding')
    return _filter_bound_documents(document_paths,policy,job,manifest,authority,source_root,tokenizer_root,interpreter)


def _filter_bound_documents(document_paths,policy,job,manifest,authority,source_root,tokenizer_root,interpreter):
    from subnet.native_math_prompt import NativeMathPromptSession
    from transformers import AutoTokenizer
    source_root=Path(source_root)
    # No source-root imports from arbitrary supplied paths. Operator startup
    # must already be running the approved source loader, checked here.
    from subnet.committed_training_inputs import admitted_submission, training_document_cap
    if len(document_paths)!=len(job['submissions']) or len(document_paths)>training_document_cap(manifest):
        raise ValueError('exact original committed document paths')
    limits=validate_limits(policy['limits'],manifest=manifest)
    if limits['version']==K2L2_VERSION and (type(manifest.get('K'))is not int or type(manifest.get('L'))is not int or manifest['K']!=2 or manifest['L']!=2):
        raise ValueError('K2L2 native filter explicit signed quotas')
    if limits['version']==MULTI_VERSION:
        document_pair_quota(manifest)
    if limits['version']==VERSION and (manifest.get('K',1)!=1 or manifest.get('L',1)!=1):
        raise ValueError('K2L2 requires new native filter version')
    pairs=[];documents=[]
    for path,obj in zip(document_paths,job['submissions']):
        summary, admitted_pairs=admitted_submission(path,obj,manifest,authority,retire=False)
        documents.append((summary,[digest(list(pair)) for pair in admitted_pairs]))
        pairs.extend(admitted_pairs)
    import subnet.native_math_prompt as native_prompt
    execution_root=Path(policy.get('execution_root',source_root))
    if Path(native_prompt.__file__).resolve()!= (execution_root/'subnet/native_math_prompt.py').resolve():
        raise ValueError('native filter approved source loader')
    for name, expected in job['source_files'].items():
        relative=Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('native source member path')
        _read(source_root/relative,expected,16*1024**2)
        # The CPU import tree is distinct; scientific bytes remain the original source.
        if relative.name == 'native_math_prompt.py':
            _read(execution_root/relative,expected,16*1024**2)
    tokenizer_root=Path(tokenizer_root)
    if set(policy['tokenizer_binding'])!={'tokenizer.json','tokenizer_config.json','chat_template.jinja'}:
        raise ValueError('native tokenizer asset closure')
    if {p.name for p in tokenizer_root.iterdir()}!=set(policy['tokenizer_binding']):
        raise ValueError('no extra tokenizer executable/config inputs')
    for name, expected in policy['tokenizer_binding'].items():
        _read(tokenizer_root/name,expected,32*1024**2)
    tokenizer=AutoTokenizer.from_pretrained(tokenizer_root,local_files_only=True,trust_remote_code=False)
    model_vocab=_model_token_bounds(policy,manifest,len(tokenizer))
    sessions={}
    try:
        for definition,_,_ in pairs:
            if definition['env_id'] not in sessions:
                spec=definition['spec']
                if definition not in manifest['environments']:
                    raise ValueError('original manifest environment')
                session=NativeMathPromptSession(spec)
                _read(session.path,policy['snapshot_sha256'],128*1024**2)
                sessions[definition['env_id']]=session
        def resolve(definition,p,n):
            session=sessions[definition['env_id']]
            if (p.get('index')!=n.get('index') or p.get('env_seed')!=n.get('env_seed') or
                p.get('env_seed')!=int(definition['spec'].get('config',{}).get('seed',0))):
                raise ValueError('original paired task context')
            task=session.reset(p['index'],p['env_seed'])
            return (session.rows[p['index']]['data']['answer'],task['task_hash'],
                    min(definition['spec']['max_output_tokens'],definition['harness']['max_output_tokens']),
                    {tokenizer.eos_token_id},model_vocab)
        script=source_root/'subnet/vendor/legacy/rollouts/envs/affine_math_v1/affine_math_v1/verify.py'
        grader=PinnedGrader(interpreter,script,policy['grader_sha256'])
        accepted,receipt=_filter_admitted_pairs(pairs,limits,resolve,
                           lambda tokens:tokenizer.decode(tokens,skip_special_tokens=True),grader)
        decisions={r['pair_sha256']:r['status'] for r in receipt['rows']}
        receipt['document_decisions']=[dict(document_sha256=summary['document_sha256'],
            learner_admission_sha256=summary['learner_admission_sha256'],
            batch_sha256=summary['batch_sha256'],pair_sha256=identities,
            accepted=all(decisions[i]=='accepted_native_labels' for i in identities))
            for summary,identities in documents]
        if limits['version']in(K2L2_VERSION,MULTI_VERSION):
            accepted=_complete_document_pairs(pairs,receipt['document_decisions'],document_pair_quota(manifest))
        receipt.update(original_job_sha256=policy['job_sha256'],source_sha256=policy['source_sha256'],
                       checkpoint=policy['checkpoint'],snapshot_sha256=policy['snapshot_sha256'],
                       tokenizer_binding=policy['tokenizer_binding'],grader_sha256=policy['grader_sha256'])
        return accepted,receipt
    finally:
        for session in sessions.values():session.close()


CONTEXT_VERSION='native-outcome-eligibility-context-v1'
AUTHORIZATION_VERSION='native-outcome-preselection-authorization-v1'

def filter_eligibility_context(document_paths,context_envelope,authorization_envelope,authority,
                               source_root,tokenizer_root,interpreter):
    """Non-dispatchable, signed preselection context; no train job is created."""
    from subnet.distributed_roles import authenticate
    context=authenticate(context_envelope,authority)
    authorization=authenticate(authorization_envelope,authority)
    fields={'version','original_signed_manifest','submissions','source_files','authorization_sha256',
            'original_population_file_sha256','original_selection_file_sha256','parent_binding_sha256'}
    authfields={'version','limits','source_sha256','source_root','source_files','snapshot_sha256',
                'tokenizer_binding','grader_sha256','model_config_binding','sampling_assurance','no_credit','no_relabel'}
    if 'execution_root' in authorization:authfields.add('execution_root')
    if 'benchmark_scope' in authorization:
        authfields.add('benchmark_scope')
        scope=authorization['benchmark_scope']
        keys={'version','created_at','expires_at','helper_sha256','plan_sha256','filter_sha256','benchmark_root','original_job_file_sha256','max_documents','input_bytes','dispatchable','model_operations','optimizer_operations'}
        if (type(scope)is not dict or set(scope)!=keys or scope['version']!='bounded-native-outcome-readonly-benchmark-v1' or
            type(scope['created_at'])is not int or type(scope['expires_at'])is not int or not 0<scope['expires_at']-scope['created_at']<=600 or
            not scope['created_at']<=time.time()<scope['expires_at'] or scope['max_documents']!=256 or
            type(scope['input_bytes'])is not int or not 0<scope['input_bytes']<=512_000_000 or
            any(scope[k]is not False for k in ('dispatchable','model_operations','optimizer_operations'))):
            raise ValueError('bounded non-dispatchable native benchmark profile')
    if set(context)!=fields or context['version']!=CONTEXT_VERSION:
        raise ValueError('non-dispatchable eligibility context schema')
    if (set(authorization)!=authfields or authorization['version']!=AUTHORIZATION_VERSION or
        authorization['sampling_assurance']!='unaudited' or authorization['no_credit'] is not True or
        authorization['no_relabel'] is not True or context['authorization_sha256']!=digest(authorization_envelope)):
        raise ValueError('explicit ROOT native preselection authorization')
    manifest=authenticate(context['original_signed_manifest'],authority)
    if (context['source_files']!=authorization['source_files'] or
        manifest['source_bundle']['sha256']!=authorization['source_sha256'] or
        str(source_root)!=authorization['source_root'] or
        manifest['tokenizer_binding']!=authorization['tokenizer_binding'] or
        digest(manifest['trainer_state_binding'])!=context['parent_binding_sha256']):
        raise ValueError('native eligibility source/tokenizer/parent binding')
    for key in ('original_population_file_sha256','original_selection_file_sha256','parent_binding_sha256'):
        if type(context[key]) is not str or len(context[key])!=64 or any(c not in '0123456789abcdef' for c in context[key]):
            raise ValueError('native eligibility original digest')
    # This internal object is never signed, saved or dispatchable. It only
    # supplies original document/source data to the private admitted builder.
    inputs={'submissions':context['submissions'],'source_files':context['source_files']}
    policy=dict(authorization,job_sha256=digest(context_envelope),epoch=manifest['epoch'],
                checkpoint=manifest['checkpoint']['id'])
    accepted,receipt=_filter_bound_documents(document_paths,policy,inputs,manifest,authority,
                                             source_root,tokenizer_root,interpreter)
    receipt.pop('original_job_sha256')
    receipt.update(context_sha256=digest(context_envelope),authorization_sha256=digest(authorization_envelope),
                   original_population_file_sha256=context['original_population_file_sha256'],
                   original_selection_file_sha256=context['original_selection_file_sha256'],
                   parent_binding_sha256=context['parent_binding_sha256'])
    return accepted,receipt
