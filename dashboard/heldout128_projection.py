"""Allowlisted complete native heldout128 evidence; no dispatch or private URLs."""
import hashlib, io, json, math, tarfile
from pathlib import Path
from dashboard.learner_projection import AUTHORITY, authenticated, canonical
from dashboard.cached_evaluator_projection import POLICY

def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def read(path, expected=None):
    p=Path(path)
    if not p.is_absolute() or p.resolve()!=p or p.is_symlink():raise ValueError('canonical scoped evidence path')
    raw=p.read_bytes()
    if expected is not None and hashlib.sha256(raw).hexdigest()!=expected:raise ValueError('scoped evidence digest')
    return raw

def original(ack, source, files_sha):
    a=authenticated(ack,AUTHORITY);j=authenticated(a['original_job'],AUTHORITY);m=authenticated(j['manifest'],AUTHORITY);r=a['original_report'];t=a['original_terminal']
    if(a.get('version')!='owned-cached-evaluation-durable-ack-v1' or a.get('durable_report_full_readback')is not True or
       j.get('role')!='evaluate' or j.get('owned_evaluation_policy')!=POLICY or m['source_bundle']['sha256']!=source or
       digest(j['source_files'])!=files_sha or a['job_sha256']!=digest(j) or a['report_sha256']!=digest(r) or
       a['original_terminal_sha256']!=digest(t) or a['checkpoint']!=m['checkpoint']):raise ValueError('original signed ACK provenance')
    for name,obj in [('original-job',a['original_job']),('original-report',r),('original-terminal',t)]:
        binding=a['full_readback_objects'][name];raw=canonical(obj)
        if binding['sha256']!=hashlib.sha256(raw).hexdigest() or type(binding['bytes'])is not int or binding['bytes']!=len(raw):raise ValueError('full original readback receipt')
    if(t.get('phase')!='complete' or type(t.get('exit_code'))is not int or t['exit_code']!=0 or t['job_id']!=j['job_id'] or
       not j['created_at']<=t['started_at']<=t['finished_at']<j['expires_at']):raise ValueError('actual original terminal')
    if(t.get('job_sha256',digest(j))!=digest(j) or r.get('success')is not True or r.get('heldout_failures')!=[] or
       r.get('job_id')!=j['job_id'] or r.get('job_sha256')!=digest(j) or r.get('checkpoint')!=m['checkpoint']['id'] or
       r.get('epoch')!=m['epoch'] or r.get('source_files')!=j['source_files'] or r.get('runtime_versions')!=j['runtime_versions'] or
       r.get('role')!='evaluate' or r.get('chain_transactions')is not False or not j['created_at']<=r['completed_at']<j['expires_at']):raise ValueError('complete original report')
    if len(j['heldout'])!=1:raise ValueError('one original32 suite')
    plan=j['heldout'][0];values=r['heldout'];indices=plan['indices'];seeds=plan['seeds']
    if(len(indices)!=32 or len(set(indices))!=32 or any(type(i)is not int for i in indices) or
       seeds!=[20261002+i*1000 for i in indices] or plan['harness']!={'version':'text-tools-long-kv-v3','policy':'autoregressive','max_output_tokens':1024,'temperature':.7,'top_p':1.} or
       len(values)!=32 or [(v['index'],v['seed'])for v in values]!=list(zip(indices,seeds))):raise ValueError('exact original32 task seeds and cap')
    train=next(e for e in m['environments']if e['env_id']=='affine_math')['indices']
    if len(train)!=6746 or len(set(train))!=6746 or set(train)&set(indices):raise ValueError('heldout excludes mining split')
    for v in values:
        if(v.get('native_graded')is not True or v.get('verified')is not False or v.get('proof_verification_performed')is not False or
           v.get('trust_scope')!='operator-owned-process-native-grader' or v.get('checkpoint')!=m['checkpoint']['id'] or
           type(v.get('reward'))not in(int,float) or v['reward']not in(0,1) or
           v.get('classification')!=('positive' if v['reward']==1 else 'negative')):raise ValueError('native outcomes are not verified proofs')
    return a,j,m,plan,values

def comparison_binding(original):
    _,job,manifest,plan,values=original
    report=original[0]['original_report'];revision=manifest['model_runtime_revision'];profile=manifest['backend_profile'];numerical=manifest['numerical_policy']
    versions=job['runtime_versions']
    if not isinstance(versions,dict)or not versions or any(type(k)is not str or type(v)is not str for k,v in versions.items()):raise ValueError('complete runtime versions')
    if type(revision)is not str or not revision or not isinstance(profile,dict)or not profile or not isinstance(numerical,dict)or not numerical:raise ValueError('explicit execution profile')
    if (report.get('execution_runtime_revision')!=revision or report.get('generation_runtime_revision')!=revision or
        digest(report.get('backend_profile'))!=digest(profile)or digest(report.get('numerical_policy'))!=digest(numerical)):raise ValueError('report execution matches signed manifest')
    envs=[e for e in manifest['environments']if e['env_id']==plan['env_id']]
    if len(envs)!=1:raise ValueError('single exact native definition')
    env=envs[0];spec=env['spec']
    if any(k not in spec for k in ('version','id','adapter','source_hash','config','num_samples','max_turns','max_output_tokens','success_reward')):raise ValueError('complete native taskset and grading settings')
    # Ordering of the mining set is irrelevant; every other task/grader setting
    # remains exact. Checkpoint, epoch, and request clocks are intentionally absent.
    normalized=dict(env_id=env['env_id'],spec=spec,harness=env['harness'],indices=sorted(env['indices']))
    binding=dict(runtime_versions=versions,model_runtime_revision=revision,backend_profile=profile,numerical_policy=numerical,
                 environment_revision=manifest['environment_revision'],harness_source_hash=manifest['harness_source_hash'],
                 native_environment=normalized,heldout_harness=plan['harness'])
    return digest(binding)

def rows(pointer, production):
    c=authenticated(pointer,AUTHORITY)
    if c.get('version')!='heldout128-dashboard-sources-v1':raise ValueError('explicit heldout128 source scope')
    epochs={}
    for p in Path(production).glob('*-first-signed-manifest.json'):
        m=authenticated(json.loads(p.read_bytes()),AUTHORITY);epochs.setdefault(m['checkpoint']['id'],[]).append((m['start'],m['epoch']))
    result=[];seen=set();common_binding=None;task_hashes={}
    for entry in c['evaluations']:
        p=Path(entry['summary_path'])
        if not p.exists():continue # A running/partial group has no public row.
        raw=read(p,entry['summary_sha256']);s=authenticated(json.loads(raw),AUTHORITY)
        if(s['cohort_sha256']!=c['cohort_sha256'] or s['source_sha256']!=c['source_sha256'] or
           s.get('production_checkpoints_changed')is not False or s.get('normal_evaluator_B_paused')is not False):raise ValueError('signed summary cohort source scope')
        if type(s.get('completed_at'))not in(int,float)or not math.isfinite(s['completed_at']):raise ValueError('actual finite completion time')
        if s.get('version')=='owned-cached-heldout128-checkpoint-actual-v1':
            if s.get('all_four_genuine_full_R2_ACKs')is not True or s.get('group_owned_model_retired')is not True or s['original_jobs']!=4 or s['task_count']!=128:raise ValueError('complete checkpoint summary')
            scoped_groups=entry.get('groups')
            if not isinstance(scoped_groups,dict) or len(scoped_groups)!=1:raise ValueError('exactly one scoped checkpoint summary group')
            groups={next(iter(scoped_groups)):s['group']}
        elif s.get('version')=='owned-cached-heldout128-paired-actual-v1':
            if s.get('all_eight_genuine_full_R2_ACKs')is not True or s.get('all_group_owned_models_retired')is not True or s['original_jobs']!=8 or s['task_count_per_checkpoint']!=128:raise ValueError('complete paired summary')
            groups=s['groups']
        else:raise ValueError('known complete summary version')
        if set(groups)!=set(entry['groups']):raise ValueError('all scoped summary groups')
        checkpoint_ids=set();job_ids=set()
        for label,g in groups.items():
            paths=entry['groups'][label];ack=json.loads(read(paths['archive_ack_path'],paths['archive_ack_sha256']));archive=read(paths['archive_path'],g['archive_sha256'])
            if(ack.get('R2_full_GET_verified')is not True or ack['archive_sha256']!=g['archive_sha256'] or ack['archive_bytes']!=len(archive)):raise ValueError('full archive readback receipt')
            if len(archive)>512*1024**2:raise ValueError('bounded archive')
            with tarfile.open(fileobj=io.BytesIO(archive))as t:
                members=t.getmembers();names=[x.name for x in members]
                if len(set(names))!=len(names) or any(x.issym()or x.islnk()or x.name.startswith('/')or '..'in Path(x.name).parts for x in members):raise ValueError('safe exact archive members')
                acks=[json.loads(t.extractfile(x).read())for x in members if x.isfile()and x.name.startswith('durable-evaluation-acks/')and x.name.endswith('.json')]
            if len(acks)!=4:raise ValueError('four original full ACKs, never partial')
            originals=[original(a,c['source_sha256'],c['source_files_sha256'])for a in acks];originals.sort(key=lambda v:v[0]['group'])
            for o in originals:
                binding=comparison_binding(o)
                if common_binding is None:common_binding=binding
                elif binding!=common_binding:raise ValueError('comparable runtime profile native taskset across every chunk and checkpoint')
                for value in o[4]:
                    key=(value['index'],value['seed'])
                    if key in task_hashes and task_hashes[key]!=value['task_hash']:raise ValueError('same native task identity across checkpoints')
                    task_hashes[key]=value['task_hash']
            if [v[0]['group']for v in originals]!=list(range(4)) or len({v[1]['job_id']for v in originals})!=4:raise ValueError('distinct four originals')
            cp=originals[0][2]['checkpoint']['id']
            ids={v[1]['job_id']for v in originals}
            if cp in checkpoint_ids or ids&job_ids:raise ValueError('paired checkpoints and originals are distinct')
            checkpoint_ids.add(cp);job_ids.update(ids)
            plans=[dict(group=i,**v[3])for i,v in enumerate(originals)]
            if digest(plans)!=c['cohort_sha256'] or any(v[2]['checkpoint']['id']!=cp for v in originals):raise ValueError('one exact ordered128 cohort checkpoint')
            values=[r for v in originals for r in v[4]];keys=[(v['index'],v['seed'])for v in values]
            if len(set(keys))!=128 or set(v['index']for v in values)&set(c['excluded_indices']):raise ValueError('no overlap or old32 reuse')
            successes=sum(int(v['reward'])for v in values);score=g['result']['result'];retirement=score['retirement'];score=score['score']
            if(score['count']!=128 or score['cohort_sha256']!=c['cohort_sha256'] or score['successes']!=successes or score['mean_reward']!=successes/128 or retirement['status']!='complete'):raise ValueError('actual full128 score and retirement')
            if s.get('version')=='owned-cached-heldout128-checkpoint-actual-v1' and((s['checkpoint']['id']if isinstance(s['checkpoint'],dict)else s['checkpoint'])!=cp or s['successes']!=successes):raise ValueError('single summary score')
            if s.get('version')=='owned-cached-heldout128-paired-actual-v1' and s[label+'_successes']!=successes:raise ValueError('paired summary score')
            run='native128-'+cp+'-'+c['cohort_sha256'];matches=epochs.get(cp,[])
            if run in seen or not matches:continue
            seen.add(run);result.append(dict(run_id=run,env_id='affine_math',dataset_id='native-math-heldout128',status='complete',timestamp=s['completed_at'],epoch_id=min(matches)[1],original_epoch_id=originals[0][2]['epoch'],checkpoint=cp,count=128,successes=successes,mean_reward=successes/128,requested_count=128,completed_count=128,taskset_hash=c['cohort_sha256'],fixed_task_ids=[v['task_hash']for v in values],harness_config=plans[0]['harness'],seed=20261002,experiment_id='owned-cached-native-heldout128-cap1024-v1',sampling_policy=POLICY['version'],native_graded=True,proof_verification_performed=False,original_report_sha256=digest([v[0]['report_sha256']for v in originals])))
    return result
