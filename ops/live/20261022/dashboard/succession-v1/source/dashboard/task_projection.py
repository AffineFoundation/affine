"""Allowlisted task counts: signed declarations and separately audited successes."""
import hashlib,json

def project(manifest, learner, verified):
    corpora=[]
    for definition in manifest.get('environments',[]):
        spec=definition.get('spec',{});count=spec.get('num_samples');env=definition.get('env_id');source=spec.get('source_hash')
        if not isinstance(env,str) or type(count)is not int or not 0<count<=100000 or not isinstance(source,str):continue
        corpora.append(dict(environment=env,size=count,key=hashlib.sha256(json.dumps([env,count,source]).encode()).hexdigest()))
    if not corpora:return None
    allowed={c['environment']:c['size'] for c in corpora}
    counts={}
    submitted=learner.get('task_counts') if learner else None
    for row in submitted or []:
        if row['environment'] not in allowed or not 0<=row['index']<allowed[row['environment']]:return None
        counts[(row['environment'],row['index'])]=dict(row,solves=0)
    observed=False
    for report in verified.values():
        if not isinstance(report,dict) or report.get('epoch')!=manifest['epoch']:continue
        audited={(o.get('env_id'),o.get('index')) for o in report.get('outcomes',[]) if isinstance(o,dict) and o.get('fully_audited') is True and o.get('valid') is True}
        for batch in report.get('accepted',[]):
            if not isinstance(batch,dict):continue
            key=(batch.get('env_id'),batch.get('index'));env,index=key
            if env not in allowed or type(index)is not int or not 0<=index<allowed[env] or key not in audited or batch.get('epoch')!=manifest['epoch']:continue
            observed=True
            task=counts.setdefault(key,dict(environment=env,index=index,submitted=0,included=0,solves=0))
            task['solves']+=sum(isinstance(r,dict) and r.get('classification')=='positive' and type(r.get('reward')) in (int,float) and r['reward']==1 for r in batch.get('rollouts',[]))
    return dict(corpora=corpora,tasks=sorted(counts.values(),key=lambda t:(t['environment'],t['index'])),submitted_available=submitted is not None,solves_available=observed)
