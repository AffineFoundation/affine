"""Read-only MiMo evidence extension for the existing hourly weight writer.

The deployed writer's calculate/fallback/recipients/transaction code is reused
unchanged. Installation requires a new signed operator composition; importing
this module neither installs a hook nor mutates a live registry or cursor.
"""
from copy import deepcopy

from .backend_jobs import signed
from .continuous_audit_policy import digest
from .mimo_audit_bridge import authenticate_population, admit_queue_reports, need

VERSION = 'mimo-current-assessment-evidence-bridge-v1'
TRANSITION = 'conservative-overlap-minimum-factors-v1'


def _admissions_for_population(admissions, records):
    """Exact original membership, indexed once instead of rescanning every row."""
    keys={(r['epoch'],r['miner'],r['batch_sha256']) for r in records}
    return {key:value for key,value in admissions.items() if all(
        (o['epoch'],o['miner'],o['batch_sha256']) in keys for o in value['observations'])}


def evidence(populations, queue_rows, *, authority, cutoff, verifiers, approved_execution,
             manifest_guard, snapshot, audit_policy, release_policy_sha256,
             current_estimates=None, observation_reader=None):
    """Use the operator's pinned existing Bayesian/penalty reducer directly."""
    need(callable(snapshot) and callable(manifest_guard), 'pinned reducer and qualified manifest gate required')
    need(type(release_policy_sha256) is str and len(release_policy_sha256)==64, 'signed release policy digest')
    need(type(cutoff) is int and cutoff%3600==0, 'original hourly cutoff')
    admitted=[]; parsed=[]; epochs=set()
    for document in populations:
        p=authenticate_population(document,authority,manifest_guard=manifest_guard)
        m=signed(p['manifest_document'],authority)
        need(m['epoch']not in epochs,'duplicate epoch population'); epochs.add(m['epoch'])
        if p['committed_at']<=cutoff:
            admitted.append(document);parsed.append((p,m))
    parsed.sort(key=lambda row:(row[0]['round'],row[1]['epoch']))
    records=[row for p,_ in parsed for row in p['records']]
    admissions,refused=admit_queue_reports(queue_rows,admitted,authority=authority,verifiers=verifiers,
        approved_execution=approved_execution,manifest_guard=manifest_guard,cutoff=cutoff)
    pointers=[dict(admitted_queue_job_sha256=k) for k in sorted(admissions)]
    result=[];timings=[]
    for p,m in parsed:
        eligible=p['eligible_evidence_ids']
        # Snapshot sees current and preceding populations, with their actual
        # immutable identities. The reducer deduplicates repeated job reports.
        preceding=[row for row in records if row['round']<=p['round']]
        allowed_ids={digest(row) for row in preceding}
        subset=_admissions_for_population(admissions,preceding)
        snap=snapshot(preceding,[r for r in pointers if r['admitted_queue_job_sha256'] in subset],verifiers,
            epoch=m['epoch'],round=p['round'],checkpoint=m['checkpoint']['id'],cutoff=cutoff,
            audit_policy=audit_policy,admitted_jobs=subset,eligible_evidence_ids=eligible,authority=authority,
            **({'claim_order':{k:p[k] for k in ('claim_order_rule','contested_claims')}} if 'claim_order_rule' in p else {}))
        need(set(eligible)<=allowed_ids and snap['training_waits_for_audits'] is False,
             'actual native eligible groups and independent training')
        result.append(snap);timings.append(dict(epoch=m['epoch'],round=p['round'],checkpoint=m['checkpoint']['id'],
            committed_at=p['committed_at'],cutoff=cutoff))
    if current_estimates is not None or observation_reader is not None:
        need(callable(current_estimates) and callable(observation_reader), 'both pinned global evidence reducers')
        audits=observation_reader(pointers,records,verifiers,cutoff,admitted_jobs=admissions,authority=authority)
        global_round=max((p['round'] for p,_ in parsed),default=0)
        details=current_estimates(audits,{r['miner'] for r in records},global_round,audit_policy)
        for snap in result:
            snap['cohort_miner_details']=deepcopy(snap['miners']);snap['current_global_round']=global_round
            for miner,row in snap['miners'].items():row.update(details[miner])
            snap['points']={m:d['unique_eligible_batches']*d['validity_probability']*d['reward_multiplier']*
                            d.get('resolution_coverage_factor',1.) for m,d in snap['miners'].items()}
            total=sum(snap['points'].values());snap['weights']={m:v/total if total else 0. for m,v in snap['points'].items()}
    return dict(snapshots=result,committed_at_by_epoch={m['epoch']:p['committed_at'] for p,m in parsed},
        population_timings=timings,evidence_hashes=dict(mimo_release_policy_sha256=release_policy_sha256,
            mimo_population_documents=[digest(d) for d in admitted],mimo_admitted_jobs=sorted(admissions),
            mimo_execution_registry_sha256=digest(approved_execution)),refused=refused,excluded=[],deferred=[])


def merge(legacy, current, *, transition_policy=TRANSITION):
    """Preserve historical credit and stricter evidence during the overlap.

    This explicitly conservative transition does not reset miner reputation at a
    model/run change. For overlapping identities it applies the lower validity,
    resolution and penalty factors, plus either blacklist, to both histories.
    The existing six-hour EMA and current-UID mapping remain the sole writer's.
    """
    need(transition_policy==TRANSITION,'explicit conservative cohort transition')
    result=deepcopy(legacy)
    prior={s['epoch']:s for s in result['snapshots']}
    newer=deepcopy(current['snapshots'])
    need(not set(prior)&{s['epoch'] for s in newer}, 'new MiMo evidence cannot rewrite a historical epoch')
    if prior and newer:
        need(min(s['round'] for s in newer)>max(s['round'] for s in prior.values()),
             'global epoch order continues across model restart')
    all_snapshots=list(prior.values())+newer
    latest=[]
    for group in (list(prior.values()),newer):
        details={}
        for snap in sorted(group,key=lambda s:(s['round'],s['epoch'])):
            details.update(snap['miners'])
        latest.append(details)
    for snap in all_snapshots:
        for miner,row in snap['miners'].items():
            applicable=[v[miner] for v in latest if miner in v]
            for name in ('validity_probability','reward_multiplier','resolution_coverage_factor'):
                vals=[v.get(name,1.) for v in applicable]
                need(all(type(v)in(int,float) and 0<=v<=1 for v in vals),'bounded evidence factor')
                row[name]=min(vals)
            row['blacklisted']=any(v.get('blacklisted',False) for v in applicable)
            if row['blacklisted']:row['reward_multiplier']=0.
        snap['points']={m:d['unique_eligible_batches']*d['validity_probability']*d['reward_multiplier']*
                        d['resolution_coverage_factor'] for m,d in snap['miners'].items()}
        total=sum(snap['points'].values());snap['weights']={m:v/total if total else 0. for m,v in snap['points'].items()}
    result['snapshots']=all_snapshots
    result['committed_at_by_epoch'].update(current['committed_at_by_epoch'])
    result.setdefault('population_timings',[]).extend(current.get('population_timings',[]))
    # Retain source_admission/numerical policy hashes expected by the unchanged
    # writer, while adding exact new evidence pins under independent names.
    overlap=set(result['evidence_hashes'])&set(current['evidence_hashes'])
    need(not overlap,'new evidence cannot overwrite legacy admission pins')
    result['evidence_hashes'].update(current['evidence_hashes'])
    result['evidence_hashes']['mimo_transition_policy']=transition_policy
    for field in ('refused','excluded','deferred'):
        result.setdefault(field,[]).extend(current.get(field,[]))
    return result


def compose_loader(legacy_loader, mimo_loader, *, transition_policy=TRANSITION):
    """Factory for run_once(evidence_loader=...), never a second weight writer.

    Both readers must succeed completely. An unavailable new registry propagates
    to the writer's existing authenticated last-best fallback; it cannot produce
    a fabricated empty population or clear old penalties.
    """
    need(callable(legacy_loader) and callable(mimo_loader),'two authenticated read-only evidence readers')
    def load(*args,**kwargs):
        old=legacy_loader(*args,**kwargs)
        new=mimo_loader(authority=kwargs['authority'],cutoff=kwargs['cutoff'],verifiers=kwargs['verifiers'])
        return merge(old,new,transition_policy=transition_policy)
    return load


class EvidenceSource:
    """Signed, bounded, read-only durable source for compose_loader.

    The owner explicitly pins this adapter/config in its new writer wrapper.
    The source cannot sign, submit, create a queue or mutate historical files.
    ``snapshot`` is the exact reducer selected by that same authenticated host.
    """
    def __init__(self, document, authority, *, snapshot, current_estimates, observation_reader, allow_public=False):
        from pathlib import Path
        from .mimo_release_succession import load_release
        p=signed(document,authority)
        need(type(p)is dict and set(p)=={'version','directory','queue_path','release_policy',
            'approved_execution','audit_policy','verifiers','transition_policy'} and
            p['version']==VERSION and p['transition_policy']==TRANSITION,'explicit evidence reader policy')
        need(type(p['verifiers'])is list and p['verifiers'] and len(set(p['verifiers']))==len(p['verifiers']),
             'explicit admitted verifier roster')
        for field in ('directory','queue_path'):
            path=Path(p[field]);need(path.is_absolute() and path.resolve()==path,'ordinary exact evidence source path')
        self.release=load_release(p['release_policy'],authority,allow_public=allow_public)
        need(callable(current_estimates) and callable(observation_reader), 'pinned deployed global assessment reducers')
        self.current_estimates,self.observation_reader=current_estimates,observation_reader
        self.policy,self.authority,self.snapshot=p,authority,snapshot
        self.config_sha256=digest(document)

    @staticmethod
    def _read(path,limit):
        import json,os,stat
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
        try:
            before=os.fstat(fd)
            need(stat.S_ISREG(before.st_mode) and before.st_uid==os.getuid() and not before.st_mode&0o077
                 and 0<before.st_size<=limit,'bounded owner-private evidence file')
            with os.fdopen(fd,'rb',closefd=False) as stream:data=stream.read(before.st_size+1)
            after=os.fstat(fd)
            need(len(data)==before.st_size and (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
                 (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),'evidence changed during read')
            return json.loads(data)
        finally:os.close(fd)

    def __call__(self, *, authority,cutoff,verifiers):
        from pathlib import Path
        import sqlite3
        from urllib.parse import quote
        need(authority==self.authority and set(self.policy['verifiers'])<=set(verifiers),
             'writer admits the configured MiMo verifiers')
        root=Path(self.policy['directory'])
        need(root.resolve()==root and root.is_dir(),'original audit directory')
        state=self._read(root/'state.json',256*1024**2)
        need(state.get('version')=='mimo-audit-scheduler-v1' and type(state.get('populations'))is dict and
             type(state.get('draws'))is dict,'original audit scheduler state')
        populations=[]
        from .complete_group_admission import is_digest
        for epoch,checksum in sorted(state['populations'].items()):
            need(is_digest(checksum),'population digest path')
            doc=self._read(root/(checksum+'.population.json'),256*1024**2)
            need(digest(doc)==checksum and signed(doc,authority)['manifest_document']['payload']['epoch']==epoch,
                 'original population file digest and epoch')
            populations.append(doc)
        queue=Path(self.policy['queue_path'])
        need(queue.resolve()==queue and queue.is_file(),'original queue path')
        ids=set()
        for draw in state['draws'].values():
            ids.update(draw.get('prior_jobs',[]))
            if draw.get('job_id'):ids.add(draw['job_id'])
        db=sqlite3.connect('file:'+quote(str(queue))+'?mode=ro',uri=True);db.row_factory=sqlite3.Row
        rows=[]
        try:
            db.execute('BEGIN')
            for jobid in sorted(ids):
                row=db.execute('SELECT * FROM jobs WHERE id=? AND status=?',(jobid,'complete')).fetchone()
                if row is not None:rows.append(dict(row))
        finally:db.close()
        value=evidence(populations,rows,authority=authority,cutoff=cutoff,
            verifiers={key:['verify'] for key in self.policy['verifiers']},
            approved_execution=self.policy['approved_execution'],manifest_guard=self.release.validate_manifest,
            snapshot=self.snapshot,audit_policy=self.policy['audit_policy'],release_policy_sha256=self.release.policy_sha256,
            current_estimates=self.current_estimates,observation_reader=self.observation_reader)
        value['evidence_hashes']['mimo_reader_policy_sha256']=self.config_sha256
        return value
