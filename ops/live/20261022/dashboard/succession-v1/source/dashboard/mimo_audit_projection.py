"""Read-only projection of authenticated closed-epoch audit observations.

Display counts only: no grading, reward estimation, penalties or public identities.
"""
import copy,json,sqlite3
from contextlib import closing
from nacl.exceptions import BadSignatureError
from pathlib import Path
from subnet.mimo_audit_bridge import authenticate_population,admit_queue_reports
from subnet.mimo_release_policy import digest
from dashboard.mimo_live_projection import read


def summarize(snapshot, admissions, available, refused=0):
    result=copy.deepcopy(snapshot)
    closed={e['id']:e for e in result['epochs'] if e.get('batches_available')}
    verdicts={eid:{} for eid in closed};evidence={eid:set() for eid in closed}
    for admission in admissions.values():
        for o in admission['observations']:
            eid=o['epoch']
            if eid not in closed or eid not in available:continue
            if o['checkpoint']!=closed[eid]['checkpoint']:raise ValueError('audit checkpoint differs')
            if o['outcome']not in ('verified_valid','confirmed_invalid'):raise ValueError('unknown audit outcome')
            key=(o['miner'],o['batch_sha256'])
            verdicts[eid].setdefault(key,set()).add(o['outcome'])
            evidence[eid].add((o['job_sha256'],admission['original_report_request_sha256']))
    for row in result['epochs']:
        eid=row['id'];known=eid in closed and eid in available
        row['audit_breakdown_available']=known
        row['audit_count_source']='authenticated-original-mimo-audits' if known else None
        row['audit_evidence_sha256']=digest(sorted(evidence[eid])) if known else None
        if not known:
            row.update(accepted=None,rejected=None,unchecked=None,audit_conflicting_batches=None)
            continue
        vv=list(verdicts[eid].values())
        accepted=sum(v=={'verified_valid'} for v in vv)
        rejected=sum(v=={'confirmed_invalid'} for v in vv)
        conflicts=sum(len(v)>1 for v in vv)
        if len(vv)>row['batches']:raise ValueError('audit count exceeds frozen population')
        row.update(accepted=accepted,rejected=rejected,unchecked=row['batches']-accepted-rejected,
            audit_conflicting_batches=conflicts)
    known=[e for e in result['epochs']if e['audit_breakdown_available']]
    required=[e for e in result['epochs']if e.get('batches_available')]
    summary=result['summary'];complete=len(known)==len(required)
    summary.update(audit_breakdown_available=complete,audit_evidence_status='available'if complete else'unavailable-or-partial',
        audit_evidence_refused=refused,audit_epochs_available=len(known),audit_epochs_expected=len(required),
        audit_conflicting_batches=sum(e['audit_conflicting_batches'] for e in known))
    for k in ('accepted','rejected','unchecked'):
        summary[k]=sum(e[k]for e in known)if complete else None
        summary['known_'+k]=sum(e[k]for e in known)
    return result


class AuditProjection:
    def __init__(self,root,queue_path,authority,workers,manifest_guard,approved_execution):
        self.root=Path(root);self.queue=Path(queue_path);self.authority=authority
        self.workers={w for w,roles in workers.items()if'verify'in roles}
        self.guard=manifest_guard;self.execution=approved_execution
        self._admitted={};self._populations={}

    def __call__(self,snapshot,openings,now):
        wanted={e['id']for e in snapshot['epochs']if e.get('batches_available')and now>=e['deadline']}
        populations=[];available=set();population_cache={}
        admission_scope=digest([self.authority,sorted(self.workers),self.execution])
        # No capture files, raw trajectories or open-epoch audit populations are read.
        for d in (self.root/'epochs').iterdir():
            if not d.is_dir():continue
            path=d/'AUDIT-POPULATION.json'
            opening_path=d/'coordinator/OPENING.json'
            if not path.exists()or not opening_path.exists():continue
            opening=read(opening_path);eid=opening.get('payload',{}).get('epoch')
            if eid not in wanted:continue
            try:
                document=read(path);dispatch=read(d/'coordinator/DISPATCH.json')
                self.guard(openings[eid]['payload'])
                cache_key=digest([admission_scope,document,openings[eid],dispatch])
                if cache_key in self._populations:p=self._populations[cache_key]
                else:p=authenticate_population(document,self.authority,manifest_guard=self.guard)
                if (p['manifest_document']!=openings[eid]or p['committed_at']>now
                        or p['dispatch_document']!=dispatch):
                    raise ValueError('original closed opening/dispatch')
                populations.append(document);available.add(eid);population_cache[cache_key]=p
            except (OSError,ValueError,KeyError,TypeError,BadSignatureError):continue
        self._populations=population_cache
        if not available:return summarize(snapshot,{},set())
        try:
            if self.queue.is_symlink()or not self.queue.is_file()or self.queue.resolve()!=self.queue.absolute():raise ValueError('ordinary existing audit queue')
            with closing(sqlite3.connect(self.queue.as_uri()+'?mode=ro',uri=True,timeout=2))as db:
                db.row_factory=sqlite3.Row
                db.execute('PRAGMA query_only=ON')
                placeholders=','.join('?'for _ in available)
                rows=[dict(r)for r in db.execute("SELECT * FROM jobs WHERE status='complete' AND role='verify' AND json_extract(envelope,'$.payload.manifest.payload.epoch') IN ("+placeholders+")",sorted(available))]
        except (OSError,ValueError,sqlite3.Error):return summarize(snapshot,{},set())
        # Successful immutable evidence only. Current rows/populations/roster/source
        # remain part of each cache key; removals and changed errors never retain credit.
        population_pins={p['payload']['manifest_document']['payload']['epoch']:digest(p)for p in populations}
        scope=digest([self.authority,sorted(self.workers),self.execution])
        admissions={};pending=[];keys={};next_cache={}
        for row in rows:
            try:
                eid=json.loads(row['envelope'])['payload']['manifest']['payload']['epoch']
                key=digest([scope,population_pins[eid],row])
            except (ValueError,TypeError,KeyError):pending.append(row);continue
            keys[row['digest']]=key
            if key in self._admitted and all(o['completed_at']<=now for o in self._admitted[key]['observations']):
                admission=self._admitted[key];admissions[row['digest']]=admission;next_cache[key]=admission
            else:pending.append(row)
        refused=[]
        if pending:
            fresh,refused=admit_queue_reports(pending,populations,authority=self.authority,verifiers=self.workers,
                approved_execution=self.execution,manifest_guard=self.guard,cutoff=now)
            admissions.update(fresh)
            for job,value in fresh.items():
                if job in keys:next_cache[keys[job]]=value
        self._admitted=next_cache
        return summarize(snapshot,admissions,available,len(refused))
