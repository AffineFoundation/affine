"""Default-off sole-current FP32 cache. R2 and ROOT remain authoritative."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time
from .cache_lifecycle import snapshot,identifier
from .storage import canonical
from .distributed_roles import authenticate
VERSION='sole-current-fp32-state-cache-v1'

def policy(manifest):
    value=manifest.get('optimizer_state_local_cache')
    if value is None:return None
    if (type(value)is not dict or set(value)!={'version','max_checkpoint_bytes'} or value['version']!=VERSION or
        type(value['max_checkpoint_bytes'])is not int or not 1<=value['max_checkpoint_bytes']<=128*1024**3):
        raise ValueError('explicit bounded optimizer cache policy')
    publication=manifest.get('persistent_publication_policy')
    if not isinstance(publication,dict)or publication.get('state_readback')!='qualified-remote-full':
        raise ValueError('optimizer cache requires full independent durable state readback')
    return dict(value)

def sha(value):return hashlib.sha256(canonical(value)).hexdigest()
def hash_file(path):
    h=hashlib.sha256();size=0
    with path.open('rb')as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk);size+=len(chunk)
    return h.hexdigest(),size

def member(name):
    if not re.fullmatch(r'state-[0-9]{6}\.safetensors',name):raise ValueError('exact optimizer cache shard name')
    return name

class StateCache:
    def __init__(self,workspace,job,manifest,authority):
        self.workspace=Path(workspace).absolute()
        if self.workspace!=self.workspace.resolve()or not self.workspace.is_dir():raise ValueError('owned existing trainer workspace')
        self.job=job;self.manifest=manifest;self.authority=authority;self.policy=policy(manifest)
        if self.policy is None:raise ValueError('optimizer cache default is off')
        self.root=self.workspace/'.optimizer-state-cache';self.root.mkdir(mode=0o700,exist_ok=True)
        if self.root!=self.root.resolve():raise ValueError('optimizer cache root symlink')
        self.fd=None;self.lock=threading.Lock();self.current=None;self.rows={};self.cache_evidence=[];self.promotion_wait_seconds=1800
    def __enter__(self):
        self.fd=os.open(self.root/'lease',os.O_CREAT|os.O_RDWR|os.O_NOFOLLOW,0o600)
        try:
            snapshot(self.root/'lease')
            fcntl.flock(self.fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BaseException:os.close(self.fd);self.fd=None;raise
        return self
    def __exit__(self,*args):
        if self.fd is not None:os.close(self.fd);self.fd=None
    def save(self,path,value):
        tmp=path.with_suffix('.tmp');data=canonical(value)
        fd=os.open(tmp,os.O_WRONLY|os.O_CREAT|os.O_TRUNC|os.O_NOFOLLOW,0o600)
        with os.fdopen(fd,'wb')as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
        os.replace(tmp,path)
    def directory(self,job_id):
        directory=self.root/('candidate-'+identifier(job_id))
        if directory!=directory.resolve():raise ValueError('optimizer candidate path symlink')
        return directory
    def discard(self,value,reason):
        directory=self.directory(value['job_id']);removed=[]
        for name,row in value.get('files',{}).items():
            path=directory/member(name)
            if not path.exists():continue
            actual=snapshot(path);original=row['stat']
            # A corrupted byte cache can be discarded only if the exact owned
            # inode still exists. Replaced/external files are never adopted.
            if any(actual[k]!=original[k]for k in ('dev','ino','uid','mode')):raise ValueError('optimizer cache ownership changed')
            path.unlink();removed.append(name)
        self.save(self.root/('retired-'+value['job_id']+'.json'),dict(job_id=value['job_id'],descriptor_sha256=value.get('descriptor_sha256'),reason=reason,removed=removed))
        return removed
    def prepare_parent(self,descriptor,source):
        promotion=self.root/'promotion.json'
        if promotion.exists():
            snapshot(promotion)
            intent=json.loads(promotion.read_bytes());confirmed=authenticate(intent['ack'],self.authority)
            if intent.get('phase')=='failed' and intent.get('child_terminal_confirmed')is True:
                pid=intent.get('child_pid');ticks=intent.get('child_ticks')
                if type(pid)is not int or (not isinstance(ticks,str)and not(ticks is None and type(intent.get('child_exit_code'))is int)):raise ValueError('terminal promotion child identity')
                proc=Path('/proc')/str(pid)/'stat'
                try:fields=proc.read_text().rsplit(')',1)[1].split()
                except FileNotFoundError:fields=None
                if fields and (ticks is None or fields[0]!='Z'and fields[19]==ticks):raise ValueError('promotion child still live')
                original=authenticate(json.loads((self.workspace/(confirmed['job_id']+'.json')).read_bytes()),self.authority)
                report=json.loads((self.workspace/'jobs'/confirmed['job_id']/'report.json').read_bytes())
                if (confirmed.get('version')!='durable-original-trainer-cache-ACK-v1'or confirmed.get('authority_state_committed')is not True or
                    confirmed['job_sha256']!=sha(original)or confirmed['report_sha256']!=sha(report)or report.get('success')is not True or
                    confirmed['trainer_state']['descriptor_sha256']!=sha(descriptor)or
                    report['persistent_training_state']['descriptor_sha256']!=sha(descriptor)or
                    confirmed['new_checkpoint']!=report['new_checkpoint']or
                    confirmed['trainer_state']['optimizer_steps']!=descriptor['optimizer_steps']):
                    raise ValueError('failed promotion cold fallback exact durable original lineage')
                # Only the optional byte cache failed. Keep original signed job,
                # ACK, publication report and candidate inventory as evidence.
                for name in ('pending.json','current.json'):
                    marker=self.root/name
                    if not marker.exists():continue
                    snapshot(marker);candidate=json.loads(marker.read_bytes())
                    candidate_job=authenticate(json.loads((self.workspace/(candidate['job_id']+'.json')).read_bytes()),self.authority)
                    if sha(candidate_job)!=candidate['job_sha256']:raise ValueError('failed cache candidate original ownership')
                    self.discard(candidate,'terminal-promotion-failure-cold-fallback')
                    self.save(self.root/('failed-'+candidate['job_id']+'-'+name),candidate);marker.unlink()
                self.cache_evidence.append(dict(outcome='cold',reason='confirmed-terminal-promotion-failure',job_id=confirmed['job_id']))
                return 0
            if intent.get('phase')=='pending':
                # Cover a promotion launched between coordinator predispatch
                # admission and worker startup. Release the cache lease while
                # waiting, so the original ACK helper can actually promote.
                if self.fd is not None:os.close(self.fd);self.fd=None
                deadline=time.monotonic()+self.promotion_wait_seconds
                while time.monotonic()<deadline:
                    time.sleep(0.1)
                    snapshot(promotion);latest=json.loads(promotion.read_bytes());authenticate(latest['ack'],self.authority)
                    if latest.get('phase')!='pending':
                        self.__enter__()
                        return self.prepare_parent(descriptor,source)
                raise ValueError('original post-ACK promotion requires terminal observation before next training')
            if intent.get('phase')!='complete':raise ValueError('original post-ACK promotion requires terminal observation before next training')
        pending=self.root/'pending.json'
        marker=self.root/'current.json'
        if pending.exists():
            abandoned=json.loads(pending.read_bytes())
            current=json.loads(marker.read_bytes())if marker.exists()else None
            if current is not None and current['job_id']==abandoned['job_id']:
                # Recovery of a crash between atomic promotion and removal of
                # the pending marker; the promoted bytes remain current.
                confirmed=authenticate(current['ROOT_ack'],self.authority)
                if ({k:v for k,v in current.items()if k!='ROOT_ack'}!=abandoned or
                    confirmed['job_id']!=current['job_id']or confirmed['job_sha256']!=current['job_sha256']or
                    confirmed['trainer_state']['descriptor_sha256']!=current['descriptor_sha256']):
                    raise ValueError('recovered promotion exact current/pending/ACK lineage')
                pending.unlink()
            elif abandoned['job_id']==self.job['job_id']:
                raise ValueError('original job has staged cache candidate; recover completion instead of retraining')
            else:
                original=authenticate(json.loads((self.workspace/(abandoned['job_id']+'.json')).read_bytes()),self.authority)
                if sha(original)!=abandoned['job_sha256']:raise ValueError('abandoned candidate original ownership')
                self.discard(abandoned,'different-original-job-abandoned-unpromoted-cache');pending.unlink()
        if not marker.exists():self.cache_evidence.append(dict(outcome='cold',reason='no-promoted-cache'));return 0
        value=json.loads(marker.read_bytes());ack=authenticate(value['ROOT_ack'],self.authority)
        if ack.get('version')!='durable-original-trainer-cache-ACK-v1'or ack.get('authority_state_committed')is not True or ack['trainer_state']['descriptor_sha256']!=value['descriptor_sha256']or ack['job_id']!=value['job_id']:
            raise ValueError('promoted cache ROOT durability binding')
        original=authenticate(json.loads((self.workspace/(value['job_id']+'.json')).read_bytes()),self.authority)
        original_manifest=authenticate(original['manifest'],self.authority)
        if (sha(original)!=ack['job_sha256']or value['job_sha256']!=ack['job_sha256']or
                original_manifest['source_bundle']['sha256']!=value['source_sha256']):
            raise ValueError('promoted cache original signed source binding')
        incoming_step=descriptor['optimizer_steps']if descriptor is not None else self.manifest.get('trainer_state_binding',{}).get('global_step_before',0)
        if ack['trainer_state']['optimizer_steps']>incoming_step:
            raise ValueError('approved optimizer cache cannot roll lineage backward')
        directory=self.directory(value['job_id']);reason=None
        if descriptor is None or sha(descriptor)!=value['descriptor_sha256']or value['source_sha256']!=source:reason='wrong-approved-parent-or-source'
        else:
            shards={r['name']:r for r in descriptor['shards']}
            if directory.exists()and set(p.name for p in directory.iterdir())-set(value['files']):
                raise ValueError('unowned optimizer cache directory member')
            if set(shards)!=set(value['files']):reason='incomplete-cache'
            else:
                for name,row in value['files'].items():
                    path=directory/member(name)
                    try:
                        if snapshot(path)!=row['stat']or hash_file(path)!=(shards[name]['sha256'],shards[name]['size']):reason='missing-or-corrupt-cache';break
                    except FileNotFoundError:reason='missing-or-corrupt-cache';break
        if reason:
            self.discard(value,reason);marker.unlink();self.cache_evidence.append(dict(outcome='cold',reason=reason));return 0
        self.current=value;self.rows={r['name']:r for r in descriptor['shards']}
        size=sum(r['size']for r in self.rows.values())
        if size>self.policy['max_checkpoint_bytes']:raise ValueError('cached parent byte cap')
        self.cache_evidence.append(dict(outcome='full-approved-cache',descriptor_sha256=sha(descriptor),size=size,all_SHA_size_verified=True))
        return size
    def admit(self,plan,*,reclaimable_parent_bytes=0,disk_available=None):
        # Existing authenticated parent bytes are already occupied on disk and
        # are consumed before new export. Credit them only after full binding,
        # SHA and size checks; preserve every ordinary reservation as well.
        verified=sum(row['size']for row in self.rows.values())
        if type(reclaimable_parent_bytes)is not int or reclaimable_parent_bytes!=verified:
            raise ValueError('only fully verified existing parent bytes are reclaimable')
        desired=plan['cpu_state_bytes']+max(1,len(self.job['persistent_training']['output_shards']))*1024**2
        if desired>self.policy['max_checkpoint_bytes']:raise ValueError('prospective optimizer cache byte cap')
        free=shutil.disk_usage(self.workspace).free if disk_available is None else disk_available
        if free+reclaimable_parent_bytes<plan['additional_disk_required_bytes']+desired:
            raise ValueError('retained optimizer cache plus transfer/BF16/reserve disk budget')
        return dict(retained_state_required_bytes=desired,ordinary_disk_required_bytes=plan['additional_disk_required_bytes'],reclaimable_verified_parent_bytes=reclaimable_parent_bytes,observed_free_disk_bytes=free)
    def fetch(self,name,path,fallback):
        if self.current is None:return fallback(name,path)
        name=member(name);source=self.directory(self.current['job_id'])/name
        if snapshot(source)!=self.current['files'][name]['stat']:raise ValueError('approved cached parent changed')
        # Destination is the ordinary restore workspace; its normal full SHA,
        # tensor-schema and finite-value checks remain in force before unlink.
        destination=Path(path).absolute()
        if destination!=destination.resolve()or not destination.is_relative_to(self.workspace):raise ValueError('owned restore destination')
        os.rename(source,destination)
    def begin_candidate(self):
        registry=self.root/'pending.json'
        if registry.exists():
            previous=json.loads(registry.read_bytes());self.discard(previous,'abandoned-unpromoted-candidate');registry.unlink()
        directory=self.directory(self.job['job_id']);directory.mkdir(mode=0o700,exist_ok=False)
        self.candidate=dict(version=VERSION,job_id=self.job['job_id'],job_sha256=sha(self.job),source_sha256=self.manifest['source_bundle']['sha256'],files={},descriptor_sha256=None)
        self.save(registry,self.candidate)
    def retain(self,name,path,digest,size):
        name=member(name);path=Path(path).absolute();snapshot(path)
        if path!=path.resolve()or not path.is_relative_to(self.workspace):raise ValueError('owned uploaded shard required')
        with self.lock:
            if name in self.candidate['files']or sum(r['size']for r in self.candidate['files'].values())+size>self.policy['max_checkpoint_bytes']:
                raise ValueError('single bounded optimizer candidate')
            target=self.directory(self.job['job_id'])/name
            if target.exists():raise ValueError('optimizer cache cannot overwrite shard')
            os.rename(path,target)
            self.candidate['files'][name]=dict(sha256=digest,size=size,stat=snapshot(target))
            self.save(self.root/'pending.json',self.candidate)
        return True
    def finish(self,descriptor):
        shards={r['name']:r for r in descriptor['shards']}
        if set(shards)!=set(self.candidate['files'])or any((self.candidate['files'][n]['sha256'],self.candidate['files'][n]['size'])!=(r['sha256'],r['size'])for n,r in shards.items()):
            raise ValueError('complete candidate descriptor binding')
        self.candidate['descriptor_sha256']=sha(descriptor);self.save(self.root/'pending.json',self.candidate)
        return dict(version=VERSION,descriptor_sha256=sha(descriptor),bytes=sum(r['size']for r in shards.values()),promoted=False,ROOT_durability_ACK_required=True,parent_cache=self.cache_evidence)


def promote(ack,authority,workspace):
    value=authenticate(ack,authority);workspace=Path(workspace)
    job_envelope=json.loads((workspace/(value['job_id']+'.json')).read_bytes());job=authenticate(job_envelope,authority);manifest=authenticate(job['manifest'],authority)
    if policy(manifest)is None:return None
    report=json.loads((workspace/'jobs'/job['job_id']/'report.json').read_bytes());state=report['persistent_training_state']
    if (value.get('version')!='durable-original-trainer-cache-ACK-v1'or value.get('authority_state_committed')is not True or sha(job)!=value['job_sha256']or sha(report)!=value['report_sha256']or report.get('success')is not True or
        state['descriptor_sha256']!=value['trainer_state']['descriptor_sha256']or state['descriptor']['optimizer_steps']!=value['trainer_state']['optimizer_steps']or
        state['namespace']!=value['trainer_state']['namespace']or sha(state['descriptor'])!=state['descriptor_sha256']or
        state['descriptor']['inference_checkpoint']!=value['new_checkpoint']['id']or report['new_checkpoint']!=value['new_checkpoint']):
        raise ValueError('exact ROOT committed optimizer cache ACK')
    with StateCache(workspace,job,manifest,authority)as cache:
        pending=cache.root/'pending.json'
        marker=cache.root/'current.json'
        already=json.loads(marker.read_bytes())if marker.exists()else None
        if already is not None and already['job_id']==job['job_id']:
            if (already.get('descriptor_sha256')!=state['descriptor_sha256'] or already.get('job_sha256')!=sha(job) or
                already.get('source_sha256')!=manifest['source_bundle']['sha256'] or
                authenticate(already['ROOT_ack'],authority)!=value):
                raise ValueError('idempotent promotion exact confirmed lineage')
            if pending.exists():
                orphan=json.loads(pending.read_bytes())
                if {k:v for k,v in already.items()if k!='ROOT_ack'}!=orphan:raise ValueError('promotion pending/current mismatch')
                pending.unlink()
            return dict(promoted=True,idempotent=True,descriptor_sha256=state['descriptor_sha256'],bytes=sum(s['size']for s in state['descriptor']['shards']))
        if not pending.exists():
            if already is not None:raise ValueError('missing candidate does not match confirmed current lineage')
            return dict(promoted=False,reason='no-owned-candidate')
        candidate=json.loads(pending.read_bytes());descriptor=state['descriptor'];shards={s['name']:s for s in descriptor['shards']}
        if candidate['job_id']!=job['job_id']or candidate['job_sha256']!=sha(job)or candidate['source_sha256']!=manifest['source_bundle']['sha256']or candidate['descriptor_sha256']!=sha(descriptor)or set(candidate['files'])!=set(shards):
            raise ValueError('original candidate/job/source/descriptor binding')
        for name,row in candidate['files'].items():
            path=cache.directory(job['job_id'])/member(name)
            if snapshot(path)!=row['stat']or hash_file(path)!=(shards[name]['sha256'],shards[name]['size']):raise ValueError('candidate cache bytes changed after publication')
        marker=cache.root/'current.json'
        if marker.exists():
            previous=json.loads(marker.read_bytes())
            previous_ack=authenticate(previous['ROOT_ack'],authority)
            if previous_ack['trainer_state']['optimizer_steps']>value['trainer_state']['optimizer_steps']:raise ValueError('optimizer cache lineage rollback')
            if previous_ack['trainer_state']['optimizer_steps']==value['trainer_state']['optimizer_steps']and previous_ack['trainer_state']['descriptor_sha256']!=value['trainer_state']['descriptor_sha256']:
                raise ValueError('same optimizer counter different promotion lineage')
            if previous['job_id']!=candidate['job_id']:cache.discard(previous,'superseded-after-durable-ROOT-ACK')
        candidate['ROOT_ack']=ack;cache.save(marker,candidate);pending.unlink()
        return dict(promoted=True,descriptor_sha256=sha(descriptor),bytes=sum(s['size']for s in shards.values()))
