"""Default-off sole-current FP32 cache. R2 and ROOT remain authoritative."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import threading
import time
from .cache_lifecycle import snapshot,identifier
from .storage import canonical
from .distributed_roles import authenticate
VERSION='sole-current-fp32-state-cache-v1'
STAT_VERSION='sole-current-fp32-state-cache-stat-v2'
STAT_VALIDATION='durable-unchanged-inode-v1'
VOLUME_VERSION='alternating-local-optimizer-volume-v1'


def volume_policy(workspace,root):
    marker=Path(root)/'local-volume.json'
    if not marker.exists():return None
    if snapshot(marker)['mode']&0o077:raise ValueError('private local volume policy')
    value=json.loads(marker.read_bytes())
    volume=Path('/dev/shm')/('affine-optimizer-'+hashlib.sha256(str(workspace).encode()).hexdigest()[:20])
    if value!=dict(version=VOLUME_VERSION,workspace=str(workspace),memory_root=str(volume)):
        raise ValueError('exact owned trainer memory volume')
    st=volume.lstat()
    import stat
    if volume!=volume.resolve() or not stat.S_ISDIR(st.st_mode) or st.st_uid!=os.getuid() or st.st_mode&0o077:
        raise ValueError('private owned trainer memory directory')
    return volume

def candidate_directory(workspace,root,job_id):
    directory=root/('candidate-'+identifier(job_id))
    if directory!=directory.resolve():raise ValueError('optimizer candidate path symlink')
    volume=volume_policy(workspace,root)
    if volume is not None:
        owned=volume/directory.name
        if owned.exists() or owned.is_symlink():
            st=owned.lstat()
            if owned!=owned.resolve() or not stat.S_ISDIR(st.st_mode) or st.st_uid!=os.getuid() or st.st_mode&0o077:
                raise ValueError('private owned optimizer memory candidate')
            if directory.exists():
                if (directory.stat().st_dev,directory.stat().st_ino)!=(st.st_dev,st.st_ino):
                    raise ValueError('ambiguous optimizer memory and disk candidate')
                return directory  # Previously acknowledged legacy bind mount.
            return owned
    return directory

def policy(manifest):
    value=manifest.get('optimizer_state_local_cache')
    if value is None:return None
    if (type(value)is not dict or not (
        (set(value)=={'version','max_checkpoint_bytes'} and value['version']==VERSION) or
        (set(value)=={'version','max_checkpoint_bytes','validation'} and value['version']==STAT_VERSION and value['validation']==STAT_VALIDATION)) or
        type(value['max_checkpoint_bytes'])is not int or not 1<=value['max_checkpoint_bytes']<=128*1024**3):
        raise ValueError('explicit bounded optimizer cache policy')
    publication=manifest.get('persistent_publication_policy')
    from .persistent_publication import local_state
    if not isinstance(publication,dict)or publication.get('state_readback')!=('trainer-local' if local_state(manifest) else 'qualified-remote-full'):
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


def fd_snapshot(fd):
    import stat
    st=os.fstat(fd)
    if not stat.S_ISREG(st.st_mode) or st.st_nlink!=1 or st.st_uid!=os.getuid():
        raise ValueError('owned immutable regular single-link state fd')
    return dict(dev=st.st_dev,ino=st.st_ino,size=st.st_size,mtime=st.st_mtime_ns,
                ctime=st.st_ctime_ns,mode=st.st_mode,uid=st.st_uid)


def verification_body(value):
    return dict(version=STAT_VALIDATION,job_id=value['job_id'],job_sha256=value['job_sha256'],
                source_sha256=value['source_sha256'],descriptor_sha256=value['descriptor_sha256'],
                ROOT_ack_sha256=sha(value['ROOT_ack']),files=value['files'])


class OwnedShardReceipt:
    """Internal opened-fd handoff from an authenticated, exclusively leased cache."""
    def __init__(self,owner,fd,path,row,before,after,journal):
        self.owner=owner;self.fd=fd;self.path=Path(path);self.row=row
        self.before=before;self.after=after;self.journal=journal;self.journal_stat=snapshot(journal)
    def validate(self,path,shard,owner):
        if (owner is not self.owner or owner.fd is None or
            owner.policy['version']!=STAT_VERSION or not owner.current or
            owner.verified_rows.get(shard['name'])!=self.before or
            self.fd is None or Path(path).absolute()!=self.path or
            (self.row['sha256'],self.row['size'])!=(shard['sha256'],shard['size'])):
            raise ValueError('exact authorized owned-cache fd handoff')
        if fd_snapshot(self.fd)!=self.after or snapshot(self.path)!=self.after:
            raise ValueError('owned renamed state identity changed')
        if snapshot(self.journal)!=self.journal_stat or self.journal_stat['mode']&0o077:
            raise ValueError('unchanged private owned rename receipt required')
        if json.loads(self.journal.read_bytes())!=dict(
            version='verified-owned-state-rename-v1',job_id=owner.job['job_id'],
            original_cache_job_id=owner.current['job_id'],descriptor_sha256=owner.current['descriptor_sha256'],
            ROOT_ack_sha256=sha(owner.current['ROOT_ack']),name=shard['name'],
            sha256=shard['sha256'],size=shard['size'],before=self.before,after=self.after,
            destination=str(self.path)):
            raise ValueError('exact immutable owned rename journal')
        return '/proc/self/fd/'+str(self.fd)
    def close(self):
        if self.fd is not None:os.close(self.fd);self.fd=None


class RetainedShardReceipt(OwnedShardReceipt):
    """Read the sole local parent through an owned fd without consuming it."""
    preserves_parent=True
    def __init__(self,owner,fd,source,requested,row,before):
        self.owner=owner;self.fd=fd;self.path=source;self.requested=requested
        self.row=row;self.before=before
    def validate(self,path,shard,owner):
        from .persistent_publication import local_state
        if (owner is not self.owner or owner.fd is None or not local_state(owner.manifest) or
                owner.policy['version']!=STAT_VERSION or not owner.current or self.fd is None or
                Path(path).absolute()!=self.requested or owner.verified_rows.get(shard['name'])!=self.before or
                self.path!=owner.directory(owner.current['job_id'])/member(shard['name']) or
                (self.row['sha256'],self.row['size'])!=(shard['sha256'],shard['size']) or
                fd_snapshot(self.fd)!=self.before or snapshot(self.path)!=self.before):
            raise ValueError('unchanged retained local parent fd binding')
        return '/proc/self/fd/'+str(self.fd)


class StateCache:
    def __init__(self,workspace,job,manifest,authority):
        self.workspace=Path(workspace).absolute()
        if self.workspace!=self.workspace.resolve()or not self.workspace.is_dir():raise ValueError('owned existing trainer workspace')
        self.job=job;self.manifest=manifest;self.authority=authority;self.policy=policy(manifest)
        if self.policy is None:raise ValueError('optimizer cache default is off')
        self.root=self.workspace/'.optimizer-state-cache';self.root.mkdir(mode=0o700,exist_ok=True)
        if self.root!=self.root.resolve():raise ValueError('optimizer cache root symlink')
        self.fd=None;self.lock=threading.Lock();self.current=None;self.rows={};self.cache_evidence=[];self.verified_rows={};self.promotion_wait_seconds=1800
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
        return candidate_directory(self.workspace,self.root,job_id)
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
        volume=volume_policy(self.workspace,self.root)
        if volume is not None and directory.exists() and directory.stat().st_dev==volume.stat().st_dev:
            owned=volume/directory.name
            if (directory.stat().st_dev,directory.stat().st_ino)!=(owned.stat().st_dev,owned.stat().st_ino) or list(directory.iterdir()):
                raise ValueError('only empty exact owned candidate mount retires')
            if directory==owned:
                owned.rmdir()
            else:
                subprocess.run(['umount',str(directory)],check=True,capture_output=True)
                directory.rmdir();owned.rmdir()
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
                self.save(self.root/('failed-promotion-'+identifier(confirmed['job_id'])+'.json'),intent)
                promotion.unlink()
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
                if descriptor is not None and abandoned.get('descriptor_sha256')==sha(descriptor):
                    raise ValueError('approved parent candidate awaits original durability ACK; refuse cold abandonment')
                self.discard(abandoned,'different-original-job-abandoned-unpromoted-cache');pending.unlink()
        if not marker.exists():self.cache_evidence.append(dict(outcome='cold',reason='no-promoted-cache'));return 0
        marker_stat=snapshot(marker)
        if marker_stat['mode']&0o077:raise ValueError('private owned optimizer catalogue required')
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
        from .persistent_publication import local_state
        local=local_state(self.manifest)
        # A signed source upgrade may consume the exact authenticated parent.
        # State identity, original job and ACK still have to match; never reset.
        if descriptor is None or sha(descriptor)!=value['descriptor_sha256']or (value['source_sha256']!=source and not local):reason='wrong-approved-parent-or-source'
        else:
            shards={r['name']:r for r in descriptor['shards']}
            if directory.exists()and set(p.name for p in directory.iterdir())-set(value['files']):
                raise ValueError('unowned optimizer cache directory member')
            if set(shards)!=set(value['files']):reason='incomplete-cache'
            else:
                fast=(self.policy['version']==STAT_VERSION and
                      value.get('promotion_verification_sha256')==sha(verification_body(value)))
                for name,row in value['files'].items():
                    path=directory/member(name)
                    try:
                        actual=snapshot(path)
                        if (actual!=row['stat'] or actual['mode']&0o077 or
                            (row['sha256'],row['size'])!=(shards[name]['sha256'],shards[name]['size'])):
                            reason='missing-or-corrupt-cache';break
                        if not fast and hash_file(path)!=(shards[name]['sha256'],shards[name]['size']):
                            reason='missing-or-corrupt-cache';break
                        if snapshot(path)!=actual:raise ValueError('state changed during cache admission')
                        self.verified_rows[name]=actual
                    except FileNotFoundError:reason='missing-or-corrupt-cache';break
        if reason:
            if local:raise ValueError('required local optimizer parent unavailable: '+reason)
            self.discard(value,reason);marker.unlink();self.cache_evidence.append(dict(outcome='cold',reason=reason));return 0
        self.current=value;self.rows={r['name']:r for r in descriptor['shards']}
        size=sum(r['size']for r in self.rows.values())
        if size>self.policy['max_checkpoint_bytes']:raise ValueError('cached parent byte cap')
        self.cache_evidence.append(dict(outcome='full-approved-cache',descriptor_sha256=sha(descriptor),size=size,all_SHA_size_verified=not fast,
            **(dict(validation=STAT_VALIDATION,previous_promotion_full_SHA_verified=fast,all_owned_stat_guards_verified=True)if self.policy['version']==STAT_VERSION else {})))
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
        from .persistent_publication import local_state
        credit=0 if local_state(self.manifest) else reclaimable_parent_bytes
        volume=self.next_volume() if local_state(self.manifest) else None
        if volume is not None:
            from .persistent_training_state import available_ram_bytes
            required=desired+plan['bounded_inflight_transfer_bytes']+plan['disk_reserve_bytes']
            ram_required=desired+plan['cpu_additional_ram_required_bytes']
            cache_free=shutil.disk_usage(volume).free
            if free<plan['additional_disk_required_bytes'] or cache_free<required or available_ram_bytes()<ram_required:
                raise ValueError('alternating optimizer memory and model disk budget')
            return dict(retained_state_required_bytes=desired,ordinary_disk_required_bytes=plan['additional_disk_required_bytes'],
                        reclaimable_verified_parent_bytes=0,observed_free_disk_bytes=free,
                        optimizer_volume_required_bytes=required,observed_free_optimizer_volume_bytes=cache_free,
                        optimizer_volume='memory',additional_ram_required_bytes=ram_required)
        if local_state(self.manifest) and self.root.stat().st_dev!=self.workspace.stat().st_dev:
            cache_free=shutil.disk_usage(self.root).free
            cache_required=desired+plan['bounded_inflight_transfer_bytes']+plan['disk_reserve_bytes']
            if free<plan['additional_disk_required_bytes'] or cache_free<cache_required:
                raise ValueError('separate local optimizer volume and model disk budget')
            return dict(retained_state_required_bytes=desired,ordinary_disk_required_bytes=plan['additional_disk_required_bytes'],
                        reclaimable_verified_parent_bytes=0,observed_free_disk_bytes=free,
                        optimizer_volume_required_bytes=cache_required,observed_free_optimizer_volume_bytes=cache_free)
        if free+credit<plan['additional_disk_required_bytes']+desired:
            raise ValueError('retained optimizer cache plus transfer/BF16/reserve disk budget')
        return dict(retained_state_required_bytes=desired,ordinary_disk_required_bytes=plan['additional_disk_required_bytes'],reclaimable_verified_parent_bytes=credit,observed_free_disk_bytes=free)
    def fetch(self,name,path,fallback):
        if self.current is None:return fallback(name,path)
        name=member(name);source=self.directory(self.current['job_id'])/name
        if snapshot(source)!=self.current['files'][name]['stat']:raise ValueError('approved cached parent changed')
        # Destination is the ordinary restore workspace; its normal full SHA,
        # tensor-schema and finite-value checks remain in force before unlink.
        destination=Path(path).absolute()
        if destination!=destination.resolve()or not destination.is_relative_to(self.workspace):raise ValueError('owned restore destination')
        from .persistent_publication import local_state
        if local_state(self.manifest):
            if destination.exists()or destination.is_symlink():raise ValueError('new retained parent restore destination')
            if self.policy['version']!=STAT_VERSION:
                shutil.copyfile(source,destination);destination.chmod(0o600);return None
            before=snapshot(source)
            if self.verified_rows.get(name)!=before:raise ValueError('retained parent changed before read')
            fd=os.open(source,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
            try:
                if fd_snapshot(fd)!=before:raise ValueError('retained parent fd changed')
                return RetainedShardReceipt(self,fd,source,destination,self.rows[name],before)
            except BaseException:os.close(fd);raise
        if self.policy['version']!=STAT_VERSION:
            os.rename(source,destination);return None
        if destination.exists()or destination.is_symlink():raise ValueError('new owned restore destination required')
        before=snapshot(source)
        if self.verified_rows.get(name)!=before:raise ValueError('admitted source state changed before rename')
        fd=os.open(source,os.O_RDONLY|os.O_NOFOLLOW|os.O_CLOEXEC)
        try:
            if fd_snapshot(fd)!=before:raise ValueError('opened state fd identity')
            os.rename(source,destination);after=snapshot(destination)
            # Only ctime may change because of this exact owned rename.
            if any(after[k]!=before[k]for k in before if k!='ctime') or fd_snapshot(fd)!=after:
                raise ValueError('known owned rename identity continuity')
            journal=self.root/('rename-'+identifier(self.job['job_id'])+'-'+name+'.json')
            if journal.exists():raise ValueError('immutable original shard rename journal')
            self.save(journal,dict(version='verified-owned-state-rename-v1',job_id=self.job['job_id'],
                original_cache_job_id=self.current['job_id'],descriptor_sha256=self.current['descriptor_sha256'],
                ROOT_ack_sha256=sha(self.current['ROOT_ack']),name=name,sha256=self.rows[name]['sha256'],size=self.rows[name]['size'],
                before=before,after=after,destination=str(destination)))
            return OwnedShardReceipt(self,fd,destination,self.rows[name],before,after,journal)
        except BaseException:os.close(fd);raise
    def begin_candidate(self):
        registry=self.root/'pending.json'
        if registry.exists():
            previous=json.loads(registry.read_bytes());self.discard(previous,'abandoned-unpromoted-candidate');registry.unlink()
        from .persistent_publication import local_state
        volume=self.next_volume()if local_state(self.manifest)else None
        directory=self.directory(self.job['job_id'])
        if volume is not None:
            if directory.exists():raise ValueError('new optimizer memory candidate required')
            owned=volume/directory.name;owned.mkdir(mode=0o700,exist_ok=False)
            if self.directory(self.job['job_id'])!=owned:raise ValueError('exact owned memory candidate path')
        else:
            directory.mkdir(mode=0o700,exist_ok=False)
        self.candidate=dict(version=VERSION,job_id=self.job['job_id'],job_sha256=sha(self.job),source_sha256=self.manifest['source_bundle']['sha256'],files={},descriptor_sha256=None)
        self.save(registry,self.candidate)
    def next_volume(self):
        volume=volume_policy(self.workspace,self.root)
        if volume is None:return None
        if self.current is None:return volume
        device=self.directory(self.current['job_id']).stat().st_dev
        if device==volume.stat().st_dev:return None
        if device!=self.workspace.stat().st_dev:raise ValueError('original current optimizer volume')
        return volume
    def retain(self,name,path,digest,size):
        name=member(name);path=Path(path).absolute();snapshot(path)
        from .persistent_publication import local_state
        owned_candidate=local_state(self.manifest) and path.is_relative_to(self.directory(self.job['job_id']))
        if path!=path.resolve()or not (path.is_relative_to(self.workspace) or owned_candidate):
            raise ValueError('owned uploaded shard required')
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
        from .persistent_publication import local_state
        return dict(version=VERSION,descriptor_sha256=sha(descriptor),bytes=sum(r['size']for r in shards.values()),promoted=False,
            ROOT_durability_ACK_required=not local_state(self.manifest),ROOT_lineage_ACK_required=True,parent_cache=self.cache_evidence)


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
                if {k:v for k,v in already.items()if k not in ('ROOT_ack','promotion_verification_sha256')}!=orphan:raise ValueError('promotion pending/current mismatch')
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
            from .persistent_publication import local_state
            if (snapshot(path)!=row['stat']or
                (row['sha256'],row['size'])!=(shards[name]['sha256'],shards[name]['size'])or
                (not local_state(manifest)and hash_file(path)!=(shards[name]['sha256'],shards[name]['size']))):raise ValueError('candidate cache bytes changed after publication')
        marker=cache.root/'current.json'
        if marker.exists():
            previous=json.loads(marker.read_bytes())
            previous_ack=authenticate(previous['ROOT_ack'],authority)
            if previous_ack['trainer_state']['optimizer_steps']>value['trainer_state']['optimizer_steps']:raise ValueError('optimizer cache lineage rollback')
            if previous_ack['trainer_state']['optimizer_steps']==value['trainer_state']['optimizer_steps']and previous_ack['trainer_state']['descriptor_sha256']!=value['trainer_state']['descriptor_sha256']:
                raise ValueError('same optimizer counter different promotion lineage')
            if previous['job_id']!=candidate['job_id']:cache.discard(previous,'superseded-after-durable-ROOT-ACK')
        candidate['ROOT_ack']=ack
        if cache.policy['version']==STAT_VERSION:candidate['promotion_verification_sha256']=sha(verification_body(candidate))
        cache.save(marker,candidate);pending.unlink()
        return dict(promoted=True,descriptor_sha256=sha(descriptor),bytes=sum(s['size']for s in shards.values()))
