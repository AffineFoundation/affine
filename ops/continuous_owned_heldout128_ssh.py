"""Pinned CPU-only SSH adapter for the default-off continuous128 service."""
import base64
import copy
import hashlib
import io
import json
import os
import shlex
import subprocess
import tarfile
import time
import urllib.parse
from pathlib import Path

from subnet.backend_jobs import canonical, signed
from subnet.storage import Bucket, Identity
from ops.continuous_owned_heldout128 import digest
from ops.continuous_owned_heldout128_factory import prepare_packet
from ops.owned_cached_group_operator import GroupACKPublisher, private_json, save
from ops.owned_cached_group_ack_relay import QualifiedGroupObserver, relay_step
from ops.owned_cached_larger_cohort import aggregate


PROBE = '''import pathlib,subprocess,hashlib,shutil,importlib.metadata
active=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:c=(p/'cmdline').read_bytes()
 except OSError:continue
 if any(x in c for x in (b'subnet.remote_runner',b'subnet.backend_jobs',b'subnet.distributed_worker',b'remote_optimizer_readback',b'group_supervisor.py',b'qualification_helper.py',b'bounded_supervisor.py',b'research_helper.py',b'control_helper.py')):active.append(int(p.name))
print(json.dumps(dict(machine=hashlib.sha256(pathlib.Path('/etc/machine-id').read_bytes()).hexdigest(),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True,timeout=10).strip(),compute=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True,timeout=10).strip(),active=active,free=shutil.disk_usage('/root').free,ram=int(next(x.split()[1]for x in pathlib.Path('/proc/meminfo').read_text().splitlines()if x.startswith('MemAvailable:')))*1024,runtime={k:importlib.metadata.version(k)for k in PLAN['runtime']})))
'''

STAGE = '''import pathlib,hashlib,tarfile,os,sys
B=pathlib.Path(PLAN['root']);assert B.resolve()==B
B.mkdir(mode=0o700,exist_ok=True)
archive=B/'source.tar.gz';assert not archive.is_symlink()and hashlib.sha256(archive.read_bytes()).hexdigest()==PLAN['source_sha256']
for directory in('source','cpu-root'):
 root=B/directory;root.mkdir(mode=0o700,exist_ok=True);assert not root.is_symlink()
 with tarfile.open(archive)as t:
  members=t.getmembers();assert len(members)==len(PLAN['inventory'])and {x.name for x in members}==set(PLAN['inventory'])
  for member in members:
   assert member.isfile()and not pathlib.PurePosixPath(member.name).is_absolute()and '..'not in pathlib.PurePosixPath(member.name).parts
   raw=t.extractfile(member).read();assert hashlib.sha256(raw).hexdigest()==PLAN['inventory'][member.name]
   p=root/member.name;p.parent.mkdir(mode=0o700,parents=True,exist_ok=True);assert not p.is_symlink()
   expected=PLAN['cpu_files'].get(member.name)if directory=='cpu-root'else None
   if expected:raw=bytes.fromhex(expected['hex'])
   if p.exists():assert p.read_bytes()==raw
   else:
    fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb')as f:f.write(raw)
 if directory=='cpu-root':
  for name,item in PLAN['cpu_files'].items():
   p=root/name;raw=bytes.fromhex(item['hex']);assert hashlib.sha256(raw).hexdigest()==item['sha256'];p.parent.mkdir(mode=0o700,parents=True,exist_ok=True)
   if p.exists():assert not p.is_symlink()and p.read_bytes()==raw
   else:
    fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    with os.fdopen(fd,'wb')as f:f.write(raw)
 expected=dict(PLAN['inventory'])
 if directory=='cpu-root':expected.update({name:item['sha256']for name,item in PLAN['cpu_files'].items()})
 actual={}
 for p in root.rglob('*'):
  assert not p.is_symlink()
  if p.is_file():actual[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
 assert actual==expected
sys.dont_write_bytecode=True;sys.path.insert(0,str(B/'source'));os.chdir(B/'source')
from subnet.environments import create_session
s=create_session(PLAN['environment']);s.reset(PLAN['index'],1);s.close();assert 'torch'not in sys.modules
print(json.dumps(dict(native_CPU_ready=True,torch_imported=False)))
'''


def emit(body, plan):
    code='import json\nPLAN=json.loads('+repr(json.dumps(plan,separators=(',',':')))+')\n'+body
    compile(code,'continuous128-emitted-private-script','exec')
    return code


class Adapter:
    def __init__(self,policy,authority):
        self.p=policy;self.authority=authority;self.endpoint=policy['endpoint']
        self.local=Path(policy['journal_directory'])/'groups';self.local.mkdir(mode=0o700,parents=True,exist_ok=True)
        self.bucket=Bucket(private_json(policy['bucket_config_path'])['bucket'])
        path=Path(policy['authority_seed_path']);s=path.lstat()
        if path.is_symlink()or s.st_uid!=os.getuid()or s.st_nlink!=1 or s.st_mode&0o077:
            raise ValueError('CPU-only private ROOT signer file')
        self.identity=Identity(bytes.fromhex(path.read_text().strip()))
        if self.identity.id!=authority:raise ValueError('approved service ROOT signer identity')
        self.archive=Path(policy['source_archive_path']);self.inventory=policy['source_inventory']
        if hashlib.sha256(self.archive.read_bytes()).hexdigest()!=policy['source_sha256']:
            raise ValueError('qualified archive bytes before staging')
        self.cpu={}
        for name,row in policy['cpu_dependencies'].items():
            raw=Path(row['path']).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=row['sha256']:raise ValueError('exact qualified CPU dependency closure')
            self.cpu[name]=dict(hex=raw.hex(),sha256=row['sha256'])
        supervisor=Path(policy['supervisor_path']).read_bytes()
        if hashlib.sha256(supervisor).hexdigest()!=policy['supervisor_sha256']:
            raise ValueError('qualified original bounded group supervisor')
        self.supervisor=supervisor
        self.observers={};self.publishers={}

    def sign(self,value):
        return dict(payload=value,signer=self.authority,
                    signature=base64.b64encode(self.identity.key.sign(canonical(value)).signature).decode())

    def remote(self,body,plan,timeout=30):
        e=self.endpoint
        args=['ssh','-p',str(e['port']),'-o','StrictHostKeyChecking=yes',
              '-o','UserKnownHostsFile='+e['known_hosts'],'-o','BatchMode=yes','-o','ConnectTimeout=10',
              e.get('user','root')+'@'+e['host'],shlex.quote(e['python'])+' -I -B -']
        result=subprocess.run(args,input=emit(body,plan),text=True,capture_output=True,timeout=timeout)
        with (self.local/'private-SSH.stderr.log').open('a')as f:f.write(result.stderr)
        if result.returncode:raise RuntimeError('private source-bound SSH failure; preserve original')
        return json.loads(result.stdout)

    def copy(self,local,remote,*,reverse=False):
        e=self.endpoint
        args=['scp','-P',str(e['port']),'-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+e['known_hosts']]
        target=e.get('user','root')+'@'+e['host']+':'+remote
        subprocess.run(args+([target,str(local)]if reverse else[str(local),target]),capture_output=True,check=True,timeout=120)

    def full(self,key,sha,limit):
        size=self.bucket.client.head_object(Bucket=self.bucket.name,Key=key)['ContentLength']
        if type(size)is not int or not 0<size<=limit:raise ValueError('bounded original durability object')
        raw=self.bucket.get(key)
        if len(raw)!=size or hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('full original durability readback')
        return raw

    def bootstrap_complete(self,entry):
        raw=Path(entry['summary_path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=entry['summary_file_sha256']:raise ValueError('immutable bootstrap summary file')
        envelope=json.loads(raw);value=signed(envelope,self.authority)
        ack=private_json(entry['archive_ack_path'])
        group=value['groups'][entry['group_label']]if value['version']=='owned-cached-heldout128-paired-actual-v1'else value['group']
        if (ack.get('R2_full_GET_verified') is not True or ack['archive_sha256']!=group['archive_sha256']):
            raise ValueError('original bootstrap full archive ACK')
        archive=self.full(ack['archive_key'],ack['archive_sha256'],512*1024**2)
        if len(archive)!=ack['archive_bytes']:raise ValueError('original bootstrap archive bytes')
        # Existing hardened projection authenticates the four exact originals,
        # their full readback receipts, runtime/cohort/native split and retirement.
        from dashboard.heldout128_projection import original
        with tarfile.open(fileobj=io.BytesIO(archive))as tar:
            members=tar.getmembers()
            if len({m.name for m in members})!=len(members)or any(not m.isfile()or '..'in Path(m.name).parts or m.name.startswith('/')for m in members):
                raise ValueError('exact safe original bootstrap archive')
            acks=[json.loads(tar.extractfile(m).read())for m in members if m.name.startswith('durable-evaluation-acks/')and m.name.endswith('.json')]
        if len(acks)!=4:raise ValueError('four original completed bootstrap ACKs')
        rows=[original(a,self.p['source_sha256'],digest(self.p['source_files']))for a in acks]
        if len({r[1]['job_id']for r in rows})!=4 or any(r[2]['checkpoint']['id']!=entry['checkpoint']for r in rows):
            raise ValueError('bootstrap original checkpoint and four identities')
        return envelope

    def publication(self,row):
        closure=signed(row['completion'],self.authority)
        metrics=private_json(Path(self.p['production_directory'])/(closure['epoch']+'-training-metrics.json'))
        model_key='public/checkpoints/'+row['checkpoint']+'/authorities/'+self.authority+'/checkpoint.json'
        model=json.loads(self.bucket.get(model_key));state=json.loads(self.bucket.get(metrics['trainer_state']['descriptor_key']))
        signed(model,self.authority);value=signed(state,self.authority)
        if digest(value)!=metrics['trainer_state']['publication_sha256']:
            raise ValueError('original state publication pointer digest')
        return dict(checkpoint_descriptor=model,optimizer_publication=state)

    def idle(self):
        v=self.remote(PROBE,dict(runtime=self.p['runtime_versions']))
        if (v['machine']!=self.p['machine_id_sha256'] or v['gpu']!=self.p['gpu_uuid']or
            v['runtime']!=self.p['runtime_versions']):raise ValueError('exact qualified physical research host')
        return not(v['active']or v['compute'])and v['free']>=self.p['minimum_free_cold_bytes']and v['ram']>=self.p['minimum_available_ram_bytes']

    def refresh(self,manifest,ttl):
        # Model key layout and original source object's bucket/path are explicit
        # signed policy bindings. Refresh capabilities only, never descriptors.
        manifest['checkpoint']['read_urls']={name:self.bucket.presign(self.p['model_file_key_template'].format(checkpoint=manifest['checkpoint']['id'],name=name),'get_object',ttl)for name in manifest['checkpoint']['files']}
        manifest['source_bundle']['read_url']=self.bucket.presign(self.p['source_object_key'],'get_object',ttl)
        return manifest

    def prepare(self,policy,publication,identity):
        local=self.local/identity;local.mkdir(mode=0o700,exist_ok=True);path=local/'packet.json'
        if path.exists():return private_json(path)
        packet=prepare_packet(policy,publication,identity,self.authority,self.sign,now=time.time(),refresh_manifest=self.refresh)
        save(path,packet);return packet

    def launch(self,packet):
        if not self.idle():raise ValueError('physical reservation changed before original launch')
        stage=Path(self.p['remote_source_path']).parent
        self.remote('import pathlib;p=pathlib.Path(PLAN["root"]);p.mkdir(mode=0o700,exist_ok=True);print("{}")',dict(root=str(stage)))
        self.copy(self.archive,str(stage/'source.tar.gz'))
        template=signed(signed(self.p['template_original_job'],self.authority)['manifest'],self.authority)
        self.remote(STAGE,dict(root=str(stage),source_sha256=self.p['source_sha256'],inventory=self.inventory,cpu_files=self.cpu,
                              environment=next(v['spec']for v in template['environments']if v['env_id']=='affine_math'),index=1280),180)
        root=packet['workspace'];scope=packet['scope']['payload']
        # All mutable inputs are exclusive-create and exact on continuation.
        objects={'group.ROOT-SIGNED.json':canonical(packet['scope']),'group_supervisor.py':self.supervisor}
        objects.update({'declared-'+str(i)+'.json':canonical(v)for i,v in enumerate(packet['original_jobs'])})
        self.remote('''import pathlib,os,hashlib
B=pathlib.Path(PLAN['root']);B.mkdir(mode=0o700,exist_ok=True);assert not B.is_symlink()
for name,item in PLAN['objects'].items():
 p=B/name;raw=bytes.fromhex(item['hex']);assert hashlib.sha256(raw).hexdigest()==item['sha256']
 if p.exists():assert not p.is_symlink()and p.read_bytes()==raw
 else:
  fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
  with os.fdopen(fd,'wb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
print('{}')
''',dict(root=root,objects={n:dict(hex=b.hex(),sha256=hashlib.sha256(b).hexdigest())for n,b in objects.items()}))
        args=[self.endpoint['python'],'-B',root+'/group_supervisor.py','--scope',root+'/group.ROOT-SIGNED.json',
              '--authority',self.authority,'--cpu-root',str(stage/'cpu-root'),'--jobs']+[root+'/declared-'+str(i)+'.json'for i in range(4)]+['--execute']
        if not self.idle():raise ValueError('physical reservation changed after CPU staging, before original GPU launch')
        self.remote('''import pathlib,os,subprocess,time
B=pathlib.Path(PLAN['root']);fd=os.open(B/'supervisor.launch-marker',os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.close(fd)
with(B/'supervisor.private.log').open('xb')as f:p=subprocess.Popen(PLAN['args'],stdin=subprocess.DEVNULL,stdout=f,stderr=f,start_new_session=True,close_fds=True,umask=0o077)
v=dict(pid=p.pid,ticks=pathlib.Path('/proc/'+str(p.pid)+'/stat').read_text().rsplit(')',1)[1].split()[19],at=time.time())
(B/'supervisor.launch.json').write_text(json.dumps(v));print(json.dumps(v))
''',dict(root=root,args=args))

    def observe(self,packet):
        scope=signed(packet['scope'],self.authority)
        identity=packet['identity']
        if identity not in self.observers:
            self.publishers[identity]=GroupACKPublisher(packet['scope'],self.authority,packet['original_jobs'],self.bucket,self.sign)
            self.observers[identity]=QualifiedGroupObserver(scope,packet['original_jobs'],self.authority)
        publisher=self.publishers[identity];observer=self.observers[identity]
        relay=relay_step(publisher,observer)
        if relay['status']=='original-infrastructure-failure':return relay
        value=self.remote('''import pathlib
B=pathlib.Path(PLAN['root']);p=B/'group-supervisor-terminal.json'
if not p.exists():print('null')
else:
 original=json.loads((B/'group-supervisor-original.json').read_bytes());proc=pathlib.Path('/proc')/str(original['pid'])/'stat';live=False
 if proc.exists():
  fields=proc.read_text().rsplit(')',1)[1].split();live=fields[0]!='Z'and fields[19]==str(original['ticks'])
 print('null'if live else p.read_text())
''',dict(root=packet['workspace']))
        if value is None:return dict(status='observing-original')
        if value['scope_sha256']!=digest(packet['scope']):raise ValueError('original group supervisor scope digest')
        result=value['result']
        if result['status']!='complete':return dict(status='original-infrastructure-failure',model_reward=None,original_result=result)
        # A complete terminal appears only after the existing group operator
        # authenticated all four ACKs and performed guarded private retirement.
        score=result['score']
        if result['retirement']['status']!='complete':raise ValueError('actual group retirement required')
        return dict(status='complete',durable_ACK_count=4,owned_model_retired=True,count=score['count'],
                    successes=score['successes'],original_group_terminal=value)

    def archive_complete(self,packet,result):
        root=packet['workspace'];local=self.local/packet['identity'];path=local/'summary.ROOT-SIGNED.json'
        if path.exists():return private_json(path)
        self.remote('''import pathlib,tarfile
B=pathlib.Path(PLAN['root']);p=B/'completed-evidence.tar.gz'
if not p.exists():
 with tarfile.open(p,'x:gz')as t:
  for x in B.rglob('*'):
   if x.is_file()and x!=p and 'checkpoints'not in x.relative_to(B).parts and not x.name.endswith('.lock'):
    assert not x.is_symlink();t.add(x,arcname=str(x.relative_to(B)),recursive=False)
assert p.stat().st_size<=512*1024**2;print('{}')
''',dict(root=root))
        archive=local/'completed-evidence.tar.gz';self.copy(archive,root+'/completed-evidence.tar.gz',reverse=True)
        raw=archive.read_bytes();sha=hashlib.sha256(raw).hexdigest();key='private/owned-cached-heldout128/'+sha+'/group-full-evidence.tar.gz'
        self.bucket.put(key,raw)
        if self.bucket.get(key)!=raw:raise ValueError('full group archive R2 readback; no summary')
        ack=dict(archive_sha256=sha,archive_bytes=len(raw),archive_key=key,R2_full_GET_verified=True);save(local/'archive-ACK.json',ack)
        scope=signed(packet['scope'],self.authority)
        with tarfile.open(fileobj=io.BytesIO(raw))as tar:
            members=tar.getmembers();names=[m.name for m in members]
            if len(set(names))!=len(names) or any(not m.isfile()or m.name.startswith('/')or '..'in Path(m.name).parts for m in members):
                raise ValueError('safe exact full original group archive')
            acks=[signed(json.loads(tar.extractfile(m).read()),self.authority)for m in members if m.name.startswith('durable-evaluation-acks/')and m.name.endswith('.json')]
        originals={signed(j,self.authority)['job_id']:j for j in packet['original_jobs']}
        if len(acks)!=4 or {a['original_report']['job_id']for a in acks}!=set(originals):
            raise ValueError('four actual original authenticated archive ACKs')
        for ack in acks:
            jid=ack['original_report']['job_id']
            if ack['original_job']!=originals[jid]or ack['durable_report_full_readback']is not True:
                raise ValueError('archive ACK identity and report provenance')
        score=aggregate(dict(dispatch_allowed=False,groups=scope['groups'],experiment_id=scope['experiment_id'],cohort_sha256=self.p['cohort_sha256']),[a['original_report']for a in acks])
        if score['successes']!=result['successes']or score['count']!=result['count']:
            raise ValueError('actual ACK-derived native128 score')
        terminal=result['original_group_terminal']
        value=dict(version='owned-cached-heldout128-checkpoint-actual-v1',checkpoint=packet['checkpoint'],
            cohort_sha256=self.p['cohort_sha256'],source_sha256=self.p['source_sha256'],
            production_checkpoints_changed=False,normal_evaluator_B_paused=False,all_four_genuine_full_R2_ACKs=True,
            group_owned_model_retired=True,original_jobs=4,task_count=128,successes=result['successes'],
            completed_at=terminal['finished_at'],group=dict(archive_sha256=sha,result=terminal))
        envelope=self.sign(value);raw=canonical(envelope);key='private/owned-cached-heldout128/'+digest(envelope)+'/checkpoint-summary.ROOT-SIGNED.json'
        self.bucket.put(key,raw)
        if self.bucket.get(key)!=raw:raise ValueError('full ROOT summary R2 readback')
        save(path,envelope)
        return envelope

    def publish_pointer(self,packet,summary):
        """Append only a completed full128 scope; expose no private URLs."""
        local=self.local/packet['identity'];path=Path(self.p['dashboard_pointer_path'])
        value=signed(summary,self.authority)
        if value['checkpoint']!=packet['checkpoint'] or value['all_four_genuine_full_R2_ACKs']is not True:
            raise ValueError('completed genuine checkpoint projection only')
        expected=dict(version='heldout128-dashboard-sources-v1',source_sha256=self.p['source_sha256'],
                      source_files_sha256=digest(self.p['source_files']),cohort_sha256=self.p['cohort_sha256'],
                      excluded_indices=self.p['old32_indices'])
        pointer=signed(private_json(path),self.authority)if path.exists()else dict(expected,evaluations=[])
        if {k:pointer.get(k)for k in expected}!=expected:
            raise ValueError('preserve actual existing signed dashboard scientific scope')
        archive_ack=private_json(local/'archive-ACK.json')
        if archive_ack['archive_sha256']!=value['group']['archive_sha256']:
            raise ValueError('exact completed archive pointer')
        entry=dict(summary_path=str(local/'summary.ROOT-SIGNED.json'),
                   summary_sha256=hashlib.sha256((local/'summary.ROOT-SIGNED.json').read_bytes()).hexdigest(),
                   groups={'CP'+str(packet['optimizer_step']):dict(archive_path=str(local/'completed-evidence.tar.gz'),
                      archive_ack_path=str(local/'archive-ACK.json'),
                      archive_ack_sha256=hashlib.sha256((local/'archive-ACK.json').read_bytes()).hexdigest())})
        matches=[v for v in pointer['evaluations']if v['summary_path']==entry['summary_path']]
        if matches and matches!=[entry]:raise ValueError('immutable original projection entry changed')
        if not matches:pointer['evaluations'].append(entry)
        # Run the complete hardened projection before signing/updating its scope.
        from dashboard.heldout128_projection import rows
        envelope=self.sign(pointer);rows(envelope,self.p['production_directory'])
        path.parent.mkdir(mode=0o700,parents=True,exist_ok=True);save(path,envelope)
