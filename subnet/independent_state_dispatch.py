"""Root-only SSH dispatch to an admitted CPU reader; no worker enrollment."""
import hashlib,json,math,os,re,shlex,subprocess,time
from pathlib import Path
from . import remote_optimizer_readback as reader
from .remote_state_commit import readback_objects
from .persistent_training_protocol import validate_report,read_json

ADMISSION='qualified-independent-state-reader-v1'

def canonical(v):return reader.canonical(v)
def file_sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_once(path,value):
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,'wb')as f:f.write(canonical(value));f.flush();os.fsync(f.fileno())

def launch_code(data):
    # Structured prefix only: values cannot replace literal ROOT filenames.
    return 'DATA='+repr(data)+'\n'+'''
import json,os,hashlib,time,subprocess
from pathlib import Path
root=Path(DATA['run']);root.mkdir(mode=0o700,exist_ok=False)
os.umask(0o077);os.nice(19);assert os.getpriority(os.PRIO_PROCESS,0)==19
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
argv=[DATA['python'],'-I','-B',DATA['supervisor'],'--launch',DATA['launch'],'--authority',DATA['authority']]
with(root/'supervisor.stdout.private.log').open('xb')as out,(root/'supervisor.stderr.private.log').open('xb')as err:
 child=subprocess.Popen(argv,stdin=subprocess.DEVNULL,stdout=out,stderr=err,start_new_session=True)
 ticks=Path('/proc',str(child.pid),'stat').read_text().rsplit(')',1)[1].split()[19]
 marker=dict(pid=child.pid,ticks=ticks,started_at=time.time(),request_file_sha256=DATA['request_sha256'],launch_file_sha256=DATA['launch_sha256'],CPU_nice=19,GPU_use=False,authority_commit=False)
 with(root/'original-supervisor-launch.private.json').open('xb')as f:f.write(json.dumps(marker,sort_keys=True,separators=(',',':')).encode());f.flush();os.fsync(f.fileno())
 print(json.dumps(marker))
'''

class IndependentStateReader:
    def __init__(self,config,controller):
        required={'endpoint','reader_host','trainer_host','trainer_known_hosts',
                  'reader_identity','module_hashes','qualification','max_wall_seconds'}
        if set(config)!=required:raise ValueError('exact admitted CPU reader configuration')
        self.config=config;self.endpoint=config['endpoint'];self.authority=controller.authority.id
        if set(self.endpoint)!={'host','port','user','known_hosts','python','workspace','namespace'}:
            raise ValueError('exact qualified reader endpoint')
        admission=reader.verify(config['qualification'],self.authority)
        expected=dict(version=ADMISSION,reader_identity=config['reader_identity'],
            reader_host_record_sha256=reader.sha(config['reader_host']),
            module_hashes=config['module_hashes'],all_23_objects_full_hash=True)
        if set(admission)!=set(expected)|{'qualification_evidence_sha256','qualified_at'} or any(canonical(admission[k])!=canonical(v) for k,v in expected.items()):
            raise ValueError('root actual full-state reader qualification admission')
        if type(admission['qualified_at']) not in (int,float) or not math.isfinite(admission['qualified_at']) or not 0<admission['qualified_at']<=time.time():
            raise ValueError('actual qualified reader admission timestamp')
        for host in (config['reader_host'],config['trainer_host']):
            if set(host)!={'provider_UUID','ssh_host_key_sha256','evidence_sha256'}:
                raise ValueError('exact original physical host records')
        if config['reader_host']['provider_UUID']==config['trainer_host']['provider_UUID']:
            raise ValueError('independent actual provider machines')
        if set(config['module_hashes'])!={'remote_optimizer_readback.py','helper.py','supervisor.py'}:
            raise ValueError('three exact CPU reader modules')
        from .persistent_cpu_adamw import checkpoint_id
        for value in [config['reader_identity'],admission['qualification_evidence_sha256'],*config['module_hashes'].values()]:checkpoint_id(value)
        if type(config['max_wall_seconds'])is not int or not 1<=config['max_wall_seconds']<=3500:
            raise ValueError('bounded reader wall time')

    def command(self,code):
        e=self.endpoint
        argv=['ssh','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
              '-o','UserKnownHostsFile='+e['known_hosts'],'-p',str(e['port']),
              e['user']+'@'+e['host'],shlex.quote(e['python'])+' -I -B -c '+shlex.quote(code)]
        result=subprocess.run(argv,capture_output=True,text=True,timeout=45)
        if result.returncode:raise RuntimeError('qualified reader SSH failed; retain original request')
        if len(result.stdout)>4_000_000:raise ValueError('bounded CPU reader observation')
        return json.loads(result.stdout)

    def preflight(self):
        c=self.config;e=self.endpoint
        if (Path(e['known_hosts']).name!=c['reader_host']['provider_UUID'] or
                file_sha(e['known_hosts'])!=c['reader_host']['ssh_host_key_sha256'] or
                Path(c['trainer_known_hosts']).name!=c['trainer_host']['provider_UUID'] or
                file_sha(c['trainer_known_hosts'])!=c['trainer_host']['ssh_host_key_sha256']):
            raise ValueError('root physical host trust bytes changed')
        data=dict(namespace=e['namespace'],workspace=e['workspace'])
        code='DATA='+repr(data)+'\n'+'''
import json,hashlib,stat,subprocess,time
from pathlib import Path
from nacl.signing import SigningKey
root=Path(DATA['namespace']);seed=root/'reader.seed';assert seed.is_file()and not seed.is_symlink()and stat.S_IMODE(seed.stat().st_mode)==0o600
hashes={n:hashlib.sha256((root/n).read_bytes()).hexdigest()for n in ('remote_optimizer_readback.py','helper.py','supervisor.py')}
assert all((root/n).is_file()and not(root/n).is_symlink()for n in hashes)
active=[]
for p in (Path(DATA['workspace'])/'runner-status').glob('*.json'):
 v=json.loads(p.read_text())
 if v.get('phase')=='complete':continue
 for field in ('runner_pid','child_pid'):
  if field not in v:continue
  q=Path('/proc',str(v[field]),'stat')
  if q.exists():
   a=q.read_text().rsplit(')',1)[1].split()
   if a[0]!='Z'and a[19]==str(v.get(field+'_ticks')):active.append(v.get('job_id'))
gpu=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
print(json.dumps(dict(observed_at=time.time(),hashes=hashes,reader_identity=bytes(SigningKey(bytes.fromhex(seed.read_text().strip())).verify_key).hex(),active_jobs=active,gpu_query_exit=gpu.returncode,gpu_pids=[s for s in gpu.stdout.splitlines()if s.strip()])))
'''
        observed=self.command(code)
        if (observed['hashes']!=c['module_hashes'] or observed['reader_identity']!=c['reader_identity'] or
                observed['active_jobs'] or observed['gpu_query_exit']!=0 or observed['gpu_pids']):
            raise ValueError('fresh qualified reader host/modules/key/GPU idle admission')
        return code,observed

    def prepare_original_readback(self,controller,report,envelope):
        c=self.config;e=self.endpoint;job=reader.verify(envelope,self.authority)
        if not isinstance(job.get('job_id'),str) or not re.fullmatch('[A-Za-z0-9_-]{1,100}',job['job_id']):raise ValueError('original safe reader dispatch job ID')
        manifest=reader.verify(job['manifest'],self.authority);descriptor=validate_report(report,job,manifest)
        namespace=job['persistent_training']['output_namespace']
        if canonical(read_json(controller.bucket,namespace+'/staged-state.json'))!=canonical(descriptor):
            raise ValueError('original production staged descriptor')
        objects=readback_objects(descriptor)
        if len(objects)!=23:raise ValueError('qualified exact full-state count')
        bucket=controller.bucket
        binding=dict(purpose='production-training-state',provenance=dict(
            signed_job_envelope_sha256=reader.sha(envelope),original_report_sha256=reader.sha(report),
            signed_manifest_envelope_sha256=reader.sha(job['manifest'])),job_id=job['job_id'],
            job_sha256=reader.sha(job),source_sha256=manifest['trainer_state_binding']['source_sha256'],
            namespace=namespace,descriptor_sha256=reader.sha(descriptor),
            reader_host_record_sha256=reader.sha(c['reader_host']),trainer_host_record_sha256=reader.sha(c['trainer_host']),
            storage_origin=bucket.client.meta.endpoint_url,storage_bucket=bucket.name,storage_addressing='path')
        root=controller.state/'independent-state-readbacks'/job['job_id'];root.parent.mkdir(mode=0o700,exist_ok=True)
        if root.exists():raise ValueError('original reader dispatch already exists; explicit original-process recovery required')
        preflight,observed=self.preflight();root.mkdir(mode=0o700,exist_ok=False)
        write_once(root/'reservation.private.json',dict(original_job_sha256=reader.sha(job),
            CPU_nice=19,hash_streams=4,chunk_bytes=1024**2,GPU_use=False,authority_commit=False,
            original_preflight=observed,created_at=time.time()))
        started=time.time();duration=c['max_wall_seconds']
        payload=dict(version=reader.VERSION,**binding,reader_identity=c['reader_identity'],
            created_at=started,expires_at=started+duration,max_wall_seconds=duration,objects=objects,
            capabilities={s['name']:bucket.presign(namespace+'/'+s['name'],'get_object',duration)for s in objects})
        request=controller.signed(payload)
        reader.validate_request(request,self.authority,now=time.time(),approved_binding=binding,
            approved_objects=objects,qualified_reader=c['reader_identity'])
        requestfile=root/'request.ROOT-SIGNED.private.json';write_once(requestfile,request)
        dispatch=e['namespace']+'/production-'+job['job_id']
        launch=dict(version='independent-state-readback-launch-v1',python=e['python'],
            helper_path=e['namespace']+'/helper.py',helper_sha256=c['module_hashes']['helper.py'],
            module_path=e['namespace']+'/remote_optimizer_readback.py',module_sha256=c['module_hashes']['remote_optimizer_readback.py'],
            supervisor_sha256=c['module_hashes']['supervisor.py'],request_path=dispatch+'/request.ROOT-SIGNED.private.json',
            request_sha256=file_sha(requestfile),reader_seed_path=e['namespace']+'/reader.seed',
            result_path=dispatch+'/run/result.private.json',workspace=dispatch+'/run',
            max_wall_seconds=duration,created_at=started,expires_at=started+duration)
        launchfile=root/'launch.ROOT-SIGNED.private.json';write_once(launchfile,controller.signed(launch))
        self.command('import json,os;from pathlib import Path;os.umask(0o077);Path('+repr(dispatch)+').mkdir(mode=0o700,exist_ok=False);print(json.dumps({"created":True}))')
        argv=['scp','-q','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+e['known_hosts'],
              '-P',str(e['port']),str(requestfile),str(launchfile),e['user']+'@'+e['host']+':'+dispatch+'/']
        subprocess.run(argv,check=True,timeout=45)
        # The same remote process rechecks idle/modules immediately before Popen.
        data=dict(run=dispatch+'/supervision',python=e['python'],supervisor=e['namespace']+'/supervisor.py',
            launch=dispatch+'/launch.ROOT-SIGNED.private.json',authority=self.authority,
            request_sha256=file_sha(requestfile),launch_sha256=file_sha(launchfile))
        gate='import contextlib,io,json\n_buf=io.StringIO()\nwith contextlib.redirect_stdout(_buf):exec('+repr(preflight)+')\n_v=json.loads(_buf.getvalue());assert not _v["active_jobs"]and _v["gpu_query_exit"]==0 and not _v["gpu_pids"]\nassert _v["hashes"]=='+repr(c['module_hashes'])+'and _v["reader_identity"]=='+repr(c['reader_identity'])+'\n'
        copied='import hashlib;from pathlib import Path\nassert hashlib.sha256(Path('+repr(dispatch+'/request.ROOT-SIGNED.private.json')+').read_bytes()).hexdigest()=='+repr(file_sha(requestfile))+'\nassert hashlib.sha256(Path('+repr(dispatch+'/launch.ROOT-SIGNED.private.json')+').read_bytes()).hexdigest()=='+repr(file_sha(launchfile))+'\n'
        code=gate+copied+launch_code(data);compile(code,'qualified-reader-final-launch','exec')
        marker=self.command(code);write_once(root/'original-supervisor-launch.private.json',marker)
        while time.time()<payload['expires_at']:
            data=dict(dispatch=dispatch)
            status=self.command('DATA='+repr(data)+'\n'+'''
import json
from pathlib import Path
p=Path(DATA['dispatch'])/'run';v={}
for name in ('child','terminal','result'):
 q=p/(name+'.private.json')
 if q.exists():v[name]=json.loads(q.read_text())
print(json.dumps(v))
''')
            if 'terminal'in status:
                write_once(root/'actual-terminal-and-result.private.json',status)
                terminal=status['terminal']
                if terminal.get('exit_code')!=0 or terminal.get('timed_out')is not False or 'result'not in status:
                    raise ValueError('original independent reader failed; no authority commit')
                reader.validate_receipt(status['result'],request,self.authority,
                    approved_binding=binding,approved_objects=objects,qualified_reader=c['reader_identity'],now=time.time())
                return dict(request_bytes=requestfile.read_bytes(),receipt=status['result'],
                    launch_envelope=json.loads(launchfile.read_text()),terminal=terminal,
                    qualified_reader=c['reader_identity'],reader_host=c['reader_host'],trainer_host=c['trainer_host'],
                    storage_binding={k:binding[k]for k in ('storage_origin','storage_bucket','storage_addressing')},
                    original_child=status['child'],now=time.time())
            time.sleep(min(5,max(.01,payload['expires_at']-time.time())))
        raise TimeoutError('original bounded independent reader request expired; retain evidence')
