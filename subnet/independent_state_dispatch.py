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
        if set(config)not in (required,required|{'stream_budget'}):raise ValueError('exact admitted CPU reader configuration')
        if 'stream_budget'in config:reader.stream_budget(config['stream_budget'])
        self.config=config;self.endpoint=config['endpoint'];self.authority=controller.authority.id
        if set(self.endpoint)!={'host','port','user','known_hosts','python','workspace','namespace'}:
            raise ValueError('exact qualified reader endpoint')
        admission=reader.verify(config['qualification'],self.authority)
        expected=dict(version=ADMISSION,reader_identity=config['reader_identity'],
            reader_host_record_sha256=reader.sha(config['reader_host']),
            module_hashes=config['module_hashes'],all_23_objects_full_hash=True)
        if 'stream_budget'in config:expected['stream_budget']=config['stream_budget']
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
        data=dict(namespace=e['namespace'],workspace=e['workspace'],stream_budget=c.get('stream_budget'))
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
resource_admission=None
if DATA['stream_budget']is not None:
 import importlib.util
 module_spec=importlib.util.spec_from_file_location('qualified_reader_resource_check',root/'remote_optimizer_readback.py')
 module=importlib.util.module_from_spec(module_spec);module_spec.loader.exec_module(module)
 resource_admission=module.admit_stream_resources(DATA['stream_budget'],module.available_ram_bytes())
print(json.dumps(dict(resource_admission=resource_admission,observed_at=time.time(),hashes=hashes,reader_identity=bytes(SigningKey(bytes.fromhex(seed.read_text().strip())).verify_key).hex(),active_jobs=active,gpu_query_exit=gpu.returncode,gpu_pids=[s for s in gpu.stdout.splitlines()if s.strip()])))
'''
        observed=self.command(code)
        if (observed['hashes']!=c['module_hashes'] or observed['reader_identity']!=c['reader_identity'] or
                observed['active_jobs'] or observed['gpu_query_exit']!=0 or observed['gpu_pids']):
            raise ValueError('fresh qualified reader host/modules/key/GPU idle admission')
        if 'stream_budget'in c:
            resource=observed.get('resource_admission')
            if not isinstance(resource,dict):raise ValueError('actual reader resource admission missing')
            expected=reader.admit_stream_resources(c['stream_budget'],resource.get('available_ram_bytes'))
            if any(resource.get(k)!=v for k,v in expected.items()):raise ValueError('actual reader resource admission mismatch')
        return code,observed

    def prepare_original_readback(self,controller,report,envelope):
        c=self.config;e=self.endpoint;job=reader.verify(envelope,self.authority)
        if not isinstance(job.get('job_id'),str) or not re.fullmatch('[A-Za-z0-9_-]{1,100}',job['job_id']):raise ValueError('original safe reader dispatch job ID')
        manifest=reader.verify(job['manifest'],self.authority);descriptor=validate_report(report,job,manifest)
        namespace=job['persistent_training']['output_namespace']
        if canonical(read_json(controller.bucket,namespace+'/staged-state.json'))!=canonical(descriptor):
            raise ValueError('original production staged descriptor')
        budget=manifest.get('independent_state_readback_budget')
        if budget is not None:
            reader.stream_budget(budget)
            if canonical(budget)!=canonical(c.get('stream_budget')):raise ValueError('manifest readback budget requires exact qualified admission')
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
        if root.exists():
            return self._recover_original(controller,root,job,binding,objects)
        # This command is read-only and occurs before any reader reservation,
        # capability signing or remote namespace/launch. Transport timeout is
        # safe to retry through the original coordinator phase. Once a request
        # exists, _recover_original alone controls observation; never redispatch.
        try:preflight,observed=self.preflight()
        except subprocess.TimeoutExpired as exc:
            from .remote_backend import RemoteObservationTimeout
            raise RemoteObservationTimeout(job['job_id'],'independent-reader-preflight')from exc
        root.mkdir(mode=0o700,exist_ok=False)
        write_once(root/'reservation.private.json',dict(original_job_sha256=reader.sha(job),
            CPU_nice=19,hash_streams=reader.concurrency({'stream_budget':budget}if budget is not None else {}),chunk_bytes=1024**2,GPU_use=False,authority_commit=False,
            original_preflight=observed,created_at=time.time()))
        started=time.time();duration=c['max_wall_seconds']
        payload=dict(version=reader.VERSION,**binding,reader_identity=c['reader_identity'],
            created_at=started,expires_at=started+duration,max_wall_seconds=duration,objects=objects,
            capabilities={s['name']:bucket.presign(namespace+'/'+s['name'],'get_object',duration)for s in objects})
        if budget is not None:payload['stream_budget']=budget
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
        return self._wait_original(root,request,launchfile,binding,objects,dispatch)

    def _recover_original(self,controller,root,job,binding,objects):
        """Observe only the original signed dispatch; never sign, copy or spawn."""
        reservation=json.loads((root/'reservation.private.json').read_bytes())
        if reservation.get('original_job_sha256')!=reader.sha(job):
            raise ValueError('original reader recovery job binding')
        requestfile=root/'request.ROOT-SIGNED.private.json';launchfile=root/'launch.ROOT-SIGNED.private.json'
        # Partial pre-dispatch preparation cannot prove an original handle. It
        # remains an explicit operator recovery condition, never an auto launch.
        if not requestfile.is_file() or not launchfile.is_file():
            raise ValueError('original reader preparation incomplete; no automatic redispatch')
        request=json.loads(requestfile.read_bytes())
        reader.validate_request(request,self.authority,now=reader.verify(request,self.authority)['created_at'],approved_binding=binding,
            approved_objects=objects,qualified_reader=self.config['reader_identity'])
        launch=reader.verify(json.loads(launchfile.read_bytes()),self.authority)
        dispatch=self.endpoint['namespace']+'/production-'+job['job_id']
        c=self.config;e=self.endpoint;payload=request['payload']
        expected=dict(version='independent-state-readback-launch-v1',python=e['python'],
            helper_path=e['namespace']+'/helper.py',helper_sha256=c['module_hashes']['helper.py'],
            module_path=e['namespace']+'/remote_optimizer_readback.py',module_sha256=c['module_hashes']['remote_optimizer_readback.py'],
            supervisor_sha256=c['module_hashes']['supervisor.py'],request_path=dispatch+'/request.ROOT-SIGNED.private.json',
            request_sha256=file_sha(requestfile),reader_seed_path=e['namespace']+'/reader.seed',
            result_path=dispatch+'/run/result.private.json',workspace=dispatch+'/run',
            max_wall_seconds=payload['max_wall_seconds'],created_at=payload['created_at'],expires_at=payload['expires_at'])
        if canonical(launch)!=canonical(expected):raise ValueError('exact original recovery launch binding')
        # Pin trust again without demanding GPU idle: the same already admitted
        # CPU-only process may overlap a later evaluation. No new process starts.
        if (file_sha(e['known_hosts'])!=c['reader_host']['ssh_host_key_sha256'] or
                file_sha(c['trainer_known_hosts'])!=c['trainer_host']['ssh_host_key_sha256']):
            raise ValueError('original recovery host trust bytes changed')
        return self._wait_original(root,request,launchfile,binding,objects,dispatch)

    def _wait_original(self,root,request,launchfile,binding,objects,dispatch):
        c=self.config;requestfile=root/'request.ROOT-SIGNED.private.json';payload=request['payload']
        data=dict(dispatch=dispatch,request_sha256=file_sha(requestfile),launch_sha256=file_sha(launchfile))
        code='DATA='+repr(data)+'\n'+'''
import hashlib,json
from pathlib import Path
p=Path(DATA['dispatch']);v={}
for name in ('request','launch'):
 q=p/(name+'.ROOT-SIGNED.private.json');assert hashlib.sha256(q.read_bytes()).hexdigest()==DATA[name+'_sha256']
q=p/'supervision'/'original-supervisor-launch.private.json'
assert q.is_file(), 'original reader handle unavailable; never redispatch'
marker=json.loads(q.read_bytes());assert marker['request_file_sha256']==DATA['request_sha256'] and marker['launch_file_sha256']==DATA['launch_sha256']
v['original_supervisor']=marker
q=Path('/proc',str(marker['pid']),'stat')
if q.exists():
 parts=q.read_text().rsplit(')',1)[1].split();v['supervisor_live']=parts[0]not in ('Z','X')and parts[19]==str(marker['ticks'])
else:v['supervisor_live']=False
for name in ('child','terminal','result'):
 q=p/'run'/(name+'.private.json')
 if q.exists():v[name]=json.loads(q.read_bytes())
print(json.dumps(v))
'''
        # Observe once even after expiry; only an already successful original
        # terminal can be adopted. This never extends execution or capabilities.
        while True:
            try:status=self.command(code)
            except (RuntimeError,subprocess.TimeoutExpired):
                # A lost SSH response is observation loss, not child failure.
                # Keep the same request/namespace; next controller retry can
                # resume the same original handle even if this turn is lost.
                if time.time()>=payload['expires_at']:
                    raise TimeoutError('expired original reader observation unavailable; never redispatch')
                time.sleep(min(5,max(.01,payload['expires_at']-time.time())))
                continue
            marker=status['original_supervisor'];markerpath=root/'original-supervisor-launch.private.json'
            if markerpath.exists():
                if canonical(json.loads(markerpath.read_bytes()))!=canonical(marker):
                    raise ValueError('original reader supervisor PID/ticks changed')
            else:write_once(markerpath,marker)
            if (type(marker.get('pid'))is not int or marker['pid']<=0 or
                    not str(marker.get('ticks','')).isdigit() or
                    marker.get('request_file_sha256')!=data['request_sha256'] or
                    marker.get('launch_file_sha256')!=data['launch_sha256']):
                raise ValueError('exact original reader supervisor handle')
            if 'terminal'in status:
                terminal=status['terminal']
                if (terminal.get('actual_child_wait_completed')is not True or
                        terminal.get('exit_code')!=0 or terminal.get('timed_out')is not False or
                        'result'not in status or 'child'not in status):
                    raise ValueError('original independent reader failed; no authority commit')
                if (terminal.get('pid')!=status['child'].get('pid') or
                        terminal.get('ticks')!=status['child'].get('ticks')):
                    raise ValueError('original actual child wait handle changed')
                observed_at=time.time()
                completed_at=reader.verify(status['result'],c['reader_identity'])['completed_at']
                reader.validate_receipt(status['result'],request,self.authority,
                    approved_binding=binding,approved_objects=objects,qualified_reader=c['reader_identity'],now=completed_at)
                if completed_at>observed_at:
                    raise ValueError('original reader completion is in the future')
                evidence=root/'actual-terminal-and-result.private.json'
                if evidence.exists():
                    prior=json.loads(evidence.read_bytes())
                    # Liveness can change after completion; signed receipt,
                    # original marker/child and terminal must remain identical.
                    if any(canonical(prior[k])!=canonical(status[k])for k in ('original_supervisor','child','terminal','result')):
                        raise ValueError('original completed reader evidence changed')
                else:write_once(evidence,status)
                return dict(request_bytes=requestfile.read_bytes(),receipt=status['result'],
                    launch_envelope=json.loads(launchfile.read_text()),terminal=terminal,
                    qualified_reader=c['reader_identity'],reader_host=c['reader_host'],trainer_host=c['trainer_host'],
                    storage_binding={k:binding[k]for k in ('storage_origin','storage_bucket','storage_addressing')},
                    original_child=status['child'],now=time.time())
            if time.time()>=payload['expires_at']:
                raise TimeoutError('original bounded independent reader request expired without successful terminal; never redispatch')
            if status.get('supervisor_live')is not True:
                raise ValueError('original reader no longer live and no actual wait receipt; retain evidence')
            time.sleep(min(5,max(.01,payload['expires_at']-time.time())))
        raise TimeoutError('original bounded independent reader request expired; retain evidence, never redispatch')
