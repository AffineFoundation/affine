"""CPU-only independent durable-object hashing. Prospective; no authority publication."""
import base64, hashlib, json, re, time
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import urlsplit, unquote
from nacl.signing import VerifyKey
VERSION='independent-optimizer-readback-v2'
BINDING_FIELDS={'purpose','provenance','job_id','job_sha256','source_sha256','namespace','descriptor_sha256','trainer_host_record_sha256','reader_host_record_sha256','storage_origin','storage_bucket','storage_addressing'}
CHUNK=1024*1024
STREAM_BUDGET_VERSION='bounded-independent-readback-streams-v1'
# Includes requests/socket/hash bookkeeping as well as the bounded byte chunk.
STREAM_BUFFER_BYTES=8*CHUNK
DEFAULT_RAM_RESERVE_BYTES=1024**3

def stream_budget(value):
    if (not isinstance(value,dict) or set(value)!={'version','concurrency','ram_reserve_bytes'} or
            value['version']!=STREAM_BUDGET_VERSION or type(value['concurrency'])is not int or
            value['concurrency']not in (4,8) or type(value['ram_reserve_bytes'])is not int or
            not DEFAULT_RAM_RESERVE_BYTES<=value['ram_reserve_bytes']<=64*1024**3):
        raise ValueError('signed independent readback stream budget')
    return dict(value)

def concurrency(request):
    return stream_budget(request['stream_budget'])['concurrency']if 'stream_budget'in request else 4

def admit_stream_resources(budget,available_ram_bytes):
    value=stream_budget(budget)
    required=value['ram_reserve_bytes']+value['concurrency']*STREAM_BUFFER_BYTES
    if type(available_ram_bytes)is not int or available_ram_bytes<required:
        raise ValueError('independent readback available RAM budget')
    return dict(available_ram_bytes=available_ram_bytes,required_ram_bytes=required,
                stream_buffer_bytes=STREAM_BUFFER_BYTES,concurrency=value['concurrency'])

def available_ram_bytes():
    from pathlib import Path
    rows={line.split(':',1)[0]:line.split(':',1)[1].strip()
          for line in Path('/proc/meminfo').read_text().splitlines()}
    value=int(rows['MemAvailable'].split()[0])*1024
    # A container's host MemAvailable can exceed its actual cgroup allowance.
    root=Path('/sys/fs/cgroup')
    if (root/'memory.max').exists():
        limit=(root/'memory.max').read_text().strip()
        if limit!='max':value=min(value,max(0,int(limit)-int((root/'memory.current').read_text())))
    return value

FIELDS={'purpose','provenance','version','job_id','job_sha256','source_sha256','namespace','descriptor_sha256','reader_identity','reader_host_record_sha256','trainer_host_record_sha256','storage_origin','storage_bucket','storage_addressing','created_at','expires_at','max_wall_seconds','objects','capabilities'}
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(v):return hashlib.sha256(canonical(v)).hexdigest()
def verify(envelope,identity):
    if set(envelope)!={'payload','signature','signer'}or envelope['signer']!=identity:raise ValueError('signer')
    VerifyKey(bytes.fromhex(identity)).verify(canonical(envelope['payload']),base64.b64decode(envelope['signature'],validate=True))
    return envelope['payload']
def sign(payload,key):return {'payload':payload,'signature':base64.b64encode(key.sign(canonical(payload)).signature).decode(),'signer':bytes(key.verify_key).hex()}
def validate_request(envelope,authority,*,now,approved_binding,approved_objects,qualified_reader):
    if set(approved_binding)!=BINDING_FIELDS:raise ValueError('independent original binding required')
    r=verify(envelope,authority)
    if set(r)not in (FIELDS,FIELDS|{'stream_budget'}) or r['version']!=VERSION:raise ValueError('request schema')
    if 'stream_budget'in r:stream_budget(r['stream_budget'])
    required_provenance={'production-training-state':{'signed_job_envelope_sha256','original_report_sha256','signed_manifest_envelope_sha256'},'cpu-transport-qualification':{'signed_control_request_envelope_sha256','control_request_payload_sha256','actual_export_evidence_sha256'}}
    if r['purpose'] not in required_provenance or not isinstance(r['provenance'],dict) or set(r['provenance'])!=required_provenance[r['purpose']] or any(not isinstance(v,str) or not re.fullmatch('[0-9a-f]{64}',v) for v in r['provenance'].values()):raise ValueError('exact truthful purpose provenance')
    for k in ['job_sha256','source_sha256','descriptor_sha256','reader_identity','reader_host_record_sha256','trainer_host_record_sha256']:
        if not isinstance(r[k],str)or not re.fullmatch('[0-9a-f]{64}',r[k]):raise ValueError('hash/identity')
    if r['reader_identity']!=qualified_reader or r['reader_host_record_sha256']==r['trainer_host_record_sha256']:raise ValueError('independent qualified reader')
    for k,v in approved_binding.items():
        if r.get(k)!=v:raise ValueError('original binding '+k)
    ns=r['namespace']
    if not isinstance(ns,str)or not ns or any(x in ('','.','..')for x in ns.split('/'))or not re.fullmatch('[a-zA-Z0-9_./-]+',ns):raise ValueError('namespace')
    import math
    if not all(type(r[k])in (int,float) and math.isfinite(r[k]) for k in ['created_at','expires_at'])or not r['created_at']<=now<r['expires_at']:raise ValueError('expired')
    if type(r['max_wall_seconds'])is not int or not 1<=r['max_wall_seconds']<=3500 or r['expires_at']-r['created_at']>3500:raise ValueError('bound')
    if canonical(r['objects'])!=canonical(approved_objects) or len(approved_objects)!=23:raise ValueError('complete original inventory')
    names=[]
    for obj in approved_objects:
        if set(obj)!={'name','size','sha256'}or not re.fullmatch('[a-zA-Z0-9_.-]+',obj['name'])or obj['name']in ('.','..')or type(obj['size'])is not int or not 1<=obj['size']<=4_000_000_000 or not re.fullmatch('[0-9a-f]{64}',obj['sha256']):raise ValueError('object')
        names.append(obj['name'])
    if len(set(names))!=23 or set(r['capabilities'])!=set(names):raise ValueError('capability inventory')
    origin=urlsplit(r['storage_origin'])
    if origin.scheme!='https'or origin.path not in ('','/')or origin.query or origin.fragment or origin.username or origin.password or not origin.hostname or origin.port not in (None,443):raise ValueError('approved storage origin')
    if not re.fullmatch('[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]',r['storage_bucket'])or r['storage_addressing']not in ('path','virtual'):raise ValueError('storage bucket addressing')
    prefix=('/'+r['storage_bucket'])if r['storage_addressing']=='path'else ''
    hosts=set()
    for name,url in r['capabilities'].items():
        u=urlsplit(url)
        if u.scheme!='https'or not u.hostname or u.username or u.password or u.fragment or u.port not in (None,443)or u.netloc!=origin.netloc or unquote(u.path)!=prefix+'/'+ns+'/'+name or not u.query:raise ValueError('exact GET capability namespace')
        hosts.add(u.hostname)
    if len(hosts)!=1:raise ValueError('single approved storage endpoint')
    return r

def execute(envelope,authority,key,*,approved_binding,approved_objects,qualified_reader,read_chunks,clock=time.time):
    start=clock();r=validate_request(envelope,authority,now=start,approved_binding=approved_binding,approved_objects=approved_objects,qualified_reader=qualified_reader)
    if bytes(key.verify_key).hex()!=qualified_reader:raise ValueError('reader signing key')
    if 'stream_budget'in r:admit_stream_resources(r['stream_budget'],available_ram_bytes())
    def check(obj):
        h=hashlib.sha256();count=0
        for part in read_chunks(r['capabilities'][obj['name']]):
            if clock()>=min(r['expires_at'],start+r['max_wall_seconds']):raise TimeoutError('bounded readback')
            if not isinstance(part,bytes)or not 0<len(part)<=CHUNK:raise ValueError('bounded byte chunks')
            count+=len(part)
            if count>obj['size']:raise ValueError('oversize')
            h.update(part)
        if count!=obj['size']or h.hexdigest()!=obj['sha256']:raise ValueError('durable object integrity')
        return dict(obj)
    with ThreadPoolExecutor(max_workers=concurrency(r))as pool:rows=list(pool.map(check,r['objects']))
    end=clock()
    if end>=min(r['expires_at'],start+r['max_wall_seconds']):raise TimeoutError('expired completion')
    return sign(dict(version=VERSION,purpose=r['purpose'],provenance=r['provenance'],request_sha256=sha(r),signed_request_sha256=sha(envelope),job_id=r['job_id'],job_sha256=r['job_sha256'],source_sha256=r['source_sha256'],namespace=r['namespace'],descriptor_sha256=r['descriptor_sha256'],reader_identity=qualified_reader,reader_host_record_sha256=r['reader_host_record_sha256'],trainer_host_record_sha256=r['trainer_host_record_sha256'],objects=rows,total_bytes=sum(x['size']for x in rows),started_at=start,completed_at=end,all_objects_full_hash=True,concurrency=concurrency(r)),key)

def validate_receipt(receipt,request,authority,*,approved_binding,approved_objects,qualified_reader,now):
    r=validate_request(request,authority,now=now,approved_binding=approved_binding,approved_objects=approved_objects,qualified_reader=qualified_reader)
    p=verify(receipt,qualified_reader)
    expected={'version':VERSION,'purpose':r['purpose'],'provenance':r['provenance'],'request_sha256':sha(r),'signed_request_sha256':sha(request),'job_id':r['job_id'],'job_sha256':r['job_sha256'],'source_sha256':r['source_sha256'],'namespace':r['namespace'],'descriptor_sha256':r['descriptor_sha256'],'reader_identity':qualified_reader,'reader_host_record_sha256':r['reader_host_record_sha256'],'trainer_host_record_sha256':r['trainer_host_record_sha256'],'objects':approved_objects,'total_bytes':sum(o['size']for o in approved_objects),'all_objects_full_hash':True,'concurrency':concurrency(r)}
    if set(p)!=set(expected)|{'started_at','completed_at'}or any(canonical(p[k])!=canonical(v) for k,v in expected.items()):raise ValueError('complete request-bound receipt')
    import math
    if not all(type(p[k])in (int,float) and math.isfinite(p[k]) for k in ['started_at','completed_at'])or not r['created_at']<=p['started_at']<=p['completed_at']<=now<r['expires_at']or p['completed_at']-p['started_at']>r['max_wall_seconds']:raise ValueError('receipt time')
    return p
