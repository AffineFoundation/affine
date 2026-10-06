"""CPU-only source-specific sampler assertions; never executes model verification."""
import hashlib,importlib,json,re,secrets,sys,types
from pathlib import Path
from .distributed_roles import authenticate,digest
VERSION='source-specific-sampling-api-admission-v1'

def filehash(path):
 if path.absolute()!=path.resolve()or not path.is_file()or path.stat().st_nlink!=1:raise ValueError('owned exact sampler source file')
 return hashlib.sha256(path.read_bytes()).hexdigest()

class SamplingAdmission:
 def __init__(self,document,authority,source_trees):
  v=authenticate(document,authority)
  if set(v)!={'version','sources'}or v['version']!=VERSION or type(v['sources'])is not dict or not v['sources']:raise ValueError('explicit source-specific API sampler registry')
  if set(v['sources'])!=set(source_trees):raise ValueError('exact admitted source tree set')
  self.rows={};self.authority=authority
  for source,row in v['sources'].items():
   if re.fullmatch('[0-9a-f]{64}',source)is None or type(row)is not dict or set(row)!={'runtime_files','runtime_versions','sampling_versions'}:raise ValueError('exact source-aware sampler admission')
   files=row['runtime_files'];versions=row['sampling_versions']
   if type(files)is not dict or not 1<=len(files)<=256 or type(row['runtime_versions'])is not dict or type(versions)is not list or not versions or len(set(versions))!=len(versions):raise ValueError('bounded source-specific runtime closure')
   tree=Path(source_trees[source]).absolute()
   if tree!=tree.resolve():raise ValueError('source tree symlink')
   for name,h in files.items():
    if not name.startswith('subnet/')or Path(name).is_absolute()or '..'in Path(name).parts or re.fullmatch('[0-9a-f]{64}',h)is None or filehash(tree/name)!=h:raise ValueError('full admitted source runtime hash')
   for name in ('forced_sampling','harness','audit_policy'):
    if 'subnet/'+name+'.py'not in files:raise ValueError('complete CPU sampler semantic closure')
   prefix='_api_sampling_'+source[:16]+'_'+secrets.token_hex(8);package=types.ModuleType(prefix);package.__path__=[str(tree/'subnet')];sys.modules[prefix]=package
   sampler=importlib.import_module(prefix+'.forced_sampling')
   fast=importlib.import_module(prefix+'.fast_prefill_audit')if 'subnet/fast_prefill_audit.py'in files else None
   supported=[sampler.VERSION,None]+([fast.VERSION,fast.SUPPORT_VERSION]+([fast.THREEWAY_VERSION]if hasattr(fast,'THREEWAY_VERSION')else [])if fast else [])
   if any(x not in supported for x in versions):raise ValueError('sampler version unsupported by exact source')
   if fast and hasattr(fast,'THREEWAY_VERSION')and fast.THREEWAY_VERSION in versions and len(files)!=177:raise ValueError('v4 exact177 admitted runtime map')
   self.rows[source]=(row,tree,sampler,fast)
 def check(self,job):
  m=authenticate(job['manifest'],self.authority);source=m.get('source_bundle',{}).get('sha256')
  if source not in self.rows:raise ValueError('unadmitted execution source sampler')
  row,tree,sampler,fast=self.rows[source]
  if job.get('source_files')!=row['runtime_files']or job.get('runtime_versions')!=row['runtime_versions']:raise ValueError('exact source runtime metadata')
  for name in ('forced_sampling','fast_prefill_audit','harness','audit_policy'):
   n='subnet/'+name+'.py'
   if n not in row['runtime_files']:continue
   if filehash(tree/n)!=row['runtime_files'][n]:raise ValueError('admitted sampler closure changed')
  contract=m.get('sampling_contract');version=contract.get('version')if isinstance(contract,dict)else None
  if version not in row['sampling_versions']:raise ValueError('sampling contract version not admitted for original source')
  context=sampler.binding(m)
  if context and version!=sampler.VERSION:
   if fast is None or not m.get('environments'):raise ValueError('complete fast sampler environment/calibration closure')
   for e in m['environments']:
    harness=e.get('harness')or e.get('config')
    if harness is None:raise ValueError('signed environment harness needed for sampler calibration')
    sampler.validate_harness(harness,contract)
    try:fast.bind(m,harness)
    except fast.CalibrationRequired as error:raise ValueError('signed source calibration binding')from error
  return m,sampler,fast
 def report(self,job,report):
  m,sampler,fast=self.check(job)
  for audit in report.get('audits',[]):
   sampler.require_report(m,audit)
   if fast and hasattr(fast,'THREEWAY_VERSION')and m.get('sampling_contract',{}).get('version')==fast.THREEWAY_VERSION:
    for outcome in audit.get('outcomes',[]):
     if outcome.get('failure_kind')=='numerical_ambiguous':
      if outcome.get('valid')is not None or outcome.get('fully_audited')is not False or outcome.get('sampling_verification_complete')is not False or outcome.get('environment_verification_complete')is not False:raise ValueError('v4 unknown cannot be accepted or confirmed invalid')
      positions=outcome.get('uncertain_token_positions');count=outcome.get('uncertain_token_position_count')
      if positions is not None or count is not None:
       if type(positions)is not list or len(positions)>128 or any(type(p)is not int or p<0 for p in positions)or type(count)is not int or not len(positions)<=count<=65536:raise ValueError('bounded v4 uncertain token metadata')
      if audit.get('accepted'):raise ValueError('unknown selected child cannot receive accepted credit')
  return True

def guarded_coordinator(base,gate):
 """Wrap both admission paths while preserving the original queue implementation."""
 class SourceAwareCoordinator(base):
  def enqueue(self,envelope):gate.check(authenticate(envelope,self.authority));return super().enqueue(envelope)
  def validate_report(self,report,job,manifest,job_digest):
   gate.report(job,report);return super().validate_report(report,job,manifest,job_digest)
 return SourceAwareCoordinator
