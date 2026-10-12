"""New-source CPU check; exact historical jobs keep their original checker."""
import hashlib,importlib.util,json,sys
from pathlib import Path
C=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
VERSION='prior12-explicit-source-pre-dispatch-v1'
def need(v,m):
 if not v:raise ValueError(m)
def load(ref,name,guards):
 p=Path(ref['path']);need(p.is_absolute()and p.resolve()==p and not p.is_symlink()and guards.file_hash(p)==ref['sha256'],'pinned pre-dispatch helper')
 s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def historical_effective_policy(old, job, row, authority, guards):
 # Reproduce H10 delta.install's authenticated policy projection. The signed
 # H10 document deliberately retains the inherited H5 top-level checker.
 need(old.get('authority') == authority, 'historical preflight authority')
 spec = old.get('independent_training_dispatch')
 need(type(spec) is dict and spec.get('version') == 'basic-eligibility-authenticated-peer-dispatch136-v1', 'exact authenticated H10 dispatch policy')
 transition = load(spec['helper'], 'historical_H10_effective_policy', guards)
 need(transition.FIELD == 'independent_training_dispatch' and transition.VERSION == spec['version'], 'exact H10 transition semantics')
 need(spec['preflight'] == transition.preflight(), 'exact authenticated H10 preflight declaration')
 need(spec['preflight']['helper'] == row['historical_helper'], 'same authenticated historical checker')
 if transition.independent(guards.signed(job['manifest'])):
  return dict(old, nll_pre_dispatch=__import__('copy').deepcopy(spec['preflight']))
 return old

def run(original,result,label,policy,guards):
 row=policy['nll_pre_dispatch'];fields={'version','helper','checker','admission','historical_helper','historical_policy','timeout_seconds'}
 need(type(row)is dict and set(row)==fields and row['version']==VERSION and row['timeout_seconds']==120,'exact prospective preflight closure')
 need(row['helper']=={'path':str(Path(__file__).resolve()),'sha256':guards.file_hash(__file__)},'exact running preflight helper')
 for k in ('checker','admission','historical_helper','historical_policy'):
  p=Path(row[k]['path']);need(p.is_absolute()and p.resolve()==p and not p.is_symlink()and guards.file_hash(p)==row[k]['sha256'],'all preflight references authenticated before import')
 admission=guards.signed(guards.read(row['admission']['path']));need(admission['checker']==row['checker']and admission['authority']==policy['authority'],'signed CPU admission exact checker/authority')
 path=Path(result['job_path']);need(path==Path(original.state)/(result['job_id']+'-job.json')and path.resolve()==path and not path.is_symlink(),'exact original prepared job path')
 record_path=Path(original.state)/(label+'.json');need(record_path.resolve()==record_path and not record_path.is_symlink(),'exact role record')
 need(path.stat().st_size<=32_000_000 and record_path.stat().st_size<=1024**2,'bounded original records')
 raw=path.read_bytes();raw_record=record_path.read_bytes();job=guards.signed(json.loads(raw));record=json.loads(raw_record)
 need(job['role']=='train'and job['job_id']==result['job_id']==record['job_id']and hashlib.sha256(C(job)).hexdigest()==result['job_sha256']==record['job_sha256'],'same original signed request')
 declaration=guards.signed(job['unaudited_training_execution']);source=declaration['training_source_bundle']['sha256']
 legacy=load(row['historical_helper'],'prior12_historical_pre_dispatch',guards)
 if source!=admission['execution_source_sha256']:
  # This is an exact authenticated historic-source lookup, never a broad skip.
  route=load(policy['fp32_training_execution_route'],'prior12_preflight_routes',guards)
  matches=[]
  for entry in policy['live_retained_nll_transition']['historical_training_routes']:
   document,value=route.validate_route(entry,policy['authority'],guards.read,guards.file_hash)
   if source==value['training_source_bundle']['sha256']and job['source_files']==value['execution_source_files']and job['runtime_versions']==value['runtime_versions']and declaration['execution_release_sha256']==hashlib.sha256(C(document)).hexdigest():matches.append(entry)
  need(len(matches)==1,'one exact historical release before original checker')
  old=guards.signed(guards.read(row['historical_policy']['path']))
  effective=historical_effective_policy(old,job,row,policy['authority'],guards)
  return legacy.run(original,result,label,effective,guards)
 need(job['source_files']==admission['source_files']and job['runtime_versions']==admission['runtime_versions'],'candidate exact source/runtime before checker')
 from subnet.unaudited_training_execution import validate
 validate(json.loads(raw),policy['authority'])
 release_document=guards.read(policy['fp32_training_execution_route']['route']['release']['path'])
 need(declaration['execution_release_sha256']==hashlib.sha256(C(release_document)).hexdigest(),'candidate exact signed execution release')
 if not result.get('prepared_only'):
  need(not record.get('prepared_only'),'issued result cannot hide prepared job')
  return {'checked':False,'reason':'already issued authenticated candidate job'}
 need(record.get('prepared_only')is True,'same original unissued job')
 cases=[];total=__import__('time').monotonic()
 for case in ('fixed','foreign-runtime'):
  remaining=row['timeout_seconds']-(__import__('time').monotonic()-total);need(remaining>0,'bounded two-case admission')
  result_case=legacy.checked_child([sys.executable,'-I','-B',row['checker']['path'],'--job',str(path),'--admission',row['admission']['path'],'--authority',policy['authority'],'--case',case],remaining)
  need(result_case.get('case')==case and result_case.get('passed')is True and result_case.get('actual_signed_future_job')is True and result_case.get('exact_actual_job_and_source_validation')is True and result_case.get('CPU_only')is True and result_case.get('GPU_execution')is False and result_case.get('model_loaded')is False and result_case.get('signatures_created')is False and result_case.get('model_factory_reached')is(case=='fixed')and result_case.get('original_job_file_sha256')==hashlib.sha256(raw).hexdigest()and result_case.get('admission_sha256')==row['admission']['sha256'],'complete actual source-to-model-boundary result')
  cases.append(result_case)
 need(path.read_bytes()==raw and record_path.read_bytes()==raw_record and guards.file_hash(row['admission']['path'])==row['admission']['sha256'],'preflight preserves request/record/admission')
 return {'checked':True,'source_sha256':source,'job_file_sha256':hashlib.sha256(raw).hexdigest(),'cases':cases,'GPU_execution':False}
