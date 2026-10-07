"""Explicit composite science admission. Existing strict v2 remains unchanged.
Old GPU evidence remains labeled its original archive; new calibration is separate.
No signatures, service operations, GPU execution, or state promotion.
"""
import ast,hashlib,importlib,sys,types
from pathlib import Path
SOURCE='k2l2-miner-bound-calibration-composite-source-approval-v1'
QUALIFICATION='k2l2-miner-bound-calibration-composite-training-qualification-v1'
POLICY='durable-pinned-k2l2-composite-learner-service-v3'
DELTA={'subnet/backend_jobs.py','subnet/successor_calibration.py'}
TEST_ADDITIONS={'tests/test_v5_calibration_independent_integration.py','tests/test_v5_successor_calibration.py','tests/test_v5_calibration_bootstrap.py'}
CALIBRATION='ROOT-K2L2-v3-current-parent-H200-calibration-attestation-v1'

def exact_delta(before,after,declared):
 if set(after)-set(before)!=TEST_ADDITIONS or set(before)-set(after):raise ValueError('only exact three nonruntime qualification tests added')
 actual={k:{'before':before.get(k),'after':after[k]}for k in after if before.get(k)!=after[k]}
 if set(actual)!=DELTA|TEST_ADDITIONS or declared!=actual:raise ValueError('only exact declared calibration dispatch and nonruntime test delta')
 return actual

def validate_backend_delta(old_path,new_path):
 class MaskCalibration(ast.NodeTransformer):
  def visit_If(self,node):
   text=ast.dump(node.test)
   if "successor_calibration" in text and "evaluate" in text:
    node.body=[ast.Pass()];node.orelse=[self.visit(x)for x in node.orelse];return node
   return self.generic_visit(node)
 def normalized(p):
  return ast.dump(MaskCalibration().visit(ast.parse(Path(p).read_text())),include_attributes=False)
 if normalized(old_path)!=normalized(new_path):raise ValueError('noncalibration backend execution/admission changed')


def original_metadata(row,guards):
 if set(row)!={'path','file_sha256'}or guards.file_hash(row['path'])!=row['file_sha256']:raise ValueError('exact genuine original metadata bytes')
 return guards.read(row['path'])

def validate_current_calibration(attestation,cfg,source,runtime,authority,verify_row,guards,calibration):
 fields={'version','source_sha256','parent_checkpoint','backend_profile','numerical_policy','worker_hardware','original_invocations','no_production_update','no_state_promotion'}
 if set(attestation)!=fields or attestation['version']!=CALIBRATION or attestation['source_sha256']!=source or attestation['no_production_update']is not True or attestation['no_state_promotion']is not True:raise ValueError('new current-parent calibration attestation only')
 cp=cfg['deployment_gate']['expected_checkpoint']
 if attestation['parent_checkpoint']!=cp or guards.digest(attestation['backend_profile'])!=cfg['sampling_policy']['calibration']['backend_profile_sha256']:raise ValueError('current parent/backend profile calibration binding')
 hw=attestation['worker_hardware']
 if set(hw)!={'name','uuid','sm'}or 'H200'not in hw['name']or not hw['uuid'].startswith('GPU-')or hw['sm']!=[9,0]:raise ValueError('actual current-parent H200 qualification')
 invocations=attestation['original_invocations']
 if type(invocations)is not list or len(invocations)<2 or len(invocations)>5:raise ValueError('genuine seed and final confirmation required')
 last=None
 for row in invocations:
  if set(row)!={'job','report'}:raise ValueError('exact original calibration invocation descriptors')
  job=verify_row(row['job'],authority);report=original_metadata(row['report'],guards);m=guards.signed(job['manifest'],authority)
  req=job.get('successor_calibration');calibration.request(req)
  if job['role']!='evaluate'or req['version']!=calibration.MINER_CALIBRATION_VERSION or job['source_files']!=runtime or m['source_bundle']['sha256']!=source or m['checkpoint']['id']!=cp:raise ValueError('genuine new-source current-parent calibration job')
  if report.get('success')is not True or report.get('role')!='evaluate'or report.get('job_id')!=job['job_id']or report.get('operator')!=authority or report.get('job_sha256')!=guards.digest(job)or report.get('source_files')!=runtime or report.get('runtime_versions')!=job['runtime_versions']or report.get('checkpoint')!=cp or report.get('chain_transactions')is not False or report.get('backend_profile')!=attestation['backend_profile']or report.get('numerical_policy')!=attestation['numerical_policy']:raise ValueError('exact genuine new-source original report binding')
  result=report['successor_calibration'];calibration.admitted_policy(result,m,req)
  last=(req,result)
 req,result=last;expected=cfg['sampling_policy']['calibration']
 if req['draw_contract']['calibration']!=expected or expected['checkpoint']!=cp:raise ValueError('final confirmation binds installed current-parent policy')
 if any(x['measured_cdf_abs_error']>expected['cdf_abs_error']or x['measured_logprob_abs_error']>expected['logprob_atol']for x in result['reports']):raise ValueError('actual final confirmation within installed bounds')
 return True

def validate_original(source,qualification,cfg,source_sha256,authority,verify_row,*,guards,strict_admission,source_root):
 # An original04b source approval is evidence only; never a phantom live policy.
 if source.get('version')!=SOURCE or source.get('approved')is not True or source.get('source_sha256')!=source_sha256 or source.get('optimizer_reset')is not False or source.get('historical_relabel')is not False:raise ValueError('explicit new composite source approval')
 extra={'core_source_approval','core_qualification_approval','core_config','calibration_dispatch_delta','current_parent_calibration','core_source_root'}
 ordinary=set(source)-extra
 if ordinary!={'version','approved','source_sha256','optimizer_reset','historical_relabel','full_source_files','runtime_source_files','evidence','predecessor_source_approval','runtime_execution_files','runtime_changes','contract_sha256','predecessor_learner_policy'}or not extra<=set(source):raise ValueError('exact composite source fields')
 core=verify_row(source['core_source_approval'],authority);coreq=verify_row(source['core_qualification_approval'],authority);corecfg=original_metadata(source['core_config'],guards)
 strict_admission.validate(core,coreq,corecfg,core['source_sha256'],authority,verify_row)
 if core['predecessor_learner_policy']!=source['predecessor_learner_policy']or core['predecessor_source_approval']!=source['predecessor_source_approval']:raise ValueError('same authentic actual f213 predecessor, no phantom core-live policy')
 exact_delta(core['full_source_files'],source['full_source_files'],source['calibration_dispatch_delta'])
 for root,files in ((source['core_source_root'],core['full_source_files']),(source_root,source['full_source_files'])):
  guards.pinned_files(root,files)
 validate_backend_delta(Path(source['core_source_root'])/'subnet/backend_jobs.py',Path(source_root)/'subnet/backend_jobs.py')
 if set(core['runtime_source_files'])!=set(source['runtime_source_files'])or source['runtime_execution_files']!=sorted(source['runtime_source_files'])or any(source['full_source_files'].get(k)!=v for k,v in source['runtime_source_files'].items()):raise ValueError('new complete179 runtime inventory')
 old=verify_row(source['predecessor_source_approval'],authority)
 changes={k:{'before':old['runtime_source_files'].get(k),'after':v}for k,v in source['runtime_source_files'].items()if old['runtime_source_files'].get(k)!=v}
 if source['runtime_changes']!=changes or source['contract_sha256']!=guards.digest(strict_admission.contract(cfg)):raise ValueError('new actual runtime/contract delta')
 # Parameter genesis remains exact; production current Adam is not smoke state.
 for k in ('parameters','parameters_sha256','genesis_round','genesis_checkpoint','genesis_sha256'):
  if cfg['persistent_training_admission'].get(k)!=corecfg['persistent_training_admission'].get(k):raise ValueError('original optimizer genesis preserved')
 qfields={'version','approved','candidate_source_sha256','translation_path','translation_file_sha256','core_qualification_sha256','current_parent_calibration_sha256'}
 if set(qualification)!=qfields or qualification['version']!=QUALIFICATION or qualification['approved']is not True or qualification['candidate_source_sha256']!=source_sha256 or qualification['core_qualification_sha256']!=guards.digest(coreq):raise ValueError('explicit composite training qualification')
 parentcal=verify_row(source['current_parent_calibration'],authority)
 if qualification['current_parent_calibration_sha256']!=guards.digest(parentcal):raise ValueError('composite current-parent provenance')
 prefix='_composite_calibration_'+source_sha256[:16]
 pkg=types.ModuleType(prefix);pkg.__path__=[str(Path(source_root)/'subnet')];sys.modules[prefix]=pkg
 calibration=importlib.import_module(prefix+'.successor_calibration')
 if guards.file_hash(Path(calibration.__file__))!=source['runtime_source_files']['subnet/successor_calibration.py']:raise ValueError('new calibration module exact source')
 validate_current_calibration(parentcal,cfg,source_sha256,source['runtime_source_files'],authority,verify_row,guards,calibration)
 return {'source':source_sha256,'core_source':core['source_sha256'],'current_parent':parentcal['parent_checkpoint'],'old_evidence_relabelled':False,'runtime_count':len(source['runtime_source_files'])}

BOOTSTRAP_SOURCE='k2l2-miner-bound-bootstrap-successor-source-approval-v1'
BOOTSTRAP_QUALIFICATION='k2l2-miner-bound-bootstrap-successor-training-qualification-v1'
BOOTSTRAP_POLICY='durable-pinned-k2l2-bootstrap-successor-learner-service-v4'
BOOTSTRAP_TEST='tests/test_v5_mining_bootstrap.py'

def validate_bootstrap_delta(before,after,declared,old_root,new_root):
 if set(after)-set(before)!={BOOTSTRAP_TEST} or set(before)-set(after):raise ValueError('only exact mining bootstrap CPU test added')
 actual={k:{'before':before.get(k),'after':after[k]}for k in after if before.get(k)!=after[k]}
 if set(actual)!={'subnet/backend_jobs.py',BOOTSTRAP_TEST}or declared!=actual:raise ValueError('bootstrap only backend and CPU test delta')
 old=(Path(old_root)/'subnet/backend_jobs.py').read_text()
 new=(Path(new_root)/'subnet/backend_jobs.py').read_text()
 needle="        from .forced_sampling import MINER_VERSION\n        miner_bound=manifest.get('sampling_contract',{}).get('version')==MINER_VERSION"
 replacement="        # Admission runs before authenticated fresh runtime imports.\n        miner_bound=manifest.get('sampling_contract',{}).get('version')=='forced-inverse-cdf-prefill-miner-bound-v5'"
 if old.count(needle)!=1 or old.replace(needle,replacement)!=new:raise ValueError('only exact same version literal before fresh runtime finder')
 return actual

def validate(source,qualification,cfg,source_sha256,authority,verify_row,*,guards,strict_admission,source_root):
 if source.get('version')!=BOOTSTRAP_SOURCE:
  return validate_original(source,qualification,cfg,source_sha256,authority,verify_row,guards=guards,strict_admission=strict_admission,source_root=source_root)
 extra={'bootstrap_predecessor_source','bootstrap_predecessor_qualification','bootstrap_predecessor_config','bootstrap_predecessor_source_root','mining_bootstrap_delta'}
 prior=verify_row(source['bootstrap_predecessor_source'],authority)
 priorq=verify_row(source['bootstrap_predecessor_qualification'],authority)
 priorcfg=original_metadata(source['bootstrap_predecessor_config'],guards)
 if set(source)!=set(prior)|extra or source['approved']is not True or source['source_sha256']!=source_sha256 or source['optimizer_reset']is not False or source['historical_relabel']is not False:raise ValueError('exact bootstrap successor source fields')
 validate_original(prior,priorq,priorcfg,prior['source_sha256'],authority,verify_row,guards=guards,strict_admission=strict_admission,source_root=source['bootstrap_predecessor_source_root'])
 validate_bootstrap_delta(prior['full_source_files'],source['full_source_files'],source['mining_bootstrap_delta'],source['bootstrap_predecessor_source_root'],source_root)
 guards.pinned_files(source_root,source['full_source_files'])
 preserved=set(prior)-{'version','source_sha256','full_source_files','runtime_source_files','runtime_changes','contract_sha256'}
 if any(source[k]!=prior[k]for k in preserved):raise ValueError('original provenance and calibration remain unchanged, not relabelled')
 if set(prior['runtime_source_files'])!=set(source['runtime_source_files'])or any(source['full_source_files'].get(k)!=v for k,v in source['runtime_source_files'].items()):raise ValueError('same complete179 runtime inventory')
 old=verify_row(source['predecessor_source_approval'],authority)
 changes={k:{'before':old['runtime_source_files'].get(k),'after':v}for k,v in source['runtime_source_files'].items()if old['runtime_source_files'].get(k)!=v}
 if source['runtime_changes']!=changes or source['contract_sha256']!=guards.digest(strict_admission.contract(cfg))or strict_admission.contract(cfg)!=strict_admission.contract(priorcfg):raise ValueError('same qualified model and sampling computation contract')
 for k in ('parameters','parameters_sha256','genesis_round','genesis_checkpoint','genesis_sha256'):
  if cfg['persistent_training_admission'].get(k)!=priorcfg['persistent_training_admission'].get(k):raise ValueError('same optimizer lineage')
 expected=dict(priorq,version=BOOTSTRAP_QUALIFICATION,candidate_source_sha256=source_sha256,bootstrap_predecessor_qualification_sha256=guards.digest(priorq))
 if qualification!=expected:raise ValueError('explicit bootstrap qualification preserves original evidence')
 return {'source':source_sha256,'bootstrap_predecessor':prior['source_sha256'],'current_parent':priorcfg['deployment_gate']['expected_checkpoint'],'old_evidence_relabelled':False,'runtime_count':len(source['runtime_source_files'])}
