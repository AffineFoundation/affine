from pathlib import Path
import importlib.util,json,time,signal,os,hashlib,threading,traceback
O=Path(__file__).parent;P=O;A='3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd';os.umask(0o077)
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def expire(*_):raise TimeoutError('readonly qualification720s')
signal.signal(signal.SIGALRM,expire);signal.alarm(720)
s=importlib.util.spec_from_file_location('final_signed_candidate',P/'hourly_population_index_entry_v6.py');h=importlib.util.module_from_spec(s);s.loader.exec_module(h);doc=json.loads((P/'HOST-COMPOSITION.ROOT-SIGNED.private.json').read_text());p,_,c=h.prepare(doc,A);writer=c['writer'];policy=writer.validate_policy(c['parent_policy'],A,c['cutover'],c['anchor'])
summary=dict(cutoff=1791986400,python_threads=threading.active_count(),os_threads=len(list(Path('/proc/self/task').iterdir())),entry_sha256=p['entry']['sha256'],helper_sha256=p['parallel_module']['sha256'],signed_composition_sha256=hashlib.sha256(canonical(doc)).hexdigest(),chain_calls=0,writer_calls=0,cursor_writes=0,budget_seconds=600)
assert summary['python_threads']==summary['os_threads']==1
kw={}
if 'authenticated_admission_cache'in policy:
 from ops.current_assessment_cache import factory
 kw['admission_cache_factory']=factory(policy['authenticated_admission_cache'],A)
if 'compact_queue_evidence'in policy:kw['compact_queue_evidence']=policy['compact_queue_evidence']
t=time.monotonic();print(json.dumps(dict(event='started',**summary)),flush=True)
try:
 with writer.evidence_budget():
  result=c['evidence_loader'](policy['audit_config'],authority=A,cutoff=1791986400,verifiers=policy['verifiers'],expected_source_admission_sha256=policy['source_admission_sha256'],expected_numerical_resolution_policy_sha256=policy['numerical_resolution_policy_sha256'],**kw)
  summary['evidence_seconds']=time.monotonic()-t
  fresh=writer.calculate(result['snapshots'],result['committed_at_by_epoch'],1791986400)
  fresh.update(evidence_cutoff=1791986400,assessment_stale=False,evidence_hashes=result['evidence_hashes'],evidence_refusals=result.get('refused',[]),evidence_exclusions=result.get('excluded',[]),evidence_deferrals=result.get('deferred',[]));writer.validate_assessment(fresh,policy,1791986400)
 summary.update(completed=True,snapshots=len(result['snapshots']),private_assessment_validated=True,evidence_sha256=hashlib.sha256(canonical(result)).hexdigest(),recipients=len(fresh['weights']),mimo_populations=len(result['evidence_hashes']['mimo_population_documents']))
 (O/'FINAL-V5-EVIDENCE.private.json').write_bytes(canonical(result));(O/'FINAL-V5-ASSESSMENT.private.json').write_bytes(canonical(fresh))
except BaseException as e:summary.update(completed=False,error=type(e).__name__,detail=str(e),traceback=traceback.format_exc());raise
finally:
 summary['total_seconds']=time.monotonic()-t;(O/'FINAL-V5-QUALIFICATION.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
