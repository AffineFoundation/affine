"""Logging-only equivalence against the exact qualified pre-instrumentation AST.

Reference source SHA256:4ba6373cbbb15e996603e57108b833a42e4b4e3d79669a68be14773fc8856421.
Removing only recorder calls must recover that entire unchanged AST; the CPU
control executes both original and instrumented sequence paths and compares
losses, gradients, Adam moments, updates and forward counts exactly.
"""
import ast,copy,hashlib,importlib.util,json,math,sys,types,unittest
from pathlib import Path
import torch
D=Path(__file__).resolve().parents[1]
# Keep this CPU-only suite bounded without changing global interop settings.
torch.set_num_threads(2)
C=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
H=lambda x:hashlib.sha256(C(x)).hexdigest()
spec=importlib.util.spec_from_file_location('isolated_recorder',D/'subnet/sequence_component_diagnostics.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
Recorder=mod.SequenceComponentDiagnostics


class Strip(ast.NodeTransformer):
 def visit_ImportFrom(self,node):
  if node.module=='sequence_component_diagnostics':return None
  return node
 def visit_Assign(self,node):
  if any(isinstance(t,ast.Name)and t.id in('sequence_diagnostics','mean')for t in node.targets):return None
  if any(isinstance(t,ast.Subscript)and isinstance(t.value,ast.Name)and t.value.id=='diagnostics'and isinstance(t.slice,ast.Constant)and t.slice.value=='sequence_components'for t in node.targets):return None
  return self.generic_visit(node)
 def visit_Expr(self,node):
  if isinstance(node.value,ast.Call)and isinstance(node.value.func,ast.Attribute)and isinstance(node.value.func.value,ast.Name)and node.value.func.value.id=='sequence_diagnostics':return None
  return self.generic_visit(node)
 def visit_Return(self,node):
  if isinstance(node.value,ast.Name)and node.value.id=='mean':return ast.Return(value=ast.BinOp(left=ast.Name(id='total',ctx=ast.Load()),op=ast.Div(),right=ast.Name(id='tokens',ctx=ast.Load())))
  return self.generic_visit(node)

REFERENCE_AST_SHA256 = '5bde839dfcf561521b23aadd195430f8ac7434b0daff2cc5b79122d891ac25b1'

def trainer_tree(kind):
 tree=ast.parse((D/"subnet/task_normalized_training.py").read_text())
 if kind=="baseline":
  tree=Strip().visit(tree)
  assert hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()==REFERENCE_AST_SHA256
  ast.fix_missing_locations(tree)
 return tree

def functions(path,names,env):
 tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in names];assert {n.name for n in nodes}==set(names)
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)

class CPUFacade:
 def __getattr__(self,name):return getattr(torch,name)
 def tensor(self,*a,**kw):
  if kw.get('device')=='cuda':kw['device']='cpu'
  return torch.tensor(*a,**kw)

class Model(torch.nn.Module):
 def __init__(self,dtype):
  super().__init__();self.embedding=torch.nn.Embedding(11,7,dtype=dtype);self.head=torch.nn.Linear(7,11,dtype=dtype);self.calls=0
 def forward(self,tokens,use_cache=False):
  assert use_cache is False;self.calls+=1;return types.SimpleNamespace(logits=self.head(self.embedding(tokens)))

def pairs():
 result=[]
 for i in range(4):
  task=0 if i<3 else 1
  def rollout(role):
   outputs=([2,3+i,4]if role=='positive'else[6+i,4])
   turns=[dict(prompt=[1,2],output=outputs,done=True)]
   if i==1:turns.append(dict(prompt=[1,4,2],output=[3,2],done=True))
   return dict(classification=role,env_id='test',index=task,task_hash=H(task),turns=turns)
  result.append((dict(env_id='test',spec={}),rollout('positive'),rollout('negative')))
 return result

def environment(kind,dtype=torch.float32,positive_nll=0,maxbytes=mod.MAX_DOCUMENT_BYTES):
 root=D/'subnet';env=dict(hashlib=hashlib,math=math,canonical=C,checkpoint_id=lambda x:None)
 functions(root/'covered_epoch_optimizer.py',['pair_identity','distinct_verified_pairs'],env)
 functions(root/'epoch_optimizer.py',['preference_loss'],env)
 functions(root/'task_normalized_training.py',['task_groups','validate_positive_nll_weight','positive_nll_loss','capture_components','accumulate_tasks'],env)
 pp,tt,gg,ids=env['task_groups'](pairs(),1,'0'*64)
 torch.manual_seed(711);model=Model(dtype)
 recorder=Recorder(pp,tt,ids,max_document_bytes=maxbytes)
 env.update(torch=CPUFacade(),model=model,sequence_diagnostics=recorder,pairs=pp,runtime=types.SimpleNamespace(configure=lambda *a:None),harness_for=lambda *a:{})
 parent=next(n for n in trainer_tree(kind).body if isinstance(n,ast.FunctionDef)and n.name=='train_epoch')
 nodes=[n for n in parent.body if isinstance(n,ast.FunctionDef)and n.name in('sequence','margin','components')]
 assert len(nodes)==3
 exec(compile(ast.Module(body=nodes,type_ignores=[]),str(root/'task_normalized_training.py'),'exec'),env)
 if kind=='candidate':recorder.start('before')
 if positive_nll:
  before=env['capture_components'](torch,env['components'],len(pp));refs=[x['margin']for x in before]
 else:
  with torch.no_grad():refs=[float(env['margin'](i))for i in range(len(pp))]
 if kind=='candidate':recorder.stop()
 optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4,betas=(.9,.999),eps=1e-8,weight_decay=.01,foreach=False)
 obs=env['accumulate_tasks'](torch,env['margin'],refs,tt,gg[0],positive_nll_weight=positive_nll,components=env['components']if positive_nll else None)
 loss=sum(x['loss']*x['gradient_weight']for x in obs)
 grads=[p.grad.detach().clone()for p in model.parameters()];optimizer.step();weights=[p.detach().clone()for p in model.parameters()]
 if kind=='candidate':recorder.start('after')
 if positive_nll:
  after=env['capture_components'](torch,env['components'],len(pp),references=refs);post=[x['margin']for x in after]
 else:
  with torch.no_grad():post=[float(env['margin'](i))for i in range(len(pp))]
 if kind=='candidate':recorder.stop()
 update={'pairs':[dict(x,pair_sha256=ids[x['pair_index']])for x in obs]}
 report=recorder.report(refs,post,[update],input_checkpoint='b'*64,global_step_before=71,global_step_after=72)if kind=='candidate'else None
 return dict(refs=refs,post=post,loss=loss,obs=obs,grads=grads,weights=weights,state=optimizer.state_dict(),calls=model.calls,report=report,pairs=pp,tasks=tt,identities=ids)

class Controls(unittest.TestCase):
 def equivalent(self,dtype=torch.float32,positive_nll=0,maxbytes=mod.MAX_DOCUMENT_BYTES):
  a=environment('baseline',dtype,positive_nll,maxbytes);b=environment('candidate',dtype,positive_nll,maxbytes)
  for k in('refs','post','loss','obs','calls'):self.assertEqual(a[k],b[k],k)
  self.assertEqual(a['calls'],30) # five positive turns + five negative, reference/gradient/post
  for k in('grads','weights'):
   for x,y in zip(a[k],b[k]):self.assertTrue(torch.equal(x,y),k)
  self.assertEqual(a['state']['param_groups'],b['state']['param_groups'])
  for i,x in a['state']['state'].items():
   for k,v in x.items():self.assertTrue(torch.equal(v,b['state']['state'][i][k]),k)
  return b
 def test_fp32_exact_gradient_and_adam_equivalence(self):self.assertEqual(self.equivalent()['report']['status'],'complete')
 def test_bf16_exact_gradient_and_adam_equivalence(self):self.assertEqual(self.equivalent(torch.bfloat16)['report']['status'],'complete')
 def test_unit_NLL_branch_equivalence(self):self.assertEqual(self.equivalent(positive_nll=1)['report']['status'],'complete')
 def test_budget_failure_does_not_change_update(self):
  r=self.equivalent(maxbytes=1)['report'];self.assertEqual(r['status'],'unavailable');self.assertEqual(r['reason'],'document_byte_budget')
 def test_sums_lengths_raw_margins_and_additive_report(self):
  r=self.equivalent()['report'];self.assertEqual(r['extra_model_forward_passes'],0);self.assertFalse(r['raw_margin_beta_scaled']);self.assertFalse(r['objective_changed']);self.assertEqual(r['pair_count'],4)
  self.assertEqual(r['rows'][1]['positive']['output_tokens'],5);self.assertEqual(r['rows'][1]['positive']['prompt_tokens'],5);self.assertEqual(r['rows'][1]['negative']['output_tokens'],4)
  for row in r['rows']:
   for phase in('before','after'):
    for role in('positive','negative'):
     v=row[phase][role];n=row[role]['output_tokens'];self.assertAlmostEqual(v['total_logprob']/n,v['mean_logprob'],places=5)
    self.assertAlmostEqual(row[phase]['positive']['mean_logprob']-row[phase]['negative']['mean_logprob'],row[phase]['raw_margin'],places=6)
  self.assertAlmostEqual(sum(x['total_gradient_weight']for x in r['rows']),1)
  self.assertEqual([x['total_gradient_weight']for x in r['rows']],[1/6,1/6,1/6,1/2])
  encoded=json.dumps(r,allow_nan=False);self.assertLess(len(encoded),10000)
  self.assertNotIn('tensor(',encoded);self.assertNotIn('"output":',encoded);self.assertNotIn('"prompt":',encoded)
 def new_recorder(self,**kw):
  e=environment('candidate');return Recorder(e['pairs'],e['tasks'],e['identities'],**kw),e
 def test_inactive_gradient_path_does_not_read_scalars(self):
  rec,e=self.new_recorder()
  class Bomb:
   def detach(self):raise AssertionError('must not touch gradient scalar')
  rec.begin_pair(0);rec.observe(Bomb(),Bomb(),0);self.assertIsNone(rec.error)
 def test_duplicate_pair_disables_only_diagnostics(self):
  rec,e=self.new_recorder();rec.start('before');rec.begin_pair(0);rec.begin_pair(0);self.assertEqual(rec.error,'pair_order_invalid')
 def test_missing_pair_reports_unavailable(self):
  rec,e=self.new_recorder();rec.start('before');rec.stop();self.assertEqual(rec.error,'incomplete_phase')
 def test_nonfinite_and_wrong_count_do_not_raise(self):
  for total,tokens in((float('nan'),3),(-1.,999)):
   rec,e=self.new_recorder();rec.start('before');rec.begin_pair(0);rec.observe(torch.tensor(total),torch.tensor(-1.),tokens);self.assertEqual(rec.error,'component_unavailable')
 def test_pair_budget_disables_reporting_not_training(self):
  rec,e=self.new_recorder(max_pairs=1);self.assertEqual(rec.error,'metadata_unavailable');rec.start('before');rec.begin_pair(0);self.assertIsNone(rec.phase)
 def test_no_freeform_secrets_or_error_text(self):
  rec,e=self.new_recorder();rec.start('before');rec.begin_pair(0)
  class Secret:
   def detach(self):raise ValueError('SECRET_DO_NOT_PUBLISH')
  rec.observe(Secret(),Secret(),3);r=rec.report([],[],[],input_checkpoint='b'*64,global_step_before=71,global_step_after=72);self.assertNotIn('SECRET',json.dumps(r));self.assertEqual(set(r),{'version','status','reason','extra_model_forward_passes','objective_changed','output_checkpoint_binding'})
 def test_full_training_AST_only_contains_diagnostic_additions(self):
  before=trainer_tree('baseline');after=Strip().visit(trainer_tree('candidate'))
  self.assertEqual(ast.dump(before,include_attributes=False),ast.dump(after,include_attributes=False))
 def test_current_2048_pair_report_is_bounded(self):
  count=2048;pp=[];tasks=[];ids=[]
  for i in range(count):
   z=(dict(env_id='e'),dict(turns=[dict(prompt=[1],output=[2,3],done=True)]),dict(turns=[dict(prompt=[1],output=[4],done=True)]))
   pp.append(z);ids.append(H(('pair',i)))
   if i%4==0:tasks.append(dict(task_sha256=H(('task',i//4)),pair_indices=list(range(i,i+4))))
  rec=Recorder(pp,tasks,ids)
  class Scalar:
   def __init__(self,v):self.v=v
   def detach(self):return self.v
  for phase in('before','after'):
   rec.start(phase)
   for i in range(count):
    rec.begin_pair(i);rec.observe(Scalar(-4.),Scalar(-2.),2);rec.observe(Scalar(-3.),Scalar(-3.),1)
   rec.stop()
  updates=[dict(pairs=[dict(pair_index=i,pair_sha256=ids[i],task_sha256=H(('task',i//4)),gradient_weight=1/count)for i in range(count)])]
  r=rec.report([1.]*count,[1.]*count,updates,input_checkpoint='c'*64,global_step_before=71,global_step_after=72)
  self.assertEqual(r['status'],'complete');self.assertEqual(len(r['rows']),count);self.assertLess(len(C(r)),3*1024*1024)
  updates[0]['pairs'][0]['pair_sha256']='0'*64
  bad=rec.report([1.]*count,[1.]*count,updates,input_checkpoint='c'*64,global_step_before=71,global_step_after=72)
  self.assertEqual(bad['status'],'unavailable')
 def test_scientific_functions_unchanged(self):
  names=('task_groups','accumulate_tasks','positive_nll_loss','capture_components','weighted_component_summary')
  defs=[]
  for kind in('baseline','candidate'):
   tree=trainer_tree(kind);defs.append({n.name:ast.dump(n,include_attributes=False)for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in names})
  self.assertEqual(defs[0],defs[1]);self.assertEqual(set(defs[0]),set(names))

if __name__=='__main__':unittest.main(verbosity=2)
