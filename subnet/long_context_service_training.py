"""Full-parameter preference gradients with one live turn graph at a time."""
import hashlib,math
REVISION='bf16-full-sequential-agent-turn-dpo-mean-v1'
POLICY={'revision':REVISION,'objective':'reference-relative-all-agent-token-mean-preference',
        'parameters':'bfloat16','gradients':'bfloat16','adam_moments':'bfloat16',
        'master_parameters':False,'gradient_checkpointing':True,'use_cache':False,
        'turn_gradient_accumulation':'sequential-before-single-optimizer-step',
        'learning_rate':5e-5,'beta':.1,'weight_decay':0.,'eps':1e-8,'clip_norm':1.,
        'max_peak_allocated_bytes':8*1024**3,'max_parameters':600000000}

def output_count(rollout):
 turns=rollout.get('turns',[])
 if not turns or any(t.get('model_role','agent')!='agent' or not isinstance(t.get('output'),list) or not t['output'] for t in turns):raise ValueError('all-agent output tokens required')
 return sum(len(t['output']) for t in turns)

def turn_mean(runtime,turn):
 from .long_context_training import sequence_logprob
 return sequence_logprob(runtime,turn['prompt'],turn['output'])

def rollout_mean(runtime,rollout):
 count=output_count(rollout);total=0
 for turn in rollout['turns']:total=total+turn_mean(runtime,turn)*len(turn['output'])/count
 return total

def accumulate_pair(runtime,pos,neg,coefficient):
 """dL/dMargin is evaluated at unchanged weights, then each turn is freed."""
 counts=[output_count(pos),output_count(neg)]
 for sign,roll,count in zip((1.,-1.),(pos,neg),counts):
  for turn in roll['turns']:
   term=turn_mean(runtime,turn)*(coefficient*sign*len(turn['output'])/count)
   term.backward();del term

def parameter_sha(param):
 raw=param.detach().cpu().contiguous().view(__import__('torch').uint8).numpy()
 return hashlib.sha256(memoryview(raw)).hexdigest()

def full_parameter_train(runtime,pairs,steps=1,lr=5e-5,beta=.1):
 import gc,torch
 from .long_context_training import configure_full_training
 if not pairs or type(steps) is not int or not 1<=steps<=32 or lr!=POLICY['learning_rate'] or beta!=POLICY['beta']:raise ValueError('signed sequential training policy/steps')
 parameters=list(runtime.model.parameters());count=sum(p.numel() for p in parameters)
 if count>POLICY['max_parameters'] or any(p.dtype!=torch.bfloat16 for p in parameters):raise ValueError('full-model BF16 parameter policy')
 free,total=torch.cuda.mem_get_info();required=count*6+3*1024**3
 torch.cuda.set_per_process_memory_fraction(POLICY['max_peak_allocated_bytes']/total)
 if free<required:raise ValueError('full optimizer additional VRAM reserve')
 references=[]
 with torch.no_grad():
  for pos,neg in pairs:references.append(float(rollout_mean(runtime,pos)-rollout_mean(runtime,neg)))
 before=[parameter_sha(p) for p in parameters];configure_full_training(runtime)
 optimizer=torch.optim.AdamW(parameters,lr=lr,betas=(.9,.999),weight_decay=0.,eps=1e-8,foreach=False);losses=[];norms=[];margins=[];torch.cuda.reset_peak_memory_stats()
 try:
  for step in range(steps):
   pos,neg=pairs[step%len(pairs)];optimizer.zero_grad(set_to_none=True)
   with torch.no_grad():margin=float(rollout_mean(runtime,pos)-rollout_mean(runtime,neg))-references[step%len(pairs)]
   if not math.isfinite(margin):raise ValueError('nonfinite preference margin')
   scalar=torch.tensor(margin,device='cuda',dtype=torch.float32);loss=-torch.nn.functional.logsigmoid(beta*scalar)
   coefficient=-beta*float(torch.sigmoid(-beta*scalar))
   accumulate_pair(runtime,pos,neg,coefficient)
   if any(p.grad is None or not torch.isfinite(p.grad).all() for p in parameters):raise ValueError('missing/nonfinite full gradient')
   norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.))
   if not math.isfinite(norm) or norm<=0:raise ValueError('zero/nonfinite full gradient norm')
   optimizer.step();losses.append(float(loss));norms.append(norm);margins.append(margin)
   if torch.cuda.max_memory_allocated()>POLICY['max_peak_allocated_bytes']:raise ValueError('sequential training memory cap')
  dtypes=sorted({str(v.dtype) for state in optimizer.state.values() for k,v in state.items() if k!='step' and hasattr(v,'dtype')})
  if dtypes!=['torch.bfloat16']:raise ValueError('BF16 Adam moment policy')
  changed=sum(parameter_sha(p)!=sha for p,sha in zip(parameters,before))
  if not changed:raise ValueError('optimizer did not change checkpoint parameters')
  return {'steps':steps,'losses':losses,'reference_margins':references,'before_step_reference_relative_margins':margins,'gradient_norms':norms,
          'weights_changed':True,'gradient_tensors':len(parameters),'changed_parameter_tensors':changed,'full_model_finetune':True,'trainable_parameters':count,
          'training_policy':POLICY,'objective':POLICY['objective'],'learning_rate':lr,'beta':beta,'gradient_checkpointing':True,
          'parameter_dtype':'torch.bfloat16','optimizer_state_dtypes':dtypes,'auxiliary_tokens_in_loss':False,
          'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved(),
          'gpu_free_before_bytes':free,'gpu_required_additional_bytes':required,'quality_improvement_claimed':False}
 finally:
  optimizer.zero_grad(set_to_none=True);del optimizer;runtime.model.eval();runtime.model.gradient_checkpointing_disable();gc.collect();torch.cuda.empty_cache()
