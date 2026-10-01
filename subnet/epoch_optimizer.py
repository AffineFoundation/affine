"""Prospective full BF16 preference updates with a fixed epoch reference.

Not selected by existing workers. Each pair is already fully model/environment
verified; this module never accepts an unverified uploaded training view.
"""
from pathlib import Path
import math

POLICY='bf16-full-adamw-fixed-epoch-reference-v2'

def preference_loss(torch,margin,reference,beta=.1):
    if type(reference) not in (int,float) or not math.isfinite(reference) or not 0<beta<=1:
        raise ValueError('fixed reference policy')
    return -torch.nn.functional.logsigmoid(beta*(margin-reference))

def train_epoch(runtime,verified_pairs,destination_root,steps=3):
    import gc
    import torch
    from .backend_jobs import pair_attribution
    if not verified_pairs or type(steps) is not int or not 1<=steps<=32:raise ValueError('verified epoch pair budget')
    model=runtime.model
    if not next(model.parameters()).is_cuda:raise ValueError('approved CUDA optimizer required')
    if any(isinstance(m,torch.nn.Dropout) and m.p>0 for m in model.modules()) or getattr(model.config,'attention_dropout',0)!=0:
        raise ValueError('dropout-free fixed reference')
    parameters=list(model.parameters());count=sum(p.numel() for p in parameters)
    if any(p.dtype!=torch.bfloat16 for p in parameters):raise ValueError('BF16 parameter profile')
    free,total=torch.cuda.mem_get_info();required=count*6+3*1024**3
    if free<required:raise ValueError('persistent full optimizer GPU reserve')
    def sequence(rollout):
        total_lp=0;tokens=0
        for turn in rollout['turns']:
            prompt,output=turn['prompt'],turn['output']
            logits=model(torch.tensor([prompt+output],device='cuda'),use_cache=False).logits[0,len(prompt)-1:len(prompt)+len(output)-1]
            total_lp=total_lp+torch.log_softmax(logits.float(),-1).gather(1,torch.tensor(output,device='cuda')[:,None]).sum();tokens+=len(output)
        if not tokens:raise ValueError('empty approved training trajectory')
        return total_lp/tokens
    # Capture EVERY reference from the one input checkpoint before ANY update.
    references=[]
    model.eval()
    with torch.no_grad():
        for definition,pos,neg in verified_pairs:
            runtime.configure(definition['spec'],definition['harness'])
            reference=float(sequence(pos)-sequence(neg))
            if not math.isfinite(reference):raise ValueError('nonfinite initial reference')
            references.append(reference)
    for param in parameters:param.requires_grad_(True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.train()
    optimizer=torch.optim.AdamW(parameters,lr=1e-5,foreach=False)
    updates=[];destination=None;torch.cuda.reset_peak_memory_stats()
    try:
        for step in range(steps):
            pair_index=step%len(verified_pairs);definition,pos,neg=verified_pairs[pair_index]
            runtime.configure(definition['spec'],definition['harness'])
            optimizer.zero_grad(set_to_none=True)
            margin=sequence(pos)-sequence(neg);loss=preference_loss(torch,margin,references[pair_index])
            if not torch.isfinite(loss):raise ValueError('nonfinite epoch preference loss')
            before=float(margin.detach());loss.backward()
            gradient_tensors=sum(p.grad is not None for p in parameters)
            if gradient_tensors!=len(parameters):raise ValueError('full epoch optimizer gradient coverage')
            torch.nn.utils.clip_grad_norm_(parameters,1);optimizer.step()
            destination=Path(destination_root)/('checkpoint-step-'+str(step+1))
            if destination.exists():raise ValueError('refuse checkpoint overwrite')
            destination.mkdir(parents=True);model.save_pretrained(destination,safe_serialization=True);runtime.tokenizer.save_pretrained(destination)
            state_steps=sorted({int(row['step'].item()) for row in optimizer.state.values() if 'step' in row})
            if len(optimizer.state)!=len(parameters) or state_steps!=[step+1]:raise ValueError('persistent epoch optimizer state coverage')
            update=dict(steps=1,losses=[float(loss.detach())],training_policy=POLICY,objective='fixed-input-checkpoint reference-relative sequence preference',
                full_model_finetune=True,trainable_parameters=count,learning_rate=1e-5,beta=.1,parameter_dtype='torch.bfloat16',gradient_checkpointing=True,
                reference_margin=references[pair_index],reference_pair_index=pair_index,reference_scope='immutable-epoch-input-before-all-updates',margin_before=before,reference_deviation=before-references[pair_index],
                gradient_tensors=gradient_tensors,total_parameter_tensors=len(parameters),
                optimizer_lifecycle='one-AdamW-instance-all-epoch-steps',optimizer_state_steps=state_steps,
                optimizer_state_dtypes=sorted({str(v.dtype) for row in optimizer.state.values() for k,v in row.items() if k!='step' and hasattr(v,'dtype')}),
                gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),gpu_free_before_bytes=free,gpu_required_additional_bytes=required)
            update.update(pair_attribution(definition,pos,neg,step));updates.append(update)
        return destination,updates
    finally:
        optimizer.zero_grad(set_to_none=True);del optimizer;model.eval();model.gradient_checkpointing_disable();gc.collect();torch.cuda.empty_cache()
