"""Differentiable selective-head contract; isolated from deployed trainers."""
from .long_context_runtime import validate_tokens,prediction_rows

REVISION='cuda-bf16-sdpa-flash-selective-head-full-gradient-v1'

def selective_logprob(base_model,lm_head,input_ids,prompt_length,output):
    """Keep gradients through every prefix layer, never prefix-vocabulary logits."""
    import torch
    hidden=base_model(input_ids,use_cache=False).last_hidden_state[0]
    logits=lm_head(prediction_rows(hidden,prompt_length,len(output)))
    return torch.log_softmax(logits.float(),-1).gather(1,torch.tensor(output,device=logits.device)[:,None]).mean()

def sequence_logprob(runtime,prompt,output):
    import torch
    from torch.nn.attention import sdpa_kernel,SDPBackend
    validate_tokens(prompt,output,runtime.model.config.vocab_size)
    if runtime.model.config._attn_implementation!='sdpa' or any(p.device.type!='cuda' or p.dtype!=torch.bfloat16 for p in runtime.model.parameters()):raise ValueError('long-context training BF16 CUDA SDPA profile')
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return selective_logprob(runtime.model.base_model,runtime.model.lm_head,torch.tensor([prompt+output],device='cuda'),len(prompt),output)

def configure_full_training(runtime):
    import torch
    model=runtime.model
    if getattr(model.config,'attention_dropout',0.)!=0 or any(isinstance(m,torch.nn.Dropout) and m.p!=0 for m in model.modules()):raise ValueError('deterministic zero-dropout full training')
    for p in model.parameters():p.requires_grad_(True)
    model.config.use_cache=False;model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.train()
    if not model.is_gradient_checkpointing:raise ValueError('long-context gradient checkpointing not active')
