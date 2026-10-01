"""Test whether measured cross-runtime tolerances admit actual modified-weight computation."""
import copy,itertools,json,time
from pathlib import Path
import torch
import pipeline as P
import extended_checks as E

root=Path('prototype/artifacts');out=root/'extended';torch.set_num_threads(4);torch.set_num_interop_threads(1)
policy=json.loads((root/'trusted-policy-local.json').read_text());tok,model=P.load(policy,'cpu')
doc,arrays=P.unpack(root/'honest.zip');cache=[P.compute(model,t['prompt'],t['output']) for t in doc['turns']]
report=json.loads((out/'cross-runtime.json').read_text());grid=[]
for threshold in itertools.product([0,1,2,3,4,8],[0.,.5,1.,1.5,2.,3.,4.],[0.,1.,2.,4.],[1e-5,.01,.05,.1,.2,.5,.65,.75,1.]):
 honest=all(E.good(x,threshold) for x in report['honest_segments'])
 accepted={name:all(E.good(x,threshold) for x in values) for name,values in report['attack_segments'].items()}
 if honest:grid.append({'threshold':list(threshold),'honest_accepted':True,'attack_accepted':accepted,'separates_initial_attacks':not any(accepted.values())})
# A measured separating point; explicitly post-hoc on one honest trajectory, not deployment calibration.
threshold=(3,2.,1.,.65);rows=[]
weight=model.model.layers[0].mlp.down_proj.weight;saved=weight.detach().clone()
for scale in [1.003,1.01,1.05,1.1]:
 d=copy.deepcopy(doc);a=[]
 with torch.no_grad():weight.copy_(saved*scale)
 changed=int((weight!=saved).sum());total=weight.numel()
 for i,t in enumerate(d['turns']):
  acts,probs=P.compute(model,t['prompt'],t['output']);t['proofs']=P.build_proofs_base64(acts,decode_batching_size=P.BATCH,topk=P.TOPK);a.append(probs)
 with torch.no_grad():weight.copy_(saved)
 name=f'tolerance-weight-scale-{scale}.zip';seal=P.pack(out/name,d,a)
 frozen_d,frozen_a=P.unpack(out/name);E.validate_document(frozen_d,frozen_a,policy,tok)
 metrics=E.metrics(frozen_d,frozen_a,cache)
 row={'layer':'model.layers[0].mlp.down_proj.weight','scale':scale,'changed_bfloat16_values':changed,'total_layer_values':total,'sha256':seal,'artifact':name,'strict_accepted':all(E.good(x,E.THRESHOLDS[0]) for x in metrics),'posthoc_cross_runtime_threshold_accepted':all(E.good(x,threshold) for x in metrics),'segments':metrics}
 rows.append(row);print({k:v for k,v in row.items() if k!='segments'},flush=True)
 assert P.digest(out/name)==seal
result={'threshold_axes_order':['exp_mismatches','mantissa_mean','mantissa_median','logprob_atol'],'honest_accepting_cartesian_grid':grid,'posthoc_candidate':{'threshold':list(threshold),'honest_accepted':all(E.good(x,threshold) for x in report['honest_segments']),'initial_attack_accepted':{name:all(E.good(x,threshold) for x in values) for name,values in report['attack_segments'].items()},'warning':'Selected after observing one honest trajectory. It is not an independently calibrated or deployment-safe threshold.'},'modified_weight_tests':rows,'conclusion':'Any tolerance admits a class of approximate computations. Verify acceptable error definitions before treating a passed fingerprint as exact approved-weight execution.'}
counterexample=(8,2.,2.,.65)
result['counterexample_candidate']={'threshold':list(counterexample),'honest_accepted':all(E.good(x,counterexample) for x in report['honest_segments']),'initial_attack_accepted':{name:all(E.good(x,counterexample) for x in values) for name,values in report['attack_segments'].items()},'modified_weight_accepted':{str(x['scale']):all(E.good(r,counterexample) for r in x['segments']) for x in rows},'warning':'Post-hoc diagnostic, not deployed. It separates the original four attacks but admits some actually changed-weight computations.'}
(out/'tolerance-attacks.json').write_text(json.dumps(result,indent=2))
