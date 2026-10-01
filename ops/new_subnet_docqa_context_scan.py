"""Bounded original task index scan; report prompt lengths without altering tasks."""
import json,pathlib
from subnet.environments import build_spec,_taskset
spec=build_spec('affine_docqa',{'taskset':{'epoch':1}},num_samples=1);ts=_taskset(spec)
from affine_docqa_v1.taskset import GenTaskStore,SOURCE,PACKAGE_DIR,build_prompt,list_catalog
from transformers import AutoTokenizer
store=GenTaskStore(SOURCE,PACKAGE_DIR,1);tokenizer=AutoTokenizer.from_pretrained('state/multi-environment/checkpoint-5',local_files_only=True);rows=[]
for index,rec in enumerate(store.tasks()):
 if index>=12:break
 bundle=store.read_json(f"bundles/{rec['bundle']}.json.gz");prompt=build_prompt(bundle['docs'],rec['question']);count=len(tokenizer.encode(prompt))
 rows.append({'original_index':index,'task_name':rec['uid'],'reported_tokens':rec.get('tokens'),'measured_prompt_tokens':count,'fits_8192_pilot':count+128<8192})
p=pathlib.Path('state/multi-environment/docqa-original-context-scan.json');p.write_text(json.dumps({'source':'affine_docqa','epoch':1,'bounded_original_indices':rows,'source_hash':spec.source_hash,'no_truncation':True},indent=2));print(json.dumps(rows))
