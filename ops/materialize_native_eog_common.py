"""Operator-private original Calendar collection for prospective portable v4.

Run ONLY from the isolated reviewed portable source. Private fixtures stay
outside public source bundles; no task is replaced by a synthetic environment.
"""
import copy,hashlib,json,shutil
from pathlib import Path
from subnet.native_eog_split import OperatorBroker,sha
from subnet.native_eog_deployment import private_fixture_hash,PRIVATE_HASH_POLICY

ROOT=Path('/home/const/subnet120-rewrite')
def main():
 from subnet.native_eog_split import REVISION
 if REVISION!='original-eog-public-actor-private-grader-v4-portable-terminal':raise ValueError('isolated portable v4 source required')
 state=ROOT/'state/native-eog-common';operator=state/'operator';operator.mkdir(exist_ok=True);operator.chmod(0o700)
 original=ROOT/'state/native-eog-isolation'
 relocation=json.loads((original/'original-calendar-relocation.private.json').read_text());four=json.loads((original/'original-calendar-four.private.json').read_text())
 runtime=json.loads((original/'runtime-descriptor.json').read_text())
 selected=[(54,relocation),(0,four[0]),(3,four[3])];records=[];public=[];bindings=[]
 for index,raw in selected:
  task=copy.deepcopy(raw);seed=Path(task['data']['services'][0]['seed_file']);sha_seed=hashlib.sha256(seed.read_bytes()).hexdigest();destination=operator/(sha_seed+'.sql')
  if not destination.exists():shutil.copy2(seed,destination);destination.chmod(0o600)
  task['data']['services'][0]['seed_file']=str(destination)
  broker=OperatorBroker(task,runtime)
  try:descriptor=copy.deepcopy(broker.public)
  finally:broker.close()
  record=dict(original_index=index,original_task_id=task['data']['name'],private=task)
  binding=dict(original_index=index,original_task_id=task['data']['name'],public_descriptor_sha256=sha(descriptor),private_fixture_sha256=private_fixture_hash(task),seed_sha256=sha_seed)
  records.append(record);public.append(descriptor);bindings.append(binding)
 collection=dict(revision='original-eog-calendar-portable-v4-collection',private_hash_policy=PRIVATE_HASH_POLICY,tasks=records)
 p=operator/'private-tasks.json';p.write_text(json.dumps(collection,indent=2));p.chmod(0o600)
 materialization=json.loads((original/'fixture-materialization.json').read_text())
 result=dict(revision=collection['revision'],original_dataset=materialization['dataset'],original_dataset_revision=materialization['revision'],original_arrow_sha256=materialization['arrow_sha256'],private_hash_policy=PRIVATE_HASH_POLICY,training_indices=[0],heldout_indices=[1,2],original_indices=[54,0,3],public_tasks=public,task_bindings=bindings,common_model_proof=False,common_training=False)
 (state/'public-collection.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(tasks=len(records),original_indices=result['original_indices'],private_fixture_public=False)),flush=True)
if __name__=='__main__':main()
