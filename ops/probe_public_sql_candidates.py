"""Grade training-only public proposals using the isolated ORIGINAL grader.

Private reference SQL goes only to the separate grader; the proposal function
receives original public messages. These controls are not mined model proofs.
"""
import hashlib,json,time
from pathlib import Path
from subnet.public_sql_candidates import candidates,REVISION
from subnet.native_sql_isolation import grade

ROOT=Path('/home/const/subnet120-rewrite')
def main():
    folder=ROOT/'state/native-sql-tasks';public=json.loads((folder/'public-tasks.json').read_text());private=json.loads((folder/'private-tasks.json').read_text());runtime=json.loads((ROOT/'state/native-sql-isolation/runtime.json').read_text())
    assert public['train_indices']==list(range(16)) and public['heldout_indices']==list(range(16,32))
    rows=[]
    for index in public['train_indices']:
        visible=public['tasks'][index]['public']['messages'];proposals=candidates(visible)
        results=[dict(query_sha256=hashlib.sha256(q.encode()).hexdigest(),reward=grade(private['tasks'][index]['private'],q,runtime)['reward']) for q in proposals]
        row=dict(index=index,question=visible[1]['content'].split('Question: ',1)[1].split('\n\n',1)[0],public_prompt_sha256=hashlib.sha256(json.dumps(visible,sort_keys=True).encode()).hexdigest(),proposals=proposals,results=results,positive_negative_control=any(r['reward']==1 for r in results) and any(r['reward']==0 for r in results))
        rows.append(row);print(json.dumps(dict(index=index,rewards=[r['reward'] for r in results],positive_negative_control=row['positive_negative_control'])),flush=True)
    result=dict(revision=REVISION,generator_sha256=hashlib.sha256((ROOT/'subnet/public_sql_candidates.py').read_bytes()).hexdigest(),probe_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),completed_at=time.time(),training_only=True,heldout_examples_processed=False,full_operator_collections_loaded=True,proposal_private_grader_access=False,native_grader_original=True,model_proof=False,training=False,rows=rows)
    out=folder/'public-query-research';out.mkdir(exist_ok=True);(out/'native-controls-v1.json').write_text(json.dumps(result,indent=2))
if __name__=='__main__':main()
