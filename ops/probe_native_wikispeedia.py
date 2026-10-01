"""Qualify original Wikispeedia tools on frozen original tasks without a model."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from subnet.backend_jobs import canonical
from subnet.environments import build_spec, snapshot_spec, create_session
from subnet.native_wikispeedia import execute, public_path, replay, verify_public_resources


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state',type=Path,required=True)
    parser.add_argument('--count',type=int,default=4)
    args=parser.parse_args()
    if not 1 <= args.count <= 32:
        raise ValueError('bounded native prerequisite population')
    args.state.mkdir(parents=True,exist_ok=True)
    spec=build_spec('affine_wikispeedia',num_samples=args.count,max_turns=30,max_output_tokens=256)
    spec=snapshot_spec(spec,args.state/'original-tasks.json')
    (args.state/'environment.json').write_bytes(canonical(spec.to_dict()))
    from wikispeedia_v1.graph import WikiGraph,DEFAULT_CACHE_DIR
    graph=WikiGraph.load(include_text=False)
    resource_check=verify_public_resources(DEFAULT_CACHE_DIR)
    tasks=json.loads((args.state/'original-tasks.json').read_bytes())
    controls=[]
    for index,task in enumerate(tasks):
        data=task['data']
        # Both endpoints and the link graph are original public task inputs.
        session=create_session(spec)
        try:
            reset=session.reset(index,20261001+index)
            click_names=[t['function']['name'] for t in reset['tools'] if t['function']['name'].endswith('click_link')]
            if len(click_names)!=1:
                raise ValueError('unambiguous original click tool schema required')
            click_name=click_names[0]
        finally:
            session.close()
        route=public_path(data['source'],data['target'],graph.links,30)
        positive=[dict(text='Follow the public article link toward the target.',
                       tool_calls=[dict(name=click_name,arguments=dict(article=article))])
                  for article in route]
        negative=[dict(text='Test an unavailable article.',tool_calls=[dict(name=click_name,
                   arguments=dict(article='__affine_invalid_article__'))]),dict(text='Done.')]
        row=dict(index=index,registered_click_tool=click_name,task_hash=None,route_hops=len(route),positive_reward=None,negative_reward=None,mutations_rejected=[])
        for label,actions in [('positive',positive),('negative',negative)]:
            artifact=execute(spec,index,20261001+index,actions)
            replay(spec,artifact)
            assert artifact['reward']==(1.0 if label=='positive' else 0.0)
            row['task_hash']=artifact['task_hash'];row[label+'_reward']=artifact['reward']
            (args.state/f'{index}-{label}.json').write_bytes(canonical(artifact))
            for mutation in ('reward','tool_observation','source','task'):
                altered=copy.deepcopy(artifact)
                if mutation=='reward':altered['reward']=1-artifact['reward']
                elif mutation=='tool_observation':altered['turns'][0]['result']['observations'][0]['content']='TARGET REACHED forged'
                elif mutation=='source':altered['source_hash']='0'*64
                else:altered['task_hash']='0'*64
                try:replay(spec,altered)
                except ValueError:row['mutations_rejected'].append(label+'-'+mutation)
                else:raise AssertionError('mutation accepted: '+mutation)
        controls.append(row)
        (args.state/'controls.json').write_bytes(canonical(controls))
        print(json.dumps(dict(index=index,positive=1,negative=0,mutation_rejections=len(row['mutations_rejected']))),flush=True)
    resources={}
    for name in ('wikispeedia_paths-and-graph.tar.gz','wikispeedia_articles_plaintext.tar.gz'):
        path=DEFAULT_CACHE_DIR/name
        resources[name]=dict(size=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    result=dict(schema=1,source='affine_wikispeedia',source_hash=spec.source_hash,
        taskset_sha256=hashlib.sha256((args.state/'original-tasks.json').read_bytes()).hexdigest(),
        original_taskset_defaults_preserved=True,controls=controls,resources=resources,
        extracted_resource_inventory_sha256=resource_check['extracted_inventory_sha256'],
        extracted_resource_files=len(resource_check['extracted_files']),
        model_execution=False,TOPLOC_generated=False,optimizer_ran=False,chain_transactions=False,
        scope='Original native tool/reward and independent replay prerequisite only')
    (args.state/'qualification.json').write_bytes(canonical(result))


if __name__=='__main__':main()
