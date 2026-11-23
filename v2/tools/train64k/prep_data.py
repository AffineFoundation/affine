import json, random, collections
src='/home/box/sn120-scratch/self-distill/live/r000104/raw/captures/null.jsonl'
by=collections.defaultdict(list)
for l in open(src):
    r=json.loads(l)
    user=[m['content'] for m in r['prompt'] if m['role']=='user'][0]
    by[r['task_id']].append(dict(prompt=user,output=r['completion_text'],reward=float(r['reward']),tok=r['output_tokens'],capped=r['capped'],outcome=r['outcome']))
print('tasks',len(by), 'keys sample', list(json.loads(open(src).readline()).keys()))
mixed=[t for t,v in by.items() if len(v)>=2 and len({x['reward'] for x in v})>1]
print('mixed tasks',len(mixed))
# long: one pass >=60k tok and one capped fail
passes=sorted([x for v in by.values() for x in v if x['outcome']=='P' and x['tok']>=60000],key=lambda x:-x['tok'])
fails=sorted([x for v in by.values() for x in v if x['capped']],key=lambda x:-x['tok'])
print('long passes',len(passes),'capped fails',len(fails))
long=[[passes[0]['prompt'],[[passes[0]['output'],1.0],[fails[0]['output'],0.0]]]]
json.dump(long,open('data/long.json','w'))
random.seed(0); pick=random.sample(mixed,64)
batch=[[by[t][0]['prompt'],[[x['output'],x['reward']] for x in by[t][:2]]] for t in pick]
toks=[x['tok'] for t in pick for x in by[t][:2]]
print('batch rows',len(toks),'tokens',sum(toks),'>32k',sum(x>32768 for x in toks),'beyond32k',sum(max(0,x-32768) for x in toks))
json.dump(batch,open('data/batch128.json','w'))
