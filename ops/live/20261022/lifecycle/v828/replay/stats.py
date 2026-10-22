import json,sys
from collections import Counter
E=sys.argv[1]
d=json.load(open(f'/home/const/subnet120-rewrite/state/mimo-mainnet-v1/epochs/{E}/coordinator/DISPATCH.json'))['payload']
p=json.load(open(f'/home/const/subnet120-rewrite/state/mimo-mainnet-v1/epochs/{E}/coordinator/POPULATION.json'))['payload']
ndocs=sum(len(c['payload']['documents']) for c in p['captures'])
print(dict(epoch=E,captures=len(p['captures']),documents=ndocs,selected=d['selected_task_count'],rollouts=d['selected_rollout_count'],excluded=Counter(x['reason'] for x in d['excluded']),sealed_at=p['sealed_at']))
