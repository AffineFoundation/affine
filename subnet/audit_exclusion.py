"""Temporary exclusion based solely on authenticated, confirmed-invalid epochs."""
VERSION='confirmed-invalid-temporary-exclusion-v1'
def validate(p):
 if type(p)is not dict or set(p)!={'version','threshold_epochs','lookback_epochs','exclusion_epochs'}or p['version']!=VERSION:raise ValueError('temporary exclusion policy')
 for k in ('threshold_epochs','lookback_epochs','exclusion_epochs'):
  if type(p[k])is not int or not 1<=p[k]<=1000:raise ValueError('exclusion epoch bound')
 if p['threshold_epochs']>p['lookback_epochs']:raise ValueError('exclusion threshold window')
 return dict(p)
def confirmed(report):
 return any(o.get('valid')is False and o.get('fully_audited')is True and o.get('failure_kind')=='confirmed_invalid'for o in report.get('outcomes',[]))
def snapshot(history,policy):
 p=validate(policy);epochs=history['epochs'];events={};excluded=set();now=len(epochs)
 if len({row['epoch']for row in epochs})!=len(epochs):raise ValueError('unique history epochs')
 for seq,row in enumerate(epochs):
  for miner,report in row['reports'].items():
   if not confirmed(report):raise ValueError('history must contain confirmed invalid only')
   recent=[i for i in events.get(miner,[])if seq-i<p['lookback_epochs']]+[seq];events[miner]=recent
   if len(recent)>=p['threshold_epochs']and now<=seq+p['exclusion_epochs']:excluded.add(miner)
 return sorted(excluded)
