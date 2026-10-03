"""Prospective operator reward sidecar, no chain execution/remote commands.

Root's SINGLE writer may consume output units via ChainAdapter.submit_hour.
This exporter signs only new derived reward records; original compute documents
are never edited. All operator files remain on the operator host.
"""
import sys
sys.dont_write_bytecode=True
import argparse,base64,fcntl,json,os,time,tempfile
from pathlib import Path
from nacl.signing import SigningKey
from subnet.live_reward_bridge import canonical,signed,sha,need,export_epoch,hourly_reward_units

def sign(payload,key):return dict(payload=payload,signer=key.verify_key.encode().hex(),signature=base64.b64encode(key.sign(canonical(payload)).signature).decode())
def atomic(path,value):
 path=Path(path);fd,name=tempfile.mkstemp(prefix=path.name+'.',dir=path.parent)
 try:
  with os.fdopen(fd,'wb') as f:f.write(canonical(value));f.flush();os.fsync(f.fileno())
  os.chmod(name,0o600);os.replace(name,path)
 finally:Path(name).unlink(missing_ok=True)
def run_once(compute_state,reward_state,anchor_document,authority,key,fresh_registrations,window_end):
 need(key.verify_key.encode().hex()==authority,'operator reward authority key binding')
 compute_state=Path(compute_state);reward_state=Path(reward_state);reward_state.mkdir(parents=True,exist_ok=True);reward_state.chmod(0o700)
 with (reward_state/'reward-ledger.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  ledgerpath=reward_state/'signed-reward-ledger.json';ledger=json.loads(ledgerpath.read_text()) if ledgerpath.exists() else []
  by_epoch={signed(doc,authority)['epoch_id']:doc for doc in ledger};need(len(by_epoch)==len(ledger),'unique reward ledger epochs')
  # Only original first-manifest attestations produced by the new hook qualify.
  for first in sorted(compute_state.glob('*-first-signed-manifest.json')):
   m=signed(json.loads(first.read_text()),authority);epoch=m['epoch']
   if not (compute_state/(epoch+'-signed-compute-scores.json')).exists():continue
   report=export_epoch(compute_state,epoch,anchor_document,authority);document=sign(report,key)
   reward_epoch=report['epoch_id']
   if reward_epoch in by_epoch:need(canonical(document)==canonical(by_epoch[reward_epoch]),'immutable reward ledger collision')
   else:ledger.append(document);by_epoch[reward_epoch]=document
  atomic(ledgerpath,ledger)
  hourly=hourly_reward_units(ledger,authority,window_end,fresh_registrations=fresh_registrations)
  # This is a signed proposal, not a chain transaction or activation receipt.
  atomic(reward_state/('hour-'+str(window_end)+'-reward-units.json'),sign(hourly,key))
 return hourly

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--compute-state',required=True);p.add_argument('--reward-state',required=True);p.add_argument('--anchor',required=True);p.add_argument('--authority',required=True);p.add_argument('--operator-authority-seed-file',required=True);p.add_argument('--fresh-registrations-file',required=True);p.add_argument('--window-end',type=int)
 a=p.parse_args();end=a.window_end or int(time.time())//3600*3600;need(end<=int(time.time()),'only completed payout hours')
 seed=Path(a.operator_authority_seed_file);need(seed.is_file() and not seed.is_symlink() and seed.stat().st_mode&0o077==0,'private operator seed file')
 key=SigningKey(bytes.fromhex(seed.read_text().strip()));anchor=json.loads(Path(a.anchor).read_text());observation=json.loads(Path(a.fresh_registrations_file).read_text());need(0<=time.time()-observation['observed_at']<=60,'fresh registration observation')
 result=run_once(a.compute_state,a.reward_state,anchor,a.authority,key,observation['registrations'],end)
 print(json.dumps(dict(window_end=end,miners_with_positive_units=sum(v>0 for v in result['points'].values()),reward_records=len(result['source_reward_records']),chain_executed=False)))
if __name__=='__main__':main()
