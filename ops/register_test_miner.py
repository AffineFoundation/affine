"""One explicitly authorized paid Ed25519 test registration; never set weights."""
import argparse, dataclasses, json, os, time
from pathlib import Path
import bittensor as bt

p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');a=p.parse_args()
state=Path('state/registered-test-registration');state.mkdir(parents=True,exist_ok=True,mode=0o700)
wallet=bt.Wallet(name='miner',hotkey='affine-ed25519-test-20260930',path='/home/const/subnet120/mining/wallets')
if not Path(wallet.hotkey_file.path).exists():
    key=bt.sp_core.Keypair.create_from_seed(os.urandom(32),crypto_type=0)
    wallet.hotkey_file.set_keypair(key,encrypt=False,overwrite=False)
os.chmod(wallet.hotkey_file.path,0o600)
assert wallet.hotkey.crypto_type==0
assert wallet.coldkey.ss58_address=='5CZscRf3nZmGspyqs2ZvFXSjnnondpjNU5QbJWVFFT92FC98'
chain=bt.Subtensor(network='finney',policy=bt.Policy(max_fee_tao='0.05',allowed_netuids=[120]))
block=int(chain.block)
def q(name,params): return chain.query(getattr(bt.storage.SubtensorModule,name),params,block=block)
account=chain.query(bt.storage.System.Account,[wallet.coldkey.ss58_address],block=block)
burn=int(q('Burn',[120])); free=int(account['data']['free'])
allowed=q('NetworkRegistrationAllowed',[120]);uid=q('Uids',[120,wallet.hotkey.ss58_address])
info={'block':block,'hotkey':wallet.hotkey.ss58_address,'public_key':bytes(wallet.hotkey.public_key).hex(),'coldkey':wallet.coldkey.ss58_address,'estimated_burn_rao':burn,'free_before_rao':free,'registration_allowed':allowed,'existing_uid':uid,'wallet_path':str(wallet.hotkey_file.path),'test_payable':False,'weight_submission':False}
assert allowed and uid is None,'registration unavailable or already registered'
assert burn<5_000_000_000 and burn<free//10,'unexpected expensive registration'
intent=bt.BurnedRegister(netuid=120)
plan=chain.plan(intent,wallet);info['plan']=str(plan);info['plan_ok']=plan.ok
(state/'preflight.json').write_text(json.dumps(info,indent=2)+'\n')
print(json.dumps(info),flush=True)
if a.execute:
    if not plan.ok: raise RuntimeError('SDK plan denied')
    marker=state/'submission-attempted';fd=os.open(marker,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600);os.write(fd,str(time.time()).encode());os.close(fd)
    result=chain.execute(intent,wallet,wait_for_inclusion=True,wait_for_finalization=True,retries=0)
    result.raise_for_failure()
    after=int(chain.block); newuid=chain.query(bt.storage.SubtensorModule.Uids,[120,wallet.hotkey.ss58_address],block=after)
    owner=chain.query(bt.storage.SubtensorModule.Keys,[120,int(newuid)],block=after)
    balance=chain.query(bt.storage.System.Account,[wallet.coldkey.ss58_address],block=after)
    receipt={**info,'status':'confirmed','confirmed_block':after,'uid':int(newuid),'owner_confirmed':str(owner)==wallet.hotkey.ss58_address,'block_hash':str(result.block_hash),'free_after_rao':int(balance['data']['free']),'actual_balance_decrease_rao':free-int(balance['data']['free'])}
    assert receipt['owner_confirmed']
    (state/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
