"""Prospective SINGLE writer handoff; import has no chain/wallet/service operations.

Root must run with a global owner/netuid lock and freshly observed cutover proof.
The actual ChainAdapter retains owner/wallet/rate/UID/old-writer final checks.
"""
import sys
sys.dont_write_bytecode=True
from pathlib import Path
from subnet.live_reward_bridge import writer_gate,need,signed

def submit_hour(adapter,hourly_document,authority,writer_receipt,*,now,boot_id,writer_pid,writer_ticks,execute=False):
 hourly=signed(hourly_document,authority)
 need(hourly.get('version')=='live-reward-hour-units-v1','signed hourly reward proposal version')
 need(hourly.get('chain_executed') is False and hourly.get('units_per_point')==1_000_000,'authenticated integer reward units proposal only')
 need(type(execute)is bool,'explicit execution flag')
 if execute:writer_gate(writer_receipt,authority,now=now,boot_id=boot_id,writer_pid=writer_pid,writer_ticks=writer_ticks)
 # Fresh identity records are supplied by authenticated hourly bridge; adapter
 # independently rechecks actual chain registrations and UID/key ownership.
 return adapter.submit_hour(hourly['points'],hourly['registrations'],hourly['window_end'],execute=execute)
