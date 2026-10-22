"""SN120 v9 public post-commit reference seed (READ-ONLY chain access, no wallet).

Rule (sn120-v9-seed-rule-v1):
  deadline     = the epoch's signed manifest 'deadline' (unix s; miner presigned upload URLs expire at it)
  anchor_block = first finalized Finney block whose Timestamp.Now >= deadline * 1000
  seed_block   = anchor_block + MARGIN (10 blocks, ~2 min finality / clock-skew margin)
  seed         = sha256("sn120-v9-ref-seed-v1:" + epoch_id + ":" + seed_block_hash)  (hex)
Everything in the reference (task draw from the full offered pool, base samples, 2-6/8 band filter, chunking) is
derived from `seed` AFTER this block exists, so it is unpredictable before the commit deadline and reproducible by
anyone afterwards from (epoch_id, seed_block, seed_block_hash).  No private reference, no secret salt.
usage: seed_block.py <epoch> [out.json]"""
import hashlib, json, sys, time
from pathlib import Path
RULE = 'sn120-v9-seed-rule-v1'; MARGIN = 10
EPOCHS = Path('/home/const/subnet120-rewrite/state/mimo-mainnet-v1/epochs')

def manifest(E):
    d = json.loads((EPOCHS / str(E) / 'PLAN.json').read_text()); d = d.get('payload', d)
    m = d['manifest']; m = m.get('payload', m)
    return m['epoch'], int(m['deadline'])

def derive(epoch_id, block_hash):
    return hashlib.sha256(f'sn120-v9-ref-seed-v1:{epoch_id}:{block_hash}'.encode()).hexdigest()

class Chain:
    def __init__(self):
        import bittensor as bt
        self.s = bt.subtensor(network='finney', retry_forever=False); _ = self.s.block
        self.sub = self.s._client._substrate; self.C = self.s._call; self.raw = self.sub.raw
    def rpc(self, m, p):
        v = self.C(self.raw.rpc_request(m, p)); return v.get('result', v) if isinstance(v, dict) else v
    def finalized(self):
        h = self.rpc('chain_getFinalizedHead', []); n = self.rpc('chain_getHeader', [h])['number']
        return (int(n, 16) if isinstance(n, str) else int(n)), h
    def hash(self, n): return self.rpc('chain_getBlockHash', [n])
    def ts(self, n):
        v = self.C(self.sub.query('Timestamp', 'Now', block_hash=self.hash(n))); return int(getattr(v, 'value', v))
    def close(self): self.s.close()

def seed_for(E, ch=None):
    epoch_id, deadline = manifest(E); own = ch is None; ch = ch or Chain()
    try:
        head, _ = ch.finalized(); target = deadline * 1000
        if ch.ts(head) < target: raise RuntimeError('deadline not yet reached on finalized chain')
        lo = max(1, head - int((ch.ts(head) - target) / 12000) - 400); hi = head
        while ch.ts(lo) >= target: lo = max(1, lo - 2000)
        while hi - lo > 1:                       # invariant: ts(lo) < target <= ts(hi)
            mid = (lo + hi) // 2
            if ch.ts(mid) >= target: hi = mid
            else: lo = mid
        anchor = hi; sb = anchor + MARGIN
        if sb > head: raise RuntimeError('seed block not finalized yet')
        bh = ch.hash(sb)
        return dict(rule=RULE, epoch=E, epoch_id=epoch_id, deadline=deadline, anchor_block=anchor, anchor_ts_ms=ch.ts(anchor),
                    margin_blocks=MARGIN, seed_block=sb, seed_block_hash=bh, seed_block_ts_ms=ch.ts(sb), seed=derive(epoch_id, bh),
                    finalized_head_at_derivation=head, derived_at=time.time())
    finally:
        if own: ch.close()

if __name__ == '__main__':
    r = seed_for(int(sys.argv[1]))
    if len(sys.argv) > 2: Path(sys.argv[2]).write_text(json.dumps(r, indent=1, sort_keys=True))
    print(json.dumps(r, sort_keys=True))
