"""READ-ONLY: inspect the Commitments pallet on Finney for the staged probe commitment (no wallet, no signing, no submit)."""
import json, signal, time
signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError('120s'))); signal.alarm(120)
import bittensor as bt
from bittensor._generated.calls import Call
OWNER = '5HmYnmUYT6qe3yFMg1Ad8WLLqDvnwtjYakXBpDvoRW1Qqzb8'
s = bt.subtensor(network='finney', retry_forever=False); sub = s._client._substrate; C = s._call
out = dict(observed_at=time.time(), block=s.block)
def q(mod, item, params=()):
    try: v = C(sub.query(mod, item, list(params))); return getattr(v, 'value', v)
    except Exception as e: return f'ERR {type(e).__name__}: {str(e)[:160]}'
def k(mod, name):
    try: v = C(sub.constant(mod, name)); return getattr(v, 'value', v)
    except Exception as e: return f'ERR {type(e).__name__}: {str(e)[:120]}'
for item, params in [('RateLimit', ()), ('MaxSpace', ()), ('CommitmentOf', (120, OWNER)), ('LastCommitment', (120, OWNER)),
                     ('UsedSpaceOf', (120, OWNER)), ('TimelockedIndex', ())]:
    v = q('Commitments', item, params); out[item] = v if len(str(v)) < 600 else str(v)[:600]
for name in ('MaxFields', 'CommitmentDeposit', 'InitialDeposit', 'FieldDeposit', 'DefaultRateLimit'):
    out['const_' + name] = k('Commitments', name)
out['tempo_120'] = q('SubtensorModule', 'Tempo', (120,))
out['owner_uid_120'] = q('SubtensorModule', 'Uids', (120, OWNER))
class PK:  # public-key-only stand-in; get_payment_info needs an address, never a secret
    def __init__(self, a):
        self.ss58_address = a; self.crypto_type = 1
        B58 = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'; n = 0
        for ch in a: n = n * 58 + B58.index(ch)
        raw = n.to_bytes(35, 'big'); self.public_key = raw[1:33]
for label, info in [('sha256_field', {'fields': [[{'Sha256': '0x' + '11' * 32}]]}), ('raw32_field', {'fields': [[{'Raw32': '0x' + '11' * 32}]]})]:
    try:
        call = C(sub.compose(Call('Commitments', 'set_commitment', {'netuid': 120, 'info': info})))
        out['fee_' + label] = str(C(sub.estimate_fee(call, PK(OWNER))))
    except Exception as e: out['fee_' + label] = f'ERR {type(e).__name__}: {str(e)[:300]}'
print(json.dumps(out, default=str, indent=1))
