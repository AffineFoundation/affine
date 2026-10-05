"""Public miner emission projection from the chain's exact incentive event.

Amounts are alpha allocated by Yuma in the latest completed chain epoch, not
combined miner/validator emissions or claimed post-burn wallet payouts.
"""
import json
import os
import time
from pathlib import Path


def normalize(document):
    if not isinstance(document, dict):
        return None
    integers = ('block', 'observed_block', 'tempo')
    if any(type(document.get(key)) is not int or document[key] <= 0 for key in integers):
        return None
    if document['observed_block'] < document['block']:
        return None
    for key in ('timestamp', 'observed_at'):
        value = document.get(key)
        if type(value) not in (int, float) or not 0 < value < 1e11:
            return None
    amounts = document.get('emission_rao')
    if not isinstance(amounts, list) or not 1 <= len(amounts) <= 4096:
        return None
    if any(type(value) is not int or not 0 <= value <= 2**53-1 for value in amounts):
        return None
    if sum(amounts) > 2**53-1:
        return None
    return {key:document[key] for key in (*integers, 'timestamp', 'observed_at', 'emission_rao')}


def normalize_history(rows):
    if not isinstance(rows, list):
        return []
    valid = [normalize(row) for row in rows[-72:]]
    return sorted({row['block']:row for row in valid if row is not None}.values(),
                  key=lambda row:row['block'])[-72:]


def normalize_prices(document):
    if not isinstance(document, dict):
        return None
    def positive(key):
        value=document.get(key)
        return type(value) in (int,float) and 0 < value < 1e11
    if not positive('alpha_tao') or not positive('alpha_tao_at'):
        return None
    output={key:document[key] for key in ('alpha_tao','alpha_tao_at')}
    if positive('tao_usd') and positive('tao_usd_at') and document.get('source')=='coingecko':
        output.update({key:document[key] for key in ('tao_usd','tao_usd_at','source')})
    return output


def market_prices(graph, previous):
    import urllib.request
    reserves=graph.get('alpha_in',0)
    if not reserves or not graph.get('tao_in'):
        return normalize_prices(previous)
    prices=dict(previous or {},alpha_tao=graph['tao_in']/reserves,alpha_tao_at=time.time())
    try:
        url='https://api.coingecko.com/api/v3/simple/price?ids=bittensor&vs_currencies=usd&include_last_updated_at=true'
        request=urllib.request.Request(url,headers={'User-Agent':'Affine-dashboard/1.0'})
        with urllib.request.urlopen(request,timeout=8) as response:
            quote=json.load(response)['bittensor']
        candidate=normalize_prices(dict(prices,tao_usd=quote['usd'],tao_usd_at=quote['last_updated_at'],source='coingecko'))
        if candidate:
            return candidate
    except (OSError,ValueError,KeyError,TypeError):
        pass
    return normalize_prices(prices)


def collect(destination, backfill=24):
    import bittensor as bt
    from bittensor._generated.storage import System
    from bittensor._generated.runtime_apis import SubnetInfoRuntimeApi
    destination = Path(destination)
    try:
        previous = json.loads(destination.read_text())
    except (OSError, ValueError):
        previous = {}
    history = normalize_history(previous.get('epochs', []))
    old = normalize(previous)
    if old:
        history = normalize_history([*history, old])
    known = {row['block']:row for row in history}
    previous_blocks=set(known)
    with bt.Subtensor('finney') as chain:
        # Read the runtime record directly: no commitments or private data needed.
        graph = chain.runtime(SubnetInfoRuntimeApi.get_metagraph, [120])
        block, observed, tempo = graph['last_step'], graph['block'], graph['tempo']
        prices=market_prices(graph,normalize_prices(previous.get('prices')))
        checked = time.time()
        newest = None
        for index in range(backfill):
            try:
                if block in known:
                    document = dict(known[block], observed_block=observed, observed_at=checked)
                else:
                    snapshot = chain.at(block)
                    events = snapshot.query(System.Events)
                    matches = [event['attributes']['emissions'] for event in events
                               if event.get('module_id') == 'SubtensorModule'
                               and event.get('event_id') == 'IncentiveAlphaEmittedToMiners'
                               and event.get('attributes', {}).get('netuid') == 120]
                    if len(matches) != 1:
                        raise ValueError('Expected exactly one miner incentive event')
                    document = normalize(dict(block=block, observed_block=observed,
                        tempo=tempo, timestamp=snapshot.timestamp().timestamp(),
                        observed_at=checked, emission_rao=matches[0]))
                    if document is None:
                        raise ValueError('Invalid miner incentive event')
                known[block] = document
                if index == 0:
                    newest = document
                # Persist each real event, so a slow archive cannot discard progress.
                output = dict(newest, epochs=normalize_history(list(known.values())), prices=prices)
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_suffix('.tmp')
                temporary.write_text(json.dumps(output, allow_nan=False))
                os.chmod(temporary, 0o600)
                os.replace(temporary, destination)
                if index+1 >= backfill or (block in previous_blocks and len(known) >= backfill):
                    break
                # Follow actual completed chain steps; never invent timestamps or
                # assume that the epoch offset is a simple block-number multiple.
                prior = chain.runtime(SubnetInfoRuntimeApi.get_metagraph, [120], block=block-1)
                previous_block = prior['last_step']
                if not 0 < previous_block < block:
                    break
                block, tempo = previous_block, prior['tempo']
            except Exception:
                if newest is None:
                    raise
                # Archive gaps do not prevent publishing the current chain epoch.
                break
        print(f"Published {len(known)} miner incentive epochs; latest block {newest['block']}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    collect(parser.parse_args().output)
