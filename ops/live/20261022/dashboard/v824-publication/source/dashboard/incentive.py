"""Per-UID miner alpha settlements for completed chain epochs.

Read the Yuma miner event, recovering zero event vectors from role-filtered
TaoMarketCap receipts. Never substitute combined miner/validator emission.
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
    result = {key:document[key] for key in (*integers, 'timestamp', 'observed_at', 'emission_rao')}
    if document.get('emission_source') in ('chain-event', 'taomarketcap-receipts'):
        result['emission_source'] = document['emission_source']
    return result


def normalize_history(rows):
    if not isinstance(rows, list):
        return []
    valid = [normalize(row) for row in rows]
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


def miner_receipts(graph, block, cache, persist=None):
    """Recover allocations omitted by the Yuma server-emission event.

    The metagraph's emission includes validators; it only identifies candidates.
    Amounts come exclusively from TMC's role=miner settlement receipts.
    """
    import urllib.request
    from urllib.parse import urlencode, quote
    result = [0] * len(graph['emission'])
    headers = {'Accept': 'application/json', 'User-Agent': 'Affine-dashboard/1.0'}
    key = os.environ.get('TAOMARKETCAP') or os.environ.get('TMC_API_KEY')
    if key:
        headers['Authorization'] = key
    for uid, amount in enumerate(graph['emission']):
        if not amount:
            continue
        hotkey = graph['hotkeys'][uid]
        if hotkey not in cache or not any(row.get('tempo_end_block')==block and row.get('uid')==uid for row in cache[hotkey]):
            url = ('https://api.taomarketcap.com/public/v1/subnets/120/miners/'
                   + quote(hotkey, safe='') + '/emission-epochs/?'
                   + urlencode({'role': 'miner', 'limit': 72}))
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=15) as response:
                    rows = json.load(response)
            except urllib.error.HTTPError as error:
                if error.code==429 and persist:
                    try:delay=max(60,int(error.headers.get('Retry-After','60')))
                    except (ValueError,TypeError):delay=60
                    persist(time.time()+delay)
                raise
            if not isinstance(rows, list):
                raise ValueError('Invalid miner settlement receipts')
            cache[hotkey] = rows
            if persist:
                persist()
                time.sleep(1)
        matches = [row for row in cache[hotkey] if row.get('netuid') == 120
                   and row.get('tempo_end_block') == block and row.get('uid') == uid]
        if len(matches) != 1:
            raise ValueError('Miner settlement receipt not yet available')
        amount = matches[0].get('emission_alpha_rao')
        if not isinstance(amount, str) or not amount.isdigit():
            raise ValueError('Invalid miner settlement amount')
        result[uid] = int(amount)
    return result


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
    cache_path=destination.with_suffix('.receipts-cache.json')
    try:
        cache_document=json.loads(cache_path.read_text())
        receipts=cache_document.get('receipts',{})
        if not isinstance(receipts,dict):receipts={}
    except (OSError,ValueError):cache_document={};receipts={}
    if cache_document.get('retry_at',0)>time.time():
        print('Settlement API cooling down; preserving last successful chain snapshot')
        return
    def persist(retry_at=0):
        temporary=cache_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(dict(receipts=receipts,retry_at=retry_at)))
        os.chmod(temporary,0o600)
        temporary.replace(cache_path)
    with bt.Subtensor('finney') as chain:
        # Read the runtime record directly: no commitments or private data needed.
        graph = chain.runtime(SubnetInfoRuntimeApi.get_metagraph, [120])
        block, observed, tempo = graph['last_step'], graph['block'], graph['tempo']
        prices=market_prices(graph,normalize_prices(previous.get('prices')))
        checked = time.time()
        newest = None
        for index in range(backfill):
            try:
                if block in known and (sum(known[block]['emission_rao']) or known[block].get('emission_source')):
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
                    amounts, source = matches[0], 'chain-event'
                    if not sum(amounts):
                        settled = chain.runtime(SubnetInfoRuntimeApi.get_metagraph, [120], block=block)
                        if any(settled.get('emission', [])):
                            amounts = miner_receipts(settled, block, receipts, persist)
                            source = 'taomarketcap-receipts'
                    document = normalize(dict(block=block, observed_block=observed,
                        tempo=tempo, timestamp=snapshot.timestamp().timestamp(),
                        observed_at=checked, emission_rao=amounts, emission_source=source))
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
                if index+1 >= backfill or (block in previous_blocks and len(known) >= backfill
                    and all(sum(row['emission_rao']) or row.get('emission_source') for row in known.values())):
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
