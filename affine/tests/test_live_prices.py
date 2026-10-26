import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from affine.dash import tmc
from affine.dash.live_prices import public_prices, price_events


class LivePriceTests(unittest.IsolatedAsyncioTestCase):
    def market(self, **changes):
        return dict(netuid=120,price_tao=.05,tao_price_usd=300,
                    alpha_tao_at=1000,tao_usd_at=999,**changes)

    def test_public_quotes_only_export_price_fields(self):
        quote=public_prices(self.market(credential='PRIVATE',raw={'headers':'PRIVATE'}))
        self.assertEqual(set(quote),{'source','tao_usd_source','alpha_tao','tao_usd','alpha_tao_at','tao_usd_at'})
        self.assertEqual(quote['source'],'taomarketcap')
        self.assertNotIn('PRIVATE',json.dumps(quote))
        for value in (None,True,float('nan'),float('inf'),-1,0):
            market=self.market();market['price_tao']=value
            self.assertIsNone(public_prices(market))

    async def test_stale_sse_cannot_overwrite_newer_rest_price(self):
        original=self.market(block_number=200)
        with patch.object(tmc,'_market',original),patch.object(tmc,'_persist_market',new_callable=AsyncMock) as persist:
            await tmc._apply(None,120,{'latest_snapshot':{'price':.01,'block_number':199}})
            self.assertEqual(tmc.current_market(),original)
            persist.assert_not_awaited()

    async def test_subnet_sse_preserves_independent_usd_freshness(self):
        with patch.object(tmc,'_market',self.market(block_number=200)),patch.object(tmc,'_persist_market',new_callable=AsyncMock),patch.object(tmc,'_tip_reg_history',new_callable=AsyncMock):
            await tmc._apply(None,120,{'latest_snapshot':{'price':.06,'block_number':201}})
            self.assertEqual(tmc.current_market()['price_tao'],.06)
            self.assertEqual(tmc.current_market()['tao_usd_at'],999)
            self.assertEqual(tmc.current_market()['tao_price_usd'],300)

    async def test_rest_updates_both_prices_with_independent_timestamps(self):
        class Response:
            status_code=200
            def __init__(self,data):self.data=data
            def json(self):return self.data
        client=SimpleNamespace(get=AsyncMock(side_effect=[
            Response({'tao_price_usd':301,'timestamp':'2026-10-05T00:42:00Z'}),
            Response({'latest_snapshot':{'price':'.051','block_number':201}})]))
        cfg=SimpleNamespace(netuid=120,secrets=SimpleNamespace(taomarketcap='PRIVATE'))
        with patch.object(tmc,'_market',self.market(block_number=200)),patch.object(tmc,'_persist_market',new_callable=AsyncMock):
            await tmc._refresh_tao_usd(cfg,client)
            market=tmc.current_market()
            self.assertEqual((market['price_tao'],market['tao_price_usd'],market['block_number']),(.051,301,201))
            self.assertGreater(market['alpha_tao_at'],market['tao_usd_at'])
            self.assertNotIn('PRIVATE',json.dumps(public_prices(market)))

    async def test_older_or_invalid_usd_quote_does_not_block_new_alpha(self):
        class Response:
            status_code = 200
            def __init__(self, data): self.data = data
            def json(self): return self.data
        cfg = SimpleNamespace(netuid=120, secrets=SimpleNamespace(taomarketcap='PRIVATE'))
        for timestamp in ('1970-01-01T00:00:01Z', 'invalid'):
            with self.subTest(timestamp=timestamp):
                client = SimpleNamespace(get=AsyncMock(side_effect=[
                    Response({'tao_price_usd': 1, 'timestamp': timestamp}),
                    Response({'latest_snapshot': {'price': .07, 'block_number': 201}})]))
                with patch.object(tmc, '_market', self.market(block_number=200)), patch.object(tmc, '_persist_market', new_callable=AsyncMock):
                    await tmc._refresh_tao_usd(cfg, client)
                    market = tmc.current_market()
                    self.assertEqual((market['tao_price_usd'], market['tao_usd_at']), (300, 999))
                    self.assertEqual((market['price_tao'], market['block_number']), (.07, 201))

    async def test_events_include_changes_without_duplicate_ticks(self):
        initial=self.market();changed={**initial,'price_tao':.06}
        request=SimpleNamespace(is_disconnected=AsyncMock(side_effect=[False,False,False,True]))
        with patch('affine.dash.live_prices.market_for_snapshot',side_effect=[initial,initial,changed]),patch('affine.dash.live_prices.asyncio.sleep',new_callable=AsyncMock):
            events=[event async for event in price_events(None,request)]
        self.assertEqual(len(events),2)
        self.assertIn('"alpha_tao": 0.05',events[0])
        self.assertIn('"alpha_tao": 0.06',events[1])
        self.assertTrue(all(event.startswith('event: prices\n') for event in events))

    async def test_refresh_cadence_accounts_for_request_time_and_backs_off(self):
        from affine.dash.live_prices import run_quote_refresh
        cfg=SimpleNamespace(secrets=SimpleNamespace(taomarketcap='PRIVATE'))
        for failure,expected in [(False,1.1),(True,3.0)]:
            refresh=AsyncMock(side_effect=RuntimeError('Unavailable') if failure else None)
            sleep=AsyncMock(side_effect=asyncio.CancelledError)
            with patch('affine.dash.live_prices._refresh_tao_usd',refresh),patch('affine.dash.live_prices.time.monotonic',side_effect=[10,10.4]),patch('affine.dash.live_prices.asyncio.sleep',sleep):
                with self.assertRaises(asyncio.CancelledError):
                    await run_quote_refresh(cfg)
            self.assertAlmostEqual(sleep.call_args.args[0],expected)

    async def test_usd_quote_uses_price_timestamp_instead_of_snapshot_timestamp(self):
        class Response:
            status_code=200
            def __init__(self,data):self.data=data
            def json(self):return self.data
        client=SimpleNamespace(get=AsyncMock(side_effect=[
            Response({'tao_price_usd':301,'timestamp':'2026-10-06T18:18:00Z','usd_quote':{'last_updated':'2026-10-06T18:17:03Z'}}),
            Response({'latest_snapshot':{'price':'.051','block_number':201}})]))
        cfg=SimpleNamespace(netuid=120,secrets=SimpleNamespace(taomarketcap='PRIVATE'))
        with patch.object(tmc,'_market',self.market(block_number=200)),patch.object(tmc,'_persist_market',new_callable=AsyncMock):
            await tmc._refresh_tao_usd(cfg,client)
            from datetime import datetime
            self.assertEqual(tmc.current_market()['tao_usd_at'],datetime.fromisoformat('2026-10-06T18:17:03+00:00').timestamp())

    def test_live_usd_selection_preserves_alpha_and_exact_trade_time(self):
        from affine.dash import live_prices
        with patch.object(live_prices, '_exchange_quote', {'price':303.12, 'at':1001}):
            quote=public_prices(self.market())
            self.assertEqual((quote['tao_usd'],quote['tao_usd_at'],quote['tao_usd_source']),(303.12,1001,'kraken'))
            self.assertEqual((quote['alpha_tao'],quote['alpha_tao_at']),(.05,1000))
        with patch.object(live_prices, '_exchange_quote', {'price':1, 'at':998}):
            self.assertEqual(public_prices(self.market())['tao_usd_source'],'taomarketcap')

    def test_exchange_rejects_invalid_prices_wrong_pair_old_and_future_trades(self):
        from affine.dash import live_prices
        trade={'symbol':'TAO/USD','price':303.12,'timestamp':'1970-01-01T00:16:41Z'}
        with patch.object(live_prices,'_exchange_quote',None),patch.object(live_prices.time,'time',return_value=1005):
            live_prices.apply_exchange_trades({'channel':'trade','data':[trade]})
            self.assertEqual(live_prices._exchange_quote,{'price':303.12,'at':1001})
            for changes in [{'price':-1},{'price':float('nan')},{'price':True},{'symbol':'BTC/USD'},{'timestamp':'invalid'},{'timestamp':'1970-01-01T00:20:00Z'},{'timestamp':'1970-01-01T00:16:40Z'}]:
                live_prices.apply_exchange_trades({'channel':'trade','data':[{**trade,**changes,'credential':'PRIVATE'}]})
                self.assertEqual(live_prices._exchange_quote,{'price':303.12,'at':1001})
            self.assertNotIn('PRIVATE',json.dumps(public_prices(self.market())))
