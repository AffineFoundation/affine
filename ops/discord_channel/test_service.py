import asyncio
from copy import deepcopy
import unittest
from unittest.mock import patch

from . import service as s


class ProtocolTests(unittest.TestCase):
    def request(self, **changes):
        return {'id': '17887900000001234567890', 'channel_id': s.CHANNEL, 'content': 'hello @everyone', **changes}

    def test_mentions_suppressed_and_nonce_enforced(self):
        body = s.send_body(self.request(reply_id='1546522347203199166'))
        self.assertEqual(body['allowed_mentions'], {'parse': [], 'replied_user': False})
        self.assertTrue(body['enforce_nonce'])
        self.assertEqual(body['message_reference']['channel_id'], s.CHANNEL)
        self.assertTrue(body['message_reference']['fail_if_not_exists'])

    def test_reject_empty_long_and_invalid_nonce(self):
        for changes in ({'content': ''}, {'content': ' '}, {'content': 'a'*2001}, {'content': '😀'*1001}, {'id': 'x'}, {'reply_id': '../other'}):
            with self.subTest(changes=str(changes)[:60]), self.assertRaises(s.DiscordError):
                s.send_body(self.request(**changes))

    def test_scope(self):
        client = s.Discord('test-token')
        for channel, metadata, allowed in [
            (s.CHANNEL, {'guild_id':s.GUILD,'type':0}, True),
            ('1546522347203199166', {'guild_id':s.GUILD,'parent_id':s.CHANNEL,'type':11}, True),
            ('1546522347203199166', {'guild_id':s.GUILD,'parent_id':'other','type':11}, False),
            (s.CHANNEL, {'guild_id':'other','type':0}, False),
            ('1546522347203199166', {'guild_id':s.GUILD,'parent_id':s.CHANNEL,'type':0}, False),
        ]:
            with patch.object(client, 'request', return_value=metadata):
                if allowed:self.assertEqual(client.channel(channel),metadata)
                else:
                    with self.assertRaises(s.DiscordError):client.channel(channel)

    def test_sanitized_message(self):
        item = s.message({'id':'1','channel_id':s.CHANNEL,'content':'<script>bad()</script>', 'author':{'id':'2','username':'a','token':'secret'},'private':'secret'})
        self.assertEqual(item['content'],'<script>bad()</script>')
        self.assertNotIn('secret',str(item))


class SendTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.doc = {'send_request': {'id':'17887900000001234567890','channel_id':s.CHANNEL,'content':'hello'}}
        outer = self
        class Document:
            async def __aenter__(self):self.doc=deepcopy(outer.doc);return self
            async def __aexit__(self,*args):pass
            async def patch(self, values):outer.doc.update(deepcopy(values));self.doc=deepcopy(outer.doc);return self.doc
        self.patches=[patch.object(s,'Document',Document),patch.object(s,'lock')]
        for p in self.patches:p.start();self.addCleanup(p.stop)
        self.client=s.Discord('test-token')
        self.client_patch=patch.object(s,'Discord',return_value=self.client)
        self.client_patch.start();self.addCleanup(self.client_patch.stop)

    async def test_send_and_duplicate_is_noop(self):
        with patch.object(self.client,'channel',return_value={}), patch.object(self.client,'request',return_value={'id':'123'}) as request:
            await s.send(self.doc['send_request']['id'])
            self.assertEqual(self.doc['delivery']['status'],'sent')
            await s.send(self.doc['send_request']['id'])
            self.assertEqual(request.call_count,1)

    async def test_stale_request_never_posts(self):
        with patch.object(self.client,'request') as request:
            with self.assertRaises(s.DiscordError):await s.send('wrong')
            request.assert_not_called()

    async def test_unknown_send_is_not_retried(self):
        with patch.object(self.client,'channel',return_value={}), patch.object(self.client,'request',side_effect=s.DiscordError('network')) as request:
            with self.assertRaises(s.DiscordError):await s.send(self.doc['send_request']['id'])
            self.assertEqual(self.doc['delivery']['status'],'unknown')
            with self.assertRaises(s.DiscordError):await s.send(self.doc['send_request']['id'])
            self.assertEqual(request.call_count,1)

    async def test_scope_failure_recorded_without_post(self):
        with patch.object(self.client,'channel',side_effect=s.DiscordError('outside allowed channel')), patch.object(self.client,'request') as request:
            with self.assertRaises(s.DiscordError):await s.send(self.doc['send_request']['id'])
            self.assertEqual(self.doc['delivery']['status'],'failed')
            request.assert_not_called()


if __name__ == '__main__':unittest.main()
