import tempfile
import unittest
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from affine.dash.document import document_response


class DocumentTests(unittest.IsolatedAsyncioTestCase):
    async def test_html_and_raw_are_distinct_and_never_cached(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            source='# Affine\n\nDashboard: https://affine.io/\nCode: https://github.com/AffineFoundation/affine\n\n## Details\n\nText <script>alert(1)</script>\n\nCUBLAS_WORKSPACE_CONFIG=:4096:8 python -m worker \\\n  --state /private/state\n'
            (root/'llms.txt').write_text(source)
            (root/'llms-page.html').write_text('<html><article><!--DOCUMENT--></article></html>')
            app=FastAPI()
            @app.api_route('/llms.txt',methods=['GET','HEAD'])
            def document(request: Request):return document_response(request,root)
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url='http://test') as client:
                for accept in ('*/*','text/plain','text/plain, text/html;q=.5','text/html;q=0','application/json'):
                    response=await client.get('/llms.txt',headers={'Accept':accept})
                    self.assertEqual(response.content,source.encode())
                    self.assertTrue(response.headers['content-type'].startswith('text/plain'))
                    self.assertEqual(response.headers['vary'],'Accept')
                    self.assertIn('no-store',response.headers['cache-control'])
                html=await client.get('/llms.txt',headers={'Accept':'text/html,application/xhtml+xml,*/*;q=.8'})
                self.assertTrue(html.headers['content-type'].startswith('text/html'))
                self.assertIn('<h1>Affine</h1>',html.text)
                self.assertIn('<h2>Details</h2>',html.text)
                self.assertIn('href="https://affine.io/"',html.text)
                self.assertIn('<br />',html.text)
                self.assertNotIn('<script>',html.text)
                self.assertIn('&lt;script&gt;',html.text)
                self.assertIn('<pre><code',html.text)
                self.assertIn('  --state /private/state',html.text)
                raw=await client.get('/llms.txt?raw=1',headers={'Accept':'text/html'})
                self.assertEqual(raw.content,source.encode())
                (root/'llms.txt').write_text('# Updated document\n')
                updated=await client.get('/llms.txt',headers={'Accept':'text/html'})
                self.assertIn('Updated document',updated.text)
                head=await client.head('/llms.txt')
                self.assertEqual(head.status_code,200)
                self.assertEqual(head.content,b'')
                (root/'llms.txt').unlink()
                missing=await client.get('/llms.txt')
                self.assertEqual(missing.status_code,404)
