"""Fake vLLM /v1/completions: returns <think>..</think> final + EOS token ids."""
import json, sys
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
class H(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers['Content-Length']); self.rfile.read(n)
        body = json.dumps(dict(choices=[dict(token_ids=[248068, 5, 248069, 7, 248046], finish_reason='stop', stop_reason=248046)])).encode()
        self.send_response(200); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(body))); self.end_headers(); self.wfile.write(body)
    def log_message(self, *a): pass
ThreadingHTTPServer(('127.0.0.1', int(sys.argv[1])), H).serve_forever()
