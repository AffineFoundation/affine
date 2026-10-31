#!/usr/bin/env python3
"""A fake vLLM /v1/completions server for CPU tests of the kit plumbing (NOT a model).
It decodes the prompt ids, finds the pool task in them, and answers correctly with probability p(task) seeded by the request
seed, so head/candidate runs are reproducible. usage: fake_vllm.py TOKENIZER_DIR PORT SKILL   (SKILL shifts p up/down)."""
import hashlib, json, sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import prompts  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
tok = AutoTokenizer.from_pretrained(sys.argv[1], local_files_only=True); PORT = int(sys.argv[2]); SKILL = float(sys.argv[3])
POOL = prompts.pool(); BY = {t['prompt']: (i, t['answer']) for i, t in enumerate(POOL)}


def u(*a):
    return int(hashlib.sha256(':'.join(map(str, a)).encode()).hexdigest()[:12], 16) / 16 ** 12


class Hd(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_POST(self):
        b = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        text = tok.decode(b['prompt'])
        body = text.split('<|im_start|>user\n', 1)[1].rsplit('<|im_end|>', 1)[0]
        if body.startswith(prompts.MANDATE + '\n\n'):
            body = body[len(prompts.MANDATE) + 2:]
        i, ans = BY[body]
        p = min(0.98, max(0.02, u('task', i) + SKILL))
        r = u('req', b['seed'])
        if r < 0.05:   # ran out of budget
            ids = tok('<think>\n' + 'hmm ' * 20, add_special_tokens=False)['input_ids']; fr = 'length'
        else:
            out = '<think>\nreasoning\n</think>\n\n\\boxed{%s}' % (ans if r < 0.05 + p * 0.95 else '123456789')
            ids = tok(out, add_special_tokens=False)['input_ids'] + [248046]; fr = 'stop'
        raw = json.dumps({'choices': [{'token_ids': ids, 'finish_reason': fr, 'stop_reason': None}]}).encode()
        self.send_response(200); self.send_header('Content-Type', 'application/json'); self.send_header('Content-Length', str(len(raw)))
        self.end_headers(); self.wfile.write(raw)


ThreadingHTTPServer(('127.0.0.1', PORT), Hd).serve_forever()
