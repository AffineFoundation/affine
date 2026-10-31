"""Minimal async vLLM /v1/completions client with the gate's request shape (same body and stop handling as
validator/probe/probe_worker.py: one request per sample, n=1, its own seed, stop_token_ids = EOS, token ids returned).
Start a server per GPU (or one with tensor parallel) using the gate's flags (validator/probe/serve.sh):
  vllm serve ./head --served-model-name m --dtype bfloat16 --max-model-len 12288 --enable-prefix-caching --max-num-seqs 512
"""
import asyncio, hashlib, random
EOS = (248046, 248044)
H = dict(temperature=1.0, top_p=0.95, top_k=20, max_output_tokens=8192, max_context_tokens=12288)


def request_seed(plan_seed, env, task, j):
    """The gate's request seed rule: int(sha256('P:env:task:j')[:12], 16)."""
    return int(hashlib.sha256(f"{plan_seed}:{env}:{task}:{j}".encode()).hexdigest()[:12], 16)


async def sample_all(urls, jobs, *, model='m', concurrency=512):
    """jobs: [(key, prompt_ids, seed)] -> {key: (out_ids_with_eos, stop)}; stop is 'eos' or 'budget' (hit 8192 / context)."""
    import httpx
    sem = asyncio.Semaphore(concurrency); inflight = [0] * len(urls); res = {}
    async with httpx.AsyncClient(timeout=httpx.Timeout(3600.0), limits=httpx.Limits(max_connections=concurrency + 8)) as client:
        async def one(key, ids, seed):
            mt = min(H['max_output_tokens'], H['max_context_tokens'] - len(ids))
            body = dict(model=model, prompt=list(ids), n=1, max_tokens=mt, temperature=H['temperature'], top_p=H['top_p'],
                        top_k=H['top_k'], stop_token_ids=list(EOS), skip_special_tokens=False, return_token_ids=True, seed=seed)
            for attempt in range(200):
                async with sem:
                    ri = min(range(len(urls)), key=lambda i: inflight[i]); inflight[ri] += 1
                    try:
                        r = await client.post(urls[ri].rstrip('/') + '/v1/completions', json=body); r.raise_for_status()
                        ch = r.json()['choices'][0]; break
                    except (httpx.TransportError, httpx.HTTPStatusError):
                        pass
                    finally:
                        inflight[ri] -= 1
                await asyncio.sleep(2 + random.random() * 3)
            else:
                raise RuntimeError('vLLM request kept failing')
            out = list(ch.get('token_ids') or []); stop = 'budget' if ch.get('finish_reason') == 'length' else 'eos'
            if stop == 'eos' and (not out or out[-1] not in EOS):
                sr = ch.get('stop_reason'); out.append(sr if isinstance(sr, int) and sr in EOS else EOS[1])
            res[key] = (out, stop)
        await asyncio.gather(*(one(*j) for j in jobs))
    return res
