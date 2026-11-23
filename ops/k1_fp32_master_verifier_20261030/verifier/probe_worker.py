"""SN120 env probe worker (runs on the evaluator pod with the EVALUATOR's venv + pinned source for env sessions/graders).
Generation goes to a local vLLM server (separate venv/process). Grading reuses subnet.environments sessions and, for
affine_math, the evaluator's exact final-channel grader (subnet.mimo_answer_projection.grade).
usage: python probe_worker.py PLAN.json OUT.json [--progress P.json]
Plan fields: checkpoint_id, ckpt_dir, vllm_url, served_model, source_root, concurrency, grade_workers, harness,
             manifest_projection (manifest subset for the math grader), envs: [{env_id, spec, kind, indices, k, max_tokens}]
Output: per-env aggregates + per-task counts (private) + throughput stats. No rollout text is stored."""
import resource, random, asyncio, json, math, os, sys, time, multiprocessing as mp, statistics, hashlib, signal, traceback
PLAN = json.load(open(sys.argv[1])); OUT = sys.argv[2]
PROG = sys.argv[sys.argv.index('--progress') + 1] if '--progress' in sys.argv else None
PAUSE = PLAN.get('pause_file')
_OUT_DIR = os.path.dirname(os.path.abspath(OUT)) or '.'
FINALIZE = os.path.join(_OUT_DIR, 'FINALIZE')
GRADES_JL = os.path.join(_OUT_DIR, 'grades.jsonl')
sys.path.insert(0, PLAN['source_root']); sys.path.append('/root/affine-env-probe/extra-site')
EOS = (248046, 248044); OPEN_T, CLOSE_T = 248068, 248069
_G = {}
def _init(plan_path, verified=None):
    os.nice(10)
    p = json.load(open(plan_path)); sys.path.insert(0, p['source_root']); sys.path.append('/root/affine-env-probe/extra-site')
    if verified:
        # the parent already ran the evaluator's own _source_hash integrity check once per env for this run;
        # reuse that result instead of re-hashing the whole source tree on the (encrypted, slow) fs in every worker x env
        import subnet.environments as _envmod
        _orig = _envmod._source_hash
        _envmod._source_hash = lambda spec, _o=_orig: verified.get(spec.id) if spec.id in verified else _o(spec)
    from transformers import AutoTokenizer
    _G['tok'] = AutoTokenizer.from_pretrained(p['ckpt_dir']); _G['plan'] = p; _G['sessions'] = {}
    _G['envs'] = {e['env_id']: e for e in p['envs']}
    import atexit, multiprocessing.util
    def _close():
        for s in list(_G['sessions'].values()):
            try: s.close()
            except Exception: pass
        _G['sessions'].clear()
    atexit.register(_close)
    # pool workers leave via os._exit (atexit skipped); multiprocessing finalizers do run -> sandbox containers get removed
    multiprocessing.util.Finalize(None, _close, exitpriority=10)
def _session(env_id):
    from subnet.environments import EnvironmentSpec, create_session
    s = _G['sessions'].get(env_id)
    if s is None:
        s = _G['sessions'][env_id] = create_session(EnvironmentSpec.from_dict(_G['envs'][env_id]['spec']))
    return s
def prep(env_id, index):
    """Reset one task -> chat-template token ids (exact evaluator kwargs)."""
    s = _session(env_id); r = s.reset(index, 0)
    try: s.close_runtime()          # free the sandbox container right away (reset recreates it when grading)
    except Exception: pass
    if r.get('tools'): raise RuntimeError('probe is single-turn text only')
    ids = _G['tok'].apply_chat_template(r['messages'], tokenize=True, return_dict=False, add_generation_prompt=True,
                                        enable_thinking=_G['plan']['harness']['enable_thinking'])
    if isinstance(ids, dict): ids = ids['input_ids']
    return env_id, index, r['task_hash'], list(ids)
def _final_channel(tok, out, stop):
    if stop != 'eos' or not out or out[-1] not in EOS: return None, 'not-completed-eos'
    body = out[:-1]; special = set(getattr(tok, 'all_special_ids', ()))
    if any(t in special or t in EOS for t in body): return None, 'unexpected-control-token'
    if body.count(OPEN_T) != 1 or body.count(CLOSE_T) != 1: return None, 'thinking-marker-count'
    a, b = body.index(OPEN_T), body.index(CLOSE_T)
    if a >= b or tok.decode(body[:a], skip_special_tokens=False).strip(): return None, 'thinking-marker-order'
    final = tok.decode(body[b + 1:], skip_special_tokens=False).strip()
    if not final: return None, 'empty-final'
    return final, 'ok'
def grade(env_id, index, out, stop):
    try: return _grade(env_id, index, out, stop)
    finally:
        s = _G['sessions'].get(env_id)
        if s is not None:
            try: s.close_runtime()
            except Exception: pass
def _grade(env_id, index, out, stop):
    try:
        e = _G['envs'][env_id]; s = _session(env_id); s.reset(index, 0); tok = _G['tok']
        if e['kind'] == 'live-math':
            from subnet.mimo_answer_projection import grade as g
            _, kind, reward = g(s, _G['plan']['manifest_projection'], _G['plan']['harness'], tok, out, stop)
            if kind != 'positive':
                try:
                    if _lenient_positive(index, tok, out, stop): kind = 'positive'
                except Exception:
                    pass
            return env_id, index, kind, None
        final, why = _final_channel(tok, out, stop)
        if final is None: return env_id, index, 'neutral', why
        st = s.step({'text': final})
        if not st.get('done'): return env_id, index, 'neutral', 'not-done'
        return env_id, index, ('positive' if st['classification'] == 'positive' else 'negative' if st['classification'] == 'negative' else 'neutral'), None
    except Exception as x:
        return env_id, index, 'indeterminate', f'{type(x).__name__}: {str(x)[:120]}'


# ---- affine_math verifier fix (Allan catoneone/sn120-k1-fixes 786cdb9 code/lenient_grader.py, verbatim P1/P2/E1-E7; Jacob t831u 2026-10-30)
# Only runs when the live native grade is NOT positive and the reply finished with EOS (capped replies stay unresolved). If the lenient
# projection + equivalence rules find the last complete box equal to the gold, the sample is graded 'positive'. Head and candidate runs
# use this same file (it is part of the gate's probe fingerprint). Revert = restore probe_worker.py.bak-pre-verifierfix-*.
VERIFIER_FIX = 'allan-lenient-v1-786cdb9'
_AV = {}
def _allan():
    if 'ns' not in _AV:
        ns = {'__name__': 'allan_lenient_v1'}
        exec(compile('import re\nfrom math_verify import parse, verify\n' + _ALLAN_SRC, 'allan_lenient_v1', 'exec'), ns)
        _AV['ns'] = ns
        _AV['tasks'] = json.load(open(os.path.join(PLAN['source_root'], 'assets', 'original-math7496.tasks.json')))
    return _AV['ns'], _AV['tasks']
def _lenient_positive(index, tok, out, stop):
    if stop != 'eos' or not out or out[-1] not in EOS: return None
    ns, tasks = _allan()
    text = tok.decode(out[:-1], skip_special_tokens=False)
    data = tasks[index]['data']
    gbox, viol, pfix = ns['lenient_project'](text)
    if gbox is None: return None
    ok2, efix = ns['lenient_equal'](data['answer'], gbox, data['prompt'])
    if not ok2: return None
    return '+'.join(x for x in (pfix, efix) if x) or 'live'
_ALLAN_SRC = 'OPEN = "\\\\boxed{"\n\n\n# ---------- live: final-channel projection (mimo_answer_projection.project / _boxed_final) ----------\ndef _box_end(text, start):\n    depth = 1\n    for at in range(start, len(text)):\n        if text[at] == "{":\n            depth += 1\n        elif text[at] == "}":\n            depth -= 1\n            if depth == 0:\n                return at\n    return None\n\n\ndef live_project(text):\n    """Return (grading_text, reason). grading_text is None when the live rule marks the reply unresolved."""\n    if text.count("<think>") != 1 or text.count("</think>") != 1:\n        return None, "think"\n    a, b = text.index("<think>"), text.index("</think>")\n    if a >= b or text[:a].strip():\n        return None, "think"\n    final = text[b + len("</think>"):].strip()\n    if final.count(OPEN) != 1:\n        return None, "n_boxed"\n    begin = final.index(OPEN)\n    at = _box_end(final, begin + len(OPEN))\n    if at is None or not final[begin + len(OPEN):at].strip():\n        return None, "unclosed"\n    tail, prefix = final[at + 1:].strip(), final[:begin].rstrip()\n    if tail:\n        opening = {"\\\\]": "\\\\[", "\\\\)": "\\\\(", "$$": "$$", "$": "$"}.get(tail)\n        if opening is None or not prefix.endswith(opening):\n            return None, "tail"\n    elif prefix.endswith(("\\\\[", "\\\\(", "$")):\n        return None, "prefix"\n    return final[begin:at + 1], "ok"\n\n\n# ---------- fixes: projection ----------\ndef _unclosed_dollar(prefix):\n    """True if the text before the box leaves a `$` / `$$` open (parity of unescaped `$`)."""\n    return re.sub(r"\\\\\\$", "", prefix).count("$") % 2 == 1\n\n\ndef lenient_project(text):\n    """Return (grading_text, format_violation, fix). format_violation is None when the reply follows the protocol."""\n    box, why = live_project(text)\n    if box is not None:\n        return box, None, None\n    if why == "prefix":  # P1\n        final = text.split("</think>")[-1].strip()\n        begin = final.index(OPEN)\n        if not _unclosed_dollar(final[:begin]) and not final[:begin].rstrip().endswith(("\\\\[", "\\\\(")):\n            at = _box_end(final, begin + len(OPEN))\n            return final[begin:at + 1], None, "P1"\n    if why in ("tail", "n_boxed", "think", "prefix"):  # P2\n        # no <think>: the whole reply is the answer channel; several </think>: use the last segment\n        final = text.split("</think>")[-1]\n        at0 = final.rfind(OPEN)\n        if at0 >= 0:\n            at = _box_end(final, at0 + len(OPEN))\n            if at is not None and final[at0 + len(OPEN):at].strip():\n                return final[at0:at + 1], why, "P2"\n    return None, why, None\n\n\n# ---------- live: verify.py ----------\n_cache = {}\n\n\ndef mv_equal(gold, pred_box):\n    """verify.py: True / False, or None where verify.py exits 75 (indeterminate)."""\n    key = (gold, pred_box)\n    if key not in _cache:\n        try:\n            g = parse(OPEN + gold + "}", raise_on_error=True)\n            p = parse(pred_box, raise_on_error=True)\n            _cache[key] = bool(g) and bool(verify(g, p, raise_on_error=True))\n        except Exception:\n            _cache[key] = None\n    return _cache[key]\n\n\n# ---------- fixes: answer equivalence ----------\ndef inner(box):\n    return box[len(OPEN):-1]\n\n\ndef strip_text(s):\n    s = re.sub(r"\\\\(?:text|textbf|mathrm|mbox)\\s*\\{([^{}]*)\\}", r"\\1", s)\n    for _ in range(2):  # 1,000,000 needs two passes\n        s = re.sub(r"(\\d),\\\\!(\\d{3})", r"\\1\\2", s)\n        s = re.sub(r"(\\d)\\{,\\}(\\d{3})", r"\\1\\2", s)\n    s = s.replace("\\\\dfrac", "\\\\frac").replace("\\\\tfrac", "\\\\frac")\n    return s.strip()\n\n\nLETTERS = re.compile(r"^\\(?([A-Z])\\)?(\\s*,\\s*\\(?([A-Z])\\)?)*$")\n\n\ndef letter_set(s):\n    s = strip_text(s).replace("\\\\ ", " ").strip().strip("$")\n    if LETTERS.match(s.replace(" ", "")):\n        return frozenset(re.findall(r"[A-Z]", s))\n    return None\n\n\ndef mc_options(prompt):\n    """Options written as \\\\textbf{(A)}\\\\ xxx or \\\\text A. xxx -> {letter: text}."""\n    opts = {}\n    for m in re.finditer(r"\\\\textbf\\{\\(([A-E])\\)\\}\\\\?\\s*(.*?)(?=\\\\qquad|\\\\textbf\\{\\([A-E]\\)\\}|\\$\\s*$|$)", prompt, re.S):\n        opts[m.group(1)] = m.group(2).strip().rstrip("$").strip()\n    if not opts:\n        for m in re.finditer(r"\\\\text\\s*([A-E])\\.\\s*(?:\\\\ )*\\s*(.*?)(?=\\\\qquad|\\\\text\\s*[A-E]\\.|\\$\\$|$)", prompt, re.S):\n            opts[m.group(1)] = m.group(2).strip()\n    return opts\n\n\ndef pmatrix_to_tuple(s):\n    def col(m):\n        rows = [r.strip() for r in m.group(1).split("\\\\\\\\")]\n        return "(" + ", ".join(rows) + ")" if all("&" not in r for r in rows) else m.group(0)\n    return re.sub(r"\\\\begin\\{[pb]matrix\\}(.*?)\\\\end\\{[pb]matrix\\}", col, s, flags=re.S)\n\n\ndef ineq_to_interval(s):\n    s = strip_text(s).replace(" ", "")\n    m = re.fullmatch(r"[a-z](?:\\\\neq|\\\\ne)(-?[\\w\\\\{}.]+)", s)\n    if m:\n        c = m.group(1)\n        return f"(-\\\\infty,{c})\\\\cup({c},\\\\infty)"\n    m = re.fullmatch(r"(-?[\\w\\\\{}.]+)<[a-z]<(-?[\\w\\\\{}.]+)", s)\n    if m:\n        return f"({m.group(1)},{m.group(2)})"\n    m = re.fullmatch(r"[a-z](<|>|\\\\le|\\\\ge|\\\\leq|\\\\geq)(-?[\\w\\\\{}.]+)", s)\n    if m:\n        op, c = m.groups()\n        return {"<": f"(-\\\\infty,{c})", ">": f"({c},\\\\infty)", "\\\\le": f"(-\\\\infty,{c}]", "\\\\leq": f"(-\\\\infty,{c}]",\n                "\\\\ge": f"[{c},\\\\infty)", "\\\\geq": f"[{c},\\\\infty)"}[op]\n    return None\n\n\ndef scalar_identity(pred, gold):\n    m = re.fullmatch(r"(-?\\d+)\\s*\\\\?(?:mathbf\\{I\\}|I)", strip_text(pred).replace(" ", ""))\n    g = re.fullmatch(r"\\\\begin\\{pmatrix\\}(.*)\\\\end\\{pmatrix\\}", gold.strip(), re.S)\n    if not (m and g):\n        return None\n    n, k = len(g.group(1).split("\\\\\\\\")), m.group(1)\n    rows = [" & ".join(k if i == j else "0" for j in range(n)) for i in range(n)]\n    return "\\\\begin{pmatrix} " + " \\\\\\\\ ".join(rows) + " \\\\end{pmatrix}"\n\n\ndef complex_equal(pred, gold):\n    try:\n        import sympy\n        from latex2sympy2_extended import latex2sympy\n        a = latex2sympy(strip_text(pred)).subs(sympy.Symbol("i"), sympy.I)\n        b = latex2sympy(strip_text(gold)).subs(sympy.Symbol("i"), sympy.I)\n        return sympy.simplify(sympy.expand(a - b)) == 0\n    except Exception:\n        return False\n\n\ndef lenient_equal(gold, pred_box, prompt):\n    """Return (correct, fix). The live verify.py is tried first; fixes only run when it says no."""\n    if mv_equal(gold, pred_box):\n        return True, None\n    pred = inner(pred_box)\n    if mv_equal(strip_text(gold), OPEN + strip_text(pred) + "}"):\n        return True, "E1"\n    gs, ps = letter_set(gold), letter_set(pred)\n    if gs and ps and gs == ps:\n        return True, "E2"\n    opts = mc_options(prompt)\n    if opts:\n        p = strip_text(pred)\n        if re.match(r"^\\s*\\\\?(?:text|textbf)?\\{?\\(?[A-E]\\)", p):\n            p = re.sub(r"^\\s*\\\\?(?:text|textbf)?\\{?\\(?([A-E])\\)?\\}?\\s*(\\\\\\s)?\\s*", "", p)\n        ls = letter_set(pred)\n        if ls and len(ls) == 1:\n            p = opts.get(next(iter(ls)), p)\n        if p and mv_equal(gold, OPEN + p + "}"):\n            return True, "E3"\n    if mv_equal(pmatrix_to_tuple(gold), OPEN + pmatrix_to_tuple(pred) + "}"):\n        return True, "E4"\n    iv = ineq_to_interval(pred)\n    if iv and mv_equal(gold, OPEN + iv + "}"):\n        return True, "E5"\n    m = scalar_identity(pred, gold)\n    if m and mv_equal(gold, OPEN + m + "}"):\n        return True, "E6"\n    if "i" in gold and "i" in pred and complex_equal(pred, gold):\n        return True, "E7"\n    return False, None\n'

PREP_SWITCH = '/root/affine-env-probe/PREP-CACHE.json'; PREP_DIR = '/root/affine-env-probe/prepcache'
def _prep_cache_key(P, e, verified):
    h = hashlib.sha256()
    tf = {}
    for n in sorted(os.listdir(P['ckpt_dir'])):
        if n.startswith('tokenizer') or n in ('chat_template.jinja', 'special_tokens_map.json', 'added_tokens.json', 'vocab.json', 'merges.txt'):
            tf[n] = hashlib.sha256(open(os.path.join(P['ckpt_dir'], n), 'rb').read()).hexdigest()
    h.update(json.dumps(dict(v=1, env=e['env_id'], source_hash=verified.get(e['env_id']), spec=e['spec'], tok=tf,
                             thinking=P['harness']['enable_thinking'], harness=P['harness'].get('version'),
                             source_root=P['source_root']), sort_keys=True).encode())
    return h.hexdigest()
def _prep_cache_load(P, verified):
    try:
        if not json.load(open(PREP_SWITCH)).get('on'): return None
    except Exception: return None
    os.makedirs(PREP_DIR, exist_ok=True); PC = {}
    for e in P['envs']:
        k = _prep_cache_key(P, e, verified); path = os.path.join(PREP_DIR, k + '.json')
        try: rows = json.load(open(path))['rows']
        except Exception: rows = {}
        PC[e['env_id']] = dict(key=k, path=path, rows=rows, new=0)
    return PC
def _prep_cache_save(PC):
    for env_id, c in PC.items():
        if not c['new']: continue
        tmp = c['path'] + '.tmp.%d' % os.getpid()
        json.dump(dict(key=c['key'], env_id=env_id, rows=c['rows']), open(tmp, 'w')); os.replace(tmp, c['path'])
def _raise_nofile():
    # thousands of concurrent HTTP connections at 8 replicas x 512: the default soft limit (1024) gives EMFILE -> ConnectError
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    want = hard if hard != resource.RLIM_INFINITY else 1 << 20
    try: resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, min(want, 1 << 20)), hard))
    except Exception: pass
async def main():
    _raise_nofile()
    import httpx
    P = PLAN; t_start = time.time()
    # separate pools so a grading backlog never starves prep (which keeps the GPUs fed)
    gw = P.get('grade_workers', 32); ctx = mp.get_context('spawn')
    from subnet.environments import EnvironmentSpec, _source_hash
    verified = {}
    for e in P['envs']:
        sp = EnvironmentSpec.from_dict(e['spec']); h = _source_hash(sp)
        if h != sp.source_hash: raise ValueError(f"trusted environment code or data hash mismatch: {e['env_id']}")
        verified[sp.id] = h
    pool = ctx.Pool(gw, initializer=_init, initargs=(sys.argv[1], verified))
    # v2d prep cache (switch file PREP-CACHE.json {"on": true}): prompt token ids of (env code/data hash, spec, tokenizer
    # files, harness) are deterministic, and the gate probes the same standing sample every turn -> reuse them instead of
    # re-running env reset + chat template per task in a cold process pool (the run's serial bottleneck).
    PC = _prep_cache_load(P, verified)
    prep_pool = None if (PC is not None and all(str(i) in PC[e['env_id']]['rows'] for e in P['envs'] for i in e['indices'])) else \
        ctx.Pool(max(4, gw // 4), initializer=_init, initargs=(sys.argv[1], verified))
    # docker-sandbox graders (i3code/scicomp run each test case in a container; minutes per grade under load) get their own
    # bounded pool so they cannot occupy every grade worker and stall the fast in-process graders of the other envs
    SANDBOX = set(P.get('sandbox_envs', ['affine_i3code', 'affine_scicomp'])) & {e['env_id'] for e in P['envs']}
    sb_pool = ctx.Pool(P.get('sandbox_grade_workers', 32), initializer=_init, initargs=(sys.argv[1], verified)) if SANDBOX else None
    loop = asyncio.get_running_loop()
    def apool(fn, *a):
        f = loop.create_future()
        (prep_pool if fn is prep else sb_pool if (sb_pool is not None and a[0] in SANDBOX) else pool).apply_async(fn, a, callback=lambda r: loop.call_soon_threadsafe(f.set_result, r),
                         error_callback=lambda x: loop.call_soon_threadsafe(f.set_exception, x))
        return f
    stats = {e['env_id']: dict(tasks=0, samples=0, correct=0, wrong=0, unresolved=0, budget_hits=0, errors=0, out_tokens=0,
                               prompt_tokens=0, per_task={}, unresolved_reasons={}) for e in P['envs']}
    # continuous-partial resume: reload grades.jsonl so a later full/partial publish includes prior work
    if os.path.exists(GRADES_JL):
        for line in open(GRADES_JL):
            try: g = json.loads(line)
            except Exception: continue
            env_id = g.get('env_id');
            if env_id not in stats: continue
            idx = str(g['index']); st = stats[env_id]
            if idx in st['per_task']: continue
            pt = {k: g[k] for k in ('n', 'c', 'w', 'u', 'b', 'x', 'tok', 'L') if k in g}
            st['per_task'][idx] = pt
            st['tasks'] += 1; st['samples'] += pt.get('n', 0); st['correct'] += pt.get('c', 0); st['wrong'] += pt.get('w', 0)
            st['unresolved'] += pt.get('u', 0); st['budget_hits'] += pt.get('b', 0); st['errors'] += pt.get('x', 0)
            st['out_tokens'] += pt.get('tok', 0)
    gen_tokens = 0; done_samples = sum(st['samples'] for st in stats.values()); lat = []; preempted = [0]; retries = [0]; gen_done = [0]
    # data-parallel: one vLLM replica per GPU; each request goes to the least-loaded replica
    URLS = P.get('vllm_urls') or [P['vllm_url']]
    inflight = [0] * len(URLS); served_by = [0] * len(URLS)
    sem = asyncio.Semaphore(P['concurrency'])
    client = httpx.AsyncClient(timeout=httpx.Timeout(3600.0), limits=httpx.Limits(max_connections=P['concurrency'] + 8))
    async def one_task(e, index):
        nonlocal gen_tokens, done_samples
        env_id = e['env_id']; st = stats[env_id]
        try:
            hit = PC[env_id]['rows'].get(str(index)) if PC is not None else None
            if hit is not None: th, ids = hit[0], list(hit[1])
            else:
                _, _, th, ids = await apool(prep, env_id, index)
                if PC is not None: PC[env_id]['rows'][str(index)] = [th, list(ids)]; PC[env_id]['new'] += 1
        except Exception as x:
            st['errors'] += e['k']; st['per_task'][str(index)] = dict(err=f'prep {type(x).__name__}'); return
        max_tok = min(e['max_tokens'], P['harness']['max_context_tokens'] - len(ids))
        if max_tok <= 0: st['errors'] += e['k']; return
        base = dict(model=P['served_model'], prompt=ids, n=1, max_tokens=max_tok, temperature=P['harness']['temperature'],
                    top_p=P['harness']['top_p'], top_k=P['harness']['top_k'], stop_token_ids=list(EOS), skip_special_tokens=False,
                    return_token_ids=True)
        async def sample(j):
            # one request per sample with its own public seed (vLLM n>1 + seed can repeat samples)
            body = dict(base, seed=int(hashlib.sha256(f"{P['seed']}:{env_id}:{index}:{j}".encode()).hexdigest()[:12], 16))
            while True:
                while PAUSE and os.path.exists(PAUSE): await asyncio.sleep(5)
                async with sem:
                    if PAUSE and os.path.exists(PAUSE): continue
                    ri = min(range(len(URLS)), key=lambda i: inflight[i]); inflight[ri] += 1
                    try:
                        t0 = time.time(); req = asyncio.create_task(client.post(URLS[ri] + '/v1/completions', json=body))
                        while not req.done():
                            await asyncio.wait({req}, timeout=5)
                            if not req.done() and PAUSE and os.path.exists(PAUSE):
                                req.cancel(); preempted[0] += 1; break    # paused: abort in-flight, retry later
                    finally:
                        inflight[ri] -= 1
                    if req.cancelled(): continue
                    try:
                        r = req.result(); r.raise_for_status()
                    except (httpx.TransportError, httpx.HTTPStatusError) as x:
                        # transient replica/connection failure: back off and resend (same seed -> same sample)
                        retries[0] += 1
                        if retries[0] > 2000: raise
                        await asyncio.sleep(2 + random.random() * 3); continue
                    lat.append(time.time() - t0); served_by[ri] += 1
                    return r.json()['choices'][0]
        choices = await asyncio.gather(*(sample(j) for j in range(e['k']))); gen_done[0] += len(choices)
        j = dict(choices=choices)
        st['tasks'] += 1; st['prompt_tokens'] += len(ids); pt = dict(n=0, c=0, w=0, u=0, b=0, x=0, tok=0, L=[])
        futs = []
        for ch in j['choices']:
            out = list(ch.get('token_ids') or [])
            fr = ch.get('finish_reason')
            stop = 'budget' if fr == 'length' else 'eos'
            if stop == 'eos' and (not out or out[-1] not in EOS):
                sr = ch.get('stop_reason'); out.append(sr if isinstance(sr, int) and sr in EOS else EOS[1])
            gen_tokens += len(out); pt['tok'] += len(out); pt['L'].append(len(out)); st['out_tokens'] += len(out)
            if stop == 'budget': st['budget_hits'] += 1; pt['b'] += 1
            futs.append(apool(grade, env_id, index, out, stop))
        for env_, idx_, kind, why in await asyncio.gather(*futs):
            st['samples'] += 1; pt['n'] += 1; done_samples += 1
            if kind == 'positive': st['correct'] += 1; pt['c'] += 1
            elif kind == 'negative': st['wrong'] += 1; pt['w'] += 1
            elif kind == 'neutral':
                st['unresolved'] += 1; pt['u'] += 1
                if why: st['unresolved_reasons'][why] = st['unresolved_reasons'].get(why, 0) + 1
            else:
                st['errors'] += 1; pt['x'] += 1; st['unresolved_reasons'][why or 'err'] = st['unresolved_reasons'].get(why or 'err', 0) + 1
        st['per_task'][str(index)] = pt
    stop_reason = [None]
    def _request_stop(why):
        if stop_reason[0] is None:
            stop_reason[0] = why
    def _on_sig(signum, _frame):
        _request_stop('signal:%s' % signum)
    signal.signal(signal.SIGTERM, _on_sig)
    signal.signal(signal.SIGINT, _on_sig)

    # append-only per-task grades so a crash/finalize can still publish a partial
    grades_lock = asyncio.Lock()
    async def _flush_grade(env_id, index, pt):
        line = json.dumps(dict(env_id=env_id, index=index, **pt), sort_keys=True) + '\n'
        async with grades_lock:
            with open(GRADES_JL, 'a') as f: f.write(line)

    _one_task_orig = one_task
    async def one_task(e, index):
        if stop_reason[0] or os.path.exists(FINALIZE):
            _request_stop(stop_reason[0] or 'finalize')
            return
        try:
            await _one_task_orig(e, index)
        except asyncio.CancelledError:
            return
        st = stats[e['env_id']]; pt = st['per_task'].get(str(index))
        if isinstance(pt, dict) and pt.get('n'):
            await _flush_grade(e['env_id'], index, pt)

    async def progress():
        while True:
            await asyncio.sleep(5)
            if os.path.exists(FINALIZE):
                _request_stop('finalize')
            if PROG:
                el = time.time() - t_start
                json.dump(dict(at=time.time(), elapsed=el, inflight=list(inflight), retries=retries[0], generated=gen_done[0],
                               preempted=preempted[0], paused=bool(PAUSE and os.path.exists(PAUSE)),
                               finalize=bool(stop_reason[0]), stop_reason=stop_reason[0],
                               samples=done_samples, gen_tokens=gen_tokens, tok_s=gen_tokens / max(el, 1)), open(PROG + '.tmp', 'w')); os.replace(PROG + '.tmp', PROG)

    pj = asyncio.create_task(progress())
    order = [(e, i) for e in P['envs'] for i in e['indices']]
    # interleave envs so partial runs still cover every env
    order.sort(key=lambda t: (t[0]['indices'].index(t[1]), t[0]['env_id']))
    pending = [asyncio.create_task(one_task(e, i)) for e, i in order]
    while pending:
        if stop_reason[0] or os.path.exists(FINALIZE):
            _request_stop(stop_reason[0] or 'finalize')
            for t in pending:
                if not t.done(): t.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            break
        done, pending_set = await asyncio.wait(pending, timeout=1.0, return_when=asyncio.FIRST_COMPLETED)
        pending = [t for t in pending_set]
    pj.cancel()
    if PC is not None: _prep_cache_save(PC)
    try: await client.aclose()
    except Exception: pass
    for pl in (pool, prep_pool, sb_pool):
        if pl is None: continue
        try: pl.terminate(); pl.join()
        except Exception:
            try: pl.close(); pl.join()
            except Exception: pass
    el = time.time() - t_start
    # drop in-flight partial tasks that never wrote a complete per_task record
    for env_id, st in stats.items():
        st['per_task'] = {k: v for k, v in st['per_task'].items() if isinstance(v, dict) and v.get('n')}
        # recompute aggregates from completed per_task only (cancel mid-task can leave counters ahead)
        st['tasks'] = len(st['per_task']); st['samples'] = st['correct'] = st['wrong'] = st['unresolved'] = st['budget_hits'] = st['errors'] = 0
        st['out_tokens'] = 0
        for v in st['per_task'].values():
            st['samples'] += v.get('n', 0); st['correct'] += v.get('c', 0); st['wrong'] += v.get('w', 0)
            st['unresolved'] += v.get('u', 0); st['budget_hits'] += v.get('b', 0); st['errors'] += v.get('x', 0)
            st['out_tokens'] += v.get('tok', 0)
    done_samples = sum(st['samples'] for st in stats.values())
    envs = {}
    for env_id, st in stats.items():
        n = st['samples']; graded = st['correct'] + st['wrong'] + st['unresolved']
        p = st['correct'] / graded if graded else None
        pts = [v for v in st['per_task'].values() if isinstance(v, dict) and v.get('n')]
        se = None
        if p is not None and len(pts) > 1:
            m = len(pts); rs = [v['c'] / v['n'] for v in pts]; mean = sum(rs) / m
            se = math.sqrt(sum((r - mean) ** 2 for r in rs) / (m - 1) / m)
        envs[env_id] = dict({k: v for k, v in st.items() if k not in ('per_task',)}, pass_rate=p, se_task_clustered=se,
                            mean_out_tokens=(st['out_tokens'] / n) if n else None, mean_prompt_tokens=(st['prompt_tokens'] / st['tasks']) if st['tasks'] else None,
                            per_task=st['per_task'])
    planned = sum(len(e['indices']) * e['k'] for e in P['envs'])
    partial = bool(stop_reason[0]) or done_samples < planned
    res = dict(version='sn120-env-probe-worker-v1', checkpoint_id=P['checkpoint_id'], seed=P['seed'], started_at=t_start, finished_at=time.time(),
               wall_seconds=el, gen_tokens=gen_tokens, samples=done_samples, tok_per_s=gen_tokens / max(el, 1e-9), rollouts_per_min=done_samples / max(el, 1e-9) * 60,
               request_latency_s=dict(p50=statistics.median(lat) if lat else None, max=max(lat) if lat else None),
               concurrency=P['concurrency'], replicas=len(URLS), requests_per_replica=served_by, transient_retries=retries[0], preempted_requests=preempted[0],
               partial=partial, stop_reason=stop_reason[0], planned_samples=planned, envs=envs,
               prep_cache=(None if PC is None else {k: dict(key=v['key'][:16], rows=len(v['rows']), new=v['new']) for k, v in PC.items()}))
    json.dump(res, open(OUT + '.tmp', 'w')); os.replace(OUT + '.tmp', OUT)
    print(json.dumps({k: v for k, v in res.items() if k != 'envs'}))
if __name__ == '__main__':
    asyncio.run(main())
