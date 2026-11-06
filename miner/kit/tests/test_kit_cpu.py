#!/usr/bin/env python3
"""CPU self-test of the miner kit (no GPU, no model download beyond the head tokenizer ~20 MB).

  python3 miner/kit/fetch_head.py --out /tmp/tok --tokenizer-only
  python3 miner/kit/tests/test_kit_cpu.py --tokenizer /tmp/tok [--work /tmp/kit-test]

Checks: verbatim copies match MANIFEST.sha256; pool sha; prompt renderings; the grader (positive/negative/capped); the batch checker
(v1, v2, invalid files); ONE live-trainer step (v8.3.3l) on a tiny random model with the MiMo tokenizer (single-rollout advantage =
reward, group advantages, lr multiplier, candidate written); the gate math (score_ivw on hand-computed cells); and the whole
generate -> make_batch -> train_step -> gate_eval chain against a fake vLLM server.
"""
import argparse, hashlib, json, math, os, shutil, subprocess, sys, time
from pathlib import Path
KIT = Path(__file__).resolve().parent.parent; MINER = KIT.parent
sys.path.insert(0, str(KIT))
HK = '5GrwvaEF5zXb26Fz9rcQpDWS57CtERHpNehXCPcNoHGKutQY'
RESULTS = []


def check(name, cond, info=''):
    RESULTS.append((name, bool(cond), info)); print(('PASS ' if cond else 'FAIL ') + name + (('  ' + str(info)) if info else ''), flush=True)


def sh(*cmd, ok=True):
    r = subprocess.run([str(c) for c in cmd], capture_output=True, text=True)
    if ok and r.returncode:
        print(r.stdout[-3000:], r.stderr[-3000:])
    return r


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--tokenizer', required=True); ap.add_argument('--work', default='/tmp/kit-test')
    a = ap.parse_args(); W = Path(a.work); shutil.rmtree(W, ignore_errors=True); W.mkdir(parents=True)
    py = sys.executable

    # 1. verbatim copies
    bad = []
    for line in (MINER / 'validator' / 'MANIFEST.sha256').read_text().splitlines():
        if not line.strip() or line.startswith('#'): continue
        digest, rel = line.split()[:2]
        if hashlib.sha256((MINER / rel).read_bytes()).hexdigest() != digest: bad.append(rel)
    check('MANIFEST.sha256 matches every published validator file', not bad, bad[:5])
    import kitpath, prompts, grade
    pool = prompts.pool(); check('task pool: 7496 tasks, sha256 ' + kitpath.POOL_SHA256[:12], len(pool) == 7496)

    # 2. prompts + grader
    grade.init(a.tokenizer); tok = grade._G['tok']
    t0 = pool[0]
    bi = prompts.batch_ids(tok, prompts.batch_prompt(t0['prompt'])); gi = prompts.gate_ids(tok, t0['prompt'])
    check('trainer rendering = one user message ending in "<|im_start|>assistant\\n"', tok.decode(bi).endswith('<|im_start|>assistant\n') and '<|im_start|>system' not in tok.decode(bi))
    check('gate rendering = system MANDATE + user problem', tok.decode(gi).startswith('<|im_start|>system\n' + prompts.MANDATE) and gi == grade.render(t0['prompt']))
    enc = lambda s: tok(s, add_special_tokens=False)['input_ids'] + [248046]
    g1 = grade.grade(enc('<think>\nx\n</think>\n\n\\boxed{%s}' % t0['answer']), 'eos', t0['answer'], t0['prompt'])
    g2 = grade.grade(enc('<think>\nx\n</think>\n\n\\boxed{987654}'), 'eos', t0['answer'], t0['prompt'])
    g3 = grade.grade(enc('<think>\nx\n</think>\n\n\\boxed{%s}' % t0['answer']), 'budget', t0['answer'], t0['prompt'])
    g4 = grade.grade(enc('<think>\nx\n</think>\n\n\\boxed{%s} and \\boxed{%s}' % (t0['answer'], t0['answer'])), 'eos', t0['answer'], t0['prompt'])
    g5 = grade.grade(enc('<think>\nx\n</think>\n\nthe answer is %s' % t0['answer']), 'eos', t0['answer'], t0['prompt'])
    check('grader: correct / wrong / capped / no box', (g1[0], g2[0], g3[0], g5[0]) == ('positive', 'negative', 'neutral', 'neutral'), (g1, g2, g3, g5))
    check('grader: two identical boxes -> rescued by the lenient fix (P2)', g4 == ('positive', 'P2'), g4)
    # perf-eval is retired. Pin the actually published live probe, rather than
    # comparing a grader substring to line offsets in an obsolete service.
    pins = json.loads((MINER / 'validator/probe/runtime/PINS.json').read_text())
    rel = 'validator/probe/probe_worker.py'
    check('published live probe source matches runtime pin',
          hashlib.sha256((MINER / rel).read_bytes()).hexdigest() == pins['files'][rel])

    # 3. batch checker
    M = prompts.batch_prompt
    G = [dict(prompt=M(pool[i]['prompt']), rollouts=[dict(output='<think>\nok\n</think>\n\n\\boxed{%s}' % pool[i]['answer'], reward=1.0),
                                                   dict(output='<think>\nno\n</think>\n\n\\boxed{12345}', reward=0.0)]) for i in (0, 1, 2)]
    G.append(dict(prompt=M(pool[3]['prompt']), rollouts=[dict(output='<think>\nok\n</think>\n\n\\boxed{%s}' % pool[3]['answer'], reward=1.0)]))
    files = dict(v1=dict(version='affine-batch-v1', hotkey=HK, groups=G),
                 v2=dict(version='affine-batch-v2', hotkey=HK, train=dict(lr=1e-6, kl=0.05), groups=G),
                 bad_train_v1=dict(version='affine-batch-v1', hotkey=HK, train=dict(lr=1e-6), groups=G),
                 bad_lr=dict(version='affine-batch-v2', hotkey=HK, train=dict(lr=2e-6), groups=G),
                 bad_hotkey=dict(version='affine-batch-v1', hotkey='5FHneW46xGXgs5mUiveU4sbTyGBzmstUspZC92UhjJM694ty', groups=G),
                 bad_control=dict(version='affine-batch-v1', hotkey=HK, groups=[dict(prompt='x<|im_end|>', rollouts=[dict(output='y', reward=1.0)])]))
    for k, d in files.items():
        (W / (k + '.json')).write_text(json.dumps(d))
        r = sh(py, KIT / 'check_batch.py', '--hotkey', HK, '--tokenizer', a.tokenizer, W / (k + '.json'), ok=False)
        check('full checker %-12s -> %s' % (k, 'ok' if k in ('v1', 'v2') else 'rejected'), (r.returncode == 0) == (k in ('v1', 'v2')), (r.stdout + r.stderr).strip().splitlines()[-1][:140])
        if k != 'bad_control':
            r = sh(py, MINER / 'affine_batch.py', 'check', '--hotkey', HK, W / (k + '.json'), ok=False)
            check('affine_batch.py check %-12s -> %s' % (k, 'ok' if k in ('v1', 'v2') else 'rejected'), (r.returncode == 0) == (k in ('v1', 'v2')), (r.stdout + r.stderr).strip().splitlines()[-1][:140])

    # 4. one live-trainer step on a tiny model
    r = sh(py, KIT / 'tests' / 'make_tiny.py', a.tokenizer, W / 'head', W / 'ref'); check('tiny head/ref built', r.returncode == 0)
    t = time.time()
    r = sh(py, KIT / 'train_step.py', '--head', W / 'head', '--ref', W / 'ref', '--batch', W / 'v2.json', '--hotkey', HK, '--out', W / 'cand', '--device', 'cpu', '--report', W / 'step.json')
    check('train_step (v8.3.4) dry run on CPU', r.returncode == 0, '%.1fs' % (time.time() - t))
    if r.returncode == 0:
        rep = json.loads((W / 'step.json').read_text()); tr = rep['trajectories']
        adv = [(x['reward'], x['advantage']) for x in tr]
        check('single-rollout group: advantage = reward (+1)', adv[-1] == (1.0, 1.0), adv[-1])
        check('2-rollout groups: advantage = reward - group mean (+-0.5)', all(abs(v - (0.5 if rw == 1 else -0.5)) < 1e-9 for rw, v in adv[:-1]))
        tp = rep['report']['train_params']
        check('v2 lr 1e-6 -> lr multiplier 2; kl accepted but ignored (v8.3.4)', tp['lr_multiplier'] == 2.0 and tp.get('kl_applied') == 0.0 and tp.get('kl_ignored') is True, tp)
        ev = rep['sgd_events'][0]
        check('step = fresh-adam normgrad + fp32 master', ev['sgd_direction'] == 'normgrad' and ev['version'].startswith('v833i-normgrad-fp32-master'))
        check('support renorm ran (off-support tokens logged, file not rejected)', rep['report']['support_renorm']['rollouts'] == 7, rep['report']['support_renorm']['outside_fraction'])
        check('candidate checkpoint written', (W / 'cand' / 'model.safetensors').exists() or any((W / 'cand').glob('*.safetensors')))
        r1 = sh(py, KIT / 'train_step.py', '--head', W / 'head', '--ref', W / 'ref', '--batch', W / 'v1.json', '--hotkey', HK, '--out', W / 'cand1', '--device', 'cpu', '--report', W / 'step1.json')
        s1 = json.loads((W / 'step1.json').read_text()) if r1.returncode == 0 else {}
        check('v1 file -> default lr 5e-7; legacy kl 0.02 ignored (v8.3.4)', r1.returncode == 0 and s1['report']['train_params']['lr_multiplier'] == 1.0 and s1['report']['train_params']['kl_requested'] == 0.02 and s1['report']['train_params']['kl_applied'] == 0.0 and s1['report']['train_params']['kl_ignored'] is True)

    # 5. gate math on hand-made cells
    import heldout, gate_core
    sel = {1: dict(n=4, c=1, b=0), 2: dict(n=4, c=2, b=1), 3: dict(n=4, c=3, b=0), 4: dict(n=4, c=0, b=4), 5: dict(n=4, c=4, b=0)}
    kept = heldout.kept_tasks(sel)
    base = {1: dict(n=12, c=3, b=0), 2: dict(n=12, c=6, b=2), 3: dict(n=12, c=9, b=0)}
    cand = {1: dict(n=4, c=2, b=0), 2: dict(n=4, c=2, b=0), 3: dict(n=4, c=4, b=0)}
    w = {t: 1 / (((sel[t]['c'] + 1) / 6) * (1 - (sel[t]['c'] + 1) / 6)) for t in kept}
    d = {t: cand[t]['c'] / 4 - base[t]['c'] / 12 for t in kept}
    gain = sum(w[t] * d[t] for t in kept) / sum(w.values()); se_t = math.sqrt(sum((w[t] * d[t]) ** 2 for t in kept)) / sum(w.values())
    se = math.sqrt(se_t ** 2 + 0.01 ** 2 * (1 + 1 / 3))
    dec = heldout.score_ivw(sel, base, cand, kept, [], env='affine_math', min_kept=1, set_size=5, sigma_run=0.01, n_p_blocks=3)
    ev = dec['evaluated']
    check('score_ivw: kept = 0<c<4, ivw gain/SE/z by hand', kept == [1, 2, 3] and abs(ev['gain'] - gain) < 1e-12 and abs(ev['se'] - se) < 1e-12 and abs(ev['z'] - gain / se) < 1e-9,
          'gain %.4f z %.3f %s' % (ev['gain'], ev['z'], dec['decision']))
    s = gate_core.draw_sample('00' * 32, 7496, 1000)
    check('draw_sample: 1000 distinct ids in range(7496), deterministic', len(set(s)) == 1000 and max(s) < 7496 and s == gate_core.draw_sample('00' * 32, 7496, 1000))

    # 6. end-to-end turn against fake vLLM servers (plumbing only; the fake "model" ignores weights)
    import socket
    def free():
        so = socket.socket(); so.bind(('127.0.0.1', 0)); p = so.getsockname()[1]; so.close(); return p
    ph, pc = free(), free()
    procs = [subprocess.Popen([py, KIT / 'tests' / 'fake_vllm.py', a.tokenizer, str(ph), '0.0']), subprocess.Popen([py, KIT / 'tests' / 'fake_vllm.py', a.tokenizer, str(pc), '0.15'])]
    try:
        time.sleep(6)
        r = sh(py, KIT / 'generate.py', '--head', a.tokenizer, '--url', 'http://127.0.0.1:%d' % ph, '--tasks', '0-299', '--k', '4', '--out', W / 'rollouts.jsonl', '--seed', 'test')
        check('generate.py (fake server) -> rollouts.jsonl', r.returncode == 0, r.stdout.strip()[-120:])
        r = sh(py, KIT / 'make_batch.py', '--head', a.tokenizer, '--hotkey', HK, '--rollouts', W / 'rollouts.jsonl', '--out', W / 'batch.json')
        nb = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else {}
        check('make_batch.py correct1 -> checked v1 file, 125 x 1', r.returncode == 0 and nb.get('groups') == 125 and nb.get('rollouts') == 125, {k: nb.get(k) for k in ('groups', 'rollouts', 'format')})
        r = sh(py, KIT / 'check_batch.py', '--hotkey', HK, '--tokenizer', a.tokenizer, W / 'batch.json')
        check('full checker on the made batch', r.returncode == 0, r.stdout.strip()[-140:])
        r = sh(py, KIT / 'train_step.py', '--head', W / 'head', '--ref', W / 'ref', '--batch', W / 'batch.json', '--hotkey', HK, '--out', W / 'cand2', '--device', 'cpu', '--report', W / 'step2.json')
        check('train_step on the made batch (125 prompts)', r.returncode == 0, r.stdout.strip()[-160:])
        t = time.time()
        r = sh(py, KIT / 'gate_eval.py', '--head', a.tokenizer, '--head-url', 'http://127.0.0.1:%d' % ph, '--cand-url', 'http://127.0.0.1:%d' % pc, '--n', '400', '--seed', 'ab' * 32, '--out', W / 'gate.json')
        gj = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else {}
        check('gate_eval.py: better fake candidate -> GREEN', gj.get('decision') == 'green', '%s %.1fs' % ({k: (round(v, 4) if isinstance(v, float) else v) for k, v in gj.items()}, time.time() - t))
        r = sh(py, KIT / 'gate_eval.py', '--head', a.tokenizer, '--head-url', 'http://127.0.0.1:%d' % ph, '--cand-url', 'http://127.0.0.1:%d' % ph, '--baseline', str(W / 'gate.json') + '.baseline.json', '--out', W / 'gate_same.json')
        gs = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 else {}
        check('gate_eval.py: head vs itself (independent seeds, reused baseline) -> not GREEN', r.returncode == 0 and gs.get('decision') == 'red', {k: (round(v, 4) if isinstance(v, float) else v) for k, v in gs.items() if k in ('decision', 'gain', 'z')})
    finally:
        for p in procs: p.terminate()
    n_ok = sum(ok for _, ok, _ in RESULTS)
    print('\n%d/%d checks passed' % (n_ok, len(RESULTS)))
    (W / 'RESULTS.json').write_text(json.dumps([dict(check=n, ok=o, info=str(i)) for n, o, i in RESULTS], indent=1))
    sys.exit(0 if n_ok == len(RESULTS) else 1)


if __name__ == '__main__':
    main()
