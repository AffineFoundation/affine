"""Every consumer of the SN120 token caps (caps.json), in one table.

A consumer = one file / pod file / live endpoint that carries a cap value. Each has rules; a rule says which caps key a value
must equal and how to find it:
  R(key, regex)          regex with ONE group around the number; every match must equal the key ("{:,}" with fmt=',')
  J(key, path)           JSON value at path ('*' = every dict value / list item) must equal the key
  E(key, value)          the key must equal a literal (for the signed-legacy values: they are known and must NOT move)
Places (where the consumer lives):
  repo='rel/path'        path in the AffineFoundation/affine repo (CI / pre-push: tools/caps/caps_check.py --scope repo)
  host='/abs/path'       path on the validator host (--scope host)
  pod=('role', 'cmd')    stdout of a read-only command on a live pod of roles.json (--scope pods; excluded pods never touched)
  url='https://...'      a public URL (--scope public)
Classes:
  live     the value IS the cap somewhere (generation, validation, training, gate). Mismatch = FAIL.
  derived  a value computed from the cap (context, max-model-len, overlong window, published copies). Mismatch = FAIL.
  docs     prose for miners (llms.txt, READMEs, docstrings). Mismatch = FAIL (rendered by caps_apply like the rest).
  signed   root-signed legacy values (lifecycle job settings / manifests): must stay at caps.json signed_manifest. INFO only.
render=True: caps_apply.py rewrites the matched numbers in place (byte-identical when nothing changed).
render='sign': part of a root-signed release (trainer): caps_apply stages a patched copy; activation = the signed-release flow.
"""
import re

H = '/home/const'
REPO = H + '/subnet120-rewrite'
RA = REPO + '/state/root-audits'
WEB = H + '/subnet120/affine/website'
FF = RA + '/20261030-free-file/release'
TRH = RA + '/20261102-12k-everywhere/release-v833m-signed/trainer-host-v833m'
TRP = '/workspace/affine-mimo-trainer-prod/host-v833m'


def R(key, rx, fmt='d', flags=re.M):
    return dict(kind='re', key=key, rx=rx, fmt=fmt, flags=flags)


def J(key, path):
    return dict(kind='json', key=key, path=path)


def E(key, sub=None):
    return dict(kind='signed', key=key, sub=sub)


MAXOUT, CTX, MML, SOFT, OLMAX, LEG, EOS1 = ('max_output_tokens', 'max_context_tokens', 'max_model_len', 'overlong_soft_start',
                                           'overlong_max_tokens', 'legacy_capped_output_tokens', 'last_eos_output_tokens')
CLIP = [R(SOFT, r'clip\(\(len ?[-−] ?(\d+)\) ?/ ?\(\d+ ?[-−] ?\d+\)'), R(OLMAX, r'clip\(\(len ?[-−] ?\d+\) ?/ ?\((\d+) ?[-−] ?\d+\)'),
        R(SOFT, r'clip\(\(len ?[-−] ?\d+\) ?/ ?\(\d+ ?[-−] ?(\d+)\)')]
LLMS = [R(EOS1, r'output of up to ([\d,]+) tokens gets', ','), R(MAXOUT, r'Exactly ([\d,]+) tokens = cut at the cap', ','),
        R(MAXOUT, r'More than ([\d,]+) is invalid', ','), R(CTX, r'Prompt \+ output can be at most ([\d,]+) tokens', ','),
        R(MAXOUT, r'^\s*([\d,]+) output tokens apply everywhere', ','), R(MAXOUT, r'sample with max_tokens ([\d,]+)\)', ','),
        R(SOFT, r'-0\.5 x clip\(\(len - (\d+)\)'), R(OLMAX, r'-0\.5 x clip\(\(len - \d+\) / \((\d+) - '),
        R(SOFT, r'overlong window is (\d+)\.\.\d+'), R(OLMAX, r'overlong window is \d+\.\.(\d+)'),
        R(CTX, r'rollouts up to ([\d,]+) tokens of prompt \+ output', ','),
        R(MAXOUT, r'top_k 20, ([\d,]+) output tokens \(the task cap\)', ','), R(CTX, r'output tokens \(the task cap\), ([\d,]+) context', ','),
        R(MML, r'--max-model-len (\d+)'), R(MAXOUT, r'a capped reply \(([\d,]+) tokens, no EOS\)', ',')]

CONSUMERS = [
    # ---------------- batch-file validation (validator + miner kit) ----------------
    dict(id='ff_format', cls='live', render=True, pinned=True,
         repo='miner/validator/batch/ff_format.py', host=FF + '/ff_format.py',
         rules=[R(MAXOUT, r'^MAX_OUTPUT = (\d+)'), R(CTX, r'^MAX_CONTEXT = (\d+)'), R(LEG, r'^LEGACY_CAP = (\d+)')],
         note='batch-file format check (pinned by FF-PINS.json; the miner kit imports it)'),
    dict(id='affine_batch', cls='docs', render=True, site_asset=True,
         repo='miner/affine_batch.py', host=[FF + '/affine_batch.py', WEB + '/ff/affine_batch.py'], url='https://affine.io/ff/affine_batch.py',
         rules=[R(MAXOUT, r'tokenizes: <= (\d+) output tokens'), R(CTX, r'output tokens, <= (\d+) prompt\+output')]),
    dict(id='ff_public', cls='derived', render=True, host=FF + '/FF-PUBLIC.json',
         rules=[J(MAXOUT, ['max_output_tokens']), R(EOS1, r'output of up to (\d+) tokens gets'), R(MAXOUT, r'Exactly (\d+) tokens = cut'),
                R(MAXOUT, r'More than (\d+) tokens is invalid'), R(CTX, r'must be at most (\d+) tokens'), R(MAXOUT, r'Generate with max_tokens (\d+)'),
                R(LEG, r'output of exactly (\d+) tokens \(an 8K')],
         note='mining.json batch_file block + max_output_tokens (the publisher copies FF-PUBLIC.max_output_tokens)'),
    dict(id='kit_sampler', cls='live', render=True, repo='miner/kit/sampler.py',
         rules=[R(MAXOUT, r'^H = dict\(.*max_output_tokens=(\d+)'), R(CTX, r'^H = dict\(.*max_context_tokens=(\d+)'), R(MML, r'--max-model-len (\d+)')]),
    dict(id='kit_grade', cls='live', render=True, repo='miner/kit/grade.py',
         rules=[R(CTX, r'^H = dict\(.*max_context_tokens=(\d+)'), R(MAXOUT, r'^H = dict\(.*max_output_tokens=(\d+)'),
                R(MAXOUT, r'top_k 20, (\d+) output / \d+ context'), R(CTX, r'top_k 20, \d+ output / (\d+) context')]),
    dict(id='kit_make_batch', cls='live', render=True, repo='miner/kit/make_batch.py',
         rules=[R(MAXOUT, r"'--max-tokens', type=int, default=(\d+)"), R(MAXOUT, r'\(default (\d+) = the cap'),
                R(SOFT, r'0\.5\*clip\(\(len-(\d+)\)/\d+'), R('overlong_ramp_tokens', r'0\.5\*clip\(\(len-\d+\)/(\d+)')]),
    dict(id='kit_check_batch', cls='docs', render=True, repo='miner/kit/check_batch.py',
         rules=[R(MAXOUT, r'output <= (\d+) tokens incl\. EOS'), R(MAXOUT, r'incl\. EOS \(exactly (\d+) = capped'), R(LEG, r'exactly (\d+) = capped too')]),
    dict(id='kit_gate_eval', cls='live', render=True, repo='miner/kit/gate_eval.py',
         rules=[R(MAXOUT, r'^GATE_MAX_TOKENS = (\d+)'), R(MML, r'--max-model-len (\d+)')]),
    dict(id='kit_simulate_turn', cls='derived', render=True, repo='miner/kit/simulate_turn.sh', rules=[R(MML, r'--max-model-len (\d+)')]),
    # ---------------- training (trainer v8.3.3m: root-signed release) ----------------
    dict(id='trainer_v8211_batched', cls='live', render='sign',
         repo='miner/validator/trainer-v833m/v8211_batched.py', host=TRH + '/v8211_batched.py', pod=('trainer', 'cat %s/v8211_batched.py' % TRP),
         rules=[R(CTX, r'^TRAIN_MAX_CONTEXT = (\d+)'), R(SOFT, r'^FF_OVERLONG = \((\d+), \d+\)'), R(OLMAX, r'^FF_OVERLONG = \(\d+, (\d+)\)')]),
    dict(id='trainer_v83_dp', cls='live', render='sign', repo='miner/validator/trainer-v833m/v83_dp.py', host=TRH + '/v83_dp.py',
         pod=('trainer', 'cat %s/v83_dp.py' % TRP), rules=[R(CTX, r'max\(int\(runtime\.max_context_tokens\), (\d+)\)')]),
    dict(id='trainer_v83_fast_kernels', cls='live', render='sign', repo='miner/validator/trainer-v833m/v83_fast_kernels.py',
         host=TRH + '/v83_fast_kernels.py', pod=('trainer', 'cat %s/v83_fast_kernels.py' % TRP),
         rules=[R(CTX, r'max\(int\(runtime\.max_context_tokens\), (\d+)\)')]),
    dict(id='trainer_config_public', cls='derived', render=True, repo='miner/validator/trainer-v833m/trainer-config.public.json',
         rules=[J(SOFT, ['free_file_overlong', 'overlong_soft_start']), J(OLMAX, ['free_file_overlong', 'overlong_max_tokens']),
                R(SOFT, r'clip\(\(len - (\d+)\)'), R(OLMAX, r'clip\(\(len - \d+\) / \((\d+) -'), R(SOFT, r'clip\(\(len - \d+\) / \(\d+ - (\d+)\)')]),
    dict(id='signed_lifecycle', cls='signed', host=RA + '/20261013-mimo-cutover-v1/deployment-finalization-v4/lifecycle.ACTIVE.private.json',
         rules=[E(MAXOUT, r'"max_output_tokens": (\d+)'), E(CTX, r'"max_context_tokens": (\d+)')],
         note='root-signed lifecycle job settings (8192/12288, overlong 6000/8192); overridden by trainer v8.3.3m + ff_format + training_params'),
    # ---------------- gate / probe / perf ----------------
    dict(id='gate_flag', cls='live', render='json', host=RA + '/20261023-toploc-credit-train-gate/staged/PROMOTION-GATE.json',
         rules=[J(MAXOUT, ['heldout', 'prior_v1', 'max_tokens'])], lock=RA + '/20261023-toploc-credit-train-gate/staged/PROMOTION-GATE.json.lock',
         note='gate task cap: frozen into each round at open; the gate raises the probe pod context to max_tokens + 4096 itself'),
    dict(id='gate_cap_switch', cls='derived', render=False, host=RA + '/20261102-task-cap-12k/SWITCH.json',
         rules=[J(MAXOUT, ['max_tokens']), J(CTX, ['max_context_tokens'])], note='written by the gate when a round with a larger cap opens'),
    dict(id='probe_config', cls='live', render=True, host=RA + '/20261022-env-probe/code/config.json',
         rules=[R(MAXOUT, r'^ "max_tokens": (\d+)'), R(CTX, r'"max_context_tokens": (\d+)'), R(MAXOUT, r'"max_output_tokens": (\d+)')],
         note='env-probe defaults (the gate SWITCH.json wins while active; these are the fallback)'),
    dict(id='probe_serve_repo', cls='derived', render=True, repo='miner/validator/probe/serve.sh', rules=[R(MML, r'--max-model-len (\d+)')]),
    dict(id='probe_pod_serve', cls='derived', render=False, pod=('probe', "grep -o -- '--max-model-len [0-9]*' /root/affine-env-probe/serve.sh"),
         rules=[R(MML, r'--max-model-len (\d+)')]),
    dict(id='probe_pod_manifest', cls='derived', render=False,
         pod=('probe', "cat /root/affine-env-probe/live_math_manifest_subset.json"), rules=[J(CTX, ['harness', 'max_context_tokens'])],
         note='harness.max_output_tokens there is the signed manifest copy (8192, unused: the probe samples with the round max_tokens)'),
    dict(id='probe_pod_vllm', cls='derived', render=False, optional=True,
         pod=('probe', "curl -s -m 5 http://127.0.0.1:18555/v1/models"), rules=[J(MML, ['data', 0, 'max_model_len'])]),
    # ---------------- prior table sampler ----------------
    dict(id='prior_sampler_config', cls='live', render=True, repo='priors/pod/sampler.config.json', host=REPO + '/priors/pod/sampler.config.json',
         rules=[R(CTX, r'"max_context_tokens": (\d+)'), R(MAXOUT, r'"max_output_tokens": (\d+)')]),
    dict(id='prior_serve_sh', cls='derived', render=True, repo='priors/pod/serve.sh', host=REPO + '/priors/pod/serve.sh',
         rules=[R(MML, r'MML=\$\{MAX_MODEL_LEN:-(\d+)\}'), R(MML, r'serve\.sh \(bf16, max-model-len (\d+)')]),
    dict(id='prior_readme', cls='docs', render=True, repo='priors/README.md', host=REPO + '/priors/README.md',
         rules=[R(MAXOUT, r'max output (\d+), context \d+'), R(CTX, r'max output \d+, context (\d+)'), R(MML, r'bf16, max-model-len (\d+)')]),
    dict(id='prior_pod_sampler', cls='live', render='pod', pod=('prior', 'cat /opt/prior/code/sampler.json'),
         rules=[J(MAXOUT, ['harness', 'max_output_tokens']), J(CTX, ['harness', 'max_context_tokens'])]),
    dict(id='prior_pod_serve_env', cls='derived', render='pod', pod=('prior', 'cat /opt/prior/serve/serve.env'),
         rules=[R(MML, r'^MAX_MODEL_LEN=(\d+)')]),
    dict(id='prior_pod_serving', cls='derived', render=False, pod=('prior', 'cat /opt/prior/serve/SERVING.json'),
         rules=[J(MML, ['max_model_len'])]),
    dict(id='prior_pod_vllm', cls='derived', render=False, optional=True, pod=('prior', 'curl -s -m 5 http://127.0.0.1:8100/v1/models'),
         rules=[J(MML, ['data', 0, 'max_model_len'])]),
    # ---------------- published (mining.json, training.json, llms.txt, dashboards) ----------------
    dict(id='training_json', cls='derived', render=False, host=WEB + '/training.json',
         rules=[J(SOFT, ['loss', 'overlong_soft_start']), J(OLMAX, ['loss', 'overlong_max_tokens']),
                J(MAXOUT, ['sampling', 'harness', '*', 'max_output_tokens']), J(CTX, ['sampling', 'harness', '*', 'max_context_tokens']),
                J(MAXOUT, ['k1', 'caps', 'max_output_tokens']), J(CTX, ['k1', 'caps', 'max_context_tokens']), J(LEG, ['k1', 'caps', LEG])],
         note='written by training_params.py (reads caps.py)'),
    dict(id='mining_json', cls='derived', render=False, host=WEB + '/mining.json', url='https://affine.io/mining.json',
         rules=[J(MAXOUT, ['max_output_tokens']), J(SOFT, ['training', 'loss', 'overlong_soft_start']), J(OLMAX, ['training', 'loss', 'overlong_max_tokens']),
                J(MAXOUT, ['training', 'sampling', 'harness', '*', 'max_output_tokens']), J(MAXOUT, ['training', 'k1', 'caps', 'max_output_tokens'])]),
    dict(id='mining_json_caps', cls='derived', render=False, host=WEB + '/mining.json', url='https://affine.io/mining.json', after='caps-v1',
         rules=[J(MAXOUT, ['training', 'caps', 'max_output_tokens']), J(CTX, ['training', 'caps', 'max_context_tokens']),
                J(SOFT, ['training', 'caps', 'overlong_soft_start'])], note='the canonical caps as published (training.caps)'),
    dict(id='llms_txt', cls='docs', render=True, site_asset=True, repo='dashboard/public/llms.txt', host=WEB + '/llms.txt',
         url='https://affine.io/llms.txt', rules=LLMS),
    dict(id='miner_readme', cls='docs', render=True, repo='miner/README.md',
         rules=[R(MAXOUT, r'Exactly ([\d,]+) tokens counts as capped', ','), R(MAXOUT, r'More than ([\d,]+) is invalid', ','),
                R(CTX, r'Prompt \+ output can be at most ([\d,]+) tokens', ','), R(MAXOUT, r'\*\*([\d,]+) output tokens apply everywhere', ','),
                R(MAXOUT, r'sample with `max_tokens` ([\d,]+)', ','), R(MAXOUT, r'top_k 20, ([\d,]+) output tokens \(the task cap\)', ','),
                R(CTX, r'output tokens \(the task cap\), ([\d,]+) context', ','), R(MML, r'--max-model-len (\d+)')] + CLIP[:2]),
    dict(id='site_metrics', cls='derived', render=False, host=WEB + '/site-metrics.json', after='caps-v1',
         rules=[J(MAXOUT, ['epochs', 'last_trained', 'training', 'output_token_cap'])],
         note='dashboard per-epoch output cap (caps.at_epoch); was the signed 8192 for every epoch'),
]

# Hard-coded live cap numbers that must not come back in code that reads caps.py (checked by caps_check --scope host).
NO_LITERALS = [
    dict(id='training_params_code', host=RA + '/20261026-promotion-gate/code/training_params.py', after='caps-v1', keys=(MAXOUT, CTX, SOFT)),
    dict(id='site_metrics_code', host=REPO + '/dashboard/site_metrics.py', after='caps-v1', keys=(MAXOUT, CTX, SOFT)),
    dict(id='roles_json', host=REPO + '/ops/pods/roles.json', after='caps-v1', keys=(MAXOUT, CTX, SOFT)),
]
