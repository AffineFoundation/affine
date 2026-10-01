"""Fail-closed trusted source/package/interpreter closure for native Tau2 audits."""
import base64,hashlib,json,pathlib,sys
from importlib.metadata import version
from nacl.signing import VerifyKey

ROOT=pathlib.Path(__file__).resolve().parent.parent
VERIFIERS=('subnet/native_tau2_model.py','subnet/native_tau2_replay.py','subnet/native_tau2_attestation.py')
REQUIRED={'subnet/native_tau2_model.py','subnet/native_tau2_replay.py','subnet/native_tau2_probe.py','subnet/model.py','subnet/harness.py','subnet/proofs.py','subnet/vendor/legacy/rollouts/envs/affine_tau2_v1/affine_tau2_v1/harness.py','subnet/vendor/research/environments/tool_use/tau2_bench_v1/tau2_bench_v1/harness.py'}
PACKAGES={'torch','transformers','toploc','tau2','litellm','verifiers'}
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha(x):return hashlib.sha256(x).hexdigest()
def authenticated(x,authority):
    if x['signer']!=authority:raise ValueError('closure authority')
    VerifyKey(bytes.fromhex(authority)).verify(canonical(x['payload']),base64.b64decode(x['signature'],validate=True));return x['payload']
def require_source_closure(out,plan,authority):
    out=pathlib.Path(out);raw=(out/'operator-source-closure.json').read_bytes();closure=authenticated(json.loads(raw),authority)
    if closure['plan_hash']!=sha(canonical(plan)) or set(closure['sources'])!=REQUIRED or set(closure['package_versions'])!=PACKAGES:raise ValueError('source closure contract')
    if closure['sources']['subnet/native_tau2_model.py']!=plan['source_hash'] or closure['sources']['subnet/harness.py']!=plan['harness_source_hash']:raise ValueError('original generation source binding')
    if closure['python_version']!=sys.version or closure['interpreter_hash']!=sha(pathlib.Path(sys.executable).resolve().read_bytes()):raise ValueError('interpreter closure')
    if any(version(n)!=v for n,v in closure['package_versions'].items()):raise ValueError('package closure')
    current={n:sha((ROOT/n).read_bytes()) for n in VERIFIERS}
    supplement=authenticated(json.loads((out/'verifier-supplement.json').read_text()),authority)
    if supplement.get('schema')!=1 or supplement.get('upgrade')!='exact-derived-response-v3' or supplement.get('generation_plan_hash')!=sha(canonical(plan)) or supplement.get('generation_source_hash')!=plan['source_hash'] or supplement.get('original_source_closure_sha256')!=sha(raw) or supplement.get('verifier_sources')!=current:raise ValueError('reviewed verifier supplement')
    if sha((out/'generation-native_tau2_model.py').read_bytes())!=plan['source_hash']:raise ValueError('preserved original generation source')
    if sha((out/'generation-native_tau2_replay.py').read_bytes())!=closure['sources']['subnet/native_tau2_replay.py']:raise ValueError('preserved original replay source')
    for n,expected in closure['sources'].items():
        if n not in VERIFIERS and sha((ROOT/n).read_bytes())!=expected:raise ValueError('runtime source closure')
    return {'upgrade':supplement['upgrade'],'generation_source_hash':plan['source_hash'],'verifier_sources':current,'interpreter_hash':closure['interpreter_hash'],'package_versions':closure['package_versions']}
