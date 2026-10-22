"""Signed host composition v7c = the exact signed v7b composition (unchanged evidence, budgets, writer) + ONE additive
step: the measured-impact blend around the writer's final recipients() call (after evidence, before chain submission).
The blend value lives in a separately signed config (authority key); blend 0.0 returns the original objects unchanged."""
import argparse, hashlib, importlib.util, json, signal, sys, types
from pathlib import Path

VERSION = 'hourly-population-index-host-composition-v7c-measured-impact'
HERE = Path(__file__).resolve().parent


def prepare(document, authority):
    import base64
    from nacl.signing import VerifyKey
    canonical = lambda v: json.dumps(v, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    def need(c, m):
        if not c: raise ValueError(m)
    need(type(document) is dict and set(document) == {'payload', 'signer', 'signature'} and document['signer'] == authority, 'explicit operator authority')
    VerifyKey(bytes.fromhex(authority)).verify(canonical(document['payload']), base64.b64decode(document['signature'], validate=True))
    p = document['payload']
    need(type(p) is dict and set(p) == {'version', 'execute_enabled', 'entry', 'base_entry', 'base_composition', 'blend_module',
                                        'blend_config_path', 'maximum_seconds', 'evidence_timeout_seconds'}, 'exact v7c composition')
    need(p['version'] == VERSION and type(p['execute_enabled']) is bool and p['maximum_seconds'] == 1600 and p['evidence_timeout_seconds'] == 1400,
         'unchanged writer budgets')
    def load(name, ref):
        need(type(ref) is dict and set(ref) == {'path', 'sha256'}, 'exact reference')
        need(hashlib.sha256(Path(ref['path']).read_bytes()).hexdigest() == ref['sha256'], 'pinned digest ' + name)
        spec = importlib.util.spec_from_file_location(name, ref['path']); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
    need(Path(p['entry']['path']) == Path(__file__).resolve() and hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == p['entry']['sha256'], 'exact host entry')
    base = load('_v7b_host_composition', p['base_entry'])
    base_raw = base.pinned(p['base_composition'])
    base_p, parent, context = base.prepare(json.loads(base_raw), authority)       # exact, unchanged v7b preparation
    need(base_p['maximum_seconds'] == p['maximum_seconds'] and base_p['evidence_timeout_seconds'] == p['evidence_timeout_seconds'], 'same budgets as v7b')
    blend = load('_sn120_measured_impact_blend', p['blend_module'])
    blend.load_config(p['blend_config_path'], authority)                          # config must authenticate at start
    writer = context['writer']; run = writer.run_once; g = run.__globals__
    wrapped = blend.wrap_recipients(g['recipients'], p['blend_config_path'], authority, g['OWNER'])
    new_run = types.FunctionType(run.__code__, dict(g, recipients=wrapped), run.__name__, run.__defaults__, run.__closure__)
    new_run.__kwdefaults__ = run.__kwdefaults__
    context = dict(context, writer=types.SimpleNamespace(**dict(vars(writer), run_once=new_run)), evidence_loader=blend.timed(context['evidence_loader']))
    return p, parent, context, base, blend

def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--policy', required=True); ap.add_argument('--authority', required=True)
    m = ap.add_mutually_exclusive_group(required=True); m.add_argument('--check', action='store_true'); m.add_argument('--execute', action='store_true')
    a = ap.parse_args()
    def expired(*_): raise TimeoutError('bounded existing hourly writer')
    signal.signal(signal.SIGALRM, expired); signal.alarm(1600)
    try:
        raw = Path(a.policy).absolute().read_bytes()
        p, parent, context, base, blend = prepare(json.loads(raw), a.authority)
        if a.check:
            c = blend.load_config(p['blend_config_path'], a.authority)
            print(json.dumps({'checked': True, 'chain_calls': 0, 'assessment_writes': 0, 'version': VERSION,
                              'base_composition_sha256': p['base_composition']['sha256'], 'blend': c['blend'], 'blend_config_sequence': c['sequence'],
                              'unchanged_writer_policy_sha256': hashlib.sha256(json.dumps(context['parent_policy'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()}))
            return
        if not p['execute_enabled']: raise ValueError('explicit signed host enablement')
        result = parent.run(context, execute=True)
        print(json.dumps({k: result[k] for k in ('status', 'window_end', 'remaining_blocks') if k in result}))
    finally: signal.alarm(0)

if __name__ == '__main__': main()
