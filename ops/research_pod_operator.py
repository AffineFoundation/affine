"""Prospective ROOT-reviewed rental/launch integration; no automatic retirement."""
import argparse
import base64
import hashlib
import importlib.util
import json
import math
import re
import os
from pathlib import Path
import subprocess
import time
from nacl.signing import VerifyKey
from ops.research_pod_ownership import VERSION, rent_once, before_launch, heartbeat

AUTHORITY = '3301134b38401196d006a621ac4a772bb4b0e6afa15a7d34a0d1ae5f2c630bcd'
REGISTRY = '/home/const/subnet120/ops/pods/registry.py'
canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
digest = lambda value: hashlib.sha256(canonical(value)).hexdigest()


def authenticate(document, authority=AUTHORITY):
    if document['signer'] != authority:
        raise ValueError('ROOT operator authority required')
    VerifyKey(bytes.fromhex(authority)).verify(canonical(document['payload']), base64.b64decode(document['signature'], validate=True))
    return document['payload']


def exclusive_receipt(path, value):
    path = Path(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'wb') as stream:
        stream.write(canonical(value)); stream.flush(); os.fsync(stream.fileno())
    parent = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(parent)
    finally: os.close(parent)


def rental_intent(document, authority=AUTHORITY, fresh=True):
    p = authenticate(document, authority)
    fields = {'version', 'execute_allowed', 'created_at', 'expires_at', 'name', 'purpose',
              'price_usd_h', 'offer_id', 'template_id', 'gpu_type', 'gpu_count',
              'retention_authorized', 'registry_path', 'registry_sha256',
              'operator_sha256', 'ownership_module_sha256', 'output_directory'}
    if set(p) != fields or p['version'] != 'retained-research-original-rental-intent-v1' or p['execute_allowed'] is not True:
        raise ValueError('exact executable original rental intent')
    if type(p['created_at']) is not int or type(p['expires_at']) is not int or not 0 < p['expires_at']-p['created_at'] <= 1200:
        raise ValueError('bounded original rental authorization')
    if fresh and not p['created_at'] <= time.time() < p['expires_at']:
        raise ValueError('fresh original rental authorization')
    if p['registry_path'] != REGISTRY or p['gpu_count'] != 1 or p['gpu_type'] not in ('H100', 'H200'):
        raise ValueError('exact approved local registry/single GPU')
    if type(p['price_usd_h']) not in (int, float) or not math.isfinite(p['price_usd_h']) or p['price_usd_h'] <= 0:
        raise ValueError('finite positive signed rental price')
    if any(type(p[k]) is not str or re.fullmatch(r'[0-9a-f]{64}', p[k]) is None for k in ('registry_sha256', 'operator_sha256', 'ownership_module_sha256')):
        raise ValueError('exact lowercase SHA256 source pins')
    if type(p['gpu_count']) is not int:
        raise ValueError('exact integer GPU count')
    if p['retention_authorized'] is not True:
        raise ValueError('reviewed retention required')
    return p


def validate_provider_hardware(pod, p):
    # Actual Lium `up --json` receipt fields, observed read-only on the
    # original H200 rental. Qualification remains a separate scientific gate.
    if pod.get('gpu_type') != p['gpu_type'] or type(pod.get('gpu_count')) is not int or pod['gpu_count'] != p['gpu_count']:
        raise ValueError('provider hardware differs from signed original intent')
    actual = pod.get('price_per_hour')
    if type(actual) not in (int, float) or not math.isfinite(actual) or actual <= 0 or actual != p['price_usd_h']:
        raise ValueError('provider price differs from signed original intent')


def binding(p, document, pod_id='pending-original-rental'):
    return dict(version=VERSION, name=p['name'], pod_id=pod_id, purpose=p['purpose'],
                price_usd_h=p['price_usd_h'], rental_intent_sha256=digest(document), retention_authorized=True)


def rent_original(registry, document, provider, directory, authority=AUTHORITY):
    """One signed intent, fsynced journal, name registration before provider."""
    p = rental_intent(document, authority)
    directory = Path(directory)
    if directory != Path(p['output_directory']) or directory.parent.resolve() != directory.parent or directory.parent.stat().st_uid != os.getuid():
        raise ValueError('exact owned signed original journal directory')
    if directory.exists(): raise ValueError('original namespace already exists; observe instead')
    directory.mkdir(mode=0o700)
    exclusive_receipt(directory/'original-intent.json', document)
    def once():
        # The registry reservation exists before this callback; an exception
        # is evidence to observe, never permission for another provider call.
        try:
            receipt = provider(p)
            exclusive_receipt(directory/'original-provider-receipt.json', receipt)
            pod = receipt['pod']
            if type(pod['id']) is not str or not pod['id'] or pod['name'] != p['name']:
                raise ValueError('actual original provider identity')
            validate_provider_hardware(pod, p)
            return dict(pod_id=pod['id'])
        except BaseException as error:
            exclusive_receipt(directory/'original-provider-error.json', dict(reason=type(error).__name__, at=time.time(), reissue_allowed=False))
            raise
    receipt, actual = rent_once(registry, binding(p, document), once)
    exclusive_receipt(directory/'original-owned-binding.json', dict(binding=actual, intent_sha256=digest(document), node_qualified=False))
    return actual


def observe_heartbeat(registry, document, directory, authority=AUTHORITY):
    # A retained original lease can be heartbeated after rental capability TTL;
    # expiry cannot authorize a new provider or scientific execution.
    p = rental_intent(document, authority, fresh=False)
    original = json.loads((Path(directory)/'original-intent.json').read_bytes())
    if original != document: raise ValueError('original operator journal changed')
    actual = json.loads((Path(directory)/'original-owned-binding.json').read_bytes())
    provider = json.loads((Path(directory)/'original-provider-receipt.json').read_bytes())['pod']
    validate_provider_hardware(provider, p)
    if provider.get('name') != p['name']: raise ValueError('original provider name drift')
    expected = binding(p, document, provider['id'])
    if actual['binding'] != expected or actual['intent_sha256'] != digest(document):
        raise ValueError('actual provider/intent owner binding drift')
    return heartbeat(registry, expected)


def guarded_launch(registry, actual_binding, original_launch_journal, launch):
    """Wrap a separately ROOT-authorized, qualified original supervisor launch.

    This function grants no scientific capability. The existing reviewed
    launcher still authenticates its fresh scope/runtime/GPU/checkpoint lease.
    """
    def original():
        exclusive_receipt(original_launch_journal, dict(binding=actual_binding, at=time.time(), one_original_callback=True))
        return launch()
    return before_launch(registry, actual_binding, original)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['rent', 'heartbeat']); parser.add_argument('--intent', required=True); args=parser.parse_args()
    document=json.loads(Path(args.intent).read_bytes()); p=rental_intent(document, fresh=args.action=='rent')
    from ops import research_pod_ownership as ownership
    for file,pin in [(Path(__file__),p['operator_sha256']),(Path(ownership.__file__),p['ownership_module_sha256']),(Path(REGISTRY),p['registry_sha256'])]:
        if file.is_symlink() or hashlib.sha256(file.read_bytes()).hexdigest()!=pin: raise ValueError('exact ROOT operator/registry source pin')
    spec=importlib.util.spec_from_file_location('_owned_research_registry',REGISTRY); registry=importlib.util.module_from_spec(spec);spec.loader.exec_module(registry)
    directory=Path(p['output_directory'])
    if args.action=='heartbeat':observe_heartbeat(registry,document,directory);print(json.dumps({'heartbeat':True,'provider_actions':0}));return
    def provider(v):
        command=['lium','up',v['offer_id'],'--name',v['name'],'--template_id',v['template_id'],'--count','1','--yes','--json','--verify-gpus','--strict-gpus','--timeout','240','--ready-timeout','180']
        result=subprocess.run(command,capture_output=True,text=True,timeout=270)
        exclusive_receipt(directory/'original-provider-process.json',dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
        if result.returncode:raise RuntimeError('original provider command failed; observe without reissue')
        return json.loads(result.stdout)
    actual=rent_original(registry,document,provider,directory);print(json.dumps({'pod_id':actual['pod_id'],'ownership_bound':True,'node_qualified':False}))

if __name__=='__main__':main()
