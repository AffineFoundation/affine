"""Bounded child entrypoint; no authority key, dispatch or GPU/model creation."""
import argparse
import json
from pathlib import Path
import sys

def required_memory_bytes(request):
    """Authenticate capacity before reserving decode plus isolated child memory."""
    from subnet.distributed_roles import authenticate
    from ops.native_training_outcome_filter import digest, validate_limits
    authority = request['authority']
    context = authenticate(request['context'], authority)
    authorization = authenticate(request['authorization'], authority)
    manifest = authenticate(context['original_signed_manifest'], authority)
    if context['authorization_sha256'] != digest(request['authorization']):
        raise ValueError('native wave memory authorization binding')
    limits = validate_limits(authorization['limits'], manifest=manifest)
    raw = sum(o['size'] for o in context['submissions'])
    return 64 * raw + 512 * 1024**2 + limits['workers'] * 1073741824


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from ops.native_training_eligibility import _load,_create
    from ops.native_training_outcome_filter import filter_eligibility_context
    request=json.loads(_load(args.request))
    from subnet.persistent_training_state import available_ram_bytes
    required=required_memory_bytes(request)
    available=available_ram_bytes()
    if available<required:raise ValueError('native representative decode and grader memory admission')
    _,grades=filter_eligibility_context(request['paths'],request['context'],request['authorization'],request['authority'],
        request['source_root'],request['tokenizer_root'],request['interpreter'])
    _create(args.output,grades)

if __name__=='__main__':main()
