"""Bounded child entrypoint; no authority key, dispatch or GPU/model creation."""
import argparse
import json
from pathlib import Path
import sys

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--request',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from ops.native_training_eligibility import _load,_create
    from ops.native_training_outcome_filter import filter_eligibility_context
    request=json.loads(_load(args.request))
    from subnet.persistent_training_state import available_ram_bytes
    raw=sum(o['size'] for o in request['context']['payload']['submissions'])
    required=64*raw+512*1024**2
    available=available_ram_bytes()
    if available<required:raise ValueError('native representative decode memory admission')
    _,grades=filter_eligibility_context(request['paths'],request['context'],request['authorization'],request['authority'],
        request['source_root'],request['tokenizer_root'],request['interpreter'])
    _create(args.output,grades)

if __name__=='__main__':main()
