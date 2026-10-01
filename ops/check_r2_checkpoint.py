"""Hash published checkpoint bodies without downloading weights to disk."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import time

from subnet.storage import Bucket, canonical


def check(bucket, checkpoint, chunk_size=1024*1024):
    files = checkpoint['files']
    if not files or hashlib.sha256(canonical(files)).hexdigest() != checkpoint['id']:
        raise ValueError('checkpoint filemap identity mismatch')
    for name, expected in files.items():
        if re.fullmatch(r'[A-Za-z0-9_.-]+', name) is None or name in ('.','..'):
            raise ValueError('unsafe checkpoint filename')
        if not isinstance(expected,str) or re.fullmatch(r'[0-9a-f]{64}', expected) is None:
            raise ValueError('invalid checkpoint file digest')
    checked = {}
    for name, expected in files.items():
        key=f"public/checkpoints/{checkpoint['id']}/{name}"
        response=bucket.client.get_object(Bucket=bucket.name, Key=key)
        stream=response['Body']; digest=hashlib.sha256(); size=0
        try:
            length=response['ContentLength']
            if type(length) is not int or not 0 < length <= 16*1024**3:
                raise ValueError('invalid checkpoint object size')
            while True:
                chunk=stream.read(chunk_size)
                if not chunk: break
                size+=len(chunk)
                if size > length: raise ValueError('checkpoint response size mismatch')
                digest.update(chunk)
            if size != length: raise ValueError('checkpoint response truncated')
            if digest.hexdigest() != expected: raise ValueError('published checkpoint bytes changed: '+name)
            checked[name]={'bytes':size,'sha256':expected}
        finally:
            stream.close()
    return {'timestamp':time.time(), 'checkpoint':checkpoint['id'],
            'filemap_identity_verified':True, 'published_file_hashes_verified':True,
            'files':checked, 'model_weights_written_to_operator_disk':False,
            'fresh_model_execution_performed':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bucket-config', type=Path, default=Path('state/r2-direct.json'))
    parser.add_argument('--report', type=Path, default=Path('state/multi-environment/gpu-training-pilot/report.json'))
    parser.add_argument('--output', type=Path, default=Path('state/multi-environment/gpu-training-pilot/r2-checkpoint-independent-check.json'))
    args=parser.parse_args()
    # This report must already be accepted as an operator trust anchor. The tool
    # authenticates byte publication; it does not approve untrusted model reports.
    report=json.loads(args.report.read_text())
    result=check(Bucket(json.loads(args.bucket_config.read_text())), report['new_checkpoint'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix('.tmp');temporary.write_bytes(canonical(result));temporary.replace(args.output)
    print(json.dumps({'checkpoint':result['checkpoint'], 'files_verified':len(result['files']),
                      'bytes_verified':sum(f['bytes'] for f in result['files'].values())}))


if __name__=='__main__':main()
