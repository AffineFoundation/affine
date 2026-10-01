"""Operator-only rehydration of authenticated, immutable duplicate ZIP caches."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
from subnet.long_context_runtime import AUTHORITY
from subnet.native_tau2_common_search_contract import authenticate
from subnet.storage import Bucket

MAX_BYTES = 250000000


def rehydrate(inventory, bucket, output, authority=AUTHORITY, reserve_bytes=1073741824):
    value = authenticate(inventory, authority)
    if value.get('version') != 'owned-duplicate-cumulative-cache-storage-v1' or value.get('exact_r2_readback_verified') is not True or value.get('native_or_model_admission_claimed_by_storage') is not False or value.get('active_role_arrays_touched') is not False or value.get('payable') is not False or value.get('chain_transactions') is not False:
        raise ValueError('approved duplicate cache storage inventory')
    epoch = value.get('epoch', '')
    match = re.fullmatch(r'nonpayable-native-tau2-mixed-([0-9]+)', epoch)
    name = value.get('local_cache_name')
    size = value.get('size')
    expected = value.get('sha256', '')
    key = value.get('private_rehydration_key', '')
    if not match or name not in ('cumulative-0.zip', 'cumulative-1.zip') or type(size) is not int or not 0 < size <= MAX_BYTES or not re.fullmatch('[0-9a-f]{64}', expected):
        raise ValueError('bounded epoch cache identity')
    folder = 'epoch-' + match[1]
    final = 'private/native-tau2-common-live/' + epoch + '/final-frozen-' + expected + '.zip'
    duplicate = 'private/native-tau2-common-live/' + folder + '/duplicate-cache/' + name + '-' + expected
    if key != duplicate and not (name == 'cumulative-1.zip' and key == final):
        raise ValueError('exact private rehydration key')
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise ValueError('refuse existing output')
    if shutil.disk_usage(output.parent).free < size + reserve_bytes:
        raise RuntimeError('waiting-disk-capacity')
    fd, temporary = tempfile.mkstemp(prefix='.tau2-cache-', dir=output.parent)
    try:
        response = bucket.client.get_object(Bucket=bucket.name, Key=key)
        try:
            total = 0
            digest = hashlib.sha256()
            with os.fdopen(fd, 'wb') as handle:
                fd = None
                while True:
                    chunk = response['Body'].read(min(1048576, size - total + 1))
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > size:
                        raise ValueError('cache exceeds authenticated length')
                    digest.update(chunk)
                    handle.write(chunk)
                handle.flush()
                os.fsync(handle.fileno())
            if total != size or digest.hexdigest() != expected:
                raise ValueError('exact cache size/hash mismatch')
            # Exclusive hard-link publication cannot overwrite a raced output.
            os.link(temporary, output)
        finally:
            response['Body'].close()
    finally:
        if fd is not None:
            os.close(fd)
        Path(temporary).unlink(missing_ok=True)
    return {'sha256': expected, 'size': size, 'output': str(output), 'model_or_native_verification_performed': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--inventory', required=True)
    parser.add_argument('--bucket-config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(rehydrate(json.loads(Path(args.inventory).read_text()), Bucket(json.loads(Path(args.bucket_config).read_text())), args.output), sort_keys=True))

if __name__ == '__main__':
    main()
