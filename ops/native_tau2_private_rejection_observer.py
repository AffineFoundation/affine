"""Private rejection observer; never changes native/model responses or policy."""
import base64
import json
import os
from pathlib import Path
import shutil
import time
import traceback


def install(base, output, key, binding):
    original = base.rejection_diagnostic
    output = Path(output)
    records = []
    def observe(endpoint, request, error):
        result = original(endpoint, request, error)
        if len(records) >= 32:
            return result
        try:
            records.append({'exception_type':type(error).__name__, 'exception_message':str(error)[:4096], 'private_traceback':''.join(traceback.format_exception(type(error),error,error.__traceback__))[-16384:], 'disk_free_bytes':shutil.disk_usage(output.parent).free, 'completed_at':time.time(), 'public_diagnostic':result})
            payload = {'version':'private-native-rejection-observer-v1','binding':binding,'records':records,'model_or_native_policy_changed':False}
            raw = json.dumps(payload,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
            envelope={'payload':payload,'signer':key.verify_key.encode().hex(),'signature':base64.b64encode(key.sign(raw).signature).decode()}
            fd=os.open(output,os.O_WRONLY|os.O_CREAT|os.O_TRUNC|os.O_NOFOLLOW,0o600)
            with os.fdopen(fd,'wb') as handle:handle.write(json.dumps(envelope,sort_keys=True,separators=(',',':'),allow_nan=False).encode())
        except Exception:
            # Observation must never change the original diagnostic semantics.
            pass
        return result
    base.rejection_diagnostic=observe
    return original
