"""Build the fixed original Calendar runtime without seed/grader assets."""
import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from subnet.native_eog_isolation import IMAGE
from subnet.native_eog_clock import REVISION

def main():
    p=argparse.ArgumentParser();p.add_argument('--state',type=Path,default=Path('state/native-eog-isolation'))
    p.add_argument('--seed',default=hashlib.sha256(b'controlled-original-eog-calendar-native-v1').hexdigest())
    p.add_argument('--clock',default='2026-01-01T00:00:00+00:00');args=p.parse_args()
    root=args.state;context=root/'runtime-build';context.mkdir(parents=True,exist_ok=True)
    source=Path('subnet/native_eog_clock.py').read_bytes();(context/'clock.py').write_bytes(source)
    (context/'Dockerfile').write_text('FROM '+IMAGE+'\nCOPY --chown=calendar clock.py /app/affine_clock.py\nCMD ["python", "-B", "/app/affine_clock.py"]\n')
    with (root/'runtime-build.log').open('wb') as log:
        subprocess.run(['docker','build','-t','affine-eog-fixed-clock:controlled-v1',str(context)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=180)
    image=subprocess.check_output(['docker','image','inspect','affine-eog-fixed-clock:controlled-v1','--format','{{.Id}}']).decode().strip()
    descriptor={'revision':REVISION,'image':image,'base_image':IMAGE,
        'shim_sha256':hashlib.sha256(source).hexdigest(),'seed':args.seed,'clock':args.clock}
    from subnet.native_eog_isolation import validate_runtime
    validate_runtime(descriptor)
    (root/'runtime-descriptor.json').write_text(json.dumps(descriptor,indent=2))
    print(json.dumps({'immutable_runtime_image':image,'revision':REVISION}))

if __name__=='__main__':main()
