"""Real fresh-process negative controls for the isolated native Agent pilot."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
import numpy as np

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--plan',required=True);parser.add_argument('--artifact',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    original=Path(args.artifact);output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    probe=Path(__file__).with_name('probe_native_agent.py');rows=[]
    for kind,expected in [('probabilities','native full logprobs'),('proof','native strict TOPLOC'),('tool_observation','original native tool replay'),('reward','native full artifact equality')]:
        target=output/kind
        if target.exists():raise ValueError('mutation history must be immutable; choose new output')
        shutil.copytree(original,target)
        if kind=='probabilities':
            p=target/'negative-0.npy';values=np.load(p,allow_pickle=False);values[0,0]+=.01;np.save(p,values)
        else:
            p=target/('negative.json' if kind=='proof' else 'positive.json');value=json.loads(p.read_text())
            if kind=='proof':
                raw=bytearray(base64.b64decode(value['turns'][0]['proofs'][0]));raw[-1]^=1;value['turns'][0]['proofs'][0]=base64.b64encode(raw).decode()
            elif kind=='tool_observation':value['turns'][0]['observations'][0]['content']='[]'
            else:value['outcome']['grade']['reward']=0
            p.write_text(json.dumps(value))
        result=subprocess.run([sys.executable,'-B',str(probe),'verify','--plan',args.plan,'--output',str(target)],capture_output=True,text=True,timeout=300)
        (target/'rejection.log').write_text(result.stdout+result.stderr)
        if result.returncode==0 or expected not in result.stderr:raise ValueError('mutation was not rejected by expected check: '+kind)
        rows.append(dict(kind=kind,rejected=True,returncode=result.returncode,expected_check=expected))
    summary=dict(timestamp=time.time(),approved_plan_sha256=hashlib.sha256(Path(args.plan).read_bytes()).hexdigest(),mutation_runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),controls=rows,original_preserved=True,production_adapter=False,chain_transactions=False)
    (output/'results.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
