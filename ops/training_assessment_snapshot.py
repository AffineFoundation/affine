"""Refresh learning eligibility from authenticated audits without chain writes."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    a=argparse.ArgumentParser();a.add_argument('--runtime',required=True);a.add_argument('--writer-policy',required=True);a.add_argument('--authority-seed',required=True);a.add_argument('--output',required=True);args=a.parse_args()
    sys.path.insert(0,args.runtime)
    from nacl.signing import SigningKey
    from subnet.storage import canonical
    from subnet.live_reward_bridge import signed,sha
    from ops.current_assessment_evidence import load_evidence
    from subnet.current_assessment import calculate
    document=json.loads(Path(args.writer_policy).read_bytes());key=SigningKey(bytes.fromhex(Path(args.authority_seed).read_text().strip()));authority=key.verify_key.encode().hex();p=signed(document,authority)
    for path,expected in p['module_hashes'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:raise ValueError('pinned assessment implementation')
    cutoff=int(time.time())//3600*3600
    kwargs=dict(expected_numerical_resolution_policy_sha256=p['numerical_resolution_policy_sha256'])if 'numerical_resolution_policy_sha256'in p else {}
    evidence=load_evidence(p['audit_config'],authority=authority,cutoff=cutoff,verifiers=p['verifiers'],expected_source_admission_sha256=p['source_admission_sha256'],**kwargs)
    assessment=calculate(evidence['snapshots'],evidence['committed_at_by_epoch'],cutoff)
    assessment.update(evidence_cutoff=cutoff,assessment_stale=False,evidence_hashes=evidence['evidence_hashes'],evidence_refusals=evidence.get('refused',[]),evidence_exclusions=evidence.get('excluded',[]),evidence_deferrals=evidence.get('deferred',[]),writer_policy_sha256=sha(document))
    envelope=dict(payload=assessment,signer=authority,signature=base64.b64encode(key.sign(canonical(assessment)).signature).decode())
    out=Path(args.output);out.parent.mkdir(mode=0o700,parents=True,exist_ok=True);tmp=out.with_suffix('.tmp');tmp.write_bytes(canonical(envelope));tmp.chmod(0o600);tmp.replace(out)
    print(json.dumps(dict(assessment_refreshed=True,cutoff=cutoff,miners=len(assessment['miner_estimates']),chain_transactions=False)))


if __name__=='__main__':main()
