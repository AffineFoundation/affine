"""Reproduce the qualified portable variant in an independent source copy.

Never run against a live or shared source tree. This creates no service, fixture,
credential, model or container; subsequent operator qualification is separate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

BASELINE={
    'subnet/native_eog_isolation.py':'8ad947d004e2e9cd9d03143d95a133bb7575d39c0e75304ffbe4baed19715e48',
    'subnet/native_eog_split.py':'a2690b7ea6eee32d9bc0927060b4603606c400ec9c8f7f55b30a4171c202cea2',
}
PORTABLE={
    'subnet/native_eog_isolation.py':'b61ab172fb004880a02597aabad328a921d84c0a44d5d570ad51879703152906',
    'subnet/native_eog_split.py':'39d57b148898f7fe47870e12f758fecf04c6fb22398c01d3664fd1b3dfccda3c',
}


def transform(name, raw):
    if name not in BASELINE or hashlib.sha256(raw).hexdigest()!=BASELINE[name]:
        raise ValueError('unreviewed portable source baseline')
    text=raw.decode()
    if name.endswith('native_eog_isolation.py'):
        old="        self.database_id='vf_'+(sha(private_task)[:32] if runtime else uuid.uuid4().hex)"
        new="        from .native_eog_deployment import PRIVATE_HASH_POLICY,private_fixture_hash\n        self.database_identity_policy=PRIVATE_HASH_POLICY\n        self.database_id='vf_'+(private_fixture_hash(private_task)[:32] if runtime else uuid.uuid4().hex)"
        replacements=[(old,new)]
    else:
        replacements=[("REVISION='original-eog-public-actor-private-grader-v3-terminal'",
            "REVISION='original-eog-public-actor-private-grader-v4-portable-terminal'"),
            ("'subnet/native_eog_split.py',\n", "'subnet/native_eog_split.py','subnet/native_eog_deployment.py','subnet/native_eog_adapter.py',\n")]
    for old,new in replacements:
        if text.count(old)!=1:raise ValueError('ambiguous portable source transformation')
        text=text.replace(old,new)
    result=text.encode()
    if hashlib.sha256(result).hexdigest()!=PORTABLE[name]:
        raise ValueError('portable source output differs from qualified variant')
    return result


def prepare(source_root, protected_root):
    source_root=Path(source_root).resolve();protected_root=Path(protected_root).resolve()
    if source_root==protected_root or not source_root.is_dir():
        raise ValueError('independent source copy required')
    results={}
    for name in BASELINE:
        target=source_root/name;original=protected_root/name
        if target.is_symlink() or target.resolve()!=target or not target.is_file():
            raise ValueError('independent regular source file required')
        if original.exists() and os.path.samefile(target,original):
            raise ValueError('shared or hard-linked source rejected')
        results[name]=transform(name,target.read_bytes())
    # Validate both inputs and both qualified outputs before any write.
    for name,raw in results.items():
        target=source_root/name;temporary=target.with_suffix('.portable.tmp')
        with temporary.open('xb') as stream:stream.write(raw)
        temporary.replace(target)
    evidence=dict(revision='portable-eog-common-source-v4',baseline=BASELINE,
        qualified_source_files=PORTABLE,changes_only=list(BASELINE),services_started=False,
        fixture_or_model_accessed=False,operator_execution_qualified_here=False)
    (source_root/'portable-source-preparation.json').write_text(json.dumps(evidence,indent=2)+'\n')
    return evidence


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',required=True,type=Path)
    args=p.parse_args()
    result=prepare(args.source_root,Path(__file__).resolve().parents[1])
    print(json.dumps(result))


if __name__=='__main__':main()
