"""Prepare the reviewed controlled SQL dispatcher/policy in a NEW source tree.

This operator-only source construction never changes the checkout used by active
services. Private databases/queries, model files and state are not copied.
"""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path
BASE={
 'environments.py':'84037f39eff92bc42362cfe785fded522670e1ffbdde56ea0754d9c0c2a0e141',
 'harness.py':'4b565259163fc1bf6df3775a89d10fd953a65cccd9423910b56a398f183bec04',
 'gpu_runtime.py':'dff049849c3cfd05edabedf1637f2e51c57d2f4085ff6ff1339b1afe69c9d357',
 'native_sql_adapter.py':'956bde89205586680962059d35c5c1fb0600c4d48378e72c89234bdc64d8a771',
 'native_sql_actor.py':'6a29c04f6e43e80d86139320ea47ac2e40eecb2204f578d2d7289b55b54f2df4',
 'native_sql_isolation.py':'ffd7d51cf69ec9a82c5d1e45038ab537620e6adf5da307798047e9ea3330e9e5',
 'native_sql_deployment.py':'85e9ae9764b19fe88d42bd0f091fefc5f1b947bdca002599faa1e20a1d6bf67b',
 'public_sql_candidates.py':'fd66b20ea68da8d84bb82d76488f5e22af1172a34df8501cf4ca82dd6500f4e0'}

def replace_once(text,old,new):
    if text.count(old)!=1:raise ValueError('qualified source patch context')
    return text.replace(old,new,1)

def transformed(source,private_tasks):
    for name,digest in BASE.items():
        p=source/'subnet'/name
        if p.is_symlink() or not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=digest:raise ValueError('unqualified base source: '+name)
    if not Path(private_tasks).is_absolute():raise ValueError('operator private collection must be absolute')
    env=(source/'subnet/environments.py').read_text()
    env=replace_once(env,"('prime_v1', 'legacy_mastermind', 'resource_prime_v1', 'resource_prime_v1_controlled')","('prime_v1', 'legacy_mastermind', 'resource_prime_v1', 'resource_prime_v1_controlled', 'native_sql_controlled')")
    env=replace_once(env,"    files.append(('adapter',_hash(__file__)))\n","    files.append(('adapter',_hash(__file__)))\n    if spec.adapter=='native_sql_controlled':\n        for name in ('native_sql_adapter.py','native_sql_actor.py','native_sql_isolation.py','native_sql_deployment.py'):\n            files.append(('native_sql',name,_hash(PACKAGE_ROOT/name)))\n")
    env=replace_once(env,"    checked=EnvironmentSpec.from_dict(spec) if isinstance(spec,dict) else spec\n","    checked=EnvironmentSpec.from_dict(spec) if isinstance(spec,dict) else spec\n    if checked.adapter=='native_sql_controlled':\n        if _source_hash(checked)!=checked.source_hash:raise ValueError('SQL approved source/resource hash')\n        from .native_sql_deployment import deployment\n        return deployment(checked,"+repr(str(private_tasks))+')\n')
    harness=(source/'subnet/harness.py').read_text()
    harness=replace_once(harness,"('autoregressive', 'candidates', 'visible-copy-candidates')","('autoregressive', 'candidates', 'visible-copy-candidates', 'public-sql-candidates')")
    harness=replace_once(harness,"    if value['policy'] == 'candidates':\n","    if value['policy'] == 'public-sql-candidates':\n        from .public_sql_candidates import REVISION\n        digest=hashlib.sha256((Path(__file__).parent/'public_sql_candidates.py').read_bytes()).hexdigest()\n        if value.get('public_policy_revision')!=REVISION or value.get('public_policy_sha256')!=digest:\n            raise ValueError('public SQL proposal policy pin')\n    if value['policy'] == 'candidates':\n")
    harness=replace_once(harness,"return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()","return hashlib.sha256(Path(__file__).read_bytes()+(Path(__file__).parent/'public_sql_candidates.py').read_bytes()).hexdigest()")
    harness=replace_once(harness,"    if config['policy'] == 'visible-copy-candidates':\n","    if config['policy'] == 'public-sql-candidates':\n        from .public_sql_candidates import candidates as public_sql_candidates\n        proposals=public_sql_candidates(messages)\n        if len(proposals)!=2:raise ValueError('unsupported public SQL question')\n        config=dict(config,policy='candidates',candidates=proposals)\n    if config['policy'] == 'visible-copy-candidates':\n")
    gpu=(source/'subnet/gpu_runtime.py').read_text()
    gpu=replace_once(gpu,"        if config['policy']=='visible-copy-candidates':\n","        if config['policy']=='public-sql-candidates':\n            from .public_sql_candidates import candidates as public_sql_candidates\n            proposals=public_sql_candidates(messages)\n            if len(proposals)!=2:raise ValueError('unsupported public SQL question')\n            config={**config,'policy':'candidates','candidates':proposals}\n        if config['policy']=='visible-copy-candidates':\n")
    return {'environments.py':env,'harness.py':harness,'gpu_runtime.py':gpu}

def prepare(source,destination,private_tasks='/root/native-sql-common-v1/operator/private-tasks.json'):
    source=Path(source).resolve();destination=Path(destination).resolve()
    if destination.exists() or destination.is_relative_to(source):raise ValueError('new separate destination required')
    # Never follow links into credentials, private resources, or the live tree.
    for root in (source/'subnet',source/'prototype/vendor'):
        if root.resolve()!=root or root.is_symlink():raise ValueError('source inventory symlink')
        if root.exists() and any(p.is_symlink() for p in root.rglob('*')):
            raise ValueError('source inventory symlink')
    # Public preparation must start from tracked, unmodified source, not a
    # working directory containing arbitrary regular private files.
    tracked=set(subprocess.check_output(['git','-C',str(source),'ls-files','-z','--','subnet','prototype/vendor']).decode().split('\0'))
    for root in (source/'subnet',source/'prototype/vendor'):
        if root.exists():
            for path in root.rglob('*'):
                if path.is_file() and '__pycache__' not in path.parts and path.suffix!='.pyc' and str(path.relative_to(source)) not in tracked:
                    raise ValueError('untracked source inventory file')
    for args in (['diff','--quiet','--'],['diff','--cached','--quiet','--']):
        if subprocess.run(['git','-C',str(source),*args,'subnet','prototype/vendor'],check=False).returncode:
            raise ValueError('clean committed source required')
    patches=transformed(source,private_tasks)
    destination.mkdir(parents=True)
    ignore=shutil.ignore_patterns('__pycache__','*.pyc')
    shutil.copytree(source/'subnet',destination/'subnet',ignore=ignore)
    vendor=source/'prototype/vendor'
    if vendor.exists():shutil.copytree(vendor,destination/'prototype/vendor',ignore=ignore)
    for name,text in patches.items():(destination/'subnet'/name).write_text(text)
    record={'schema':1,'scope':'controlled-cohost-original-Spider-public-question-derived-query-policy-v1','base_source_files':BASE,'patched_source_files':{n:hashlib.sha256((destination/'subnet'/n).read_bytes()).hexdigest() for n in patches},'operator_private_assets_copied':False,'models_copied':False,'state_copied':False,'live_source_changed':False,'model_or_native_execution':False}
    (destination/'source-preparation.json').write_text(json.dumps(record,sort_keys=True,indent=2));return record

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--destination',type=Path,required=True);p.add_argument('--private-tasks',default='/root/native-sql-common-v1/operator/private-tasks.json');a=p.parse_args();print(json.dumps(prepare(a.source,a.destination,a.private_tasks)))
if __name__=='__main__':main()
