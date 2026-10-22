import sys
from pathlib import Path
p=Path(sys.argv[1]);s=p.read_text()
helper='''

def install_host_validator(ref,authority,pinned):
    """Current release validator: the ROOT-admitted v8.2.4 host override that the live
    lifecycle runs (signed uid-operator admission -> pinned mimo_uid_capacity.py).
    Installs ONLY the v1+v2 model-publication receipt check; no lifecycle/trainer methods."""
    import importlib.util,inspect
    from subnet.backend_jobs import signed
    from subnet.storage import canonical,sha
    operator_path=pinned(ref['operator_config']);operator=json.loads(operator_path.read_bytes())
    proposal={k:v for k,v in operator.items() if k!='admission'}
    if signed(operator['admission'],authority)!=dict(version=operator['version'],configuration_sha256=sha(canonical(proposal)),
            source_unchanged=True,scope='host-registration-capacity-only',capacity=operator['capacity']):
        raise ValueError('dashboard exact ROOT host override admission')
    helper_path=pinned(ref['helper'])
    if helper_path.parent!=operator_path.parent or operator['host_files'].get(helper_path.name)!=ref['helper']['sha256']:
        raise ValueError('dashboard host override file not ROOT-admitted')
    spec=importlib.util.spec_from_file_location('affine_dashboard_host_validator',helper_path)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    import subnet.mimo_training_worker as worker
    import subnet.mimo_grpo.publication as publication
    if helper.method_sha(worker.validate_result)!=helper.V824_PINS['mimo_training_worker.validate_result']:
        raise ValueError('dashboard exact original validate_result required')
    source=helper.v824_validate_result_source(inspect.getsource(worker.validate_result))
    exec(compile(helper.V824_PUBLICATION_VERIFIED,inspect.getsourcefile(publication),'exec'),publication.__dict__)
    exec(compile(source,inspect.getsourcefile(worker),'exec'),worker.__dict__)

'''
anchor="\n\ndef main(argv=None):"
assert s.count(anchor)==1; s=s.replace(anchor,helper+"\ndef main(argv=None):")
old="    from subnet.mimo_worker import validate_report\n"
new=old+"    if 'host_validator' in cfg:install_host_validator(cfg['host_validator'],cfg['authority'],pinned)\n"
assert s.count(old)==1; s=s.replace(old,new)
s=s.replace("(v7 r189 -> v8.2 r228 -> v8.2.1 r229 -> v8.2.2/3 r231).","(v7 r189 -> v8.2 r228 -> v8.2.1 r229 -> v8.2.2/3/4 r231).")
p.write_text(s)
