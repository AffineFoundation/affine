"""Operator overlay: preserve original backend source and signed inventory.

Invoke this immutable wrapper with --backend-source /original/pinned/source.
No original source file is replaced. Uses that source's protocol/authentication
modules, and only adds local ownership lifecycle around backend job completion.
"""
import importlib.util
from pathlib import Path
import sys


def main():
    if '--backend-source' not in sys.argv:raise ValueError('pinned backend source required')
    source=Path(sys.argv[sys.argv.index('--backend-source')+1]).resolve(strict=True)
    sys.path.insert(0,str(source))
    import subnet
    overlay=Path(__file__).resolve().parents[1]
    if '--resident-backend' in sys.argv:
        helper=Path(sys.argv[sys.argv.index('--resident-backend')+1])
        if helper != overlay/'ops'/'resident_verifier_backend.py':
            raise ValueError('resident helper must be the operator overlay member')
        spec=importlib.util.spec_from_file_location('ops.resident_verifier_backend',helper)
        resident=importlib.util.module_from_spec(spec);sys.modules[spec.name]=resident
        spec.loader.exec_module(resident)
    for name in ('cache_lifecycle','distributed_worker'):
        if name=='distributed_worker':
            spec=importlib.util.spec_from_file_location('ops.verifier_capacity_admission',overlay/'ops'/'verifier_capacity_admission.py')
            capacity=importlib.util.module_from_spec(spec);sys.modules[spec.name]=capacity;spec.loader.exec_module(capacity)
        qualified='subnet.'+('operator_lifecycle_worker' if name=='distributed_worker' else name)
        spec=importlib.util.spec_from_file_location(qualified,overlay/'subnet'/(name+'.py'))
        module=importlib.util.module_from_spec(spec);sys.modules[qualified]=module;spec.loader.exec_module(module)
    module.main()


if __name__=='__main__':main()
