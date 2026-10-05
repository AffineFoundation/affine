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
    for name in ('cache_lifecycle','distributed_worker'):
        qualified='subnet.'+('operator_lifecycle_worker' if name=='distributed_worker' else name)
        spec=importlib.util.spec_from_file_location(qualified,overlay/'subnet'/(name+'.py'))
        module=importlib.util.module_from_spec(spec);sys.modules[qualified]=module;spec.loader.exec_module(module)
    module.main()


if __name__=='__main__':main()
