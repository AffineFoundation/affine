#!/usr/bin/env python3
"""PE outer wrapper atop the admitted lifecycle launcher (RA/20261106-parallel-cutover).
2026-11-07 rebase: wraps RA/20261107-timeout-promote/release/ff_launcher_ntp.py (the k1-squat-trim ff_launcher unchanged +
non-promoting gate timeout), not the bare squat-trim ff_launcher, so the timeout -> no_decision guard stays in force whenever
parallel_eval is off. The ntp launcher still performs all admission checks and composes every existing wrapper; only after
v829v is installed do we install PE._training outermost (pe_lifecycle.install_pe unwraps ntp -> v829i -> un-held training).
Revert: bash RA/20261106-parallel-cutover/REVERT.sh (removes the PE drop-in; the ntp drop-in is then the active ExecStart).
"""
import importlib.util
import sys
from pathlib import Path

NTP = Path('/home/const/subnet120-rewrite/state/root-audits/20261107-timeout-promote/release/ff_launcher_ntp.py')
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))


def main():
    spec = importlib.util.spec_from_file_location('pe_ntp_launcher', NTP)
    ntp = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ntp
    spec.loader.exec_module(ntp)
    ntp_patch = ntp.patch

    def patch(L):
        ntp_patch(L)                       # ntp: v829i inner timeout = never; real timeout -> no_decision
        prior = L.install_v829v_preopen

        def finish_composition(Lifecycle):
            meta = prior(Lifecycle)
            import pe_lifecycle
            meta['parallel_eval'] = pe_lifecycle.install_pe(Lifecycle)
            return meta

        L.install_v829v_preopen = finish_composition

    ntp.patch = patch                      # ntp.main -> load_launcher -> patch(L) (module-global lookup)
    ntp.main()


if __name__ == '__main__':
    main()
