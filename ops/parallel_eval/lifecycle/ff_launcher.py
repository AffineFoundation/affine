"""SN120 lifecycle launcher: the live v8.2.9x launcher (L) + free-file turns (ff_lifecycle.py). Jacob 2026-10-30 pt898u.

Same arguments as L. Runs L's own composition unchanged and installs ff_lifecycle right after L's K1 wrapper, so the
promotion gate (v829i), early seal (v829t) and pre-open (v829v) wrap it exactly as they wrap K1.
Revert = the previous unit drop-in (activate_ff.sh MODE=revert)."""
import argparse, importlib.util, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
L_PATH = Path('/home/const/subnet120-rewrite/state/root-audits/20261029-v829u-125task/release-v829x-v823u-signed/mimo_v829_train_gate_launcher.py')
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))


def load_launcher():
    spec = importlib.util.spec_from_file_location('affine_v829_launcher', L_PATH)
    L = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = L
    spec.loader.exec_module(L)
    return L


def main():
    L = load_launcher()
    import ff_lifecycle
    spec = importlib.util.spec_from_file_location('affine_v828_launcher', L.V828_DIR / 'mimo_v828_launcher.py')
    v828 = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = v828
    if str(L.V828_DIR) not in sys.path:
        sys.path.insert(0, str(L.V828_DIR))
    spec.loader.exec_module(v828)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--operator-config', required=True)
    p.add_argument('--check', action='store_true')
    p.add_argument('--v828-operator-config', default=str(L.V828_DIR / 'v828-operator.ROOT-ADMITTED.private.json'))
    args = p.parse_args()
    uid, parent, operator, module, Lifecycle, Coordinator, methods, workers = v828.checked(v828._paths.private_json(args.v828_operator_config))
    v824, v828_meta = v828.compose(uid, parent, operator, Lifecycle, Coordinator, methods, module, workers)
    meta = L.install_v829(module, Lifecycle)
    meta['v829h'] = L.install_v829h_collection(Lifecycle)
    meta['v829l'] = L.install_v829l_k1(Lifecycle)
    meta['ff'] = ff_lifecycle.install(L, Lifecycle)          # free-file turns, inside the gate wrapper
    meta['v829p_cutoff'] = L.install_v829p_served_cutoff(Lifecycle)
    meta['v823_lifetime'] = L.install_v823_lifetime(Lifecycle)
    meta['v829i'] = L.install_v829i_promotion_gate(Lifecycle)
    meta['v829j'] = L.install_v829j_eval_memo()
    meta['v829k'] = L.install_v829k_seal_prefetch(Coordinator)
    meta['v829u'] = L.install_v829u_dual_accept(Coordinator)
    meta['v829x'] = L.install_v829x_batch_cap(Coordinator)
    meta['v829n'] = dict(L.install_v829n_capture_put(), **L.install_v829n_pregrade(Lifecycle))
    meta['v829t'] = L.install_v829t_red_skip(Lifecycle, Coordinator)
    meta['v829v'] = L.install_v829v_preopen(Lifecycle)
    import subnet.distributed_roles as roles
    meta['v829d_verify_workers'] = L.install_v829d_verifier_roster(roles.Coordinator)
    if args.check:
        print(json.dumps(dict(version=L.VERSION, scope=L.SCOPE, ff=True, v828=v828_meta, v829=meta, gate_enabled=L.gate_enabled(),
                              production_services_started=False, operator_admission_passed=True, v829_check=True), default=str))
        return
    print(json.dumps(dict(component='host-v824', **v824)), flush=True)
    print(json.dumps(dict(component='host-v828', **v828_meta)), flush=True)
    print(json.dumps(dict(component='host-v829', **meta), default=str), flush=True)
    sys.argv = [operator['base']['entry']['path']] + operator['base']['argv']
    module.main()


if __name__ == '__main__':
    main()
