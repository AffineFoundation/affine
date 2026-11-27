"""SN120 lifecycle launcher = RA/20261105-k1-squat-trim/release-signed/ff_launcher.py, unchanged, plus a non-promoting gate timeout.
RA/20261107-timeout-promote (overnight autonomy pt1363u; Jacob 2026-11-06 policy: fail-closed timeouts are no_decision, old head kept).

v8.2.9i (the signed launcher) promotes the candidate when the gate has not decided within decision_timeout_seconds
('timeout-promote'). E836 b0beb5a4 became head that way without a GREEN. This wrapper:
  * gives the inner v829i wrapper an effectively infinite decision_timeout_seconds (it can never reach its promote branch);
  * applies the real timeout itself as no_decision: marker decision 'timeout-no-decision', NO_DECISION sentinel for the gate,
    K1 grant (key stays queued, red_credit zero) and lifecycle result no-update on the current head (exactly the RED path's
    return shape, reason 'promotion-gate-timeout-no-decision-...').
Everything else (argument parsing, composition order, every other install) is the original ff_launcher.main()."""
import importlib.util, json, sys, time
from pathlib import Path

ORIG = Path('/home/const/subnet120-rewrite/state/root-audits/20261105-k1-squat-trim/release-signed/ff_launcher.py')
GATE_EP = Path('/home/const/subnet120-rewrite/state/root-audits/20261026-promotion-gate/state/epochs')
GRANTS = Path('/home/const/subnet120-rewrite/state/root-audits/20261023-toploc-credit-train-gate/staged/K1-GRANTS.json')
TURNS = Path('/home/const/subnet120-rewrite/state/root-audits/20261027-k1-per-uid/state/K1-TURNS.jsonl')
NEVER = 10 ** 12


def _rj(p):
    try:
        return json.loads(Path(p).read_bytes())
    except Exception:
        return None


def _wj(p, obj, mode=0o644):
    p = Path(p); t = p.with_name(p.name + '.tmp-ntp')
    t.write_text(json.dumps(obj, indent=2, sort_keys=True) + chr(10)); t.chmod(mode); t.replace(p)


def _grant(e, why):
    """Same row shape as gate_service._k1_grant_no_decision (key stays queued, no_decision=True)."""
    try:
        uid = hotkey = None
        for line in TURNS.read_text().splitlines():
            if line.strip():
                o = json.loads(line)
                if int(o.get('epoch_index', -1)) == int(e):
                    uid, hotkey = o.get('uid'), o.get('hotkey'); break
        if not hotkey:
            return 'no-hotkey'
        g = _rj(GRANTS) or {}
        rows = list(g.get('grants') or [])
        if any(int(r.get('red_epoch', -1)) == int(e) and r.get('hotkey') == hotkey for r in rows):
            return 'already-granted'
        rows.append(dict(red_epoch=int(e), uid=uid, hotkey=hotkey, weighted_z=None, reason='no_decision: %s' % why,
                         authorized='Jacob 2026-11-06 12:36 CR: fail-closed/infra no_decision is not a served turn; key stays queued via K1-GRANTS',
                         set_at=time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()), no_decision=True))
        g['grants'] = rows
        _wj(GRANTS, g)
        return 'granted'
    except Exception as ex:
        return 'error %r' % (ex,)


def patch(L):
    real_flag = L.pg_flag

    def inner_flag():   # the signed v829i wrapper reads this every tick: it must never reach its timeout-promote branch
        f = real_flag()
        return None if f is None else dict(f, decision_timeout_seconds=NEVER)
    L.pg_flag = inner_flag
    install = L.install_v829i_promotion_gate

    def install_ntp(Lifecycle):
        meta = install(Lifecycle)
        inner = Lifecycle._training

        def _training(self, directory, opening, dispatch, current, now):
            f = real_flag()
            if f is not None:
                d = Path(directory); e = int(d.name); key = str(d)
                res = self.__dict__.get('_v829i_results', {}).get(key)
                if res is not None and res.get('status') == 'updated':
                    H, C = current['checkpoint']['id'], res['checkpoint']['id']
                    dec = _rj(GATE_EP / str(e) / 'DECISION.json') or {}
                    decided = dec.get('head') == H and dec.get('candidate') == C and dec.get('decision') in ('green', 'red')
                    rp = d / 'TRAINING-RESULT.json'
                    t0 = rp.stat().st_mtime if rp.exists() else time.time()
                    if not decided and time.time() - t0 > float(f['decision_timeout_seconds']):
                        why = 'no gate decision within %ss of TRAINING-RESULT (lifecycle backstop)' % f['decision_timeout_seconds']
                        sentinel = GATE_EP / str(e) / 'LIFECYCLE-TIMEOUT.json'
                        sentinel.parent.mkdir(parents=True, exist_ok=True)
                        _wj(sentinel, dict(epoch=e, head=H, candidate=C, at=time.time(), why=why, no_decision=True,
                                           note='lifecycle kept the head; the gate must not decide this epoch (RA/20261107-timeout-promote)'))
                        g = _grant(e, why)
                        marker = dict(version='sn120-promotion-gate-v829i-ntp', epoch_index=e, head=H, candidate=C,
                                      decision='timeout-no-decision', no_decision=True, served_turn=False, red_credit='zero',
                                      reason=why, k1_grant=g)
                        _wj(d / 'PROMOTION-GATE.json', marker)
                        print(json.dumps(dict(component='promotion-gate-v829i-ntp', **marker)), flush=True)
                        self.__dict__['_v829i_results'].pop(key, None)
                        return dict(status='no-update', checkpoint=current['checkpoint'], optimizer_step=current['optimizer_step'],
                                    checkpoint_authorization=current.get('checkpoint_authorization'),
                                    reason='promotion-gate-timeout-no-decision-e%d-%s-kept-%s' % (e, C[:8], H[:8]))
            return inner(self, directory, opening, dispatch, current, now)

        _training.__qualname__, _training.__module__ = inner.__qualname__, inner.__module__
        Lifecycle._training = _training
        return dict(meta, ntp=True, inner_timeout=NEVER)
    L.install_v829i_promotion_gate = install_ntp


def main():
    spec = importlib.util.spec_from_file_location('ff_launcher_orig', ORIG)
    ff = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = ff
    spec.loader.exec_module(ff)
    orig_load = ff.load_launcher

    def load_launcher():
        L = orig_load()
        patch(L)
        return L
    ff.load_launcher = load_launcher
    ff.main()


if __name__ == '__main__':
    main()
