# mining.json fix — 2026-10-21 (times CR, UTC-6)

## Root cause
mining.json is written by `affine-mining-discovery.service` (user unit on Arbos Life; static file
`~/subnet120/affine/website/mining.json` served by Caddy; no R2 write). The active drop-in was
`zzzz~~~~mimo-observer-v824.conf`: observer `20261013-null-pow-v1/uid1000-host-prospective/publish_mimo_public_discovery.py`
with `20261017-v824/observer.v8.2.4.private.json`. Its `selector()` authenticates the *physical* lifecycle argv and
only knew the v8.2.4 chain (uid-operator v8.2.4 ef82a9f0 -> capture-operator v8.2.4 baba0e55, capture launcher version
`...-v1`). v8.2.6 (12:56:55) relaunched the lifecycle via uid-operator v8.2.6 (103af4f6...) / capture v2, so every
5 s tick raised `ValueError: physical MiMo lifecycle config/entry/activation differs` -> base document
`discovery_binding_error` (first at 12:57:03). v8.2.7 (capture `-v3`) and v8.2.8 (new `mimo_v828_launcher.py`
layer) were also unknown to the observer. No release rebound the observer after v8.2.4. Openings themselves
were fine: e80 OPENING/DISCOVERY authenticate.
Impact: e77 visible only 12:52–12:57; e78, e79, e80 never shown as open.

## Fix (observer only; lifecycle, trainer, v9, chain untouched; nothing re-signed)
- `observer-v828/publish_mimo_public_discovery.py` sha256 4ab43459fc0e83d383e633f86b564144fbeca97b2be582234880e222dd42861a
  (copy of original 9b599d56...; +capture versions v2/v3; +`host_v828_execution` verifying the ROOT v8.2.8 admission,
  host files, and re-verifying the full v8.2.7 uid -> capture -> lifecycle v8.2.4 chain; no wrapper code executed).
- `observer.v8.2.8.private.json` sha256 e6296a29eef820e5ed33d12ae037a33cfd8dc75d07de0e2bfb729035894c523c pins:
  lifecycle c38a9fcf (unchanged), entry d068abbd, capture-operator v8.2.7 1f1d7586, uid-operator v8.2.7 4b8b081f,
  v828-operator 56e57851 (matches v828 SIGNING-RESULT).
- Drop-in `zzzz~~~~~mimo-observer-v828.conf` installed (overrides v824 drop-in, left in place). Observer restarted 19:02:14.
- Dry run (DRYRUN.txt): live v8.2.8 process accepted; old v8.2.4 config refused; tampered argv refused.

## Result
19:02:14 mining.json authenticated again: `epoch_closed`, epoch e80 (e80 window 18:26:43–18:56:43 already over).
Stock `miner_supervisor.opening()` returns None now (correct: closed); forced in-window probe authenticates the
signed R2 pointer + manifest for e80 -> PASS. e81 will show `mining` / accepting_submissions=true automatically
when the lifecycle signs its opening.

## Recommendations (not done)
- Consider longer collection for e81 (e.g. collection_seconds 1800 -> 3600) so idle miners can re-hydrate — lifecycle config change, needs approval.
- Monitor: alert if mining.json status==discovery_binding_error > 60 s or status_updated_at stale.
- Release checklist: every lifecycle-launcher release must ship + dry-run an observer rebind.
