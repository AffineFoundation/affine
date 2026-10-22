# seq7 progress notes (CR = UTC-6)
- 11:10 task received. 11:12 copied live code/scores/assessments to box (/workspace/sn120-v9fix).
- 11:23 DEPLOYED (deploy7.sh, all steps completed, nothing half-applied): backup/pre-seq7-20261022T172302Z; pod v9agg.py fix;
  writer code (weights_v9.py, hourly_entry_v9.py, sign_v9.py, test_weights_v9.py, public_scores_v9.py); signed LIVE-CONFIG seq4
  (28bb2380...), HOST-COMPOSITION (d93f9c49...), RELEASE seq7 (b7e298ae...). --check OK (config_sequence 4, 165 paid).
- 11:24 --plan OK (165 UIDs; journal plan deferred by epoch-boundary guard). Side effect: overwrote live-log/weights-1792688400.json,
  restored byte-identical 11:43 (see ROLLBACK.md caution).
- 11:24 public https://affine.io/scores.json, /scores/e88.json, /scores/index.json live (refreshed by every writer run).
- 11:28 / 11:40 e85 / e87 re-scored with fixed scorer and installed.
- ~11:30 llms.txt updated live (backup website/llms.txt.before-v9-scoring-20261022).
- GitHub: commit b9f5a663 on local branch docs/v9-live-scoring in /workspace/affine (box), patch here. Push blocked: no GitHub credentials.
- Evidence feed (history.json): stale since 10-16 00:37 CR, cause: publisher aborts on "MiMo eval release" policy mismatch. NOT fixed:
  publishing writes to R2 (R2 is read-only for this task). One-line patch tested only as a scratch copy in /tmp/evpub.
- 12:00 CR run: first window under seq7, in progress at 12:01.
- 12:01 CR: interruption check. 11:00 CR vector a97e03d599 (28 UIDs) came from the seq6 rule on the e88 score (installed 10:25 CR); the 11:00 log was written 11:00:27 CR, before this release (11:23). Not caused by seq7. State consistent: all signed docs match pinned code (--check OK).
- 12:01 CR: 12:00 window submitted under seq7: score e89 (window e86-e89), 179 UIDs, burn 0, max 1.68%; commit included block 9340706, intended sha f36c7ab708, pending reveal. 0 paid UIDs with 0 e89 groups; 67/67 full-9 miners paid; UID 108 (0 groups e89) = 0.
- 12:44 CR: 12:00 vector REVEALED on chain at the 9340907 epoch boundary: f36c7ab708, 179 UIDs, UID0 0, == intent. Release verified. Remaining: GitHub push (no creds); evidence feed fix (R2 write, not done).
