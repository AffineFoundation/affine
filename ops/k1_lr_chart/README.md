# K=1 learning rate per turn (affine.io tile)

Jacob pt1039u (2026-11-01 19:54 CR): chart of the learning rate the trainer applied to each turn batch.

- Data: `https://affine.io/k1/lr.json`, written every 20 s by `affine-training-params` (`training_params.py` `write_lr()`;
  `training.json` / `mining.json` are unchanged). Per epoch: TRAINING-RESULT `report.training.train_params.lr` (the applied lr,
  miner-chosen or default), else optimizer `sgd.sgd_lr`. Verdict from the gate DECISION.json.
- Privacy: a miner-chosen lr on a RED or pending turn stays private (null), like `train_params` in RED turn.json.
- Site: `k1.js` tile `panel-k1-lr` (log y, GREEN/RED/pending dots, dashed red = private miner lr, same epoch range as the
  pass-rate chart); `tile-details.js` keeps its legend visible.
- Revert: `bash ops/k1_lr_chart/revert.sh` on the host (backups `*.bak-pre-lrchart-20261102T015639Z`).
