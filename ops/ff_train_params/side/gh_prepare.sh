#!/usr/bin/env bash
# Build the AffineFoundation/affine commit for RA/20261031-ff-train-params in a fresh clone (no push here).
set -euo pipefail
RA=/home/const/subnet120-rewrite/state/root-audits; D=$RA/20261031-ff-train-params; G=$D/gh; FE=${FE:?from epoch}
cd $G; git pull -q --rebase origin main
O=ops/ff_train_params; rm -rf $O; mkdir -p $O/release $O/host-v833j $O/activation $O/side $O/tests $O/services
cp $D/stage/release/{ff_format.py,ff_lifecycle.py,ff_service.py,affine_batch.py} $O/release/
cp $RA/20261030-free-file/release/FF-PUBLIC.json $RA/20261030-free-file/release/FF-PINS.json $O/release/
cp $D/release/patch_v833j.py $D/sign_v833j.py $O/
cp $D/release-v833j-signed/SIGNED-MANIFEST.sha256 $D/release-v833j-signed/SIGNING-RESULT.json $O/host-v833j/
cp $D/release-v833j-signed/trainer-host-v833j/{mimo_host_trainer_v83.py,ff_trainer.py} $O/host-v833j/
cp $D/activation/{activate_tp.sh,watchdog_tp.py,test_watchdog_tp.py} $O/activation/
cp $D/side/{make_ff_public.py,patch_sidecars.py,web_docs.py,test_k1t.py,gh_prepare.sh} $O/side/
cp $D/tests/{test_v833j.py,test_ff_format_v2.py,test_ff_format_v1_regression.py,test_e2e_sim.py,e2e_sim.out} $O/tests/
cp $RA/20261026-promotion-gate/code/training_params.py $RA/20261027-k1-per-uid/code/k1_transparency.py $O/services/
cp $D/stage/release/affine_batch.py miner/affine_batch.py
python3 - "$FE" <<'EOF'
import sys, re
fe = sys.argv[1]
V2 = open('/home/const/subnet120-rewrite/state/root-audits/20261031-ff-train-params/side/web_docs.py').read()
ns = {}; exec(V2.split("def llms(")[0], ns); block = ns['V2'].replace(' <!-- ff-train-params-20261031 -->', '').replace('### affine-batch-v2', '### Optional: `affine-batch-v2`')
p = 'docs/MINER_QUICKSTART.md'; s = open(p).read(); a = '\n## 3. Upload\n'
assert s.count(a) == 1 and 'affine-batch-v2' not in s; s = s.replace(a, block + a)
s = s.replace('## 2. The file: `affine-batch-v1`\n', '## 2. The file: `affine-batch-v1` (or `affine-batch-v2` with optional lr / kl)\n', 1)
open(p, 'w').write(s)
p = 'README.md'; s = open(p).read(); a = '- [Validator side of the free-file protocol (lifecycle, trainer v8.3.3h, activation/revert)](ops/free_file/README.md)\n'
assert s.count(a) == 1
s = s.replace(a, a + '- Optional per-turn `lr` (1.25e-7 to 1e-6, default 5e-7) and `kl` (0.005 to 0.1, default 0.02) in an `affine-batch-v2` file, live from e%s: '
              '[format and math](docs/MINER_QUICKSTART.md#optional-affine-batch-v2-optional-lr-and-kl-from-2026-10-31) · '
              '[validator side (trainer v8.3.3j, activation/revert)](ops/ff_train_params/README.md)\n' % fe)
open(p, 'w').write(s)
EOF
cat > $O/README.md <<EOF
# Free-file per-turn lr / kl (affine-batch-v2), trainer v8.3.3j

Jacob 2026-10-31 07:08 CR (pt902u). Live from epoch $FE. Built on trainer v8.3.3i (ops/renorm_step).

- Format: \`affine-batch-v2\` = v1 + optional top-level \`"train": {"lr", "kl"}\` (both optional; omitted = default).
  lr in [1.25e-7, 1e-6] (default 5e-7), kl in [0.005, 0.1] (default 0.02), inclusive. Out of range / non-number / extra key
  = invalid file (skipped). Never clamped. v1 unchanged; \`train\` in a v1 file is invalid. \`release/ff_format.py\`.
- Lifecycle (\`release/ff_lifecycle.py\`): v2 accepted only while \`state/FF-TRAIN.json\` is enabled; the effective params go into the
  signed sn120-ff-dispatch-v1 payload \`train\` {lr, kl, given, defaults} and K1-TURN (private). Flag off = exact previous behaviour.
- Trainer v8.3.3j (\`patch_v833j.py\` on the signed v8.3.3i host; \`host-v833j/\`): m = lr / 5e-7;
  n0_t = r_t ||d_t|| (the v8.3.3i per-tensor norm), c_t = 2e-5 (1 - 1e-6) ||M_H,t||, n_t = min(m n0_t, c_t), s = min(1, 1.0 / ||n||),
  Delta_t = s n_t g_t / ||g_t||. m = 1 runs the v8.3.3i code path unchanged. kl replaces kl_coefficient for that one update
  (L = loss_scale (1/N) sum_i (1/K_i) sum_t [pg + kl k3]); the reported config stays the signed one. Report: training.train_params.
- Public: turn.json training.train_params / turn.train / dispatch.train and mining.json training.miner_train_params.recent_green_turns,
  GREEN only (\`services/\`, \`side/patch_sidecars.py\`); batch_file.train in mining.json (\`side/make_ff_public.py\`); docs (\`side/web_docs.py\`).
- Activation: \`activation/activate_tp.sh\` (stage | switch | revert), auto-revert \`activation/watchdog_tp.py\`.
  Revert: \`MODE=revert bash RA/20261031-ff-train-params/activation/activate_tp.sh\`.
- Tests: \`tests/\` (step math vs an independent reference, m = 1 bit-identical to v8.3.3i, format v1 regression + v2, e2e selection
  simulation with a v1 file, a valid v2 file and an out-of-range v2 file), \`activation/test_watchdog_tp.py\`, \`side/test_k1t.py\`.
EOF
git add -A; git status --short | head -50
