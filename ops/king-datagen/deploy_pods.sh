#!/bin/bash
# Deploy the rollouts package to the datagen pods and (optionally) ask the
# supervisor to relaunch between batches.
#
#   ops/king-datagen/deploy_pods.sh [--restart] --all
#   ops/king-datagen/deploy_pods.sh [--restart] HOST:PORT [HOST:PORT ...]
#
# Per pod: back up the current files to /root/rollouts/rollouts/.bak/, scp
# the package files listed in FILES, import-check the registry on the pod
# (REGISTRY_OK), and with --restart touch /root/rollouts/RESTART — the
# supervisor exits cleanly at its next cycle boundary (no running batch is
# killed) and bootstrap.sh relaunches it on the new code. Never kills.
#
# The pods' /root/affine tree is NOT touched except for AFFINE_FILES — the
# stdlib-only dialect registry the pod imports to validate policies'
# action_kind (a policy in an unregistered dialect fails the registry
# check; the same file gates nothing on the pod beyond yield accounting).
#
# Env wave 1 (2026-09-11): the in-house taskset wrappers under rollouts/envs
# (ENV_PKGS) are copied to /root/rollouts/envs/ and installed editable into
# the pod's verifiers env together with the research-environments tasksets
# they subclass (RESEARCH_ENVS, from the pod's own checkout) — always
# `--no-deps`: the tasksets pin `verifiers>=0.2.0`, and letting uv resolve
# that would replace the pod's editable verifiers checkout (0.0.1.dev1) with
# a PyPI release. Their three pure-python extras are installed by name.
# Idempotent; UV_NO_SYNC pods are fine (`uv pip` never syncs the lock).
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
REPO=$PWD
KH="$REPO/ops/king-datagen/state/known_hosts"
FILES=(schema.py registry.py catalog.py run.py king.py scheduler.py policies.toml sources.toml
       loopguard.py loopguard_site/sitecustomize.py adapters/mini_swe.py adapters/verifiers.py
       dockerwrap/docker
       runners/base.py runners/verifiers.py runners/mini_swe.py)
AFFINE_FILES=(affine/dialects.py affine/corpus/trace.py)
# The mini_swe_textbased verifiers harness is a tiny package of ours installed
# editable on every pod at /root/prime-pilot/mini-swe-textbased; its source of
# truth is rollouts/harnesses/mini_swe_textbased in this repo.
HARNESS_SRC=rollouts/harnesses/mini_swe_textbased/mini_swe_textbased/__init__.py
HARNESS_DST=/root/prime-pilot/mini-swe-textbased/mini_swe_textbased/__init__.py
ENV_PKGS=(affine_logic_v1 affine_trivia_v1 affine_ifeval_v1 affine_science_v1
          affine_unscramble_v1 affine_prolog_v1 affine_needle_v1 affine_wikispeedia_v1)
RESEARCH_ENVS=(reasoning/i3_logic_v1 knowledge/triviaqa_v1 if/ifeval_v1 science/i3_science_v1
               reasoning/unscramble_v1 reasoning/prolog_v1
               long_context/patterned_needle_in_haystack_v1 reasoning/wikispeedia_v1)
ENV_EXTRA_DEPS=(immutabledict langdetect markdown)
ENV_IMAGES=(swipl:latest)
RESTART=0; TARGETS=()
for a in "$@"; do
  case "$a" in
    --restart) RESTART=1 ;;
    --all) mapfile -t rows < <("$REPO/.venv/bin/python" ops/king-datagen/kingctl.py pods)
           for r in "${rows[@]}"; do read -r _n h p <<< "$r"; TARGETS+=("$h:$p"); done ;;
    *) TARGETS+=("$a") ;;
  esac
done
[ ${#TARGETS[@]} -gt 0 ] || { echo "no targets (use --all or HOST:PORT)"; exit 2; }

rc=0
for t in "${TARGETS[@]}"; do
  H=${t%%:*}; P=${t##*:}
  SSH="ssh -o UserKnownHostsFile=$KH -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o ConnectTimeout=15 -o LogLevel=ERROR -p $P root@$H"
  SCP="scp -o UserKnownHostsFile=$KH -o StrictHostKeyChecking=accept-new -o BatchMode=yes -o LogLevel=ERROR -P $P"
  echo "== $H:$P"
  $SSH 'mkdir -p /root/rollouts/rollouts/{runners,adapters,loopguard_site,dockerwrap} /root/rollouts/rollouts/.bak/{runners,adapters,loopguard_site,dockerwrap} /root/affine/.bak/affine/corpus && cd /root/rollouts/rollouts && for f in '"${FILES[*]}"'; do [ -f "$f" ] && cp "$f" ".bak/$f"; done; cd /root/affine && for f in '"${AFFINE_FILES[*]}"'; do [ -f "$f" ] && cp "$f" ".bak/$f"; done; mkdir -p "$(dirname '"$HARNESS_DST"')/.bak" && [ -f '"$HARNESS_DST"' ] && cp '"$HARNESS_DST"' "$(dirname '"$HARNESS_DST"')/.bak/__init__.py"; echo backed-up' || { echo "SSH-FAILED"; rc=1; continue; }
  ok=1
  for f in "${FILES[@]}"; do
    $SCP "rollouts/rollouts/$f" "root@$H:/root/rollouts/rollouts/$f" || { echo "SCP-FAILED $f"; ok=0; break; }
  done
  for f in "${AFFINE_FILES[@]}"; do
    $SCP "affine/$f" "root@$H:/root/affine/$f" || { echo "SCP-FAILED $f"; ok=0; break; }
  done
  if [ $ok = 1 ]; then
    $SCP "$HARNESS_SRC" "root@$H:$HARNESS_DST" || { echo "SCP-FAILED $HARNESS_SRC"; ok=0; }
  fi
  [ $ok = 1 ] || { rc=1; continue; }
  $SSH 'chmod +x /root/rollouts/rollouts/dockerwrap/docker'
  tar -C rollouts/envs -czf - "${ENV_PKGS[@]}" | $SSH 'mkdir -p /root/rollouts/envs && tar -C /root/rollouts/envs -xzf -' \
    || { echo "ENV-COPY-FAILED"; rc=1; continue; }
  $SSH 'export PATH=$HOME/.local/bin:$PATH; cd /root/prime-pilot/verifiers || exit 1
RE=/root/prime-pilot/research-environments/environments
base=(); for d in '"${RESEARCH_ENVS[*]}"'; do [ -d "$RE/$d" ] && base+=(-e "$RE/$d") || echo "missing research env $d"; done
uv pip install --python .venv/bin/python -q --no-deps "${base[@]}" || exit 1
wrap=(); for p in '"${ENV_PKGS[*]}"'; do wrap+=(-e "/root/rollouts/envs/$p"); done
uv pip install --python .venv/bin/python -q --no-deps "${wrap[@]}" || exit 1
uv pip install --python .venv/bin/python -q '"${ENV_EXTRA_DEPS[*]}"' || exit 1
.venv/bin/python -c "import '"$(IFS=,; echo "${ENV_PKGS[*]}")"'; print(\"ENV_IMPORT_OK\")" || exit 1
for img in '"${ENV_IMAGES[*]}"'; do docker image inspect "$img" >/dev/null 2>&1 || docker pull -q "$img" >/dev/null || echo "WARNING: pull failed $img"; done' \
    || { echo "ENV-INSTALL-FAILED"; rc=1; continue; }
  $SSH "/root/prime-pilot/verifiers/.venv/bin/python -m py_compile $HARNESS_DST && echo HARNESS_OK" || { echo "HARNESS-COMPILE-FAILED (restored from .bak)"; $SSH "cp $(dirname "$HARNESS_DST")/.bak/__init__.py $HARNESS_DST"; rc=1; continue; }
  $SSH 'cd /root/rollouts && source /root/affine/.datagen_env && source /root/rollouts/.rollouts_env 2>/dev/null; PYTHONPATH=/root/affine:/root/rollouts /root/venv/bin/python - <<PY
import os
from rollouts.registry import load_registry
from rollouts.king import read_king_env, KING_ENV_PATH
from rollouts.runners.base import EndpointHealth
from rollouts.scheduler import Scheduler
import rollouts.run
r = load_registry()
ks = sorted(p for p in r.policies if p.startswith("king_"))
env = dict(os.environ)
avail = [p.id for p in r.policies.values() if p.available_endpoints(env)]
ke = read_king_env()
print("REGISTRY_OK king policies", len(ks), "available now", len(avail),
      "king env", KING_ENV_PATH, "reign", ke.get("KING_REIGN"), "digest", (ke.get("KING_DIGEST") or "")[:12])
PY' || { echo "REGISTRY-CHECK-FAILED (files restored from .bak)"; $SSH 'cd /root/rollouts/rollouts && for f in '"${FILES[*]}"'; do [ -f ".bak/$f" ] && cp ".bak/$f" "$f"; done; cd /root/affine && for f in '"${AFFINE_FILES[*]}"'; do [ -f ".bak/$f" ] && cp ".bak/$f" "$f"; done'; rc=1; continue; }
  if [ $RESTART = 1 ]; then
    $SSH 'touch /root/rollouts/RESTART && echo "RESTART flag set (supervisor exits at the next cycle boundary; bootstrap loop relaunches)"; pgrep -f -x "bash /root/rollouts/bootstrap.sh" >/dev/null || echo "WARNING: bootstrap loop not running — kingctl watchdog will relaunch it"'
  fi
done
exit $rc
