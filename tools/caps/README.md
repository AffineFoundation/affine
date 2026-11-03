# SN120 token caps: one source, rendered and checked everywhere

`caps.json` (repo root) is the only place the token caps are set. `caps.py` loads it and derives the rest:

| value | today | rule |
|---|---|---|
| `max_output_tokens` | 12288 | stored: generation `max_tokens`, batch-file output cap, gate task cap, training cap |
| `max_context_tokens` / `max_model_len` | 16384 | `max_output_tokens + prompt_headroom_tokens` (4096) |
| overlong window | 10096..12288 | `max_output_tokens - overlong_ramp_tokens` (2192) .. `max_output_tokens`, penalty 0.5 |
| `legacy_capped_output_tokens` | 8192 | an output of exactly this length still counts as capped (files generated at the old cap) |
| `signed_manifest` | 8192 / 12288 / 6000 | what the root-signed lifecycle job settings say; never edited, overridden by the above |

`tools/caps/consumers.py` lists every place that carries a cap (34 consumers with 104 checked values - 12 live, 16 derived, 5 docs, 1 signed - plus 3 files that must hold no cap literals: ff_format, kit, trainer, gate flag, probe, prior pod,
training.json / mining.json, llms.txt, READMEs, the live pods). Each one is either read from `caps.py` at run time
(training_params, site_metrics, replace_pod roles.json `{CAPS:key}` placeholders) or rendered from `caps.json` and checked against it.

## Check
    python3 tools/caps/caps_check.py --scope repo          # pre-push (git config --local core.hooksPath tools/caps/hooks); CI: copy tools/caps/ci/caps-check.yml to .github/workflows/ (needs a token with workflow scope)
    python3 tools/caps/caps_check.py --scope host,pods     # validator host: live files + prior/probe/trainer pods (read-only ssh)
    python3 tools/caps/caps_check.py --scope public        # affine.io + GitHub
The hourly SN120 health check runs `--scope host,pods` and reports CRIT on any disagreement; replacing a pod fails its health check
if the pod is not at the caps.

## Change the cap (one edit + one command)
1. Edit `max_output_tokens` in `caps.json` on the validator host (`~/subnet120-rewrite/caps.json`); set `legacy_capped_output_tokens` to
   the old cap and append a `history` entry (`from_epoch`, `label`).
2. `python3 tools/caps/caps_apply.py` shows the plan; `python3 tools/caps/caps_apply.py --apply` waits for an idle epoch boundary,
   takes the lifecycle locks, backs everything up, renders every host file (gate flag, ff_format + FF-PINS, FF-PUBLIC, probe config,
   prior sampler/serve files, docs), pushes the prior pod config and restarts its sampler on a fresh pass,
   restarts the publishers, verifies with caps_check and writes `REVERT.sh`.
   The trainer constants live in a root-signed release: if they would change, apply stops first and `--stage-trainer` writes the patched
   files for the signing step (then switch the release and re-run apply).
3. In a clone of this repo: copy the new `caps.json`, run `python3 tools/caps/caps_apply.py --scope repo --write`, commit, push (CI checks).
