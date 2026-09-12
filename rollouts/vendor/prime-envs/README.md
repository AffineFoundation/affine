# Vendored prime-envs tasksets

Tasksets from `PrimeIntellect-ai/prime-envs` (formerly `research-environments`)
that the datagen pods' pinned checkout (`b10db76`, 2026-08-05) does not carry.
Copied verbatim at the commit named below; installed editable `--no-deps` into
the pods' verifiers env by `ops/king-datagen/deploy_pods.sh` (`VENDOR_ENVS`).
Do not `git pull` the pods' checkout to get them: commit `c4d04dfe`
(2026-09-08, PR #798) also adds solver system prompts to every SWE and
terminal taskset, which would change every coding / terminal prefix in D.

| Package | Upstream path | Commit |
|---|---|---|
| `deshuffle_papers` | `environments/reasoning/deshuffle_papers` | `c4d04dfe` |
