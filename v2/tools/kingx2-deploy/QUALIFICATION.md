# Offline qualification — DEPLOY KIT (do not run against prod)

This note records offline evidence, not deployment authorization or production
health. The source base is `28c56f439c0c0fbf3554c8546cc86dbd88119125`.
No production service, chain write, R2 operation or secret access is part of
these checks.

## Implementation identities

- GPU code: `64e7bd9c10bba7eeb139ca70a68c41711da882dbee42763890f85be59a4ea3f9`
  -> `5027620615cbd7bc707668cc9f93e98e2133180613404b558e6f05a2b648eb47`.
- Params: `94f931588114f335814231409938f39f2056893708bc798e90f1f7676920a35c`
  -> `bba7a409fc12d9ceb7b85f4e935fadfbd3f4b1efccedb7a6fde90c451f6e534a`.

```json
{
  "irt_draw_rule": {"before": null, "after": "sqrt_fisher"},
  "null_samples_per_task": {"before": null, "after": 2},
  "recipe": {"code_digest": {
    "before": "64e7bd9c10bba7eeb139ca70a68c41711da882dbee42763890f85be59a4ea3f9",
    "after": "5027620615cbd7bc707668cc9f93e98e2133180613404b558e6f05a2b648eb47"
  }}
}
```

The `null` before values mean omitted fields, not stored JSON nulls: the old
rule was Fisher and the king inherited the LoRA sample count. LoRA
`samples_per_task=1`, 2,000 tasks,
six-LoRA batch cap and 10,000-task rotation are unchanged.

## Replay and synthetic evidence

- Historical report: `/home/bb/scratch/kingx2-live-history-v30/history.json`.
  All 81 draws, public entry fields, public score fields and round metadata
  matched; 33 public winner decisions and the fully reconstructible r027 IRT
  update matched. Sixty-nine rounds lack private scored-LoRA outcomes needed
  for a complete IRT recomputation. No standalone `round_scores` artifacts
  exist; the score fields in `round.json` were compared instead. This is not
  complete private decision/IRT recomputation.
- Synthetic round: `/home/bb/scratch/kingx2-live-qualification-synthetic-final-v7/synthetic.json`.
  Two king strings and one LoRA string give delta 0.75, SE
  0.14433756729740643, cap denominators 8/4 and an SE ratio 0.8628490455699164,
  near sqrt(0.75) = 0.8660254037844386 for comparable independent noise.
- Next open: `/home/bb/scratch/kingx2-live-qualification-v25/next-open.json`.
  The old/new opens have the same 10,000-task active set and rotation, the new
  params hash and sqrt-Fisher snapshot, and different intended draws.
- Timing projection: `/home/bb/scratch/kingx2-timing-final.json`. Adding one
  king sample changes `1+L` rollouts to `2+L`: +14.2857% at L=6. Projected
  totals are r074 8,138s, r080 9,640.86s and r081 7,300.43s, below the 10,200s
  scoring-before-reserve limit. This is rollout-count scaling, **not measured
  GPU throughput**. Older maxima r049 (9,455s evaluation) and r055 (10,212s
  total) used other configurations; live health remains an explicit gate.

For r081, the journal-authenticated rotation is
`289a2184869bda8a2d97cef809c3bf91c64c3db357e3108fea6ff11164e47dcf`
and round SHA is `3f49be1b91f866bd280f956be3add0b9add27d751ae6be3d65a5762cf8e365b0`.
The older cached ranking `6306d36d920380f76e02027ca5757a9f64ef991c80dffc000927bb84f9bc4b7d`
and rotation `449af9ac3fa9f2257296f507f63f5f6238265062b64e31ed9fcbb2b52ab2a7bc`
do not match that journal generation. They remain explicitly reported, not
waived; the authenticated current generation was used for replay.

## Test and rehearsal logs

- Exact-base suite: `/home/bb/scratch/kingx2-full-base-final.log`.
- Branch suite: `/home/bb/scratch/kingx2-full-branch-final-v6.log` (completed
  100%; exit 1 with only the two exact-base failures: the grader-lifecycle
  missing-`capture` failure and the public-snapshot hash mismatch). No
  branch-specific failure remains. The project's duplicate quiet flags omit an
  aggregate pass count; the final named-failure list is the recorded result.
  A branch-only contract-test assumption about default optional fields was
  corrected without changing production bytes; the test now also requires
  selected `null_samples_per_task=2` to be published.
- Focused feature/import tests: `/home/bb/scratch/kingx2-final-focused-v8.log`
  (**155 passed**, including the corrected publisher contract).
- Successful complete isolated rehearsal: `/home/bb/scratch/kingx2-r082-live/kingx2-deploy-final-v30/dry-run.log`.
  Cutover, first-round health, public tamper, sealed-result preservation,
  independent undecided rollback and the decided-round guard all passed.
- Forced abort: `/home/bb/scratch/kingx2-r082-live/kingx2-deploy-abort-final-v23/dry-run.log`.
- Sealed-result guard: `/home/bb/scratch/kingx2-r082-live/kingx2-deploy-sealed-final-v22/dry-run.log`.
- Failed-lease sealed-result guard: `/home/bb/scratch/kingx2-r082-live/kingx2-deploy-failed-lease-final-v23/dry-run.log`.
- Pre-go disarm: `/home/bb/scratch/kingx2-r082-live/kingx2-disarm-final-v5/disarm-check.log`.

The rehearsal copies journal/public objects, uses a scratch-only commit and
simulated loop/outcomes. It does not run a GPU workload or establish production
health. The implementation commit and kit-pin commit are recorded in the PR;
the scratch commit in rehearsal logs is not the implementation commit.
