# Fresh training-split evaluation

The dashboard has a separate “Training” selector in the existing performance chart. It displays fresh generations on a fixed 128-task cohort drawn from the approved training split. This is a diagnostic of the model at a checkpoint, not the 50% success fraction required by submitted K4/L4 batches and not an independent held-out confirmation.

The separate evaluator targets every four finalized optimizer updates, coalescing a backlog to the latest durable checkpoint. Its original 7B comparison reference (checkpoint 2173) and current-model evaluations use the same cohort, task seeds and runtime profile with a 2,048-token output budget. The original 7B reference differs from the actual training-run genesis (checkpoint 6493); exposure follows the actual run ancestry. Existing held-out series remain separate, including older 1,024-token profiles. Epoch progression and weight setting do not wait for this diagnostic.

Only ROOT-authenticated aggregate results whose original private artifact has passed a complete bucket readback appear in the public JSON. Actual prompt/task/seed comparability must pass. Infrastructure errors or incomplete provenance produce an unavailable observation, not zero accuracy; chart lines break across missing observations or changes of run, cohort, seed or runtime. Raw prompts, generated text, task identifiers, object capabilities and credentials are not exposed by this projection.

The fixed overall cohort can include tasks not yet used by the learner. Tooltips show the number previously trained and solved/count for previously trained, not-yet-trained and unknown-exposure groups. These changing groups are descriptive, not a fixed memorization test. An optional frozen initially-trained subset is identified separately. Baseline “zero exposure” describes this training run only, not unknown pretraining history.

The dashboard consumes finalized signed aggregates from its configured source directory under `training-evaluations/*.signed.json` and exports them as `training_evaluations` in `network-data.json`. Empty means no completed authenticated measurement is available. No fabricated points or miner labels fill the series. Producer deployment and the first numerical measurements are separate from publishing the display.

The public change also synchronizes existing deployed responsive/desktop and incentive display support. It does not redesign the website or change the mining contract.

Validation: 12 unique Python controls cover signed provenance, count/exposure accounting, comparability, unavailable results, database isolation and the actual synthetic producer-to-projection seam; six Node controls cover chart separation and gaps. Browser checks passed at 390px and 1440px without changing the deployed CSS.
