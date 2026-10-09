# Serial private held-out monitor

This operator observes genuine, durably published training checkpoints and evaluates a fixed, authenticated cohort on a separate worker. It never gates production training. A ROOT-signed configuration supplies the already qualified scientific `evaluate.py`, its exact source/runtime/cohort plan, the durable baseline, and an evidence adapter exposing `validate`, `normalized_summary`, `normalized_outcome`, and `full_readback`. The monitor does not redefine the scientific sampler or grader.

Run `monitor.py --config <signed-config> --authority <public-authority>` from this directory. The supplied local and remote roots must be private, canonical operator directories. No credentials, signed capabilities, node addresses, or deployment configuration are included here. The scientific program and evidence adapter must be supplied and pinned by the deployment.

The configured initial step and step gap select the latest eligible checkpoint, coalescing backlog. A durable dispatch intent and an inherited kernel lease prevent concurrent evaluations or replay of an ambiguous original dispatch. Completed output is authenticated and archived with full readback before an independently signed cache-retirement action.

A failed or abandoned attempt keeps its original records. Once the worker lease is free, bounded retries use distinct signed assignments and fresh read capabilities. Exhaustion records a failed diagnostic and permits later checkpoints; it does not create a score. Failed download cleanup requires the signed original assignment/read plan and a durable failed-attempt archive, preserves evidence and the baseline, and checks serial/evaluation/hydration leases. Interrupted deletions resume only under the same authorization.

Results from a different cohort, token budget, or sampler belong in a separate series. This operator does not alter the website or merge its metrics into an existing chart.
