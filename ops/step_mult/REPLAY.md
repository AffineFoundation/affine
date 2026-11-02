# Offline replay of step_mult (RA/20261102-step-mult)

For past GREEN turns H -> C with published fp32 masters, candidate(m) = bf16_rne(M_H + m (M_C - M_H)). This equals training that turn with
step_mult m exactly: one gradient, the step is linear in m and the global cap (1.0) never fires (default step norm about 0.023).
m = 0 rebuilds the head and m = 1 the published candidate bit-exactly (bf16_check). Each model is scored on the same fixed 1,897-task subset
of the private holdout (8192-token harness, 1 rollout per task, fixed seed); deltas are paired per task vs the head, in percentage points.

```
epoch     head%       m=1 dpp(se)       m=2 dpp(se)       m=4 dpp(se)       m=8 dpp(se)
401       88.19      +0.30 (0.31)      +0.38 (0.32)      +0.66 (0.33)                 -
413       88.98      +0.30 (0.30)      +0.58 (0.31)      +1.25 (0.31)                 -
419       89.77      +0.00 (0.30)      +0.30 (0.30)      -0.42 (0.32)                 -
425       89.77      +0.14 (0.30)      +0.00 (0.29)      -0.50 (0.31)                 -
432       90.14      -0.06 (0.30)      +0.24 (0.29)      -0.08 (0.32)                 -
448       89.98      -0.14 (0.29)      +0.16 (0.30)      -1.05 (0.33)                 -
458       89.62      +0.42 (0.31)      +0.70 (0.31)      +0.74 (0.29)                 -
463       90.72      +0.08 (0.29)      +0.24 (0.30)      -0.40 (0.33)                 -
481       90.93      +0.22 (0.28)      -0.22 (0.30)                 -                 -
526           -                 -                 -                 -                 -
m1 {"files": 9, "mean_delta_pp_full": 0.14067524115755625, "se_pp": 0.099345649777652, "z": 1.4160181293534748, "wins": 6}
m2 {"files": 9, "mean_delta_pp_full": 0.265719899964273, "se_pp": 0.10135775450676784, "z": 2.6216040524707003, "wins": 7}
m4 {"files": 8, "mean_delta_pp_full": 0.025120578778135064, "se_pp": 0.11255391689310765, "z": 0.22318706866498528, "wins": 3}
```

Decision (Jacob 11:48 CR: ship m in {1,2,4} unless m > 1 clearly hurts): m = 2 is above the head on almost every turn and its mean
gain is about 2x that of m = 1; m = 4 lost 1.05pp (3.2 SE) on e448, three more turns were negative and its mean gain is not above
m = 1, so only step_mult {1, 2} is accepted. m = 8 was not replayed (not shipping above 2).
