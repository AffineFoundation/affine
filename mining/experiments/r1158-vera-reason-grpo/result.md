# R1158 — result notes
- p4279: rented `mine-r1158-vera-reason-grpo-1` (`brave-matrix-2a`) **8×B200** @$5.60/h TTL→2026-08-22T07:45Z.
- p4280: bootstrap LIVE — pip/king/teacher DL ✓ but **nvidia-smi=1** (Device Minor 6).
- p4281: teacher DL done, still 1 GPU → tore brave-matrix; blind re-rent returned **same** exec **fbb1135f-cffe-4962-9389-150ec0e0852b**@192.9.163.79 (`eager-lion-45`, API `gpu_count=1`) → tore again. Lesson: **blind `lium up --gpu` ignores executor_blacklist**. Waiter now rents by **node id** + verifies **ngpu≥8** (ssh+API) before bootstrap. Fleet QUEUE HEAD=R1158. Slot open until real 8× stock.
