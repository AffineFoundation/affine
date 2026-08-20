# R944 p4096 — teacher OOM → TP4 repair → v4 n80 LIVE

- Prior p4095 n80 pid55267 **DEAD**: teacher EngineDead mid-duel (CUDA OOM allocate 3.13 GiB / 2.52 GiB free at TP2/`gpu_mem=0.88` under concurrent k=3).
- King:8001 + chall:8002@0.65 stayed up; GPUs 0,1,5,6,7 free.
- Repair: FORCE seed teacher Triton from `/tmp/r926_triton_seed/teacher` (n_so=7) + relaunch **TP=4 on GPUs 0,1,5,6 @ gpu_mem=0.85**, `max_num_batched_tokens=4096`, `expandable_segments`.
- Teacher READY poll=76 · probe_ok · **v4 n80 LIVE** pid**58462** · bh=`fb2a9be1…` · hotkey `local-r944-reign36-wvk7-p4096` · corpus epoch13 ready · T/K/C health 200.
- Script: `experiments/fleet-rent/p4096_r926_teacher_tp4_r944_n80.sh`
- Decision rule unchanged: CROWN_OK iff margin > max(2·SE, 0.002) AND thought≥80 AND B≥0.30 vs reign36 (wvk=7 k=3 τ=0.03).
