# KL research PROGRESS
Started 2026-10-21 22:06 CR.
- [ ] check lium access
- [x] Pod rented 2026-10-21 22:10 CR: affine-kl-main (huid gentle-orbit-02, id b06acfd9-7611-49aa-85bd-b8b69bd0fc49), 8xH100 secure US, $16.80/h, ssh root@143.110.230.129 -p 30063 (key ~/.ssh/affine_kl_key), auto-removal 2026-10-22 12:10 CR (ttl 14h). This is the ONLY pod we created.
- [x] Env on pod: /root/venv (vllm 0.32, torch 2.13 cu130) + cuda-compat-13 (source /root/kl/env.sh). HF_HOME=/workspace/hf.
- Models: teacher Qwen/Qwen3-8B (non-thinking), student Qwen/Qwen3-1.7B (instruct, non-thinking). Same tokenizer/chat template.
- [x] Prompts built (/root/kl/data/pool_prompts.json; 11 honest sources + 6 adversarial; heldout 80/source).
- [~] 22:21 CR: teacher pool generation running on 8 GPUs (gen_pool.py, logs /root/kl/logs/gen*.log) -> /root/kl/data/pool/shard*.pkl
- Code in /workspace/kl-research/code (mirrors /root/kl on pod).
Plan phase 1 arms (1 GPU each): fixed, random, sel(main score), selraw(raw gap), full(bandit+online mutations via teacher server :8001), adv(bandit over honest+adv sources by naive uncapped gap). GPU0/1 eval workers (+teacher server on GPU1).
Phase 2: judge masks (judge_masks.py, Qwen3-8B as pinned judge) vs tconf vs none.
- 22:57 CR: resumed after interruption; pod affine-kl-main still running (only our pod). Pool gen DONE (78M tokens, teacher ~5000 tok/GPU-s). Eval worker GPU0, teacher server GPU1:8001, smoke test GPU2. Base held-out topkKL=0.49.
- 23:05 CR: PHASE 1 RUNNING (launched ~23:00 CR): 6 arms x 400 rounds x 64k tok/round on GPUs 2-7 (logs /root/kl/logs/train_<arm>.log, runs /root/kl/runs/<arm>/log.jsonl, ckpts /root/kl/ckpt/<arm>/rNNNN at 25,50,100,150,200,300,400). ~20s/round => ETA ~01:30 CR.
  Eval worker GPU0 (restart with /root/kl/restart_eval.sh [two]); r0 evals = base copy. Judge masks running GPU1 (logs/judge.log) alongside teacher server :8001 (used by 'full' arm).
  Sync to box: /workspace/kl-research/sync.sh -> sync/; analysis: python code/analyze.py.
  Base: gsm8k .838 math .682 mmlu .58 humaneval .70.
  Known bug fixed: os._exit left vLLM EngineCore orphans holding GPU memory -> eval_ckpt/judge now kill children.
- 23:55 CR: Found benchmark drops (MMLU parse: student stopped using \boxed, answers "**A**"; HumanEval took last code block). Patched eval_ckpt (robust MMLU parse, mmlu_lp logprob variant, HumanEval block choice, saves all outputs *.samples.json); requeued all evals. Verified teacher server logprobs == offline logprobs (mut_* high gaps are real, not artifact).
  Prelim: held-out KL 0.49 -> ~0.27-0.29 in all arms by ~6-8M tokens (plateau); GSM8K/MATH slightly down vs base in all arms.
- 23:47 CR (05:47 UTC pod): all running. fixed r280 (11s/rnd), random r210, sel/selraw r140 (~21s/rnd, ETA ~01:20 CR), full r170, adv r120. Judge 69% (ETA ~00:05 CR). Eval queue ~10. Skipped evals for some r0025/r0150 (marked .skipped).
  New-parser evals: base gsm8k .838 math .682 mmlu .595 he .701; teacher .931/.74/.779/.866. Arms @r25: small changes; MATH -4..-6 pts (more unboxed/truncated answers).
  NEXT: when fixed/random finish -> start eval workers on GPU2/3 (`source env.sh; nohup python eval_worker.py 2 0.85 ...`). When judge done -> merge judge_masks.0.pkl -> data/judge_masks.pkl, eval worker GPU1 (0.4). Phase 2 when sel arms finish.
- 00:07 CR: fixed arm DONE (400 rnd). Judge DONE: 60,832 masks (mean 22% tokens) -> /root/kl/data/judge_masks.pkl. Killed judge orphan engine. Eval workers now on GPU0 (0.85), GPU1 (0.4, shares w/ teacher server), GPU2 (0.85).
  Phase 2 launcher: /root/kl/launch_p2.sh NAME GPU [args] (sel arm, judged_only, 250 rnd). Planned: p2_none, p2_judge, p2_tconf, p2_judge_s(--mask_train 0), p2_tconf_s(--mask_train 0), p2_rand(random mask) -- start as phase-1 GPUs free.
- 00:30 CR: started p2_none on GPU2 (eval worker GPU2 stopped; workers on GPU0/GPU1 keep up).
- 00:36 CR: p2_watch.sh running on pod (auto-launches p2_judge/GPU3, p2_tconf/GPU4, p2_judge_s/GPU5, p2_tconf_s/GPU6, p2_rand/GPU7 when phase-1 arm on that GPU prints DONE). Don't relaunch manually; check logs/train_p2_*.log.
  Prelim phase1 held-out topk-KL at matched tokens (2/5/8/11/15M): fixed .300/.294/.291/.290/.289; random .285/.276/.270/.267/.263; sel .291/.279/.274/.271/.267; selraw .295/.286/.280/.277/.271; full .291/.281/.274/.270/.267; adv .301/.291/.282/.278/.276  => random >= sel per token.
- 01:00 CR: Phase-1 interim evals (new parser): base avg .704 (gsm .838 math .682 mmlu .595 he .701); teacher .829. fixed@400 .682, random@400 .674, sel@200 .670, selraw@200 .666, full@300 .667, adv@200 .669 => NO benchmark gains; KL -47%. MATH drops ~4-7 pts (more unboxed/rambling). Held-out KL: random best per token.
  Added train_arm args --sources, --src_cap. Downloaded Qwen3-1.7B-Base (eval queued as 'basemodel').
  PHASE 3 (extra variants) auto-launcher p3_watch.sh running: as each p2 arm finishes -> b_random, b_sel, b_fixed (Base student), s_divcap (src_cap .2), s_mc_sel, s_mc_rand (math/code only), 200 rounds each. Log logs/p3_watch.log.
- 00:37 CR (real box clock; earlier entries after 23:47 were labeled ~1h ahead): state OK. p2_watch + p3_watch alive. sel r290, selraw r300, full r350, adv r260, p2_none r50, p2_judge r20. Evals: basemodel + p2_none r50 running.
- 00:53 CR: full DONE -> p2_tconf_s launched GPU6. sel r340, selraw r360, adv r300, p2_none r100, p2_judge r70.
- 01:09 CR: selraw DONE -> p2_judge_s GPU5. sel r390, adv r330. r0_watch.sh auto-copies r0 evals.
- 01:19 CR: sel DONE -> p2_tconf GPU4. Killed teacher server (no longer needed).
- 01:27 CR: resumed; all watchers alive; adv r380, p2_none r200, p2_judge r170, p2_tconf_s r100, p2_judge_s r60, p2_tconf r40. Teacher server killed. Nothing relaunched.
- 01:28 CR: fixed stale evals (sel r50 requeued; selraw r25 dropped; phase-1 r0 recopied from new-parser base.json). analyze.py now writes matched tables (summary.json['matched']).
- 01:39 CR: PHASE 1 COMPLETE (all 6 arms 400 rnd). adv done -> p2_rand GPU7. All phase 2 arms running.
- PHASE 1 RESULT (final, 400 rounds, ~26M tok except fixed 17M): held-out topk-KL end: fixed .290, random .255, sel .257, selraw .258, full .258, adv .265 (base .492). GPU-h: fixed 1.6, random 2.5, sel 5.1, selraw 5.0, full 3.9, adv 5.6.
  Bench avg (gsm/math/mmlu/he) @400: base .704, fixed .682, random .674, sel .667, selraw .662, full .654, adv .661, teacher .829 => no arm improves benchmarks; all slightly below base. Per-domain KL table: /workspace/kl-research/phase1_domain_kl.md (most base KL is chat/writing/science style; selection arms reduce those slightly more, math/code slightly less than random).
- 01:50 CR: p2_none DONE -> b_random (Qwen3-1.7B-Base student) on GPU2. Base-model eval: gsm .783 math .498 mmlu .535 he .311 (avg .532), held-out KL .597 => headroom for phase 3.
- 02:00 CR: p2_judge DONE -> b_sel GPU3. b_random r80.
- 02:17 CR: resumed. p2_watch finished (all p2 launched); p3_watch+r0_watch alive. p2_tconf_s r240, p2_judge_s r210, p2_tconf r180, p2_rand r130; b_random r150, b_sel r70. Hard stop 07:30 CR.
- 02:28 CR: p2_tconf_s DONE; b_random DONE; b_fixed GPU6, s_divcap GPU2 launched. Pending: s_mc_sel, s_mc_rand (auto).
- 02:49 CR: phase 2 all DONE except p2_rand (~r230). Phase 3 all launched (s_mc_rand GPU4). p4_watch.sh: seed replicates random_s1 (GPU7 after p2_rand) + sel_s1 (GPU3 after b_sel), 200 rnd, for noise estimate. ETA all training ~04:30 CR.
- 02:49 CR PHASE 2 RESULT (250 rnd, ~16.5M tok, judged pool): bench avg p2_none .658, p2_judge .653, p2_judge_s .663, p2_tconf .661, p2_tconf_s .665, p2_rand(r150) .656; held-out KL end: none .263, judge .284, judge_s .261, tconf .283, tconf_s .263. => no capability differences (all within noise, all < base .704). tconf mask shifts selected curriculum to science/code/mmlu/math (~50%) vs chat; judge mask leaves curriculum ~unchanged (chat-heavy).
- 03:00 CR: p2_rand, b_fixed, b_sel, b_random DONE. random_s1 GPU7, sel_s1 GPU3 running; s_divcap/s_mc_sel/s_mc_rand ~r90. Extra eval worker on GPU6.
- 03:06 CR: BUG found in base-student evals: Base generation_config eos=<|endoftext|> only, so vLLM didn't stop at <|im_end|> -> garbage/truncation. Fixed eval_ckpt (stop_token_ids=[151645,151643]); requeued basemodel + all b_* evals (incl r0). Instruct evals unaffected.
- 03:08 CR: resumed; s_divcap r120, s_mc_rand r120, s_mc_sel r110, random_s1 r50, sel_s1 r20 (ETA ~04:30). b_* re-evals in progress (3 workers GPU0/1/6).
- 03:08 CR: analyze.py now also makes phase3 charts (phase3_basestudent_*, phase3_variants_*). Next: wait for training ~04:30 + evals, then RESULTS.md, teardown.
- 03:29 CR: s_mc_rand DONE. s_divcap r180, s_mc_sel r180, random_s1 r150, sel_s1 r90 (ETA 04:10).
- 03:39 CR: s_divcap, s_mc_sel DONE. random_s1 r190, sel_s1 r120 (~04:10).
- 03:50 CR PHASE 3 RESULTS: Base student (Qwen3-1.7B-Base, avg .532): b_fixed@200 .666, b_random .654, b_sel .647 (most gain by r50 = chat format; sel costs 2.6x GPU-h). Variants (instruct): s_divcap .658, s_mc_sel .671, s_mc_rand .676. Seed noise: random vs random_s1 avg diff .006-.012. sel_s1 finishing (~04:10). Then: final analyze, RESULTS.md, teardown.
- 03:51 CR: RESULTS.md drafted with placeholders (TABLE_*, MATCHED1, SEL_S1_NOTE, CHARTS) to fill after sel_s1 + final sync.
- 04:06 CR: all training/evals DONE; final sync+analyze; RESULTS.md filled. Tearing down affine-kl-main.
- 04:06 CR: TORN DOWN affine-kl-main (gentle-orbit-02): 5.9h, ~$99.80 total. No affine-kl-* pods remain. DONE.
