# Why 310/270 (0 groups) and 396/108 were paid, and full-9 miners were not  (2026-10-22, CR = UTC-6)
Live writer before seq7 = weights_v9.py seq5 always-pay + seq6 share-EMA (LIVE-CONFIG seq3).
1. Eligibility came from the scorer's 4-epoch WINDOW (window_gate B >= 3 units over E-3..E), not from the newest epoch.
   310/270 sent 9 groups in e76-e79 and 0 in e80-e84: at E80/E81/E82 they still had B = 9/6/3 and z = 1.32-1.85
   (top-10 positive z), so they were in the softmax pool. Paid 22:00-00:00 CR on 10-21: 310 2.2%/3.3%, 270 1.6%/1.3%.
2. Pool = top 10 by window Stouffer z, softmax(z). On honest data z is noise (no stable per-miner signal), so the pool is a lottery
   that favours units with a few high-p hits: 396 (6,6,0,3 groups) had z 1.6-3.4 on stale e76-e77 hits; 108 had z 2.0-3.8 on e76.
   Volume does not enter z at all, so 9-group miners outside the top 10 got 0 (18-60 full-9 miners unpaid per epoch).
3. The share-EMA (half-life 4, lookback 12, min share 1%) then carried past winners for ~12 epochs (paid set 18-28 UIDs),
   requiring only "eligible in newest window", which (1) makes nearly automatic. New entrants needed to win the lottery AND
   accumulate >= 1% EMA share -> only a handful entered. Coldkeys with many UIDs got more lottery tickets: top-2 coldkeys
   held 92% (E80) -> 73% (E82) of weight.
4. Scorer bug: Null.p returned exactly 1.0 for a unit below every calibration group -> norm.isf(1) = -inf -> JSON encode
   error -> e85 and e87 were never scored (retried every 30 min, same failure). Their groups earned nothing, and the
   windows for e86/e88 were 3 and 2 epochs.
5. The vector stuck at 502bdbddd8 from 05:00 to 10:00 CR because no new score was installed between e86 (04:46 CR) and e88 (10:25 CR):
   the trainer/scorer pod migration plus the e87 failure.
Answers: post-commit seed LIVE (seed_block = first finalized block with ts >= deadline, +10; seed = sha256(rule:epoch_id:block_hash); check_seed enforced).
Fixed 3-group units LIVE in the scorer (unit = slot // 3), but partial units (1-2 groups) are scored and counted as units
(B), and the miner protocol does not enforce exact 3-group batches. seq5 always-pay = gate/alarms diagnostic only; pay softmax(z/T=1) over
top-10 positive-z miners with B >= 3 units in the window (fallback top 3), 100% to miners, burn only on stale/invalid score.
