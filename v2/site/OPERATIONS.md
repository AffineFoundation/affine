# The status site: operator steps

This file is for the operator of AffineValidator. It covers one change of the site builder: the benchmark table with one row for each king, the charts, and the source of the reference models. The cutover itself is in `v2/CUTOVER.md`; this file changes no step of it.

## What the change does

- **Benchmarks** (new place: under the live round). One row for the genesis head and one for each king. A king is the head that one applied round made. A row with no score says why: `queued`, `running`, `failed`, or `no run yet` (the benchmark file does not name the head).
- **Score cards** above the table: one card for the average and one for each benchmark, with the score over the kings in time. The genesis head is the first point.
- **Rounds in charts** (new section): best z per round, pass rate per round, batches per round, round duration. They show the closed rounds that `mining.json` lists (the newest 20).
- **Reference models.** The builder takes them from the frozen v1 history that is installed in the site (`v1-data.json`, the copy of `$V1FILE`). It does this on each build, whatever `public/benchmarks.json` holds. So:
  - No reference model needs a seed in the store's benchmark table.
  - The rows do not need v1's benchmark service, and they do not need v1's `benchmarks.json` at the root of the site.
  - They need one file: `$V1FILE` on AffineValidator. Keep it.
  - If the store's table has a row of another kind than `king` and `genesis` with the same label, the store's row is shown and the row of the v1 history is not.

The builder and these steps place no `public/benchmarks.json` in the site directory. Only the pull writes that file, as a copy of the store's object.

## Hosts

AffineValidator only. The GPU box and the benchmark host need nothing. There is no new unit, no new flag, no new credential and no new Python package. The pull's unit must have `--v1-history` in its `ExecStart` (step 12.2 of "AffineValidator" in `v2/CUTOVER.md` writes it so).

## Steps

Names as in "AffineValidator" of `v2/CUTOVER.md`: `REPO`, `SITE`, `V1FILE`, `PY`.

1. **First: look for a local change of the site in the checkout.** On 14 Nov 2026 the live `site-data.json` was not the output of the builder in `v2`: its Kings table was empty while `mining.json` listed 10 kings, and it had 22 reference rows with no benchmark file. So expect a local change of `v2/site/build.py`.

   ```
   git -C "$REPO" rev-parse HEAD
   git -C "$REPO" status --short
   git -C "$REPO" diff -- v2/site/
   grep -E '^ExecStart=' /etc/systemd/system/sn120-v2-site-pull.service
   ```

   - Right: `status` prints nothing and `diff` prints nothing. Go to step 2.
   - `status` or `diff` prints a line: a file of the checkout was changed on the host. Save the change, and discard nothing:

     ```
     git -C "$REPO" diff > "$HOME/sn120-v1-archive/local-change-$(date -u +%Y%m%dT%H%M%SZ).diff"
     git -C "$REPO" status --short > "$HOME/sn120-v1-archive/local-change-status.txt"
     ```

     Send both files and the commit of `rev-parse` to the coordinator, and wait for the answer. Do not run `git stash`, `git checkout -- <file>`, `git reset` or `git checkout -f` before it. The checkout of step 4 stops with `Your local changes to the following files would be overwritten by checkout` while a changed file is in its way; that is the wanted result.
   - A file with `??` in `status` under `v2/site/` is a file that git does not know. Copy it to `$HOME/sn120-v1-archive/` and name it to the coordinator too.
   - The line of the unit names `v2/site/pull.py` of `$REPO`, `--site-dir`, `--public-url` and `--v1-history`. If it names another script, or a second unit writes `site-data.json`, tell the coordinator. With no `--v1-history` the site has reference models only while `$SITE/v1-data.json` of an earlier run is there.

   What the site shows once this commit is in force, against the site of 14 Nov 2026:

   - **Kings** lists each king of `mining.json`, also a king that was carried over from v1 (10 rows on 14 Nov 2026, each with the link `v1 king`). The site of 14 Nov 2026 showed none of them. The code has no rule that hides them; if the empty table is wanted, that is a new rule for the coordinator, not a step for the host.
   - **Benchmarks** has the same 22 reference models with the same scores, and one more row `Genesis`. They are in the order of the v1 page: the largest model first. As on the v1 page, the 8 that lead for their size are in view; a link shows all 22.
   - The sections **Benchmarks** (now second) and **Rounds in charts** are new. Step 5 has the details.

2. **The frozen v1 history has the reference models.**

   ```
   "$PY" -c "import json, sys; d = json.load(open(sys.argv[1])); print(d['schema'], sum(r['kind'] == 'reference' for r in d['bench']['rows']))" "$V1FILE"
   ```

   Right: `sn120-v1-archive-1 22`. Another number is the number of reference models that the site will show.

3. **Look for a benchmark file that the pull did not write.**

   ```
   ls -l "$SITE/public/benchmarks.json"; grep -c '"public/benchmarks.json"' "$SITE/public/pull_index.json"
   ```

   Right, while the benchmark service has not written: `No such file or directory` and `0`. Right, after its first write: the file and `1`. A file with `0` is a file placed by hand. Leave it for now and do step 6 after the checkout.

4. **Check out the commit that the coordinator names.**

   ```
   cd "$REPO" && git fetch origin && git checkout <commit>
   grep -c '^def tried' v2/site/build.py
   ```

   Right: `1`. Nothing is restarted: the timer starts a new process each minute, and the next run builds the site with the new code.

5. **The next run of the pull.** Wait two minutes, or start one run: `sudo systemctl start sn120-v2-site-pull.service`.

   ```
   journalctl -u sn120-v2-site-pull.service -n 5 --no-pager
   curl -s https://affine.io/site-data.json | "$PY" -c "
   import collections, json, sys
   b = json.load(sys.stdin)['bench']
   print(sorted(collections.Counter((r['kind'], r['status']) for r in b['rows']).items()), len(b['columns']))"
   ```

   Right: the journal shows `failed steps 0; site built`. Before the first write of the benchmark service, with the history of step 2, the second command prints

   ```
   [(('genesis', 'none'), 1), (('reference', 'failed'), 5), (('reference', 'scored'), 17)] 11
   ```

   and one more `('king', 'none')` for each applied round that `mining.json` lists under `heads`. After the service's first write the genesis row and each king row have the state that the service gave (`queued`, `running`, `scored`, `failed`); the reference rows stay as they are.

   The page `https://affine.io` (reload once without the cache):

   - **Benchmarks** is the second section. Its cards say `No score yet.` until the first score.
   - The table ends with 8 reference models, the largest first (`Qwen3.5-9B` first, `DeepScaleR-1.5B-Preview` last), and the row `Genesis`. `Genesis` carries the label `no run yet` until the benchmark service names it. The link under the table says `Show all 22 reference models`.
   - With no applied round, the body of the table says `No king yet: no round has applied a winner.` After an applied round there is a row `<round id> <first 8 characters of the head>` with its state.
   - **Rounds in charts** has four tiles. With no closed round each says so; with one closed round each shows one bar or point.
   - **Kings** lists the kings of `mining.json`. A king imported from v1 has the link `v1 king`.
   - `https://affine.io/v1.html` has the same table for v1: the kings of v1 with a score, and the reference models.

6. **Only if step 3 found a file placed by hand.** Move it out of the site directory, after step 5 is right:

   ```
   mv "$SITE/public/benchmarks.json" "$HOME/sn120-v1-archive/benchmarks-placed-by-hand.json"
   ```

   Then repeat the checks of step 5. While that file is there, the pull does not copy the store's benchmark table: its run exits 1 with the name of the file.

## The way back

```
cd "$REPO" && git checkout <the commit before>
```

The next run of the pull builds the pages and `site-data.json` of that commit. The file set of the site is the same in both commits, so nothing is left behind. No state changes: the change is in the builder only.

## What the log can say

Each line is on stderr of the pull (the journal of `sn120-v2-site-pull.service`). None of them fails the run.

| Line | Meaning | What to do |
|---|---|---|
| `site: no reference model from the v1 history: ...` | `$SITE/v1-data.json` is not a v1 archive, or its benchmark part cannot be read. The table has no reference model; the rest is built. | Do step 2. The pull copies `$V1FILE` again on each run. |
| `site: no benchmark file, so no benchmark score: ...` | `public/benchmarks.json` is there and cannot be used (not JSON, or a field is missing). The table shows the heads with `no run yet`, and the reference models. A missing file prints nothing in the pull. | Tell the coordinator; the benchmark service wrote the file. |
| `pull: FAILED v1 history: ...` | `$V1FILE` cannot be read or is not a v1 archive. The copy in the site stays. | Do step 2. |

## Limits

- `mining.json` lists the newest 20 heads and the newest 20 closed rounds. A king that the benchmark file does not name gets its `no run yet` row only while it is among these heads. A king that the file names keeps its row.
- The benchmark service makes rows only for folded heads. With the live value `fold_every_k = 1` each king is folded.
- The scores of the reference models are those of v1's benchmark service at the time of the archive. Nothing updates them.
