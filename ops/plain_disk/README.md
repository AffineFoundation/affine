# Plain-disk moves (SN120, 2026-10-30)

Rule: [STANDING-RULES.md](../../STANDING-RULES.md) - nothing on SN120 uses the encrypted disk (Lium `/root` on `fuse.gocryptfs`).

Each hot `/root/...` path is served from a byte-identical copy on plain local disk (`/opt/plain/<pod>/...`, overlay `/` on the host NVMe)
through a symlink at the **same path**, so code, configs and fingerprints are unchanged. The original stays as `<path>.gocryptfs-orig`.

| file | what |
|---|---|
| `activate_pd.sh` | host side: `[MODE=install\|revert\|verify\|status] bash activate_pd.sh <probe\|eval\|trainer> <item>`; auto-reverts on sha256 mismatch |
| `pd_pod.sh` | pod side: final `rsync -aH --delete`, `mv P P.gocryptfs-orig`, `ln -s N P`, backfill, sha256 identity; refuses while a process has a cwd / open file / mapping under the path; `merge` mode moves a whole directory whose hot subtrees are already plain (symlinks into the target are kept, never copied); `ensure` mode for new pods |
| `pdman.py` | sha256 identity of original vs copy over files present in both (skips `*.partial` / tmp), reports pruned files |
| `items.conf` | item -> paths (`MODE_<pod>_<item>=merge`, `SKIP_<pod>_<item>=<regex>` for files live processes still append to) |
| `watchdog_pd.py` | auto-revert on epoch error / fail-closed / salvage / >5 indeterminate / serve_s>320 / EngineDead or FileNotFound in serve logs / trainer restore failure / evaluator worker gone |
| `ensure_plain_disk.sh` | **new pod setup default**: creates/moves the standard `/root` caches on plain disk and writes an env file pointing HF/Triton/Inductor/vLLM caches and TMPDIR at plain disk |
| `delete_orig.sh` | pod side: deletes `*.gocryptfs-orig` copies only when the live path is already a symlink to plain disk and nothing holds the orig (cwd/fd/mmap); prints DELETED.tsv rows |
| `pod.sh`, `podin.sh` | ssh helpers (strict host keys from the lium known_hosts) |

Safety: ckpt (`*/ckpt`) switches refuse while `gate_delta.py`/`gate_fetch.py` run; do not switch serving caches/ckpt while the gate is
fetching, serving or running a candidate (see 13:10 CR incident in the run log). Revert one item: `MODE=revert bash activate_pd.sh <pod> <item>`.

After the e276..e278 watchdog window closed clean (14:49 CR 2026-10-30) every `*.gocryptfs-orig` on the probe and trainer was deleted
(log: `state/root-audits/20261030-disk-cleanup/DELETED.tsv`), so `MODE=revert` no longer works for those items: a rollback now means a
fresh copy back, not a rename. Items moved after that (pd6..pd8) keep their `*.gocryptfs-orig` until it is unheld and deleted.
