"""Stage 1: pick Stack Exchange posts that can become terminal tasks.

Source: the Stack Exchange data dump as mirrored on the Hugging Face Hub
(`HuggingFaceH4/stack-exchange-preferences`, CC BY-SA 4.0 like the dump).
One row per question: `qid`, `question` (HTML), `answers` (list of
{text, pm_score, selected, ...}), `metadata` (list of URLs, whose host is
the site). Streamed, so nothing is downloaded whole.

A post is kept when
  * its site is in `--sites` (default: the five sysadmin / shell sites),
  * it has a selected answer whose body has a code block with a shell command,
  * the question body is <= `--max-chars`, the answer score >= `--min-score`.

Kept rows are shuffled with `--seed` (= the fold epoch) and the first `--n`
are written to `<out>/e<epoch>/posts.jsonl.gz` as
{site, qid, url, title, question, answer, score}. HTML is reduced to text
with code fences preserved.

    python posts.py --epoch 63 --n 300 --out ~/terminal_gen/out
"""

from __future__ import annotations

import argparse
import html
import logging
import random
import re
from urllib.parse import urlparse

from common import out_dir, write_jsonl

log = logging.getLogger("terminal_gen.posts")

HF_DATASET = "HuggingFaceH4/stack-exchange-preferences"
DEFAULT_SITES = ("unix.stackexchange.com", "askubuntu.com", "serverfault.com",
                 "superuser.com", "dba.stackexchange.com")

# A code block counts as "shell" when one of these appears at a line start.
SHELL_WORDS = (
    "awk", "sed", "grep", "find", "xargs", "sort", "uniq", "cut", "tr", "tar", "gzip",
    "rsync", "ssh", "scp", "chmod", "chown", "chattr", "ln", "mv", "cp", "rm", "mkdir",
    "mount", "umount", "df", "du", "ls", "cat", "tail", "head", "less", "wc", "diff",
    "patch", "git", "make", "gcc", "python", "python3", "pip", "bash", "sh", "zsh",
    "cron", "crontab", "systemctl", "service", "journalctl", "ps", "kill", "pkill",
    "top", "nohup", "screen", "tmux", "curl", "wget", "iptables", "nft", "ip", "ss",
    "netstat", "dig", "nslookup", "openssl", "gpg", "dd", "fdisk", "parted", "lsblk",
    "blkid", "mkfs", "fsck", "useradd", "usermod", "groupadd", "passwd", "sudo", "su",
    "env", "export", "alias", "for", "while", "if", "case", "echo", "printf", "date",
    "stat", "file", "which", "type", "apt", "apt-get", "dpkg", "yum", "dnf", "rpm",
    "psql", "mysql", "sqlite3", "mysqldump", "pg_dump", "jq", "perl", "ruby", "docker",
    "getfacl", "setfacl", "rename", "basename", "dirname", "readlink", "realpath",
    "split", "paste", "join", "comm", "tee", "yes", "seq", "shuf", "column", "nl",
    "fold", "expand", "column", "zip", "unzip", "7z", "xz", "bzip2", "logrotate",
)
_SHELL_RE = re.compile(
    r"(?m)^\s*(?:\$\s+|#\s+|sudo\s+)?(?:" + "|".join(re.escape(w) for w in SHELL_WORDS) + r")\b")
_CODE_RE = re.compile(r"<pre[^>]*>\s*<code[^>]*>(.*?)</code>\s*</pre>", re.S | re.I)
_INLINE_CODE_RE = re.compile(r"<code[^>]*>(.*?)</code>", re.S | re.I)
_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\n{3,}")


def html_to_text(body: str) -> str:
    """Reduce SE HTML to text; fenced blocks keep their code verbatim."""
    def fence(m: re.Match) -> str:
        code = html.unescape(_TAG_RE.sub("", m.group(1))).rstrip("\n")
        return f"\n```\n{code}\n```\n"
    text = _CODE_RE.sub(fence, body)
    text = _INLINE_CODE_RE.sub(lambda m: "`" + html.unescape(_TAG_RE.sub("", m.group(1))) + "`", text)
    text = re.sub(r"</?(p|div|br|li|h[1-6]|blockquote|ul|ol)[^>]*>", "\n", text, flags=re.I)
    text = html.unescape(_TAG_RE.sub("", text))
    return _WS_RE.sub("\n\n", text).strip()


def shell_blocks(body_html: str) -> list[str]:
    blocks = [html.unescape(_TAG_RE.sub("", m)) for m in _CODE_RE.findall(body_html)]
    return [b for b in blocks if _SHELL_RE.search(b)]


def site_of(metadata) -> tuple[str, str]:
    """(host, url) from the row's metadata URL list; ('', '') when absent."""
    urls = metadata if isinstance(metadata, list) else [metadata] if metadata else []
    for u in urls:
        if isinstance(u, str) and u.startswith("http"):
            host = urlparse(u).netloc.lower()
            if host.startswith("www."):
                host = host[4:]
            return host, u
    return "", ""


def selected_answer(answers) -> dict | None:
    if not isinstance(answers, list):
        return None
    chosen = [a for a in answers if isinstance(a, dict) and a.get("selected")]
    if not chosen:
        return None
    return max(chosen, key=lambda a: a.get("pm_score", 0) or 0)


def keep_row(row: dict, sites: set[str], min_score: int, max_chars: int) -> dict | None:
    host, url = site_of(row.get("metadata"))
    if host not in sites:
        return None
    ans = selected_answer(row.get("answers"))
    if not ans:
        return None
    score = int(ans.get("pm_score") or 0)
    if score < min_score:
        return None
    q_html = row.get("question") or ""
    a_html = ans.get("text") or ""
    if len(q_html) > max_chars or len(a_html) > max_chars:
        return None
    if not shell_blocks(a_html):
        return None
    q_text = html_to_text(q_html)
    if len(q_text) < 80:
        return None
    title = q_text.split("\n", 1)[0][:200]
    return {
        "site": host,
        "qid": str(row.get("qid")),
        "url": url,
        "title": title,
        "question": q_text,
        "answer": html_to_text(a_html),
        "score": score,
    }


def iter_hf_rows(dataset: str, split: str, sites=()):
    from datasets import load_dataset  # heavy import; only when streaming

    # The repo is laid out one directory per site (data/<site>/*.parquet) and
    # a whole-split stream walks data/Stackoverflow.com/ (335 shards) first:
    # the first run scanned 3,000,000 rows and matched 0 (2026-09-22). Stream
    # only the requested sites' shards.
    if sites:
        files = [f"data/{site}/*.parquet" for site in sorted(sites)]
        return load_dataset(dataset, data_files={split: files}, split=split, streaming=True)
    return load_dataset(dataset, split=split, streaming=True)


def collect(rows, sites: set[str], min_score: int, max_chars: int,
            scan_limit: int, want: int, seed: int) -> list[dict]:
    """Reservoir sample of `want` kept rows over at most `scan_limit` scanned rows."""
    rng = random.Random(seed)
    kept: list[dict] = []
    n_scanned = n_kept = 0
    for row in rows:
        n_scanned += 1
        if scan_limit and n_scanned > scan_limit:
            break
        rec = keep_row(row, sites, min_score, max_chars)
        if rec is None:
            continue
        n_kept += 1
        if len(kept) < want:
            kept.append(rec)
        else:
            j = rng.randrange(n_kept)
            if j < want:
                kept[j] = rec
        if n_scanned % 100_000 == 0:
            log.info("scanned %d rows, %d kept so far", n_scanned, n_kept)
    log.info("scanned %d rows, %d matched the filter, %d sampled", n_scanned, n_kept, len(kept))
    rng.shuffle(kept)
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epoch", type=int, required=True, help="fold epoch = sample seed")
    ap.add_argument("--n", type=int, default=300, help="posts to keep")
    ap.add_argument("--out", default="~/terminal_gen/out")
    ap.add_argument("--dataset", default=HF_DATASET)
    ap.add_argument("--split", default="train")
    ap.add_argument("--sites", default=",".join(DEFAULT_SITES))
    ap.add_argument("--min-score", type=int, default=3)
    ap.add_argument("--max-chars", type=int, default=6000)
    ap.add_argument("--scan-limit", type=int, default=3_000_000,
                    help="stop after this many streamed rows (0 = whole split)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    sites = {s.strip().lower() for s in args.sites.split(",") if s.strip()}
    rows = iter_hf_rows(args.dataset, args.split, sites)
    kept = collect(rows, sites, args.min_score, args.max_chars, args.scan_limit, args.n, args.epoch)
    path = out_dir(args.out, args.epoch) / "posts.jsonl.gz"
    n = write_jsonl(path, kept)
    by_site = {}
    for r in kept:
        by_site[r["site"]] = by_site.get(r["site"], 0) + 1
    log.info("wrote %d posts -> %s  %s", n, path, by_site)


if __name__ == "__main__":
    main()
