"""Stage 5: decontaminate the validated tasks against the terminal benchmarks.

Every validated task's `instruction.md` is compared with the task text of the
held-out terminal benchmarks (Terminal-Bench 2.0 AND Terminal-Bench 4.0 by
default) and with the tmax pool (a D source, so an overlap there is a
duplicate rather than a leak, still dropped). A task is dropped when

  * it shares any word 16-gram with a benchmark instruction (MiMo's rule,
    also the fold's rule for D vs the benchsuite),
  * its Stack Exchange URL or question id appears in a benchmark task's text
    or metadata (benchmarks cite the post they were built from),
  * its task name collides with a benchmark task name.

Benchmarks are Harbor task directories (`<task>/instruction.md` +
`task.toml`), given as `--against <name>=<dir>` or fetched through the Harbor
CLI with `--harbor-dataset terminal-bench@2.0` (cached under
`~/.cache/affine/terminal_gen/benchmarks/`). The stage refuses to run
without at least one benchmark set: a task set published without this report
is not admissible.

Writes `<out>/e<epoch>/decontam.jsonl` (per task: ok, hits), `decontam.json`
(summary) and `decontam_report.md` (the human-readable report that ships
inside the data package next to the tasks).

    python decontam.py --epoch 63 --out ~/terminal_gen/out \
        --harbor-dataset terminal-bench@2.0 --harbor-dataset terminal-bench@4.0 \
        --against tmax=~/.cache/affine/prime-tasks/datasets/tmax
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import shutil
import subprocess
import tomllib
from collections import defaultdict
from pathlib import Path

from common import out_dir, read_jsonl, write_jsonl

log = logging.getLogger("terminal_gen.decontam")

NGRAM = 16
# The Harbor hub (2026-09-22) registers terminal-bench@2.0 and @2.1 (plus
# -pro / -science); "terminal-bench@4.0" does not resolve there. A default
# that cannot be fetched is skipped with a warning and recorded in
# decontam.json["missing_refs"] instead of aborting the run.
DEFAULT_HARBOR = ("terminal-bench@2.0", "terminal-bench@2.1", "terminal-bench@4.0")
CACHE_ROOT = Path("~/.cache/affine/terminal_gen/benchmarks").expanduser()
_WORD_RE = re.compile(r"[a-z0-9_./-]+")
_URL_RE = re.compile(r"https?://[^\s)\]>\"']+")


def words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def shingles(text: str, n: int = NGRAM) -> set[str]:
    w = words(text)
    return {" ".join(w[i:i + n]) for i in range(0, max(0, len(w) - n + 1))}


def harbor_bin() -> str | None:
    for cand in (shutil.which("harbor"), str(Path.home() / "benchsuite" / "harborenv" / "bin" / "harbor"),
                 "/root/harborenv/bin/harbor"):
        if cand and Path(cand).exists():
            return cand
    return None


def fetch_harbor_dataset(name: str) -> Path:
    """Download a Harbor dataset's task dirs once; return the directory."""
    dest = CACHE_ROOT / name.replace("@", "-").replace("/", "_")
    if dest.is_dir() and any(dest.rglob("instruction.md")):
        return dest
    hb = harbor_bin()
    if hb is None:
        raise SystemExit(f"harbor CLI not found and {name} is not cached at {dest}; "
                         f"pass --against <name>=<dir> with a checkout of the task dirs")
    dest.mkdir(parents=True, exist_ok=True)
    attempts = (
        [hb, "datasets", "download", name, "--output-dir", str(dest)],
        [hb, "datasets", "download", "-d", name, "-o", str(dest)],
        [hb, "dataset", "download", name, "-o", str(dest)],
    )
    last = ""
    for cmd in attempts:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        if p.returncode == 0 and any(dest.rglob("instruction.md")):
            log.info("fetched %s with: %s", name, " ".join(cmd))
            return dest
        last = (p.stdout + p.stderr)[-500:]
    raise SystemExit(f"could not download {name} with the harbor CLI ({last!r}); "
                     f"download it by hand and pass --against {name}={dest}")


def load_benchmark(name: str, root: Path) -> list[dict]:
    """One row per Harbor task dir: name, text (instruction + description), urls, qids."""
    rows = []
    for inst in sorted(root.rglob("instruction.md")):
        d = inst.parent
        text = inst.read_text(errors="ignore")
        tname = d.name
        meta_text = ""
        toml_path = d / "task.toml"
        if toml_path.exists():
            raw = toml_path.read_text(errors="ignore")
            meta_text = raw
            try:
                tname = tomllib.loads(raw).get("task", {}).get("name") or tname
            except Exception:  # noqa: BLE001
                pass
        blob = text + "\n" + meta_text
        rows.append({"bench": name, "name": str(tname), "short": d.name, "text": text,
                     "urls": set(_URL_RE.findall(blob)), "blob": blob.lower()})
    if not rows:
        raise SystemExit(f"benchmark {name}: no instruction.md under {root}")
    log.info("benchmark %s: %d tasks from %s", name, len(rows), root)
    return rows


def se_qid_of(url: str) -> str:
    m = re.search(r"/(?:questions|q|a)/(\d+)", url or "")
    return m.group(1) if m else ""


def check_task(task_dir: Path, benches: list[dict], index: dict[str, list[tuple[str, str]]]) -> dict:
    inst = (task_dir / "instruction.md").read_text(errors="ignore")
    toml = tomllib.loads((task_dir / "task.toml").read_text())
    meta = toml.get("metadata", {})
    se_url = str(meta.get("se_url") or "")
    se_qid = str(meta.get("se_qid") or "") or se_qid_of(se_url)
    hits: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for sh_ in shingles(inst):
        for bench, bname in index.get(sh_, ()):
            if (bench, bname) not in seen:
                seen.add((bench, bname))
                hits.append({"kind": "16gram", "bench": bench, "task": bname, "shingle": sh_})
    for b in benches:
        if se_url and se_url in b["blob"]:
            hits.append({"kind": "se_url", "bench": b["bench"], "task": b["name"]})
        elif se_qid and re.search(rf"stackexchange\.com/(?:questions|q|a)/{re.escape(se_qid)}\b", b["blob"]):
            hits.append({"kind": "se_qid", "bench": b["bench"], "task": b["name"]})
        if b["short"] == task_dir.name or b["name"] == toml.get("task", {}).get("name"):
            hits.append({"kind": "name", "bench": b["bench"], "task": b["name"]})
    return {"task_id": task_dir.name, "ok": not hits, "hits": hits}


def write_report(path: Path, epoch: int, benches_by_name: dict[str, int], rows: list[dict],
                 validated: int) -> None:
    kinds = defaultdict(int)
    per_bench = defaultdict(int)
    for r in rows:
        for h in r["hits"]:
            kinds[h["kind"]] += 1
            per_bench[h["bench"]] += 1
    ok = sum(r["ok"] for r in rows)
    lines = [
        f"# Decontamination report — affine_terminal_gen epoch {epoch}",
        "",
        f"Rule: drop a task if its `instruction.md` shares any word {NGRAM}-gram with a benchmark task's text, "
        "or its Stack Exchange URL / question id appears in a benchmark task, or its name collides.",
        "",
        "| benchmark set | tasks compared against | tasks hit |",
        "|---|---|---|",
    ]
    for name, n in benches_by_name.items():
        lines.append(f"| {name} | {n} | {per_bench.get(name, 0)} |")
    lines += [
        "",
        f"Validated tasks checked: {validated}. Clean: {ok}. Dropped: {len(rows) - ok} "
        f"(by kind: {dict(kinds) or 'none'}).",
        "",
    ]
    dropped = [r for r in rows if not r["ok"]]
    if dropped:
        lines.append("## Dropped tasks")
        for r in dropped:
            h = r["hits"][0]
            extra = f" — `{h['shingle']}`" if h.get("shingle") else ""
            lines.append(f"- `{r['task_id']}`: {h['kind']} vs {h['bench']} `{h['task']}`{extra}")
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--out", default="~/terminal_gen/out")
    ap.add_argument("--against", action="append", default=[], metavar="NAME=DIR",
                    help="a directory of Harbor task dirs to compare against (repeatable)")
    ap.add_argument("--harbor-dataset", action="append", default=None, metavar="NAME@VERSION",
                    help=f"fetch through the harbor CLI (repeatable; default {', '.join(DEFAULT_HARBOR)})")
    ap.add_argument("--no-default-harbor", action="store_true",
                    help="do not add the default Terminal-Bench 2.0 / 4.0 sets")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = out_dir(args.out, args.epoch)

    sets: dict[str, Path] = {}
    for spec in args.against:
        name, _, d = spec.partition("=")
        if not d:
            raise SystemExit(f"--against needs NAME=DIR, got {spec!r}")
        sets[name] = Path(d).expanduser()
    harbor_names = list(args.harbor_dataset or [])
    if not args.no_default_harbor:
        for n in DEFAULT_HARBOR:
            if n not in harbor_names and n not in sets:
                harbor_names.append(n)
    missing_refs: list[str] = []
    for n in harbor_names:
        if n not in sets:
            try:
                sets[n] = fetch_harbor_dataset(n)
            except SystemExit as e:
                log.warning("reference set %s unavailable — skipped: %s", n, str(e)[:200])
                missing_refs.append(n)
    if not sets:
        raise SystemExit("no benchmark set to check against")

    benches: list[dict] = []
    for name, root in sets.items():
        benches.extend(load_benchmark(name, root))
    index: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for b in benches:
        for sh_ in shingles(b["text"]):
            index[sh_].append((b["bench"], b["name"]))

    validated = [r["task_id"] for r in read_jsonl(out / "validation.jsonl") if r.get("ok")]
    if not validated:
        raise SystemExit("no validated tasks (run validate.py first)")
    rows = [check_task(out / "tasks" / tid, benches, index) for tid in validated]
    write_jsonl(out / "decontam.jsonl", rows)
    by_name = {name: sum(1 for b in benches if b["bench"] == name) for name in sets}
    summary = {"epoch": args.epoch, "ngram": NGRAM, "benchmarks": by_name, "checked": len(rows),
               "missing_refs": missing_refs,
               "clean": sum(r["ok"] for r in rows), "dropped": sum(not r["ok"] for r in rows)}
    (out / "decontam.json").write_text(json.dumps(summary, indent=1))
    write_report(out / "decontam_report.md", args.epoch, by_name, rows, len(validated))
    log.info("decontam: %s", json.dumps(summary))


if __name__ == "__main__":
    main()
