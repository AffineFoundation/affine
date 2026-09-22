"""Stage 3: render every accepted spec into a Harbor task directory.

Reads `<out>/e<epoch>/specs.jsonl`, writes `<out>/e<epoch>/tasks/<task_id>/`:

    task.toml                 schema 1.1; [environment].docker_image = affine/terminal-gen:<id>
    instruction.md
    environment/Dockerfile    + environment/_fixtures/<path>
    tests/test.sh             tmax's convention: pytest -> /logs/verifier/reward.txt (1|0)
    tests/test_final_state.py
    solution/solve.sh         reference solution (validate.py runs it; the agent never sees it)

The dir set is exactly what `affine_tmax_v1` / Harbor's `parse_task` reads,
so the taskset side needs nothing new. `[metadata]` carries the provenance
(site, qid, url, CC BY-SA 4.0 attribution) that the licence requires.

    python harbor.py --epoch 63 --out ~/terminal_gen/out
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

from common import (GENERATOR_VERSION, IMAGE_PREFIX, LICENSE, LICENSE_URL,
                    TASK_NAME_PREFIX, out_dir, read_jsonl, task_id, write_jsonl)

log = logging.getLogger("terminal_gen.harbor")

TEST_SH = """#!/bin/bash
set -e

mkdir -p /logs/verifier

cd /tests
python3 -m pytest test_final_state.py -v 2>&1 | tee /logs/verifier/test-stdout.txt
TEST_EXIT=${PIPESTATUS[0]}

if [ $TEST_EXIT -eq 0 ]; then
    echo 1 > /logs/verifier/reward.txt
else
    echo 0 > /logs/verifier/reward.txt
fi

exit 0
"""

# The image must carry pytest for tests/test.sh; the model's Dockerfile does
# not know that, so it is appended after the model's own lines.
PYTEST_LAYER = """
# --- terminal_gen: verifier dependency (appended by ops/terminal_gen/harbor.py)
RUN if command -v apt-get >/dev/null 2>&1; then \\
      apt-get update && apt-get install -y --no-install-recommends python3 python3-pytest && rm -rf /var/lib/apt/lists/*; \\
    else python3 -m pip install --no-cache-dir pytest; fi
WORKDIR /app
"""


def _toml_str(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def render_task_toml(tid: str, post: dict, spec: dict, epoch: int) -> str:
    meta = spec["metadata"]
    cmds = ", ".join(_toml_str(c) for c in meta.get("commands", []))
    return f"""schema_version = "1.1"

[task]
name = {_toml_str(f"{TASK_NAME_PREFIX}/{tid}")}
description = {_toml_str(post["title"][:160])}

[metadata]
source = "stackexchange"
generator = {_toml_str(GENERATOR_VERSION)}
data_epoch = {epoch}
se_site = {_toml_str(post["site"])}
se_qid = {_toml_str(str(post["qid"]))}
se_url = {_toml_str(post["url"])}
license = {_toml_str(LICENSE)}
license_url = {_toml_str(LICENSE_URL)}
attribution = {_toml_str(f"Derived from a {post['site']} question and its accepted answer ({post['url']}), CC BY-SA 4.0.")}
domain = {_toml_str(meta["domain"])}
language = {_toml_str(meta["language"])}
difficulty = {_toml_str(meta["difficulty"])}
commands = [{cmds}]
synth_model = {_toml_str(spec.get("_model", ""))}

[agent]
timeout_sec = 600.0

[verifier]
timeout_sec = 120.0

[environment]
docker_image = {_toml_str(f"{IMAGE_PREFIX}:{tid}")}
cpus = 1
memory_mb = 2048
allow_internet = false
"""


def render_task(root: Path, tid: str, post: dict, spec: dict, epoch: int) -> Path:
    d = root / tid
    if d.exists():
        shutil.rmtree(d)
    (d / "environment").mkdir(parents=True)
    (d / "tests").mkdir()
    (d / "solution").mkdir()
    (d / "task.toml").write_text(render_task_toml(tid, post, spec, epoch))
    (d / "instruction.md").write_text(spec["instruction"].strip() + "\n")
    dockerfile = spec["dockerfile"].rstrip() + "\n" + PYTEST_LAYER
    (d / "environment" / "Dockerfile").write_text(dockerfile)
    for rel, content in spec["fixtures"].items():
        p = d / "environment" / "_fixtures" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
    (d / "tests" / "test.sh").write_text(TEST_SH)
    (d / "tests" / "test.sh").chmod(0o755)
    (d / "tests" / "test_final_state.py").write_text(spec["test_py"].rstrip() + "\n")
    solve = spec["solve_sh"].rstrip() + "\n"
    if not solve.startswith("#!"):
        solve = "#!/bin/bash\nset -euo pipefail\n" + solve
    (d / "solution" / "solve.sh").write_text(solve)
    (d / "solution" / "solve.sh").chmod(0o755)
    return d


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epoch", type=int, required=True)
    ap.add_argument("--out", default="~/terminal_gen/out")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = out_dir(args.out, args.epoch)
    tasks_root = out / "tasks"
    tasks_root.mkdir(exist_ok=True)
    rows = []
    n_ok = 0
    for row in read_jsonl(out / "specs.jsonl"):
        if "spec" not in row:
            continue
        post, spec = row["post"], row["spec"]
        spec["_model"] = row.get("model", "")
        tid = task_id(args.epoch, post["site"], post["qid"])
        render_task(tasks_root, tid, post, spec, args.epoch)
        rows.append({"task_id": tid, "site": post["site"], "qid": post["qid"], "url": post["url"],
                     "domain": spec["metadata"]["domain"], "difficulty": spec["metadata"]["difficulty"]})
        n_ok += 1
    write_jsonl(out / "rendered.jsonl", rows)
    log.info("rendered %d task dirs under %s", n_ok, tasks_root)


if __name__ == "__main__":
    main()
