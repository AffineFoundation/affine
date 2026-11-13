"""Host settings for the tests: scratch directories, and a Lium CLI and an ssh wrapper that do not exist."""
import os, tempfile

_tmp = tempfile.mkdtemp(prefix="affine-bench-test-")
os.makedirs(_tmp + "/site")
os.environ.update(BENCH_STATE=_tmp + "/state", BENCH_SITE=_tmp + "/site", BENCH_NODE_SSH=_tmp + "/no-ssh", BENCH_LIUM=_tmp + "/no-lium",
                  BENCH_MAX_PRICE="1", BENCH_REPO_REF="0" * 40, BENCH_STORE="pathlib:Path.cwd", BENCH_JOURNAL=_tmp + "/no-journal.jsonl")
