"""publish: an errored rollout counts as a wrong sample in the pooled score, and rows that lost samples are flagged incomplete."""
import json, os, pathlib, runpy
import pytest
from affine_bench import config as C, publish, registry as R

HERE = pathlib.Path(__file__).resolve().parent
BENCH_VIEW = runpy.run_path(str(HERE.parents[1] / "site" / "build.py"))["bench_view"]
PP0, PP15 = "qwen3.5-family-pp0", "qwen3.5-family"
KING, GENESIS = "king-10757e879efe5b91", "genesis-87de8304e9d33dde"  # row 876 and the launch model, both scored at PP15
RUNNABLE = [b["id"] for b in R.benchmarks(enabled_only=True, runnable_only=True)]
AIME = R.benchmark("aime")["pooled_from"]
LIVE = json.loads((HERE / "live_rows_876_and_genesis.json").read_text())
PROBLEMS = {r["benchmark_id"]: r["n"] for r in LIVE["results"]}  # the count of problems of each benchmark, from the live rows


def test_pooled_counts_errored_rollouts_as_wrong(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "STATE", str(tmp_path))
    os.makedirs(tmp_path / "private" / "m")
    # part a: 2 tasks x 4 rollouts, task 1 lost 2 rollouts (n=2) but both of its valid ones were right
    (tmp_path / "private" / "m" / "a.per_task.jsonl").write_text(
        json.dumps(dict(idx=0, n=4, correct=4.0, errors=0)) + "\n" + json.dumps(dict(idx=1, n=2, correct=2.0, errors=2)) + "\n")
    (tmp_path / "private" / "m" / "b.per_task.jsonl").write_text(json.dumps(dict(idx=0, n=4, correct=0.0, errors=0)) + "\n")
    results = {("m", "a"): dict(rollouts=4, tasks=2, tasks_total=2, samples=6, cap_rate=0.0, finished_at=1),
               ("m", "b"): dict(rollouts=4, tasks=1, tasks_total=1, samples=4, cap_rate=0.0, finished_at=1)}
    full = {"a": 2, "b": 1}
    p = publish.pooled("m", dict(pooled_from=["a", "b"]), results, full)
    assert abs(p["score"] - (1.0 + 0.5 + 0.0) / 3) < 1e-9  # dropping the errors would give (1 + 1 + 0) / 3
    assert p["samples_expected"] == 12 and p["errors"] == 2 and p["complete"] is False


def test_expected_uses_the_full_task_count():
    assert publish.expected(dict(tasks=393, tasks_total=393, rollouts=1), "math500", {"math500": 500}) == 500
    assert publish.expected(dict(tasks=30, tasks_total=30, rollouts=16), "aime22", {"aime22": 30}) == 480


# ------------------------------------------------------------------------- one model with results at two profiles
@pytest.fixture
def live(tmp_path, monkeypatch):
    """The public table as it was read on 2026-11-14, with the rows of UID 876 and of the launch model only, as the
    table of this host. No result file is on the host."""
    monkeypatch.setattr(C, "STATE", str(tmp_path)); monkeypatch.setattr(publish, "OUT", str(tmp_path / "benchmarks.json"))
    doc = json.loads(json.dumps(LIVE)); json.dump(doc, open(publish.OUT, "w"))
    return doc


def table(): return json.load(open(publish.OUT))
def of(doc, mid, name="results"): return [r for r in doc[name] if r["model_id"] == mid]
def view(mid): return next(r for r in BENCH_VIEW(table())["rows"] if r["head"] == next(m for m in table()["models"] if m["model_id"] == mid)["head_version_id"])
def shown(mid): return {r["benchmark_id"]: r for r in of(table(), mid)}  # the last row of each benchmark, as the site takes it


def put(mid, bid, profile, score, at):
    """A result file on the host of a run of one benchmark, as a worker stores it, with its per-problem lines."""
    b = R.benchmark(bid); n = PROBLEMS[bid]
    os.makedirs(f"{C.STATE}/results/{mid}", exist_ok=True); os.makedirs(f"{C.STATE}/private/{mid}", exist_ok=True)
    json.dump(dict(model_id=mid, model_kind=mid.split("-")[0], benchmark=bid, bench_version=R.bench_version(b), score=score, se=0.01, tasks=n, tasks_total=n,
                   samples=n * b["rollouts"], rollouts=b["rollouts"], cap_rate=0.0, budget=65536, profile=profile, mean_out_tokens=1000.0, finished_at=at,
                   wall_seconds=60.0, node=dict(gpu="8xH100")), open(f"{C.STATE}/results/{mid}/{bid}.json", "w"))
    with open(f"{C.STATE}/private/{mid}/{bid}.per_task.jsonl", "w") as f:
        f.write("".join(json.dumps(dict(idx=i, n=b["rollouts"], correct=score * b["rollouts"], errors=0)) + "\n" for i in range(n)))


@pytest.mark.parametrize("mid", [KING, GENESIS])
def test_a_second_result_of_the_same_version_at_another_profile_leaves_the_first_in_the_file(live, mid):
    """Row 876 at 0 beside its results at 1.5, and the launch model at 0 beside its results at 1.5."""
    before = of(live, mid); models = live["models"]; publish.publish(models, {})
    assert of(table(), mid) == before and table()["results_prior"] == live["results_prior"]  # the table of now, written again, is the same
    mean = view(mid)["avg"]; first = next(r for r in before if r["benchmark_id"] == "gsm8k")
    assert round(mean, 4) == {KING: 0.7661, GENESIS: 0.7226}[mid]  # the means of the public table of that day
    put(mid, "gsm8k", PP0, 0.5, first["run_at"] + 86400); publish.publish(models, {})
    after = of(table(), mid); second = [r for r in after if r not in before]
    assert all(r in after for r in before) and len(after) == len(before) + 1  # no published result has left the file
    assert [(r["benchmark_id"], r["env_version"], r["profile"], r["score"]) for r in second] == [("gsm8k", first["env_version"], PP0, 0.5)]
    assert shown(mid)["gsm8k"] == first and second[0] in table()["results_prior"]  # the row of the first profile is still the shown one
    assert {r["profile"] for r in shown(mid).values()} == {PP15} and view(mid)["avg"] == mean  # no mean over two profiles
    assert next(m for m in table()["models"] if m["model_id"] == mid)["profile"] == PP15
    signature = publish.signature(); publish.publish(models, {})  # the same result once more: no copy
    assert of(table(), mid) == after and publish.signature() == signature


def test_a_second_run_at_another_profile_is_shown_when_it_has_every_benchmark_and_the_first_stays_in_the_file(live):
    before = of(live, KING); models = live["models"]; publish.publish(models, {}); mean = view(KING)["avg"]; at = max(r["run_at"] for r in before) + 86400
    pooled = lambda: [(r["profile"], r in before) for r in of(table(), KING) if r["benchmark_id"] == "aime"]
    half = ["aime22", "amc23", "gsm8k", "math500"]
    for i, bid in enumerate(half):  # the second run is at work, or it stopped halfway: the table shows the first run alone
        put(KING, bid, PP0, 0.5, at + i); publish.publish(models, {})
        assert {r["profile"] for r in shown(KING).values()} == {PP15} and view(KING)["avg"] == mean and all(r in of(table(), KING) for r in before)
    assert len(of(table(), KING)) == len(before) + len(half) and pooled() == [(PP15, True)]  # no pooled value over parts of two profiles
    rest = [bid for bid in RUNNABLE if bid not in half]
    for i, bid in enumerate(rest[:-1]):
        put(KING, bid, PP0, 0.5, at + 10 + i)
    publish.publish(models, {})
    assert {r["profile"] for r in shown(KING).values()} == {PP15} and view(KING)["avg"] == mean  # one benchmark is not there yet
    put(KING, rest[-1], PP0, 0.5, at + 30); publish.publish(models, {})
    now = shown(KING)  # the second run has each benchmark: it is the shown one, with its own pooled value and its own mean
    assert {r["profile"] for r in now.values()} == {PP0} and set(now) == set(RUNNABLE) | {"aime"} and abs(view(KING)["avg"] - 0.5) < 1e-9
    assert all(r in of(table(), KING) and r in table()["results_prior"] for r in before) and pooled() == [(PP15, True), (PP0, False)]
    assert next(m for m in table()["models"] if m["model_id"] == KING)["profile"] == PP0
    assert of(table(), GENESIS) == of(live, GENESIS)  # the other row is as it was
    after = table()["results"]; signature = publish.signature(); publish.publish(models, {})
    assert table()["results"] == after and publish.signature() == signature


def test_the_pooled_value_is_never_over_parts_of_two_profiles(tmp_path, monkeypatch):
    monkeypatch.setattr(C, "STATE", str(tmp_path))
    for i, bid in enumerate(AIME):
        put("m", bid, PP0 if i == 0 else PP15, 0.5, 1)
    results = {("m", bid): json.load(open(f"{tmp_path}/results/m/{bid}.json")) for bid in AIME}
    assert publish.pooled("m", R.benchmark("aime"), results, {}) is None
    put("m", AIME[0], PP15, 0.5, 2); results[("m", AIME[0])] = json.load(open(f"{tmp_path}/results/m/{AIME[0]}.json"))
    assert publish.pooled("m", R.benchmark("aime"), results, {})["profile"] == PP15


def test_a_benchmark_that_a_row_has_at_the_second_profile_only_is_shown_at_that_profile(live):
    """What the publisher alone does not stop: the row then has cells of two profiles, and the site's mean is over both
    (bench_view of v2/site/build.py does not look at the profile of a cell)."""
    live["results"] = [r for r in live["results"] if (r["model_id"], r["benchmark_id"]) != (KING, "arxivmath")]
    json.dump(live, open(publish.OUT, "w")); before = of(live, KING)
    put(KING, "arxivmath", PP0, 0.5, max(r["run_at"] for r in before) + 86400); publish.publish(live["models"], {})
    assert {bid: r["profile"] for bid, r in shown(KING).items()} == {bid: PP15 for bid in RUNNABLE + ["aime"]} | {"arxivmath": PP0}
    assert all(r in of(table(), KING) for r in before) and next(m for m in table()["models"] if m["model_id"] == KING)["profile"] == PP15
    assert view(KING)["avg"] is not None
