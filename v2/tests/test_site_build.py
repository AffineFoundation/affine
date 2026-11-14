"""The status site builder (``v2/site/build.py``), run on snapshots of the CPU simulation of the controller."""

from __future__ import annotations

import hashlib
import json
import re
import runpy
import shutil
import subprocess
import sys
from pathlib import Path

from sn120_v2 import layout, sim
from sn120_v2.types import King, V1Import

BUILD = Path(__file__).parents[1] / "site" / "build.py"
V1_ARCHIVE = BUILD.with_name("v1_archive.py")
URL = "https://data.example"


def snapshot(tmp_path: Path) -> dict:
    """One applied round (uid 4 wins, 3 and 7 lose, 6 has no file), then an open round with uid 8. 9 waits with no file."""
    v1 = V1Import("test", 1, (King(5, "hk-old", "v1-e41", "", 50),), (), ())
    world = sim.SimWorld(tmp_path / "world", sim.sim_params(), v1=v1)
    for uid, quality in ((3, 0.0), (4, 0.3), (6, None), (7, -0.2)):
        world.register(uid, quality)
    assert world.round().outcome == "applied"
    world.register(8, 0.0)
    world.register(9, None)
    world.controller.open_round(world.clock(), "entropy")
    return json.loads(json.dumps(world.publisher.snapshots[-1]))


def build(tmp_path: Path, snap: object, *more: str) -> tuple[subprocess.CompletedProcess, Path, str]:
    store, out = tmp_path / "store", tmp_path / "out"
    (store / layout.MINING_JSON).parent.mkdir(parents=True, exist_ok=True)
    (store / layout.MINING_JSON).write_text(snap if isinstance(snap, str) else json.dumps(snap))
    args = [sys.executable, str(BUILD), "--source", str(store), "--public-url", URL, "--out", str(out), *more]
    done = subprocess.run(args, capture_output=True, text=True)
    return done, out, (out / "site-data.json").read_text() if (out / "site-data.json").exists() else ""


def test_build_writes_a_fixed_file_set_with_versioned_asset_links(tmp_path):
    done, out, _ = build(tmp_path, snapshot(tmp_path))
    assert done.returncode == 0, done.stderr
    assert sorted(path.name for path in out.iterdir()) == sorted(
        ["index.html", "v1.html", "site.css", "site.js", "site-data.json", "favicon.svg", "network-haffer.ttf", "network-mono.ttf"])
    page = (out / "index.html").read_text() + (out / "site.css").read_text()
    for name in ("site.js", "site.css", "favicon.svg"):  # the v1 page gets the same stamped files
        assert re.search(rf"""["']{name}\?v=[0-9a-f]{{8}}["']""", (out / "v1.html").read_text())
    assert "{{" not in (out / "v1.html").read_text() and 'href="v1.html"' in page
    assert "{{" not in page
    stamps = dict(re.findall(r"""["']([\w.-]+\.(?:js|css|svg|ttf))\?v=([0-9a-f]{8})["']""", page))
    assert set(stamps) == {"site.js", "site.css", "favicon.svg", "network-haffer.ttf", "network-mono.ttf"}
    for name, stamp in stamps.items():  # the stamp is the hash of the file a browser gets
        assert hashlib.sha256((out / name).read_bytes()).hexdigest()[:8] == stamp
    assert not re.search(r"""(?:href|src)=["'][\w.-]+\.(?:js|css|svg|ttf)["']""", page)  # no unversioned asset link
    assert f'href="{URL}/{layout.LLMS_TXT}"' in page and f'href="{URL}/{layout.MINING_JSON}"' in page


def test_each_page_has_the_shell_of_the_benchmark_table_and_of_each_chart_that_the_script_fills(tmp_path):
    done, out, _ = build(tmp_path, snapshot(tmp_path))
    assert done.returncode == 0, done.stderr
    index, v1, script = (out / "index.html").read_text(), (out / "v1.html").read_text(), (out / "site.js").read_text()
    for page in (index, v1):  # the kings scroll in the body; the reference models and the genesis head are the footer
        bench = page[page.index('<section id="bench"'):]
        bench = bench[:bench.index("</section>")]
        assert '<div class="sparks"' in bench and "<thead></thead><tbody></tbody><tfoot></tfoot>" in bench and 'id="refs"' in bench
        assert "the reference models, the largest first" in bench and "leads for its size" in bench  # the order of the v1 page
    sizes = script[script.index("const SIZE = {"):script.index("const size = ")]  # by label in lower case, in billions of parameters
    table = dict(re.findall(r"'([^']+)': ([\d.]+)", sizes))
    assert len(table) == 22 and all(name == name.lower() and 1 < float(size) < 25 for name, size in table.items())
    tiles = re.findall(r'<div class="tile" id="([\w-]+)">', index)
    assert tiles == ["c-z", "c-pass", "c-batch", "c-time"] and index.index('id="bench"') < index.index('id="charts"') < index.index('id="kings"')
    assert all(f"tile('{name}'" in script for name in tiles) and 'class="tile"' not in v1
    # a table names its columns: each name must be a column of the script, and each table must be on a page
    columns = set(re.findall(r"^    (\w+): \{ h: ", script, re.M))
    tables = dict(re.findall(r"^    (\w+): \{ cols: \[([^\]]+)\]", script, re.M))
    assert set(tables) == {"kings", "queue", "history", "rounds", "heads", "v1kings", "v1turns"}
    for name, cols in tables.items():
        assert set(re.findall(r"'(\w+)'", cols)) <= columns, name
        assert f'<section id="{name}"' in index + v1


def test_only_an_applied_winner_gets_files_and_unknown_fields_never_reach_the_page(tmp_path):
    snap = snapshot(tmp_path)
    winner = next(entry for entry in snap["history"][0]["entries"] if entry["status"] == "winner")
    snap["history"][0]["entries"][0] |= {"uri": "private/somewhere", "train": {"lora_lr": 1}, "lora_hash": "f" * 64}
    snap["queue"]["next"][0]["coldkey"] = "ck-secret"
    done, _, text = build(tmp_path, snap)
    assert done.returncode == 0, done.stderr
    data = json.loads(text)
    rows = {row["uid"]: row for row in data["history"]}
    assert rows[4]["batch"] == f"{URL}/{layout.winner_batch('r000001')}" and rows[4]["lora"] == winner["lora_hash"]
    assert rows[3]["status"] == rows[7]["status"] == "lost" and rows[7]["z"] < 0 and len(rows[3]["sha"]) == 64
    assert all(rows[uid]["batch"] is None and rows[uid]["lora"] is None for uid in (3, 7))
    for private in ("private/", "pending/", "lora_lr", "f" * 64, "ck-secret", snap["params_hash"], snap["history"][0]["round_seed"]):
        assert private not in text
    assert [(king["uid"], king["imported"], king["batch"] is None) for king in data["kings"]] == [(4, False, False), (5, True, True)]
    assert data["kings"][0]["lora"] == winner["lora_hash"] and data["kings"][1]["file_url"] is None
    params = snap["params"]  # the bar and the caps come from the snapshot's Params
    assert (data["z_bar"], data["max_output_tokens"], data["round"]["cap"]) == (params["z_bar"], params["max_output_tokens"], params["batch_cap"])
    assert data["generated_at"] == snap["generated_at"] and data["phase"] == "RUNNING" and data["round"]["seats"] == 1
    assert [head["current"] for head in data["heads"]] == [True, False, False] and data["heads"][0]["manifest"]
    assert data["rounds"][0]["outcome"] == "applied" and data["rounds"][0]["winner_uid"] == 4 and data["rounds"][0]["skipped"] == 1
    # what the charts read: the open time, and the z and the pass rate of the applied winner
    closed, chart = snap["history"][0], data["rounds"][0]
    assert (chart["opened_at"], chart["at"]) == (closed["opened_at"], closed["closed_at"]) and chart["opened_at"] <= chart["at"]
    assert (chart["best_z"], chart["winner_pass"]) == (winner["z"], winner["pass_rate"]) and chart["null_pass"] == closed["null_pass_rate"]
    assert chart["with_z"] == 3 < chart["seats"] + chart["skipped"] and chart["reason"] == closed["reason"] != ""

    snap["history"][0]["outcome"] = "abandoned"  # a named winner whose files were not verified has no public files
    data = json.loads(build(tmp_path, snap)[2])
    assert data["history"][0]["batch"] is None
    scores = [entry["z"] for entry in closed["entries"] if entry.get("z") is not None]  # no winner: the highest z of the round
    assert data["rounds"][0]["winner_pass"] is None and data["rounds"][0]["best_z"] == max(scores) and len(scores) == 3
    # the first live round: no decision; the training of one batch failed, so its entry has no score field at all
    closed.update(outcome="no_decision", winner_uid=None, reason="no entry was scored")
    failed = closed["entries"][0]
    for name in ("delta", "se", "z", "cap_rate", "n_tasks", "pass_rate"):
        failed.pop(name, None)
    data = json.loads(build(tmp_path, snap)[2])
    assert data["rounds"][0] | {"with_z": 2, "best_z": max(scores[1:]), "winner_pass": None, "reason": "no entry was scored"} == data["rounds"][0]
    row = next(row for row in data["history"] if row["uid"] == failed["uid"])
    assert (row["status"], row["z"], row["delta"], row["cap_rate"]) == (failed["status"], None, None, None)
    for entry in closed["entries"]:
        entry["z"] = None
    data = json.loads(build(tmp_path, snap)[2])
    assert data["rounds"][0]["best_z"] is None and data["rounds"][0]["with_z"] == 0  # a round with no score has no bar


def test_queue_lists_the_round_first_and_its_total_matches_the_rows(tmp_path):
    snap = snapshot(tmp_path)
    queue = json.loads(build(tmp_path, snap)[2])["queue"]
    assert [(row["line"], row["uid"], row["file_valid"], row["flags"]) for row in queue["rows"]] == [
        (1, 8, True, ["leased"]), (2, 6, False, []), (3, 9, False, [])]
    assert queue["total"] == len(queue["rows"])
    snap["queue"]["length"] = 57  # the snapshot lists only the first seats of a long line
    queue = json.loads(build(tmp_path, snap)[2])["queue"]
    assert queue["total"] - len(queue["rows"]) == 55


def test_a_bad_or_missing_snapshot_keeps_the_last_good_data(tmp_path):
    snap = snapshot(tmp_path)
    done, out, good = build(tmp_path, snap)
    assert done.returncode == 0 and good
    for bad in ("{not json", {"generated_at": 1}, snap | {"schema": "sn120-v2-public-999"}):
        done, out, text = build(tmp_path, bad)
        assert done.returncode == 1 and "kept the last good" in done.stderr
        assert text == good and not [path for path in out.iterdir() if path.name.startswith(".tmp-")]
    (tmp_path / "store" / layout.MINING_JSON).unlink()
    args = [sys.executable, str(BUILD), "--source", str(tmp_path / "store"), "--public-url", URL, "--out", str(out)]
    assert subprocess.run(args, capture_output=True).returncode == 1 and (out / "site-data.json").read_text() == good


def test_missing_arguments_fail_closed(tmp_path):
    for missing in ("--source", "--public-url", "--out"):
        args = {"--source": str(tmp_path), "--public-url": URL, "--out": str(tmp_path / "out")}
        del args[missing]
        done = subprocess.run([sys.executable, str(BUILD), *[part for pair in args.items() for part in pair]], capture_output=True, text=True)
        assert done.returncode == 2 and missing in done.stderr
    assert not (tmp_path / "out").exists()


def result(model: str, bench: str, score: float, complete: bool = True) -> dict:
    return dict(model_id=model, benchmark_id=bench, score=score, se=0.01, complete=complete, samples=90, samples_expected=100, cap_rate=0.0)


def king(model: str, crowned_at: int, round_id: str | None = None, head: str | None = None, status: str = "scored") -> dict:
    return dict(model_id=model, label=model, kind="king", status=status, round_id=round_id, head_version_id=head, crowned_at=crowned_at)


def head(version_id: str, round_id: str, applied_at: int, folded: bool) -> dict:
    """One line of ``heads`` of the snapshot (``sn120_v2.public.snapshot``)."""
    return dict(version_id=version_id, round_id=round_id, applied_at=applied_at, folded=folded)


def bench_view(*args: object) -> dict:
    return runpy.run_path(str(BUILD))["bench_view"](*args)


def test_benchmark_table_sorts_kings_by_crowned_at_and_marks_partial_runs():
    bench = {
        "updated_at": 5, "node": {"pod": "secret-pod", "usd_h": 9},
        "benchmarks": [dict(id="b2", label="B2", suite="math", order=2, enabled=True, in_avg=False),
                       dict(id="b1", label="B1", suite="math", order=1, enabled=True), dict(id="off", label="Off", suite="math", enabled=False),
                       dict(id="b3", label="B3", suite="math", order=3, enabled=True)],
        "models": [dict(model_id="genesis", label="Genesis", kind="genesis", status="scored"), king("old", 100), king("new", 200),
                   king("half", 50), dict(model_id="weak", label="Weak", kind="reference", status="scored"),
                   dict(model_id="dead", label="Dead", kind="reference", status="failed"),
                   dict(model_id="strong", label="Strong", kind="reference", status="scored")],
        "results": [result("new", "b1", 0.5, complete=False), result("new", "b2", 0.9), result("new", "b3", 0.7), result("old", "b1", 0.4),
                    result("old", "b3", 0.4), result("old", "off", 1.0), result("half", "b1", 0.8), result("half", "b2", 0.8),
                    result("weak", "b1", 0.1), result("weak", "b3", 0.3), result("strong", "b1", 0.9), result("strong", "b3", 0.9),
                    result("genesis", "b1", 1.0), result("genesis", "b3", 1.0)],
    }
    view = bench_view(bench)
    assert [column["id"] for column in view["columns"]] == ["b1", "b2", "b3"]
    # the kings, newest first; then the reference models (the page orders them by size); the genesis head is the last row at any score
    assert [(row["label"], row["at"]) for row in view["rows"]] == [
        ("new", 200), ("old", 100), ("half", 50), ("Strong", None), ("Weak", None), ("Dead", None), ("Genesis", None)]
    new, old, half = view["rows"][:3]
    assert new["cells"]["b1"]["partial"] and new["avg_partial"] and new["avg"] == 0.6  # b2 is shown but not in the average
    assert not old["cells"]["b1"]["partial"] and not old["avg_partial"] and "secret-pod" not in json.dumps(view)
    # a mean over fewer benchmarks cannot be compared: a row with a counted benchmark missing has no average
    assert half["avg"] is None and set(half["cells"]) == {"b1", "b2"} and view["rows"][5]["avg"] is None
    assert [row["status"] for row in view["rows"]] == ["scored"] * 5 + ["failed", "scored"]


def test_every_king_and_the_genesis_head_has_one_row_that_says_the_state_of_its_benchmark_run():
    """The file of the benchmark service names the folded head of a round. The snapshot lists the stacked and the folded head."""
    heads = [head("g" * 64, "genesis", 10, True), head("s1" * 32, "r000001", 20, False), head("f1" * 32, "r000001", 21, True),
             head("s2" * 32, "r000002", 30, False), head("f2" * 32, "r000002", 31, True), head("s3" * 32, "r000003", 40, False),
             head("f3" * 32, "r000003", 41, True), head("s4" * 32, "r000004", 50, False), head("f4" * 32, "r000004", 51, True),
             head("s5" * 32, "r000005", 60, False)]  # the fold of the newest winner is not made yet
    bench = {
        "updated_at": 7, "benchmarks": [dict(id="b1", label="B1", suite="math", order=1, enabled=True)],
        "models": [dict(model_id="genesis-87de", label="genesis Launch-Model-9B", kind="genesis", status="queued", round_id="genesis",
                        head_version_id="g" * 64, crowned_at=10),
                   king("king-1", 21, "r000001", "f1" * 32), king("king-2", 31, "r000002", "f2" * 32, status="failed"),
                   king("king-3", 41, None, "s3" * 32, status="queued"),  # named by a head id only
                   king("king-4", 51, "r000004", "f4" * 32, status="evaluating"),
                   dict(model_id="ref", label="Ref-8B", kind="reference", status="scored")],
        "results": [result("king-1", "b1", 0.5), result("king-2", "b1", 0.25), result("ref", "b1", 0.75)],
    }
    rows = bench_view(bench, heads)["rows"]
    assert [(row["label"], row["kind"], row["status"], row["at"]) for row in rows] == [
        ("r000005 s5s5s5s5", "king", "none", 60),  # the file does not name this king yet: the row comes from the snapshot
        ("r000004 f4f4f4f4", "king", "running", 51), ("king-3", "king", "queued", 41), ("r000002 f2f2f2f2", "king", "failed", 31),
        ("r000001 f1f1f1f1", "king", "scored", 21), ("Ref-8B", "reference", "scored", None), ("Genesis Launch-Model-9B", "genesis", "queued", 10)]
    assert [row["avg"] for row in rows] == [None, None, None, 0.25, 0.5, 0.75, None]  # a result of a failed run stays
    assert rows[0]["head"] == "s5" * 32 and rows[1]["head"] == "f4" * 32 and rows[6]["head"] == "g" * 64

    # the states of the benchmark service, and the same states in other words
    for word, state in (("scored", "scored"), ("done", "scored"), ("queued", "queued"), ("evaluating", "running"), ("running", "running"),
                        ("failed", "failed"), (None, "none"), ("paused", "paused")):
        assert bench_view(bench | {"models": [king("k", 21, "r000001", "f1" * 32, status=word)]}, heads[:3])["rows"][0]["status"] == state

    # no benchmark file: one row for each applied round, from its folded head when the snapshot has it, and the genesis row
    bare = bench_view(None, heads)
    assert bare["columns"] == [] and bare["updated_at"] is None
    assert [(row["label"], row["status"], row["avg"], row["cells"]) for row in bare["rows"]] == [
        ("r000005 s5s5s5s5", "none", None, {}), ("r000004 f4f4f4f4", "none", None, {}), ("r000003 f3f3f3f3", "none", None, {}),
        ("r000002 f2f2f2f2", "none", None, {}), ("r000001 f1f1f1f1", "none", None, {}), ("Genesis", "none", None, {})]
    assert [row["kind"] for row in bare["rows"]] == ["king"] * 5 + ["genesis"]
    # a file whose genesis row has no head and no time gets them from the snapshot; a file with reference models only keeps them
    seeded = bench_view(bench | {"models": [dict(model_id="genesis", label="Genesis", kind="genesis", status="scored"), bench["models"][-1]]}, heads[:3])
    assert [(row["label"], row["status"], row["head"], row["at"]) for row in seeded["rows"]] == [
        ("r000001 f1f1f1f1", "none", "f1" * 32, 21), ("Ref-8B", "scored", None, None), ("Genesis", "scored", "g" * 64, 10)]
    assert bench_view(bench, [])["rows"][-1]["label"] == "Genesis Launch-Model-9B" and len(bench_view(bench, [])["rows"]) == 6


def test_the_reference_models_come_from_the_frozen_v1_history_whatever_the_benchmark_file_holds():
    columns = [dict(id="b1", label="B1", suite="math", order=1, enabled=True), dict(id="b2", label="B2", suite="math", order=2, enabled=True)]
    reference = [dict(model_id="a", label="Ref-A", kind="reference", status="scored"), dict(model_id="b", label="Ref-B", kind="reference", status="failed"),
                 dict(model_id="c", label="Ref-C", kind="reference", status="scored")]
    kept = bench_view({  # as ``v1_archive.py`` froze it: the kings and the genesis model of v1, and the reference models
        "updated_at": 5, "benchmarks": columns,
        "models": [king("E12 abcd", 200), dict(model_id="g", label="Genesis", kind="genesis", status="scored"), *reference],
        "results": [result("E12 abcd", "b1", 0.9), result("E12 abcd", "b2", 0.9), result("g", "b1", 0.2), result("g", "b2", 0.2),
                    result("a", "b1", 0.5), result("a", "b2", 0.75), result("c", "b1", 1.0)]})
    heads = [head("g" * 64, "genesis", 10, True), head("f1" * 32, "r000001", 21, True)]

    bare = bench_view(None, heads, kept)  # no benchmark file: the columns and the reference models of the v1 history
    assert [column["id"] for column in bare["columns"]] == ["b1", "b2"] and bare["updated_at"] is None
    assert [(row["label"], row["kind"], row["status"], row["avg"], row.get("v1")) for row in bare["rows"]] == [
        ("r000001 f1f1f1f1", "king", "none", None, None), ("Ref-A", "reference", "scored", 0.625, True),
        ("Ref-B", "reference", "failed", None, True), ("Ref-C", "reference", "scored", None, True), ("Genesis", "genesis", "none", None, None)]
    assert bare["rows"][-1]["head"] == "g" * 64 and bare["rows"][-1]["cells"] == {}  # the genesis head of v2, not the model of v1
    assert bare["rows"][1]["cells"] == kept["rows"][1]["cells"] and "E12 abcd" not in json.dumps(bare)

    table = {  # the table of the benchmark service: one column, its own rows, and one reference model that an operator put in
        "updated_at": 9, "benchmarks": columns[:1],
        "models": [king("king-1", 21, "r000001", "f1" * 32), dict(model_id="a2", label="Ref-A", kind="reference", status="scored")],
        "results": [result("king-1", "b1", 0.75), result("a2", "b1", 0.25)]}
    both = bench_view(table, heads, kept)
    assert [column["id"] for column in both["columns"]] == ["b1"] and both["updated_at"] == 9
    # one row for each label: the row of the table wins. The average of a kept row is over the columns of the table.
    assert [(row["label"], row["status"], row["avg"], row.get("v1")) for row in both["rows"]] == [
        ("r000001 f1f1f1f1", "scored", 0.75, None), ("Ref-C", "scored", 1.0, True), ("Ref-A", "scored", 0.25, None),
        ("Ref-B", "failed", None, True), ("Genesis", "none", None, None)]
    assert bench_view(table, heads)["rows"] == [row for row in both["rows"] if not row.get("v1")]


def test_the_site_builds_from_the_publishers_file_with_or_without_the_benchmark_file(tmp_path):
    """``public/mining.json`` is the snapshot plus the publisher's own keys. ``public/benchmarks.json`` may be absent or bad."""
    snap = snapshot(tmp_path) | {"snapshot_digest": "d" * 64, "contract": {"public_url": "https://x.example", "objects": {}}, "upload": {"k": 1}}
    folded = next(line for line in snap["heads"] if line["folded"] and line["round_id"] == "r000001")
    done, _, text = build(tmp_path, snap)  # launch: the benchmark service does not run
    assert done.returncode == 0 and "no benchmark file" in done.stderr
    data = json.loads(text)
    assert data["kings"] and "d" * 64 not in text and "x.example" not in text
    bare = data["bench"]  # the king and the genesis head have their rows; there is no score and no column
    assert [(row["label"], row["kind"], row["status"]) for row in bare["rows"]] == [
        (f"r000001 {folded['version_id'][:8]}", "king", "none"), ("Genesis", "genesis", "none")]
    assert bare["columns"] == [] and bare["updated_at"] is None and bare["rows"][0]["head"] == folded["version_id"]
    path = tmp_path / "store" / layout.BENCHMARKS
    path.write_text(json.dumps({
        "updated_at": 5, "benchmarks": [dict(id="b1", label="B1", suite="math", order=1, enabled=True)],
        "models": [king("king-0123", folded["applied_at"], "r000001", folded["version_id"])], "results": [result("king-0123", "b1", 0.5)]}))
    done, _, text = build(tmp_path, snap)
    assert done.returncode == 0 and "no benchmark file" not in done.stderr
    assert [(row["label"], row["status"], row["avg"]) for row in json.loads(text)["bench"]["rows"]] == [
        (bare["rows"][0]["label"], "scored", 0.5), ("Genesis", "none", None)]  # the same name with the file, and no second row
    for bad in ("{not json", "[1, 2]", json.dumps({"benchmarks": [{"label": "no id"}], "models": [], "results": []})):
        path.write_text(bad)
        done, _, text = build(tmp_path, snap)
        assert done.returncode == 0 and "no benchmark file" in done.stderr
        assert json.loads(text)["bench"] == bare and json.loads(text)["kings"] == data["kings"]


def v1_site(root: Path) -> Path:
    """The four public files of the v1 website, with the fields that v1 writes."""
    def turn(epoch: int, status: str, verdict: str | None, **more: object) -> dict:
        return dict(epoch_index=epoch, status=status, verdict=verdict, uid=epoch % 7, hotkey=f"hk{epoch}", head="a" * 64, tasks=125,
                    rollouts=1000, candidate="c" * 64, turn_at=1000.0 + epoch, reg_block=77, queue_position=3, **more)

    files = {
        "k1.json": {
            "generated_at": 2000.0, "queue": [{"uid": 1, "hotkey": "waiting"}],
            "payees": [dict(uid=920, hotkey="hk-king", green_epoch=12, z=1.5, still_registered=True),
                       dict(uid=918, hotkey="hk-older", green_epoch=9, z=2.5, still_registered=True)],
            "turns": [turn(13, "turn", "pending"), turn(12, "turn", "green", z=1.5, c_pass=0.8, h_pass=0.7, cap=0.25),
                      turn(11, "skip", None), turn(10, "turn", "red", z=-0.4, c_pass=0.6, h_pass=0.7, cap=0.5)]},
        "k1/turns/index.json": {"turns": [dict(epoch=9, uid=918, decision="green", tasks=8, candidate="e" * 64, turn="k1/turns/9.json"),
                                          dict(epoch=10, uid=3, decision="red", tasks=8, candidate="", turn="k1/turns/10.json")]},
        "scores.json": {"miners": [dict(uid=920, hotkey="hk-king", share_now=0.5, miner_hex="secret-hex"),
                                   dict(uid=918, hotkey="someone-else", share_now=0.5)]},
        "benchmarks.json": {
            "updated_at": 5, "node": {"pod": "secret-pod"}, "benchmarks": [dict(id="b1", label="B1", suite="math", order=1, enabled=True)],
            "models": [king("E12 abcd", 200) | {"epoch": 12}, king("E9 ef01", 100) | {"epoch": 9},
                       dict(model_id="ref", label="Ref-8B", kind="reference", status="scored")],
            "results": [result("E12 abcd", "b1", 0.9), result("E9 ef01", "b1", 0.8), result("ref", "b1", 0.75)]},
    }
    for name, body in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(json.dumps(body))
    return root


def test_the_v1_history_is_frozen_once_and_then_needs_no_v1_file(tmp_path):
    site, frozen = v1_site(tmp_path / "v1"), tmp_path / "keep" / "v1-history.json"
    freeze = [sys.executable, str(V1_ARCHIVE), "--from", str(site), "--out", str(frozen)]
    done = subprocess.run(freeze, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    data = json.loads(frozen.read_text())
    assert data["schema"] == "sn120-v1-archive-1" and data["archived_at"] > data["v1_generated_at"] == 2000.0
    assert [(k["rank"], k["uid"], k["epoch"], k["z"], k["share"]) for k in data["kings"]] == [(1, 920, 12, 1.5, 0.5), (2, 918, 9, 2.5, None)]
    # newest first; the "skip" epoch is not a turn; epoch 10 is in both files and keeps its scores; epoch 9 is only in the old index
    assert [(row["epoch"], row["verdict"]) for row in data["turns"]] == [(13, "pending"), (12, "green"), (10, "red"), (9, "green")]
    assert data["turns"][2] | {"z": -0.4, "c_pass": 0.6, "cap_rate": 0.5, "hotkey": "hk10", "at": 1010.0} == data["turns"][2]
    assert data["turns"][3] == {"epoch": 9, "uid": 918, "verdict": "green", "tasks": 8, "candidate": "e" * 64}
    assert [row["label"] for row in data["bench"]["rows"]] == ["E12 abcd", "E9 ef01", "Ref-8B"] and data["bench"]["rows"][0]["avg"] == 0.9
    for private in ("secret", "waiting", "queue_position", "reg_block", "k1/turns/"):
        assert private not in frozen.read_text()

    shutil.rmtree(site)  # the v1 website is gone: the v2 site is built from the frozen file alone
    snap = snapshot(tmp_path)
    done, out, _ = build(tmp_path, snap, "--v1-history", str(frozen))
    assert done.returncode == 0, done.stderr
    assert (out / "v1-data.json").read_bytes() == frozen.read_bytes() and "v1-data.json" in (out / "v1.html").read_text()
    again = subprocess.run(freeze, capture_output=True, text=True)  # a source that cannot be read never replaces a good archive
    assert again.returncode == 1 and "nothing written" in again.stderr and json.loads(frozen.read_text()) == data

    # the v2 table takes the reference model, and no king of v1, from the installed history; there is no benchmark file
    def table(*more: str) -> tuple[subprocess.CompletedProcess, list]:
        done, _, text = build(tmp_path, snap, *more)
        bench = json.loads(text)["bench"]
        return done, [(row["label"].split()[0], row["kind"], row["status"], row["avg"], row.get("v1")) for row in bench["rows"]]

    rows = [("r000001", "king", "none", None, None), ("Ref-8B", "reference", "scored", 0.75, True), ("Genesis", "genesis", "none", None, None)]
    done, made = table("--v1-history", str(frozen))
    assert made == rows and "reference model" not in done.stderr
    assert table()[1] == rows  # the copy in the site is enough: the flag installs it, the builder reads it
    for bad in ("{not json", json.dumps({"schema": "other"}), json.dumps(data | {"bench": {"rows": [{"kind": "reference"}], "columns": []}})):
        (out / "v1-data.json").write_text(bad)  # a bad copy takes away the reference models and nothing else
        done, made = table()
        assert done.returncode == 0 and "no reference model from the v1 history" in done.stderr and made == [rows[0], rows[2]]
    (out / "v1-data.json").unlink()  # a site with no v1 history is normal: nothing is printed for it
    done, made = table()
    assert done.returncode == 0 and "reference model" not in done.stderr and made == [rows[0], rows[2]]


def test_a_partial_v1_site_or_a_file_that_is_not_an_archive_is_refused(tmp_path):
    site = v1_site(tmp_path / "v1")
    (site / "scores.json").unlink()
    done = subprocess.run([sys.executable, str(V1_ARCHIVE), "--from", str(site), "--out", str(tmp_path / "frozen.json")], capture_output=True, text=True)
    assert done.returncode == 1 and "nothing written" in done.stderr and not (tmp_path / "frozen.json").exists()
    snap = snapshot(tmp_path)
    done, out, _ = build(tmp_path, snap, "--v1-history", str(site / "k1.json"))  # a raw v1 file is not the archive
    assert done.returncode == 1 and "v1 history not installed" in done.stderr and not (out / "v1-data.json").exists()
    done, out, _ = build(tmp_path, snap, "--v1-history", str(tmp_path / "missing.json"))
    assert done.returncode == 1 and not (out / "v1-data.json").exists()
