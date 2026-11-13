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

    snap["history"][0]["outcome"] = "abandoned"  # a named winner whose files were not verified has no public files
    assert json.loads(build(tmp_path, snap)[2])["history"][0]["batch"] is None


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


def king(model: str, crowned_at: int) -> dict:
    return dict(model_id=model, label=model, kind="king", status="scored", round_id="r1", head_version_id="h", crowned_at=crowned_at)


def test_benchmark_table_sorts_kings_by_crowned_at_and_marks_partial_runs():
    bench = {
        "updated_at": 5, "node": {"pod": "secret-pod", "usd_h": 9},
        "benchmarks": [dict(id="b2", label="B2", suite="math", order=2, enabled=True, in_avg=False),
                       dict(id="b1", label="B1", suite="math", order=1, enabled=True), dict(id="off", label="Off", suite="math", enabled=False)],
        "models": [dict(model_id="genesis", label="Genesis", kind="genesis", status="scored"), king("old", 100), king("new", 200)],
        "results": [result("new", "b1", 0.5, complete=False), result("new", "b2", 0.9), result("old", "b1", 0.4), result("old", "off", 1.0)],
    }
    view = runpy.run_path(str(BUILD))["bench_view"](bench)
    assert [column["id"] for column in view["columns"]] == ["b1", "b2"]
    assert [(row["label"], row["at"]) for row in view["rows"]] == [("new", 200), ("old", 100), ("Genesis", None)]
    new, old = view["rows"][0], view["rows"][1]
    assert new["cells"]["b1"]["partial"] and new["avg_partial"] and new["avg"] == 0.5  # b2 is shown but not in the average
    assert not old["cells"]["b1"]["partial"] and not old["avg_partial"] and "secret-pod" not in json.dumps(view)


def test_the_site_builds_from_the_publishers_file_with_or_without_the_benchmark_file(tmp_path):
    """``public/mining.json`` is the snapshot plus the publisher's own keys. ``public/benchmarks.json`` may be absent or bad."""
    snap = snapshot(tmp_path) | {"snapshot_digest": "d" * 64, "contract": {"public_url": "https://x.example", "objects": {}}, "upload": {"k": 1}}
    done, _, text = build(tmp_path, snap)  # launch: the benchmark service does not run
    assert done.returncode == 0 and "no benchmark table" in done.stderr
    data = json.loads(text)
    assert data["bench"] is None and data["kings"] and "d" * 64 not in text and "x.example" not in text
    path = tmp_path / "store" / layout.BENCHMARKS
    path.write_text(json.dumps({
        "updated_at": 5, "benchmarks": [dict(id="b1", label="B1", suite="math", order=1, enabled=True)],
        "models": [king("new", 200)], "results": [result("new", "b1", 0.5)]}))
    done, _, text = build(tmp_path, snap)
    assert done.returncode == 0 and "no benchmark table" not in done.stderr
    assert json.loads(text)["bench"]["rows"][0]["avg"] == 0.5
    for bad in ("{not json", "[1, 2]", json.dumps({"benchmarks": [{"label": "no id"}], "models": [], "results": []})):
        path.write_text(bad)
        done, _, text = build(tmp_path, snap)
        assert done.returncode == 0 and "no benchmark table" in done.stderr
        assert json.loads(text)["bench"] is None and json.loads(text)["kings"] == data["kings"]


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
            "models": [king("E12 abcd", 200) | {"epoch": 12}, king("E9 ef01", 100) | {"epoch": 9}],
            "results": [result("E12 abcd", "b1", 0.9), result("E9 ef01", "b1", 0.8)]},
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
    assert [row["label"] for row in data["bench"]["rows"]] == ["E12 abcd", "E9 ef01"] and data["bench"]["rows"][0]["avg"] == 0.9
    for private in ("secret", "waiting", "queue_position", "reg_block", "k1/turns/"):
        assert private not in frozen.read_text()

    shutil.rmtree(site)  # the v1 website is gone: the v2 site is built from the frozen file alone
    done, out, _ = build(tmp_path, snapshot(tmp_path), "--v1-history", str(frozen))
    assert done.returncode == 0, done.stderr
    assert (out / "v1-data.json").read_bytes() == frozen.read_bytes() and "v1-data.json" in (out / "v1.html").read_text()
    again = subprocess.run(freeze, capture_output=True, text=True)  # a source that cannot be read never replaces a good archive
    assert again.returncode == 1 and "nothing written" in again.stderr and json.loads(frozen.read_text()) == data


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
