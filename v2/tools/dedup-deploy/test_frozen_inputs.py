#!/usr/bin/env python3
"""Offline qualification for frozen_inputs.py; no network or real credentials."""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
from pathlib import Path
import struct
import subprocess
import tempfile
from unittest import mock
import unittest

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[2]
JOURNAL = Path("/home/bb/inbox/dedup-prodstate/journal.jsonl")
PYTHON = Path("/home/bb/scratch/kingx2-genfeed-rebase/testenv/bin/python")
ROUNDS = ("r000085", "r000086", "r000087", "r000088")

spec = importlib.util.spec_from_file_location("frozen_inputs_under_test", HERE / "frozen_inputs.py")
TOOL = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(TOOL)
REAL_RESULT_REFERENCE = TOOL.result_reference


def synthetic_result(records: list[dict], round_id: str, lease: dict, public: bool = True) -> bytes:
    entries = []
    for seat in lease["entries"]:
        pos = seat["queue_pos"]
        prefix = (f"public/loras/{hashlib.sha256(f'{round_id}/{pos}'.encode()).hexdigest()}" if public
                  else f"pending/rounds/{round_id}/{lease['lease_id']}/loras/{pos:03d}")
        files = {name: hashlib.sha256(f"{round_id}/{pos}/{name}".encode()).hexdigest()
                 for name in TOOL.LORA_FILES}
        artifact = {"lora_hash": TOOL.manifest_digest(files), "files": files,
                    "sizes": {name: len(f"{round_id}/{pos}/{name}".encode()) for name in TOOL.LORA_FILES},
                    "prefix": prefix, "head_version_id": "synthetic", "batch_sha256": seat["batch"]["batch_sha256"],
                    "recipe_id": "synthetic", "rank": 1, "alpha": 1.0, "target_modules": [],
                    "report": {"seed": 1, "steps": 1, "loss": 0.0, "step_norm": 1.0, "tokens": 1,
                               "wall_s": 1.0, "gpu_class": "fake", "image_digest": "fake"}}
        entries.append({"queue_pos": pos, "status": "scored", "reason": "", "lora": artifact, "outcomes": []})
    obj = {"round_id": round_id, "lease_id": lease["lease_id"], "box_id": "fake",
           "ticket_digest": lease["ticket_digest"], "head_version_id": "fake", "image_digest": "fake",
           "null_outcomes": [], "entries": entries, "started_at": 0, "finished_at": 1}
    return TOOL.canonical_json(obj)


class FakeStore:
    def __init__(self, objects: dict[str, bytes]):
        self.objects = objects
        self.calls = []

    def get(self, key):
        self.calls.append(("get", key))
        if key not in self.objects:
            raise KeyError(key)
        return self.objects[key]

    def put(self, *args): self.calls.append(("put", args)); raise AssertionError("write reached fake store")
    def put_file(self, *args): self.calls.append(("put_file", args)); raise AssertionError("write reached fake store")
    def copy(self, *args): self.calls.append(("copy", args)); raise AssertionError("write reached fake store")
    def delete(self, *args): self.calls.append(("delete", args)); raise AssertionError("write reached fake store")
    def list(self, *args): self.calls.append(("list", args)); raise AssertionError("list reached fake store")
    def exists(self, *args): self.calls.append(("exists", args)); raise AssertionError("exists reached fake store")
    def get_file(self, *args): self.calls.append(("get_file", args)); raise AssertionError("get_file reached fake store")
    def sha256(self, *args): self.calls.append(("sha256", args)); raise AssertionError("sha256 reached fake store")


class FakeClient:
    def __init__(self):
        self.calls = []
        self.meta = object()

    def _make_api_call(self, name, params):
        self.calls.append((name, params))
        if name != "GetObject":
            raise AssertionError("non-GET reached client")
        return {"Body": io.BytesIO(b"{}")}


class FrozenInputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = TOOL.read_journal(JOURNAL)
        cls.settled = {}
        for round_id in ROUNDS[:3]:
            rnd = TOOL.round_open(cls.records, round_id)
            for lease in rnd["leases"]:
                raw = synthetic_result(cls.records, round_id, lease)
                digest = TOOL.result_digest(raw)
                key = f"private/rounds/{round_id}/{lease['lease_id']}/result.json"
                cls.settled[key] = (raw, digest)

    def plan(self, root="/operator/frozen", results=True, **kwargs):
        results_map = {key: raw for key, (raw, _) in self.settled.items()} if results else None
        with unittest.mock.patch.object(TOOL, "result_reference", side_effect=self.reference):
            return TOOL.make_plan(JOURNAL, root, ROUNDS, None, str(PYTHON), results_map, **kwargs)

    def reference(self, records, round_id, lease_id):
        key = f"private/rounds/{round_id}/{lease_id}/result.json"
        return self.settled[key][1] if key in self.settled else REAL_RESULT_REFERENCE(self.records, round_id, lease_id)

    def objects_for(self, plan, streaming=False):
        objects = {}
        batch_preimages = {}
        for obj in plan["objects"]:
            if obj["kind"] == "batch":
                # Synthetic bytes generated under the journal's real hash. No
                # real batch object or private content is read by these tests.
                objects[obj["key"]] = TOOL.canonical_json({"fixture": obj["sha256"]})
            elif obj["kind"] == "result":
                objects[obj["key"]] = self.settled[obj["key"]][0]
            elif obj["kind"] == "lora":
                pos = int(obj["path"].split("/q")[1].split("/")[0])
                name = obj["path"].rsplit("/", 1)[1]
                round_id = obj["path"].split("/", 1)[0]
                objects[obj["key"]] = f"{round_id}/{pos}/{name}".encode()
            elif obj["kind"] == "first_seen":
                objects[obj["key"]] = b"{}"
        if not streaming:
            return objects

        real_digest = hashlib.sha256
        fixture_hashes = {hashlib.sha256(raw).digest(): expected for raw, expected in
                          ((TOOL.canonical_json({"fixture": obj["sha256"]}), obj["sha256"])
                           for obj in plan["objects"] if obj["kind"] == "batch")}

        class ExpectedDigest:
            def __init__(self, data=b""):
                self.hasher = real_digest(data)

            def update(self, chunk):
                self.hasher.update(chunk)

            def hexdigest(self):
                return fixture_hashes.get(self.hasher.digest(), self.hasher.hexdigest())

        def expected_digest(data=b""):
            return ExpectedDigest(data)

        class StreamingR2:
            def __init__(self):
                self._bucket = "synthetic-bucket"
                self._prefix = ""
                self._s3 = self
                self.objects = objects
                self.calls = []

            def get_object(self, *, Bucket, Key):
                self.calls.append(("GetObject", Bucket, Key))
                if Key not in objects:
                    raise KeyError(Key)
                return {"Body": io.BytesIO(objects[Key])}

        return StreamingR2(), mock.patch.object(TOOL.hashlib, "sha256", expected_digest)

    def test_01_supplied_journal_and_commands(self):
        self.assertEqual(TOOL.sha256(JOURNAL.read_bytes()), TOOL.SUPPLIED_JOURNAL_SHA256)
        plan = self.plan(results=False)
        self.assertEqual(len(plan["seats"]), 24)
        self.assertEqual([len(c["argv"][c["argv"].index("dedup-dryrun") + 1:]) for c in plan["dryrun_commands"]], [12] * 4)
        for command in plan["dryrun_commands"]:
            self.assertEqual(command["argv"].count("--batch"), 6)
            self.assertNotIn("--lora", command["argv"])
            self.assertNotIn("--first-seen", command["argv"])
            self.assertEqual(command["argv"][:3], [str(PYTHON), "-I", str(HERE / "frozen_input_dryrun.py")])
        self.assertEqual(sum(o["kind"] == "batch" for o in plan["objects"]), 24)
        self.assertEqual(sum(o["kind"] == "result" for o in plan["objects"]), 3)

    def test_02_sealed_results_derive_loras_and_no_pending_assumption(self):
        plan = self.plan()
        for command in plan["dryrun_commands"][:3]:
            self.assertTrue(command["with_loras"])
            self.assertEqual(command["argv"].count("--lora"), 6)
        self.assertFalse(plan["dryrun_commands"][3]["with_loras"])
        self.assertEqual(sum(o["kind"] == "lora" for o in plan["objects"]), 54)
        self.assertTrue(all(o["key"].startswith("public/loras/") for o in plan["objects"] if o["kind"] == "lora"))
        pending = synthetic_result(self.records, "r000086", TOOL.round_open(self.records, "r000086")["leases"][0], public=False)
        key = next(key for key in self.settled if "/r000086/" in key)
        with unittest.mock.patch.object(TOOL, "result_reference", return_value=TOOL.result_digest(pending)):
            changed = TOOL.make_plan(JOURNAL, "/x", ("r000086",), None, str(PYTHON), {key: pending})
        self.assertEqual(changed["seats"][0]["lora_status"], "complete_candidate")
        self.assertTrue(changed["dryrun_commands"][0]["with_loras"])
        self.assertTrue(changed["seats"][0]["lora"]["prefix"].startswith("pending/rounds/"))

    def test_03_get_only_store_fetch_and_hashes(self):
        plan = self.plan()
        fake, digest_patch = self.objects_for(plan, streaming=True)
        guarded = TOOL.GetOnlyStore(fake)
        with tempfile.TemporaryDirectory() as tmp, digest_patch:
            target = Path(tmp) / "new"
            receipt = TOOL.fetch_plan(plan, target, guarded)
            self.assertEqual(len(receipt["fetched"]), len(plan["objects"]))
            self.assertTrue(all(name == "GetObject" for name, _, _ in fake.calls))
            self.assertTrue((target / "fetch-receipt.json").is_file())
            self.assertTrue((target / "r000086/q000000/lora/adapter_config.json").is_file())
            for method in ("put", "put_file", "copy", "delete", "list", "exists", "sha256", "get_file"):
                with self.assertRaisesRegex(TOOL.InputError, "GET-only guard"):
                    getattr(guarded, method)("x")
            with self.assertRaisesRegex(TOOL.InputError, "new, empty"):
                TOOL.fetch_plan(plan, target, guarded)

    def test_04_missing_optional_result_and_required_lora(self):
        plan = self.plan()
        fake, digest_patch = self.objects_for(plan, streaming=True)
        result_key = next(iter(self.settled))
        del fake.objects[result_key]
        with tempfile.TemporaryDirectory() as tmp, digest_patch:
            receipt = TOOL.fetch_plan(plan, Path(tmp) / "new", TOOL.GetOnlyStore(fake))
            self.assertEqual(len(receipt["optional_missing"]), 1)
        plan = self.plan()
        fake, digest_patch = self.objects_for(plan, streaming=True)
        del fake.objects[next(o["key"] for o in plan["objects"] if o["kind"] == "lora")]
        with tempfile.TemporaryDirectory() as tmp, digest_patch:
            with self.assertRaisesRegex(TOOL.InputError, "required object is missing"):
                TOOL.fetch_plan(plan, Path(tmp) / "new", TOOL.GetOnlyStore(fake))
            self.assertFalse(list((Path(tmp) / "new").rglob("*.part")))

    def test_05_corrupt_object_and_wrong_sealed_seat_fail(self):
        plan = self.plan()
        fake, digest_patch = self.objects_for(plan, streaming=True)
        key = next(o["key"] for o in plan["objects"] if o["kind"] == "lora")
        fake.objects[key] = b"wrong"
        with tempfile.TemporaryDirectory() as tmp, digest_patch:
            with self.assertRaisesRegex(TOOL.InputError, "SHA-256 mismatch"):
                TOOL.fetch_plan(plan, Path(tmp) / "new", TOOL.GetOnlyStore(fake))
        rnd = TOOL.round_open(self.records, "r000085")
        lease = rnd["leases"][0]
        obj = json.loads(synthetic_result(self.records, "r000085", lease))
        obj["entries"][0]["lora"]["batch_sha256"] = "0" * 64
        with self.assertRaisesRegex(TOOL.InputError, "another frozen batch"):
            TOOL.validate_result(TOOL.canonical_json(obj), rnd, lease, None)

    def test_06_first_seen_all_or_none_and_params(self):
        plan = self.plan(results=False)
        ledger = {s["batch_sha256"]: {"hotkey": "synthetic", "first_seen_unix": 100 + i}
                  for i, s in enumerate(plan["seats"])}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "first_seen.json"
            path.write_text(json.dumps(ledger))
            plan = self.plan(results=False, first_seen=path)
            self.assertIsNone(plan["params"])
            for command in plan["dryrun_commands"]:
                self.assertEqual(command["argv"].count("--first-seen"), 6)
            ledger.pop(next(iter(ledger)))
            path.write_text(json.dumps(ledger))
            plan = self.plan(results=False, first_seen=path)
            self.assertTrue(all(not c["with_first_seen"] for c in plan["dryrun_commands"]))
            self.assertTrue(all(c["first_seen_missing_seats"] for c in plan["dryrun_commands"]))
        with tempfile.TemporaryDirectory() as tmp:
            params = Path(tmp) / "params.json"
            params.write_text("{}")
            plan = TOOL.make_plan(JOURNAL, "/x", ("r000088",), str(params), str(PYTHON))
            self.assertEqual(plan["dryrun_commands"][0]["argv"][-2:], ["--params", str(params)])

    def test_07_s3_guard_allows_only_get_object(self):
        client = FakeClient()
        guarded = TOOL.GetOnlyS3(client)
        body = guarded.get_object(Bucket="fake", Key="private/fake")["Body"]
        self.assertEqual(body.read(), b"{}")
        self.assertEqual([c[0] for c in client.calls], ["GetObject"])
        for name in ("ListObjectsV2", "HeadObject", "PutObject", "DeleteObject", "CopyObject"):
            with self.assertRaisesRegex(TOOL.InputError, "GET-only transport"):
                guarded._guarded_call(name, {"Bucket": "fake", "Key": "private/fake"})
        with self.assertRaisesRegex(TOOL.InputError, "GET-only transport"):
            guarded.list_objects_v2(Bucket="fake")
        self.assertEqual([c[0] for c in client.calls], ["GetObject"])

    def test_08_real_transport_guard_precedes_network(self):
        import boto3
        client = boto3.client("s3", endpoint_url="https://example.invalid", region_name="auto",
                              aws_access_key_id="fake-not-a-credential", aws_secret_access_key="fake-not-a-credential")
        guarded = TOOL.GetOnlyS3(client)
        with self.assertRaisesRegex(TOOL.InputError, "ListObjectsV2"):
            guarded._guarded_call("ListObjectsV2", {"Bucket": "fake", "Prefix": "private/"})
        with self.assertRaisesRegex(TOOL.InputError, "GetObject"):
            guarded._guarded_call("GetObject", {"Bucket": "fake", "Key": "x", "Range": "bytes=0-1"})

    def test_09_result_digest_required_before_lora_plan(self):
        rnd = TOOL.round_open(self.records, "r000085")
        lease = rnd["leases"][0]
        raw = synthetic_result(self.records, "r000085", lease)
        with self.assertRaisesRegex(TOOL.InputError, "digest differs"):
            TOOL.validate_result(raw, rnd, lease, "0" * 64)

    def test_10_cli_summary_and_private_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "plan.json"
            base = [str(PYTHON), "-I", str(HERE / "frozen_inputs.py"), "--journal", str(JOURNAL),
                    "--journal-sha256", TOOL.SUPPLIED_JOURNAL_SHA256, "--root", "/operator/frozen"]
            summary = subprocess.run(base + ["--summary"], text=True, capture_output=True, check=True)
            report = json.loads(summary.stdout)
            self.assertEqual(report["batch_count"], 24)
            self.assertEqual(len(report["commands"]), 4)
            self.assertNotIn("private/rounds/", summary.stdout)
            subprocess.run(base + ["--output", str(output)], text=True, capture_output=True, check=True)
            plan = json.loads(output.read_text())
            self.assertEqual(oct(output.stat().st_mode & 0o777), "0o600")
            self.assertEqual(len(plan["seats"]), 24)
            with self.assertRaisesRegex(TOOL.InputError, "sha256"):
                TOOL.make_plan(JOURNAL, "/x", ROUNDS, None, str(PYTHON), journal_sha256="0" * 64)

    def test_11_real_cli_launcher(self):
        proc = subprocess.run([str(PYTHON), "-I", str(HERE / "frozen_input_dryrun.py"),
                               "--v2-root", str(WORKTREE / "v2"), "dedup-dryrun", "--help"],
                              text=True, capture_output=True, check=True)
        self.assertIn("--batch", proc.stdout)
        self.assertIn("--lora", proc.stdout)
        self.assertIn("--first-seen", proc.stdout)

    def test_11b_generated_command_executes_real_cli(self):
        plan = self.plan(root="/operator/frozen")
        self.assertTrue(plan["dryrun_commands"][0]["with_loras"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Synthetic stand-ins at the exact generated paths; no fetched batch
            # bytes or private hotkeys are used or printed by this test.
            for seat in plan["seats"]:
                if seat["round_id"] != "r000085":
                    continue
                batch = root / seat["batch_path"]
                batch.parent.mkdir(parents=True, exist_ok=True)
                batch.write_text(json.dumps({
                    "version": "affine-batch-v3", "hotkey": "synthetic-hotkey",
                    "groups": [{"prompt": "synthetic", "rollouts": [
                        {"output": "a", "reward": 1.0}, {"output": "b", "reward": 0.0}]}]}))
                lora = batch.parent / "lora"
                lora.mkdir(exist_ok=True)
                header = json.dumps({
                    "layer.lora_A.weight": {"dtype": "F32", "shape": [1, 1], "data_offsets": [0, 4]},
                    "layer.lora_B.weight": {"dtype": "F32", "shape": [1, 1], "data_offsets": [4, 8]},
                }, separators=(",", ":")).encode()
                (lora / "adapter_model.safetensors").write_bytes(
                    struct.pack("<Q", len(header)) + header + struct.pack("<2f", 2.0, 2.0))
                (lora / "adapter_config.json").write_text('{"r":1,"lora_alpha":1.0}')
                (lora / "train_report.json").write_bytes(TOOL.canonical_json({
                    "seed": 1, "steps": 1, "loss": 0.0, "step_norm": 1.0,
                    "tokens": 1, "wall_s": 1.0, "gpu_class": "synthetic", "image_digest": "synthetic"}))
            command = plan["dryrun_commands"][0]
            argv = [str(root / str(Path(arg).relative_to("/operator/frozen")))
                    if arg.startswith("/operator/frozen/") else arg for arg in command["argv"]]
            argv += ["--first-seen", "100"] * 6
            argv += ["--params", str(root / "params.json")]
            (root / "params.json").write_text("{}")
            proc = subprocess.run(argv, text=True, capture_output=True)
            self.assertEqual(proc.returncode, 0, proc.stderr)
            report = json.loads(proc.stdout)
            self.assertTrue(report["read_only"])
            self.assertEqual(len(report["entries"]), 6)
            self.assertTrue(report["stage_b"]["available"])
            self.assertTrue(all(row["step_norm"] == 1.0 for row in report["entries"]))

    def test_13_results_only_provenance_and_optional_first_seen(self):
        plan = self.plan(results=False, fetch_first_seen=True, results_only=True)
        objects = {obj["key"]: self.settled[obj["key"]][0]
                   for obj in plan["objects"] if obj["kind"] == "result"}
        objects[TOOL.FIRST_SEEN_KEY] = b"{}"
        fake = FakeStore(objects)
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "results"
            receipt = TOOL.fetch_plan(plan, target, TOOL.GetOnlyStore(fake), save_receipt=False, results_only=True)
            self.assertEqual(len(receipt["fetched"]), 4)
            self.assertFalse((target / "fetch-receipt.json").exists())
            self.assertTrue((target / "first_seen.json").is_file())
            with mock.patch.object(TOOL, "result_reference", side_effect=self.reference):
                selected = TOOL.select_sealed_results(target, self.records, ROUNDS,
                                                      expected_journal_sha256=TOOL.SUPPLIED_JOURNAL_SHA256)
            self.assertEqual(len(selected), 3)
            source = json.loads((target / "source-journal.json").read_text())
            self.assertNotIn("objects", source)
            self.assertNotIn("seats", source)

    def test_12_commands_only_fetch_reproduces_paths_without_receipt(self):
        plan = self.plan(results=False)
        plan["derive_from_journal"] = True
        fake = FakeStore({})
        # Materialize journal-referenced results in the offline source directory.
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "results-only"
            source.mkdir()
            (source / "source-journal.json").write_text(json.dumps({
                "schema": TOOL.SCHEMA + "-source", "journal_sha256": TOOL.SUPPLIED_JOURNAL_SHA256,
                "rounds": list(ROUNDS)}))
            for key, (raw, _) in self.settled.items():
                _, _, round_id, lease_id, _ = key.split("/", 4)
                path = source / "results" / round_id / f"{lease_id.removesuffix('/result.json')}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
            with unittest.mock.patch.object(TOOL, "result_reference", side_effect=self.reference):
                results = TOOL.select_sealed_results(source, self.records, ROUNDS,
                                                     expected_journal_sha256=TOOL.SUPPLIED_JOURNAL_SHA256)
                generated = TOOL.make_plan(JOURNAL, str(Path(tmp) / "target"), ROUNDS, None, str(PYTHON), results)
            fake, digest_patch = self.objects_for(generated, streaming=True)
            target = Path(tmp) / "target"
            with digest_patch:
                receipt = TOOL.fetch_plan(generated, target, TOOL.GetOnlyStore(fake), save_receipt=False)
            self.assertFalse((target / "fetch-receipt.json").exists())
            self.assertTrue((target / "r000086/q000000/batch.json").is_file())
            self.assertTrue((target / "r000086/q000000/lora/train_report.json").is_file())
            self.assertEqual(len(receipt["optional_missing"]), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
