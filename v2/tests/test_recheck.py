"""Offline re-check: read-only evidence, exact paired scoring and private parent links.

All tests use the CPU simulation and local files. None needs a GPU, R2 or the public site.
"""

from __future__ import annotations

import base64
import contextlib
import dataclasses
import functools
import http.server
import json
import sys
import threading
import types
from pathlib import Path

import pytest

from sn120_v2 import layout, recheck
from sn120_v2.canonical import canonical_json, sha256_hex
from sn120_v2.envs.math_pool import MathPool, Task
from sn120_v2.errors import ConfigError, InfraError, VerifyError
from sn120_v2.journal import JOURNAL_FILE
from sn120_v2.judging import TIE_TOLERANCE, score_confirmation
from sn120_v2.sampling import rollout_seed
from sn120_v2.sim import SimEvaluator, SimWorld, sim_params


@pytest.fixture
def headfold_without_torch(monkeypatch):
    """Load the real downloader without importing its unused GPU fold stack."""
    import sn120_v2.gpu.fold_input  # this module has no torch import
    monkeypatch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    monkeypatch.setitem(sys.modules, "sn120_v2.gpu.folder", types.SimpleNamespace(fold_files=None, read_lora=None))
    monkeypatch.setitem(sys.modules, "sn120_v2.fold_kernel", types.SimpleNamespace(Entry=object, read_index=None, read_header=None))
    import sn120_v2.headfold as headfold
    return headfold


def infra_failed(root: Path) -> tuple[SimWorld, object, object]:
    """A real crown with a public affine batch, then an infra-failed confirmation."""
    params = sim_params(confirm_next_round=True, n_tasks=8, min_scored_tasks=2, looks=(1.0,), batch_cap=1,
                        null_samples_per_task=2, z_bar=0.0)
    world = SimWorld(root, params)
    world.register(738, 0.3)
    # Store exactly the original bytes before admission freezes them. Preserve
    # the simulation's quality field so its trainer still builds the crown.
    world.admission.upload("hk738", 0.3)
    original = json.loads(world.store.get(layout.miner_object("hk738")))
    training = {"version": "affine-batch-v3", "hotkey": "hk738", "groups": [
        {"prompt": f"problem {i}", "rollouts": [{"output": f"trained answer {i}", "reward": 1.0}]}
        for i in range(12)
    ]}
    raw = canonical_json(original | training)
    world.store.put(layout.miner_object("hk738"), raw)
    crown = world.round()
    assert crown.outcome == "applied"
    assert sha256_hex(world.store.get(layout.winner_batch(crown.round_id))) == sha256_hex(raw)
    # Keep the tiny fixture's mapping deterministic without changing the file
    # or its public hash. The real mapper is covered below by the same groups.
    real = world.controller._training_task_exclusions

    def exclusions(parent):
        if parent is not None:
            return frozenset(f"sim:{i}" for i in range(12))
        return real(parent)

    world.controller._training_task_exclusions = exclusions
    world.controller.prompt_pool = MathPool(
        tuple(Task(f"sim:{i}", "affine_math", f"problem {i}", "1") for i in range(400)), "test")

    def broken(ticket, deadline=0):
        raise InfraError("parent base missing")

    world.evaluator.score_parent = broken
    world.register(922, -1.0)
    confirmation = world.round()
    assert confirmation.outcome == "null" and not confirmation.confirmation.confirmed
    assert "parent base missing" in confirmation.confirmation.reason
    return world, crown, confirmation


def exported(tmp_path: Path) -> tuple[SimWorld, object, object, dict, Path]:
    world, crown, confirmation = infra_failed(tmp_path / "state")
    bundle_path = tmp_path / "bundle.json"
    bundle = recheck.export_bundle(world.root, confirmation.round_id, bundle_path)
    assert json.loads(bundle_path.read_bytes()) == bundle
    return world, crown, confirmation, bundle, bundle_path


@contextlib.contextmanager
def private_base_server(root: Path):
    """Serve a test object mirror over HTTP; urllib's file handler rejects query strings."""
    class Quiet(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass  # a test URL may carry a fake signed query
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Quiet, directory=str(root)))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_export_is_deterministic_sealed_and_never_writes_state(tmp_path):
    world, crown, confirmation, bundle, path = exported(tmp_path)
    journal = world.root / JOURNAL_FILE
    before = journal.read_bytes()
    lock = world.root / "journal.lock"
    lock.write_bytes(b"leave this byte alone")
    locked_before = lock.read_bytes()
    locked_stat = lock.stat()
    again = recheck.export_bundle(world.root, confirmation.round_id, tmp_path / "second.json")
    assert again == bundle
    assert (tmp_path / "second.json").read_bytes() == path.read_bytes()
    assert journal.read_bytes() == before and lock.read_bytes() == locked_before
    assert lock.stat().st_mtime_ns == locked_stat.st_mtime_ns
    assert lock.stat().st_ctime_ns == locked_stat.st_ctime_ns
    body = bundle["body"]
    assert body["round_id"] == confirmation.round_id
    assert body["pending_crown"]["crown_round_id"] == crown.round_id
    assert body["pending_crown"]["uid"] == 738
    assert body["exclusions"]["excluded_training_prompts"] == 12
    training = body["exclusions"]["winner_batch"]
    raw = world.store.get(layout.winner_batch(crown.round_id))
    assert training["available"] and training["sha256"] == sha256_hex(raw)
    assert len(training["groups"]) == 12
    assert training["groups"][0] == {"prompt": "problem 0", "rollouts": [{"output": "trained answer 0", "reward": 1.0}]}
    assert training["file"]["quality"] == 0.3  # original extra fields remain sealed too
    assert base64.b64decode(training["file_json_b64"]) == raw
    mapped = recheck._training_evidence(world.store, recheck.load_bundle(path)[1].parent.pending)
    assert mapped == training
    evidence = body["sealed_evidence"]
    assert evidence["parent_error"] and not evidence.get("parent_outcomes")
    assert "parent base missing" in body["confirmation_resolved"]["reason"]
    assert len(body["null_samples"]) == 2
    assert all(len(text) == 8 for text in body["null_samples"])
    assert bundle["sha256"] == sha256_hex(canonical_json(body))
    body2, ticket, nulls = recheck.load_bundle(path)
    assert body2 == body and tuple(body["null_samples"]) == nulls
    assert ticket.draw.to_dict() == body["draw"]
    assert ticket.parent.manifest.to_dict() == body["parent_manifest"]


def test_bundle_tamper_is_rejected_even_with_a_recomputed_outer_hash(tmp_path):
    _, _, _, bundle, path = exported(tmp_path)
    body = bundle["body"]
    body["null_samples"][0] = "P" * 8
    tampered = dict(bundle, sha256=sha256_hex(canonical_json(body)))
    path.write_bytes(canonical_json(tampered))
    with pytest.raises(VerifyError, match="result bytes|bundle sha256"):
        recheck.load_bundle(path)
    _, _, _, bundle, path = exported(tmp_path / "fresh")
    body = bundle["body"]
    body["exclusions"]["winner_batch"] = {"available": True, "sha256": body["pending_crown"]["batch_sha256"],
        "file": {"version": "affine-batch-v3", "groups": [{"prompt": "original", "rollouts": [{"reward": 1.0}]}]},
        "file_json_b64": base64.b64encode(canonical_json({"version": "affine-batch-v3", "groups": [{"prompt": "original", "rollouts": [{"reward": 1.0}]}]})).decode("ascii"),
        "groups": [{"prompt": "changed", "rollouts": [{"reward": -100}]}]}
    path.write_bytes(canonical_json(dict(bundle, sha256=sha256_hex(canonical_json(body)))))
    with pytest.raises(VerifyError, match="prompts and rollouts|bundle sha256|winner batch"):
        recheck.load_bundle(path)


def test_fake_evaluator_uses_all_nulls_live_score_and_matched_seeds(tmp_path):
    world, _, confirmation, bundle, path = exported(tmp_path)
    evaluator = SimEvaluator(world.store)
    evaluator.parent_quality = -0.5
    seen = []

    def score_parent(ticket, deadline=0):
        seen.append((ticket, deadline))
        return SimEvaluator.score_parent(evaluator, ticket, deadline)

    evaluator.score_parent = score_parent
    evaluator.begin_round = lambda ticket: None
    calls = []

    def factory(body, ticket, work):
        calls.append((body, ticket, work))
        return evaluator

    verdict = recheck.score_bundle(path, tmp_path / "work", tmp_path / "verdict.json", evaluator_factory=factory,
                                  downloader=lambda ticket, work, links: None)
    assert len(seen) == 1 and seen[0][1] == 0
    ticket = seen[0][0]
    assert ticket.draw.to_dict() == bundle["body"]["draw"]
    assert ticket.params.null_samples == 2
    assert ticket.parent.manifest.base_files == bundle["body"]["parent_manifest"]["base_files"]
    assert verdict["confirmed"] and verdict["delta"] > TIE_TOLERANCE
    expected = score_confirmation(ticket.draw.weights, bundle["body"]["null_samples"], verdict["parent_outcomes"])
    assert {key: verdict[key] for key in expected} == expected
    assert verdict["null_samples"] == verdict["parent_samples"] == 2
    assert verdict["n_tasks"] == 8 and not verdict["sanity"]
    assert verdict["parent_base"]["source"] == "anonymous-public-store"
    assert "url" not in json.dumps(verdict) and "SECRET" not in json.dumps(verdict)
    assert calls[0][2] == (tmp_path / "work").resolve()


def test_exact_ties_fail_and_sanity_prefixes_all_inputs(tmp_path):
    world, _, _, bundle, path = exported(tmp_path)
    nulls = tuple(bundle["body"]["null_samples"])
    seen = []

    class Tie:
        def begin_round(self, ticket):
            seen.append(ticket)

        def score_parent(self, ticket, deadline=0):
            return tuple(text[:len(ticket.draw.task_ids)] for text in nulls)

        def close(self):
            pass

    verdict = recheck.score_bundle(path, tmp_path / "work", tmp_path / "tie.json", n_tasks=3, gpus=1,
                                  evaluator_factory=lambda body, ticket, work: Tie(), downloader=lambda *args: None)
    assert verdict["sanity"] and not verdict["confirmed"]
    assert verdict["delta"] == 0 and verdict["se"] == 0 and verdict["z"] == 0
    assert "TIE_TOLERANCE" in verdict["reason"] and "sanity prefix only" in verdict["note"]
    assert verdict["null_outcomes"] == [text[:3] for text in nulls]
    assert verdict["parent_outcomes"] == verdict["null_outcomes"]
    assert seen[0].draw.task_ids == tuple(bundle["body"]["draw"]["task_ids"][:3])
    assert seen[0].draw.weights == tuple(bundle["body"]["draw"]["weights"][:3])
    assert seen[0].draw.round_seed == bundle["body"]["draw"]["round_seed"]
    assert seen[0].params.n_gpus == 1
    assert verdict["local_overrides"]["round_n_gpus"] == 8
    assert verdict["parent_eval"] == {}


def test_parent_seed_matches_the_null_seed_for_each_sample_and_task(monkeypatch):
    # Directly exercise the live request builder: parent sample k enters the
    # same rollout_seed derivation as new-head null sample k, without a GPU.
    from sn120_v2.gpu.evaluator import VllmEvaluator
    seen = []
    monkeypatch.setattr("sn120_v2.gpu.evaluator.chat_body",
                        lambda model, messages, seed, params: seen.append(seed) or {})

    class Pool:
        def task(self, task_id):
            assert task_id == "sim:7"
            return types.SimpleNamespace(messages=lambda: [])

    evaluator = VllmEvaluator.__new__(VllmEvaluator)
    evaluator.pool = Pool()
    evaluator.http = lambda *args: (_ for _ in ()).throw(OSError("no server in this unit test"))
    evaluator.capture = None
    evaluator.settings = types.SimpleNamespace(retries=0)
    request = types.SimpleNamespace(round_seed="round-seed", params=sim_params(), deadline=0, task_ids=("sim:7",))
    for sample in range(2):
        assert evaluator._rollout(request, "http://127.0.0.1", ("parent", sample, 0), []) is None
    assert seen == [rollout_seed("round-seed", "sim:7", sample) for sample in range(2)]
    assert seen[0] != seen[1]


def test_legacy_crown_note_distinguishes_one_old_parent_from_two_live_nulls(tmp_path):
    # Crown under the historical one-sample pin, then move to live two-sample
    # params before confirmation. No hand-edited bundle or null string.
    crown_params = sim_params(confirm_next_round=True, n_tasks=8, min_scored_tasks=2, looks=(1.0,),
                              batch_cap=1, null_samples_per_task=1, z_bar=0.0,
                              recipe=dataclasses.replace(sim_params().recipe,
                              code_digest="ab2922c88558afd37e95938773a7e148e88c1a6ec543f96c1ce05483fd1b7941"))
    world = SimWorld(tmp_path / "state", crown_params)
    world.register(738, 0.3)
    crown = world.round()
    assert crown.outcome == "applied"
    live = dataclasses.replace(crown_params, null_samples_per_task=2,
                               recipe=dataclasses.replace(crown_params.recipe, code_digest="d46b54e9891355fef71becb3b860c0e355804cb414353a11e66966ac2454c5d8"))
    world.controller.set_params(live, world.clock())
    world.evaluator.score_parent = lambda *args, **kwargs: (_ for _ in ()).throw(InfraError("parent base missing"))
    world.register(922, -1.0)
    confirmation = world.round()
    assert confirmation.outcome == "null" and not confirmation.confirmation.confirmed
    path = tmp_path / "bundle.json"
    bundle = recheck.export_bundle(world.root, confirmation.round_id, path)
    body = bundle["body"]
    assert body["crown_design"]["parent_samples"] == 1
    assert body["crown_design"]["note"] == recheck.LEGACY_NOTE
    _, ticket, nulls = recheck.load_bundle(path)
    assert len(nulls) == 2
    verdict = recheck.score_bundle(path, tmp_path / "work", tmp_path / "legacy.json",
                                  evaluator_factory=lambda *args: _Fixed(nulls), downloader=lambda *args: None)
    assert verdict["note"] == recheck.LEGACY_NOTE
    assert verdict["null_samples"] == verdict["parent_samples"] == 2
    assert verdict["code_identity_matches"]
    assert verdict["confirmation_code_digest"] == live.recipe.code_digest


class _Fixed:
    def __init__(self, outcomes):
        self.outcomes = tuple(outcomes)
        self.closed = False

    def begin_round(self, ticket):
        pass

    def score_parent(self, ticket, deadline=0):
        return self.outcomes

    def close(self):
        self.closed = True


def test_links_are_private_verified_and_download_without_credentials(tmp_path, monkeypatch, capsys, headfold_without_torch):
    world, crown, confirmation, bundle, bundle_path = exported(tmp_path)
    _, ticket, _ = recheck.load_bundle(bundle_path)
    manifest = ticket.parent.manifest
    head = manifest.head
    # The simulated genesis base is already at the exact content-addressed public key.
    for name, sha in manifest.base_files.items():
        key = layout.public_base_file(head.base_hash, name)
        world.store.put(key, b"sim base")
        assert sha256_hex(world.store.get(key)) == sha
    secret = "signed-query-must-never-be-printed"

    class Store:
        def __init__(self, base_url):
            self.local, self.base_url = world.store, base_url

        def size(self, key):
            return self.local.size(key)

        def presign_get(self, key, ttl_s):
            return f"{self.base_url}/{key}?signature={secret}"

    out = tmp_path / "links.json"
    with private_base_server(world.store.root) as base_url:
        document = recheck.issue_links(world.root, confirmation.round_id, out, 21600, store=Store(base_url), now=lambda: 2_000_000_000)
        assert out.stat().st_mode & 0o777 == 0o600
        assert document["schema"] == recheck.LINKS_SCHEMA and document["expires_at"] == 2_000_021_600
        assert document["master"] is None and document["loras"] == []
        assert [row["name"] for row in document["base"]["files"]] == sorted(manifest.base_files)
        # No presigned URL can appear in the bundle, verdict or terminal output.
        assert secret not in bundle_path.read_text()
        verdict = recheck.score_bundle(bundle_path, tmp_path / "work", tmp_path / "verdict.json",
                                      links_path=out, evaluator_factory=lambda *args: _Fixed(bundle["body"]["null_samples"]),
                                      downloader=lambda ticket, work, links: recheck._fetch_parent(ticket, work, links))
        monkeypatch.setattr(recheck, "_default_evaluator", lambda *args: _Fixed(bundle["body"]["null_samples"]))
        assert recheck.main(["score", "--bundle", str(bundle_path), "--work-dir", str(tmp_path / "work"),
                             "--links", str(out), "--out", str(tmp_path / "cli-verdict.json")]) == 0
        base = tmp_path / "work" / "cache" / head.base_hash
        assert base.is_dir()
        from sn120_v2.canonical import file_sha256
        assert {name: file_sha256(base / name) for name in manifest.base_files} == manifest.base_files
        assert verdict["parent_base"]["files_verified"] == len(manifest.base_files)
        assert verdict["parent_base"]["source"] == "signed-links"
        assert secret not in (tmp_path / "verdict.json").read_text()
    captured = capsys.readouterr()
    assert secret not in captured.out and secret not in captured.err


def test_links_reject_wrong_parent_expiry_and_size_and_reuse_verified_cache(tmp_path, monkeypatch, headfold_without_torch):
    world, _, confirmation, bundle, path = exported(tmp_path)
    _, ticket, _ = recheck.load_bundle(path)
    manifest = ticket.parent.manifest
    for name in manifest.base_files:
        world.store.put(layout.public_base_file(manifest.head.base_hash, name), b"sim base")
    links_path = tmp_path / "links.json"
    issued = recheck.issue_links(world.root, confirmation.round_id, links_path,
                                store=recheck.SigningStore(world.store.root), now=lambda: 2_000_000_000)
    changed = json.loads(links_path.read_bytes())
    changed["head_version_id"] = "another-parent"
    links_path.write_bytes(canonical_json(changed))
    with pytest.raises(VerifyError, match="not for this bundle"):
        recheck._fetch_parent(ticket, tmp_path / "wrong", links_path)
    links_path.write_bytes(canonical_json(issued | {"expires_at": 1}))
    with pytest.raises(ConfigError, match="expired"):
        recheck._fetch_parent(ticket, tmp_path / "expired", links_path)
    changed = json.loads(json.dumps(issued))
    changed["base"]["files"][0]["size"] += 1
    links_path.write_bytes(canonical_json(changed))
    with pytest.raises(VerifyError, match="bytes|size"):
        recheck._fetch_parent(ticket, tmp_path / "size", links_path)
    links_path.write_bytes(canonical_json(issued))
    monkeypatch.setattr(headfold_without_torch.time, "time", lambda: 2_000_021_599)
    base = recheck._fetch_parent(ticket, tmp_path / "cached", links_path)
    # Existing files are reverified without opening the signed URLs; the
    # content is also checked by resident_parent_base inside live scoring.
    monkeypatch.setattr(headfold_without_torch, "download", lambda *args: pytest.fail("valid cache used a signed URL"))
    assert recheck._fetch_parent(ticket, tmp_path / "cached", links_path) == base


def test_links_missing_names_and_cli_secret_containment(tmp_path, monkeypatch, capsys):
    world, _, confirmation, _, _ = exported(tmp_path)
    manifest = recheck.load_bundle(tmp_path / "bundle.json")[1].parent.manifest
    for name, sha in manifest.base_files.items():
        world.store.put(layout.public_base_file(manifest.head.base_hash, name), b"sim base")
        assert sha256_hex(b"sim base") == sha
    secret = "credential-and-presigned-url-secret"

    class Missing:
        def size(self, key):
            raise KeyError(secret)

        def presign_get(self, key, ttl_s):
            raise AssertionError("a missing object must not be signed")

    with pytest.raises(ConfigError) as error:
        recheck.issue_links(world.root, confirmation.round_id, tmp_path / "links.json", store=Missing())
    assert "model.bin" in str(error.value) and secret not in str(error.value)
    monkeypatch.setattr(recheck, "_evidence_store", lambda state_dir, signing=False: recheck.SigningStore(Path(state_dir) / "store"))
    assert recheck.main(["links", "--state-dir", str(world.root), "--round", confirmation.round_id,
                         "--out", str(tmp_path / "private.json")]) == 0
    printed = capsys.readouterr()
    assert "http" not in printed.out and "signature" not in printed.out
    assert (tmp_path / "private.json").stat().st_mode & 0o777 == 0o600
    manifest = recheck.load_bundle(tmp_path / "bundle.json")[1].parent.manifest
    for name in manifest.base_files:
        world.store.delete(layout.public_base_file(manifest.head.base_hash, name))
    assert recheck.main(["links", "--state-dir", str(world.root), "--round", confirmation.round_id,
                         "--out", str(tmp_path / "absent.json")]) == 1
    printed = capsys.readouterr()
    assert "model.bin" in printed.err and "http" not in printed.err and secret not in printed.err
    # A signer error never overwrites a previously usable private document.
    for name in manifest.base_files:
        world.store.put(layout.public_base_file(manifest.head.base_hash, name), b"sim base")
    existing = tmp_path / "private.json"
    before = existing.read_bytes()
    class BrokenSigner:
        def size(self, key):
            return 8
        def presign_get(self, key, ttl_s):
            raise ConfigError("signed URL credential-and-presigned-url-secret")
    with pytest.raises(ConfigError):
        recheck.issue_links(world.root, confirmation.round_id, existing, store=BrokenSigner())
    assert existing.read_bytes() == before
    assert recheck.main(["links", "--state-dir", str(world.root), "--round", confirmation.round_id,
                         "--out", str(existing)]) == 0
    printed = capsys.readouterr()
    assert "http" not in printed.out and "signature" not in printed.out and secret not in printed.out


def test_default_evaluator_prepares_historical_base_and_live_solo_budget(tmp_path, monkeypatch):
    _, _, _, _, path = exported(tmp_path)
    body, ticket, _ = recheck.load_bundle(path)
    calls = []

    class Prepared:
        def __call__(self, base, head):
            calls.append((base, head))
            return base

    class Evaluator:
        def __init__(self, store, config, pool, grader, *, fold_input):
            self.config, self.fold_input = config, fold_input
            self.resident = types.SimpleNamespace(tree=lambda prefix, identity: calls.append(("unprepared", identity)))
            self._solo_clients = None

        def clients_for(self, params):
            return params.max_num_seqs

        def solo_client_budget(self, n_gpus, max_clients):
            calls.append(("budget", n_gpus, max_clients))
            return 37

    monkeypatch.setitem(sys.modules, "sn120_v2.envs.math_grader", types.SimpleNamespace(GradePool=object))
    monkeypatch.setattr("sn120_v2.envs.math_pool.pool_for_draw", lambda ids: calls.append(("pool", ids)) or "pool")
    monkeypatch.setattr("sn120_v2.gpu.evaluator.VllmEvaluator", Evaluator)
    prepared_path = tmp_path / "work" / "fold-inputs" / ticket.parent.manifest.head.base_hash
    prepared_path.mkdir(parents=True)
    monkeypatch.setattr("sn120_v2.gpu.fold_input.FoldInputs", lambda work, genesis: lambda base, head: Prepared()(base, head) and prepared_path)
    evaluator = recheck._default_evaluator(body, ticket, tmp_path / "work")
    assert calls[0][0] == "pool" and calls[0][1] == ticket.draw.task_ids
    assert calls[1] == (tmp_path / "work" / "cache" / ticket.parent.manifest.head.base_hash, ticket.parent.manifest.head)
    assert calls[2] == ("budget", ticket.params.n_gpus, ticket.params.max_num_seqs)
    assert evaluator._solo_clients == 37
    # This is the exact live score_parent handoff: a supplied base_dir bypasses
    # _start_one's fold_input fallback, so the prepared path must arrive here.
    from sn120_v2.gpu.resident import resident_parent_base
    assert resident_parent_base(evaluator.resident, ticket.parent.manifest) == prepared_path
    evaluator.resident.tree("masters/other", "other")
    assert calls[-1] == ("unprepared", "other")


def test_anonymous_public_fallback_checks_hash_and_size(tmp_path, monkeypatch, headfold_without_torch):
    _, _, _, bundle, path = exported(tmp_path)
    _, ticket, _ = recheck.load_bundle(path)
    manifest = ticket.parent.manifest
    head = manifest.head
    source = recheck._PublicSource(head.base_hash, "https://public.invalid")
    requested = []

    def fake_head(request, timeout=0):
        requested.append(request.full_url)
        class Reply:
            headers = {"Content-Length": "8"}
            def __enter__(self): return self
            def __exit__(self, *args): pass
        return Reply()

    monkeypatch.setattr(recheck.urllib.request, "urlopen", fake_head)
    calls = []
    monkeypatch.setattr("sn120_v2.headfold.download", lambda url, path, sha, size: calls.append((url, path, sha, size)))
    source.get("base", head.base_hash, "model.bin", "0" * 64, tmp_path / "model.bin")
    assert len(calls) == 1 and calls[0][3] == 8 and "?" not in requested[0]
    with pytest.raises(VerifyError):
        source.get("master", head.base_hash, "model.bin", "0" * 64, tmp_path / "other")
