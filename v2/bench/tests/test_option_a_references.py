"""Option A registry, official reference rows, queue priority and public HF fetching."""
import types

from affine_bench import daemon as D, fetch, registry as R

OPTION_A = ("hmmt_feb_2025", "hmmt_nov_2025", "hmmt_feb_2026", "brumo_2025", "cmimc_2025", "beyondaime")
RETIRED = ("math500", "math", "aime", "aime22", "aime23", "aime24", "aime25", "aime26", "amc23", "arxivmath", "gsm8k", "gsm8k_platinum")


def test_option_a_registry_has_exact_six_counted_benches_at_k8():
    benches = R.benchmarks()
    by_id = {bench["id"]: bench for bench in benches}
    enabled = R.benchmarks(enabled_only=True, runnable_only=True)
    assert [bench["id"] for bench in enabled] == list(OPTION_A)
    assert [bench["order"] for bench in enabled] == [10, 20, 30, 40, 50, 60]
    assert all(bench["rollouts"] == 8 and bench["in_avg"] and bench["enabled"] for bench in enabled)
    assert all(bench["suite"] == "math" and bench["source"] == f"repo:bench/environments/{bench['id']}" for bench in enabled)
    assert all(bench["harness"] == "null" and bench["runtime"] == "subprocess" for bench in enabled)
    for name in RETIRED:
        assert by_id[name]["enabled"] is False and by_id[name]["in_avg"] is False and by_id[name]["hide_column"] is True


def test_official_qwen_reference_uses_pinned_public_model_and_explicit_profile():
    (reference,) = R.references()
    assert reference == dict(model_id="ref-qwen3.5-9b", label="Qwen3.5-9B", kind="reference",
                             repo="Qwen/Qwen3.5-9B", revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a",
                             profile="qwen3.5-family", reasoning_parser="qwen3", crowned_at=None)
    settings = R.resolve_settings(reference["model_id"], reference["kind"], reference["profile"])
    assert settings["profile"] == "qwen3.5-family" and settings["presence_penalty"] == 1.5
    assert R.resolve_settings("a-king", "king")["profile"] == R.model_settings()["king_profile"]


def row(name, kind, at):
    return dict(model_id=name, label=name, kind=kind, crowned_at=at, base_hash=name + "-weights")


def test_model_rows_include_official_references_and_queue_priority(monkeypatch):
    models = [row("old", "king", 100), row("genesis", "genesis", 50), row("new", "king", 200)]
    monkeypatch.setattr(D.source, "models", lambda: models)
    monkeypatch.setattr(D, "published", lambda: set())
    monkeypatch.setattr(D, "S", {"fail": {}, "unavailable": []})
    monkeypatch.setattr(R, "benchmarks", lambda **kwargs: [dict(id=name) for name in OPTION_A])
    monkeypatch.setattr(R, "bench_version", lambda bench: bench["id"] + "/v1")
    monkeypatch.setattr(D, "rj", lambda *args: None)
    rows = D.model_rows()
    assert rows[0]["model_id"] == "ref-qwen3.5-9b"
    assert [model["model_id"] for model in rows[1:]] == ["old", "genesis", "new"]
    assert [model["model_id"] for model in D.queue()] == ["ref-qwen3.5-9b", "genesis", "new", "old"]
    # Three historical rows and a new live king: the new king is ahead of old backfill, never ahead of the reference/genesis.
    models.append(row("newest", "king", 300))
    assert [model["model_id"] for model in D.queue()] == ["ref-qwen3.5-9b", "genesis", "newest", "new", "old"]


def test_reference_fetch_is_public_pinned_hf_snapshot_with_no_credential(monkeypatch, tmp_path):
    node = types.SimpleNamespace(commands=[], run=lambda cmd, timeout: None)
    def run(cmd, timeout):
        node.commands.append(cmd)
        return types.SimpleNamespace(returncode=0, stdout="snapshot complete", stderr="")
    node.run = run
    reference = R.references()[0]
    result = fetch.fetch_reference(reference, node)
    (command,) = node.commands
    assert "snapshot_download" in command and "Qwen/Qwen3.5-9B" in command
    assert reference["revision"] in command and "token=False" in command and ".verified" in command
    assert "huggingface_hub" in command and "venv-vllm/bin/python" in command
    assert f"/{reference['model_id']}" in command
    assert not any(word in command.upper() for word in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "AWS_SECRET", "SN120_R2", "PAT="))
    assert result["status"] == "rc0"


def test_king_fetch_preserves_expiring_links_on_stdin_only(monkeypatch):
    class Store:
        def presign_get(self, key, ttl_s): return "https://signed.example/" + key
    monkeypatch.setattr(fetch.source, "store", lambda: Store())
    monkeypatch.setattr(fetch.source, "_served", {"base": types.SimpleNamespace(base_files={"model.bin": "abc"})})
    node = types.SimpleNamespace(commands=[], stdin=[])
    def run(cmd, inp, timeout):
        node.commands.append(cmd); node.stdin.append(inp)
        return types.SimpleNamespace(returncode=0, stdout="verified", stderr="")
    node.run = run
    result = fetch.fetch("base", node)
    assert "gate_fetch.py" in node.commands[0] and "signed.example" not in node.commands[0]
    assert "signed.example" in node.stdin[0] and result["status"] == "rc0"


def test_reference_uses_explicit_profile_and_its_own_model_dir(monkeypatch):
    reference = R.references()[0]
    seen = []
    monkeypatch.setattr(R, "resolve_settings", lambda *args: (seen.append(args), dict(max_model_len=65536, hf_overrides=""))[1])
    monkeypatch.setattr(D.NODE, "run", lambda cmd, timeout: types.SimpleNamespace(stdout="123\n"))
    monkeypatch.setattr(D, "save", lambda: None)
    monkeypatch.setattr(D, "log", lambda **kwargs: None)
    D.launch(reference, list(OPTION_A))
    assert seen == [("ref-qwen3.5-9b", "reference", "qwen3.5-family")]
    assert D.model_dir(reference).endswith("/ref-qwen3.5-9b")
    assert D.S["running"]["model"]["model_id"] == "ref-qwen3.5-9b"
