"""wvk-25 context-window admission rule (Jacob 2026-09-26 09:18 UTC "256k sequence length"):
validator derivation (affine/model_store.validate_repo_context), the miner mirror
(scripts/submit.py check_context_window), the config knob and the intake/history codes.

Run from a tree that has the staged code, e.g.
  cd ~/wvk25_stage && ./.venv/bin/python ops/v20/wvk25_context_tests.py [path/to/genesis/config.json]
"""
import copy, importlib.util, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE / "affine"))
from affine import model_store as ms
from affine.config import load_config

spec = importlib.util.spec_from_file_location("submit_cli", HERE / "affine/scripts/submit.py")
submit = importlib.util.module_from_spec(spec); sys.modules["submit_cli"] = submit; spec.loader.exec_module(submit)

MIN = 262144
genesis_path = sys.argv[1] if len(sys.argv) > 1 else None
if genesis_path:
    genesis = json.load(open(genesis_path))
else:
    # Shape of Qwen/Qwen3.6-35B-A3B @ 995ad96e (text_config carries the window,
    # rope_parameters.rope_type = default).
    genesis = {"architectures": ["Qwen3_5MoeForConditionalGeneration"], "model_type": "qwen3_5_moe",
               "text_config": {"model_type": "qwen3_5_moe_text", "max_position_embeddings": 262144,
                               "rope_parameters": {"rope_type": "default", "rope_theta": 10000000}}}


def with_text(cfg, **kw):
    c = copy.deepcopy(cfg); c["text_config"].update(kw); return c


def info(cfg):
    return ms.RepoInfo(files=["config.json"], config=cfg, safetensors_blobs={}, total_safetensors_bytes=0,
                       total_repo_bytes=0, committed_at=None)


text_only = {**copy.deepcopy(genesis["text_config"]), "architectures": ["Qwen3_5MoeForCausalLM"]}
cases = [
    ("genesis (262144, rope default)", genesis, True),
    ("genesis text-only extraction (root keys)", text_only, True),
    ("genesis with max_position_embeddings = 131072", with_text(genesis, max_position_embeddings=131072), False),
    ("131072 + yarn x2 (original 131072) -> 262144", with_text(genesis, max_position_embeddings=131072,
        rope_scaling={"rope_type": "yarn", "factor": 2, "original_max_position_embeddings": 131072}), True),
    ("131072 + llama3 x4 (no multiplier)", with_text(genesis, max_position_embeddings=131072,
        rope_scaling={"rope_type": "llama3", "factor": 4}), False),
    ("131072 + linear x2 -> 262144", with_text(genesis, max_position_embeddings=131072,
        rope_scaling={"type": "linear", "factor": 2.0}), True),
    ("131072 + dynamic x1.5 -> 196608", with_text(genesis, max_position_embeddings=131072,
        rope_scaling={"rope_type": "dynamic", "factor": 1.5}), False),
    ("262144 + yarn x4 in rope_parameters (original 65536 -> 262144)", with_text(genesis,
        rope_parameters={"rope_type": "yarn", "factor": 4, "original_max_position_embeddings": 65536}), True),
    ("262144 but model_max_length = 32768 (smallest key wins)", with_text(genesis, model_max_length=32768), False),
    ("no length key at all", {"architectures": ["X"], "text_config": {"hidden_size": 1}}, False),
]
fails = 0
for name, cfg, expect_pass in cases:
    reason = ms.validate_repo_context(info(cfg), MIN)
    mirror = submit.check_context_window(cfg, MIN)
    ok = (reason is None) == expect_pass and (mirror is None) == expect_pass
    fails += not ok
    print(f"{'ok ' if ok else 'BAD'} {name}: validator={reason or 'pass'} | submit.py={mirror or 'pass'}")

# knob off -> never rejects; codes
assert ms.validate_repo_context(info(with_text(genesis, max_position_embeddings=1024)), 0) is None
r = ms.validate_repo_context(info(with_text(genesis, max_position_embeddings=131072)), MIN)
assert r.startswith("context_too_short: effective window 131072 < 262144 (max_position_embeddings=131072"), r
assert ms.hygiene_fault_code(r) == "context_too_short" and ms.hygiene_history_code(r) == "rejected_context_too_short"
assert ms.hygiene_fault_code("oversized: 120 GB") == "repo_hygiene_rejected" == ms.hygiene_history_code("oversized: 120 GB")
print("ok  codes: intake rejected_context_too_short / history rejected_context_too_short; other hygiene unchanged")

c = load_config(str(HERE / "affine/affine.toml"))
print(f"ok  config: [submission].min_context_tokens = {c.submission.min_context_tokens} (0 = off until T0), "
      f"wvk {c.weight_version_key}")
assert c.submission.min_context_tokens == 0
print("FAILED" if fails else "ALL PASSED", f"({len(cases)} derivation cases)")
sys.exit(1 if fails else 0)
