"""Offline smoke test for the affine validator (no GPU, no chain, no network).

Exercises everything the 2026-08-03 review-fix pass touched:
config typing, duel floors, state locking/no-drop invariants, hygiene caps,
copy-detection multisets, SCALE decode, tail reads, lium price parsing,
and the eval-client error taxonomy. Run:

    .venv-smoke/bin/python scripts/smoke_test.py
"""

from __future__ import annotations

import datetime
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PASS = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global PASS
    if not cond:
        print(f"FAIL {name} {detail}")
        sys.exit(1)
    PASS += 1
    print(f"ok   {name}")


# -- config -----------------------------------------------------------------
from affine.config import load_config  # noqa: E402

cfg = load_config()
check("config.typed_duel",
      cfg.duel.k_sigma == 2.0 and cfg.duel.n_turns == 2080
      and cfg.duel.n_teacher_samples == 1 and cfg.duel.n_miner_samples == 1
      and cfg.duel.reason_only is True and cfg.duel.score_bank is False)
check("config.v2_floors_gone",
      not hasattr(cfg.duel, "min_se") and not hasattr(cfg.duel, "min_margin"))
check("config.v3_fork_key", cfg.weight_version_key >= 3)
check("config.submission_caps",
      cfg.submission.max_total_repo_gb > cfg.submission.max_model_size_gb)
check("config.min_submission_block_set", cfg.min_submission_block >= 0)
check("config.corpus_base_url", cfg.dataset.corpus_base_url.startswith("https://"))
check("config.manifest_key", cfg.dataset.manifest_key.endswith(".json"))
check("config.corpus_refresh_interval", cfg.dataset.refresh_interval_s > 0)
check("config.validator_new_knobs",
      cfg.validator.metagraph_max_age_s > 0 and cfg.validator.jobs_retention > 0)
check("config.max_infra_front_requeues",
      cfg.validator.max_infra_front_requeues > 0)
check("config.weight_version", cfg.weight_version_key >= 1)
check("config.raw_still_dict", isinstance(cfg.raw["duel"], dict))
check("config.bench_machine_port",
      cfg.bench_machine.port != cfg.eval_machine.port
      and cfg.bench_machine.port > 0)
check("config.bench_machine_smaller",
      cfg.bench_machine.gpu_count <= cfg.eval_machine.gpu_count)
check("config.bench_suite_swe",
      "swe_rebench_lite" in cfg.bench.suites)
swe_ids = (Path(__file__).resolve().parents[1] / "evalsrv" / "data"
           / "swe_rebench_lite_ids.json")
pin = json.loads(swe_ids.read_text())
check("swe_lite_pin_present",
      swe_ids.is_file() and len(pin.get("instance_ids", [])) >= 10)

# -- score: Reason v3 duel (purely relative, k_sigma-only) --------------------
from affine import score  # noqa: E402


def rows_for(miner: str, per_turn: list[float], lift: float = 1.0,
             teacher_own: float = 1.0) -> list[dict]:
    out = []
    for i, s in enumerate(per_turn):
        pair = {"lpA_ya_za": lift, "lpA_ya_e": 0.0,
                "lpC_yc_za": s, "lpC_yc_e": 0.0,
                "lpC_yc_zc": teacher_own,  # Λ2(z_C) = teacher_own − 0
                "lpA_yc_za": -2.0, "lpA_yc_e": -1.0,
                "z_a": "some genuine reasoning", "y_a": "ls -la"}
        out.append({"turn_id": f"t{i}", "miner": miner, "valid": True,
                    "pairs": [pair]})
    return out


# RT-4 copy null: identical per-turn scores → margin exactly 0 → no crown.
king = rows_for("king", [0.05 * ((i % 5) - 2) for i in range(20)])
copycat = rows_for("chall", [0.05 * ((i % 5) - 2) for i in range(20)])
r = score.duel(copycat, king)
check("duel.copy_null_no_crown", not r.challenger_wins and r.margin == 0.0,
      f"z={r.z:.3f} margin={r.margin}")

# A real margin with real variance wins.
flat_king = rows_for("king", [0.0] * 20)
chall_real = rows_for("chall", [0.5 + 0.05 * ((i % 5) - 2) for i in range(20)])
r2 = score.duel(chall_real, flat_king)
check("duel.real_margin_wins", r2.challenger_wins,
      f"margin={r2.margin:.3f} se={r2.se:.4f} z={r2.z:.1f}")

# v3 has no absolute floor: a statistically-real but tiny margin DOES crown
# (accepted risk — the 2026-08-05 noise-floor policy taken to its limit).
chall_sig = rows_for("chall", [0.008 + 0.0004 * ((i % 5) - 2) for i in range(20)])
r3 = score.duel(chall_sig, flat_king)
check("duel.tiny_real_margin_crowns", r3.challenger_wins,
      f"margin={r3.margin:.4f} z={r3.z:.1f}")

# A positive but sub-kσ margin does not crown (k_sigma default = 2).
chall_noise = rows_for("chall", [0.01 + 0.05 * ((i % 5) - 2) for i in range(20)])
r4 = score.duel(chall_noise, flat_king)
check("duel.sub_sigma_no_crown", not r4.challenger_wins and r4.z <= r4.k_sigma,
      f"margin={r4.margin:.4f} z={r4.z:.2f} k={r4.k_sigma}")

# score_miner is gateless: mean Reason + telemetry, no valid flag.
ms = score.score_miner(chall_real)
check("score.miner_reason_mean", abs(ms.reason - 0.5) < 1e-9
      and ms.mean_l1lift == -1.0 and ms.gate_pass_rate == 1.0,
      f"reason={ms.reason} l1={ms.mean_l1lift}")
check("score.miner_no_gating", not hasattr(ms, "valid"))
# η = Reason / Λ2(z_C); teacher_own=1.0 ⇒ mean η ≈ mean Reason.
check("score.miner_mean_eta", ms.mean_eta is not None
      and abs(ms.mean_eta - ms.reason) < 1e-9,
      f"mean_eta={ms.mean_eta} reason={ms.reason}")
# Undefined when teacher own-lift is ~0.
no_own = rows_for("x", [0.5], teacher_own=0.0)
check("score.eta_undefined_zero_denom",
      score.eta(no_own[0]["pairs"][0]) is None
      and score.score_miner(no_own).mean_eta is None)
# Reason-only pairs (no lpA): score still works; lpA telemetry is None.
ro_pair = {"lpC_yc_za": 0.4, "lpC_yc_e": 0.0, "lpC_yc_zc": 0.8,
           "z_a": "thought", "y_a": "ls"}
ro_rows = [{"turn_id": "t0", "miner": "ro", "valid": True, "pairs": [ro_pair]}]
ro = score.score_miner(ro_rows)
check("score.reason_only_ok",
      abs(ro.reason - 0.4) < 1e-9 and ro.mean_l1lift is None
      and ro.calib_ratio is None and ro.baseline_abs is None
      and score.gate_pass(ro_pair) is None,
      f"reason={ro.reason} l1={ro.mean_l1lift}")

# -- state: locking, pop/push, no-count requeue, dead code gone ---------------
from affine.state import QueueEntry, State  # noqa: E402

with tempfile.TemporaryDirectory() as td:
    st = State(Path(td))
    st.load()
    check("state.failed_models_removed", not hasattr(st, "failed_models"))
    e = st.enqueue("hk1", "user/Affine-x", "a" * 40, block=100,
                   min_submission_block=1)
    check("state.enqueue", e is not None and st.queue and
          "hk1" in st.seen_hotkeys)
    got = st.pop_next()
    check("state.pop_next", got is e and not st.queue)
    st.push_front(got)
    check("state.push_front", st.queue[0] is got)
    got = st.pop_next()
    n = st.requeue_front(got, "machine down", count_retry=False)
    check("state.requeue_no_count", n == 0 and st.queue[0].retry_count == 0)
    got = st.pop_next()
    n = st.requeue_front(got, "flaky stream", count_retry=True)
    check("state.requeue_counted", n == 1)
    # duplicate revision burns the slot of the second hotkey
    dup = st.enqueue("hk2", "user2/Affine-y", "a" * 40, block=101,
                     min_submission_block=1)
    check("state.revision_dedup_burns_slot", dup is None and "hk2" in st.seen_hotkeys)
    st.set_king("hk9", "u/Affine-king", "b" * 40, 102, "chal-x")
    st.flush()
    # crash-restart reconcile
    st2 = State(Path(td))
    st2.load()
    check("state.reload_king", st2.king is not None and st2.king.hotkey == "hk9")

    # requeue_back rotates a wedging entry to the tail so others progress.
    st3 = State(Path(td) )
    a = st3.enqueue("hkA", "u/Affine-a", "a" * 40, 200, min_submission_block=1)
    b = st3.enqueue("hkB", "u/Affine-b", "b" * 40, 201, min_submission_block=1)
    head = st3.pop_next()
    st3.requeue_back(head, "pod too small")
    check("state.requeue_back_defers_head",
          head is a and st3.queue[-1] is a and st3.queue[0] is b)

    # revert_king promotes the previous king, drops the dead one, and bumps
    # the reign monotonically (dead reign 1 -> restored reign 2).
    st4 = State(Path(td) / "sub")
    st4.set_king("hkK1", "u/Affine-k1", "1" * 40, 300, "c1")
    st4.set_king("hkK2", "u/Affine-k2", "2" * 40, 301, "c2")
    restored = st4.revert_king("k2 repo 404")
    check("state.revert_king_promotes_prev",
          restored is not None and st4.king.hotkey == "hkK1"
          and st4.king.reign_number == 2
          and "hkK2" not in st4.king_chain_hotkeys(5))
    # P0 durability: a restart must NOT resurrect the reverted-away dead king
    # via history reconciliation.
    st4b = State(Path(td) / "sub")
    st4b.load()
    check("state.revert_king_durable_across_restart",
          st4b.king is not None and st4b.king.hotkey == "hkK1"
          and "hkK2" not in st4b.king_chain_hotkeys(5))
    # genesis with no prior king → no fallback.
    st5 = State(Path(td) / "sub2")
    st5.set_king("hkGen", "u/Affine-gen", "9" * 40, 400, "seed")
    check("state.revert_king_no_prior", st5.revert_king("gen dead") is None)

# -- model_store: hygiene caps + multiset copy detection ----------------------
from affine import model_store  # noqa: E402
from affine.model_store import ModelRef, RepoInfo  # noqa: E402


def info(files, blobs, st_bytes, total_bytes, config=None, ts=None):
    return RepoInfo(files=files, config=config or {"architectures": ["X"]},
                    safetensors_blobs=blobs,
                    total_safetensors_bytes=st_bytes,
                    total_repo_bytes=total_bytes, committed_at=ts)


base_files = ["config.json", "model.safetensors"]
ok_info = info(base_files, {"model.safetensors": "d1"}, int(1e9), int(2e9))
check("hygiene.ok", model_store.validate_repo_hygiene(
    ok_info, max_size_gb=90, max_total_repo_gb=100,
    allow_python_files=False, allow_auto_map=False) is None)

# safetensors under cap but repo stuffed with 500 GB of junk → rejected.
junk = info(base_files + ["junk.bin"], {"model.safetensors": "d1"},
            int(1e9), int(500e9))
reason = model_store.validate_repo_hygiene(
    junk, max_size_gb=90, max_total_repo_gb=100,
    allow_python_files=False, allow_auto_map=False)
check("hygiene.total_repo_cap", reason is not None and "total repo" in reason)

many = info(base_files + [f"f{i}.txt" for i in range(6000)],
            {"model.safetensors": "d1"}, int(1e9), int(2e9))
check("hygiene.file_count_cap", model_store.validate_repo_hygiene(
    many, max_size_gb=90, max_total_repo_gb=100,
    allow_python_files=False, allow_auto_map=False) is not None)

big_cfg = info(base_files, {"model.safetensors": "d1"}, int(1e9), int(2e9),
               config={"architectures": ["X"], "pad": "x" * (2 << 20)})
check("hygiene.config_size_cap", model_store.validate_repo_hygiene(
    big_cfg, max_size_gb=90, max_total_repo_gb=100,
    allow_python_files=False, allow_auto_map=False) is not None)

# copy detection: rename + junk-pad must still flag
t0 = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)
t1 = datetime.datetime(2026, 2, 1, tzinfo=datetime.timezone.utc)
king_info = info(["model-00001-of-00002.safetensors",
                  "model-00002-of-00002.safetensors"],
                 {"model-00001-of-00002.safetensors": "aaa",
                  "model-00002-of-00002.safetensors": "bbb"},
                 int(2e9), int(2e9), ts=t0)
thief = info(["renamed-1.safetensors", "renamed-2.safetensors",
              "pad.safetensors"],
             {"renamed-1.safetensors": "aaa", "renamed-2.safetensors": "bbb",
              "pad.safetensors": "zzz"}, int(3e9), int(3e9), ts=t1)
cv = model_store.check_model_copy(
    ModelRef("u/Affine-thief", "c" * 40), thief,
    ModelRef("u/Affine-king", "d" * 40), king_info)
check("copy.multiset_rename_pad_flagged",
      cv is not None and cv.action == "reject")

original = info(["w.safetensors"], {"w.safetensors": "aaa"}, int(1e9),
                int(1e9), ts=t0)
king2 = info(["x.safetensors"], {"x.safetensors": "aaa"}, int(1e9),
             int(1e9), ts=t1)
cv2 = model_store.check_model_copy(
    ModelRef("u/Affine-orig", "e" * 40), original,
    ModelRef("u/Affine-king", "f" * 40), king2)
check("copy.crown_earlier", cv2 is not None and cv2.action == "crown_earlier")

different = info(["w.safetensors"], {"w.safetensors": "qqq"}, int(1e9),
                 int(1e9), ts=t1)
check("copy.different_not_flagged", model_store.check_model_copy(
    ModelRef("u/Affine-a", "1" * 40), different,
    ModelRef("u/Affine-king", "2" * 40), king2) is None)

# -- chain: reveal roundtrip, SCALE modes, fail-closed hash, staleness --------
from affine import chain  # noqa: E402

payload = chain.build_reveal("affine1", "user/Affine-m", "a" * 40,
                             "5" + "F" * 45)
check("chain.reveal_roundtrip",
      chain.parse_reveal("affine1", payload)[0] == "user/Affine-m")

short = b"hello|world"  # mode 0 covers len < 64
scale0 = bytes([len(short) << 2]) + short
_, out = chain._decode_commitment_pair(("hotkey1", [("0x" + scale0.hex(), 7)]))
check("chain.scale_mode0", out[0] == (7, short.decode()))
long_msg = b"affine1|u/Affine-m|" + b"a" * 40 + b"|" + b"5" + b"F" * 45
scale1 = int.to_bytes((len(long_msg) << 2) | 1, 2, "little") + long_msg
_, out1 = chain._decode_commitment_pair(("hk", [("0x" + scale1.hex(), 9)]))
check("chain.scale_mode1", out1[0][1] == long_msg.decode())


class _DeadSub:
    def get_block_hash(self, b):
        raise RuntimeError("rpc down")


try:
    chain.block_hash_at(_DeadSub(), 5)
    check("chain.blockhash_fail_closed", False)
except chain.BlockHashUnavailable:
    check("chain.blockhash_fail_closed", True)

mg = chain.Metagraph()
check("chain.metagraph_stale_refuses", chain.set_rolling_weights(
    None, None, 120, ["hk"], mg, 0, max_metagraph_age_s=10) is False)

# -- dashboard tail reads ------------------------------------------------------
from affine.dashboard import Dashboard  # noqa: E402

with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "h.jsonl"
    with open(p, "w") as f:
        for i in range(5000):
            f.write(json.dumps({"i": i}) + "\n")
    rows = Dashboard._tail_jsonl(None, p, 100)
    check("dashboard.tail_last_n",
          len(rows) == 100 and rows[-1]["i"] == 4999 and rows[0]["i"] == 4900)
    small = Path(td) / "s.jsonl"
    small.write_text(json.dumps({"i": 1}) + "\n")
    check("dashboard.tail_small_file",
          Dashboard._tail_jsonl(None, small, 100) == [{"i": 1}])

# -- provisioner: price parsing + redaction ------------------------------------
from affine import provisioner  # noqa: E402

ls_json = json.dumps([
    {"huid": "brave-fox-3a", "price_per_hour": 20.0, "gpu_type": "B300"},
    {"huid": "calm-owl-9b", "price_per_hour": 48.0, "gpu_type": "B300"},
    {"huid": "wise-elk-1c", "price_per_hour": 24.0, "gpu_type": "B300"},
])
ident, price = provisioner._parse_cheapest_lium(ls_json, cap=40.0)
check("provisioner.cheapest_under_cap",
      price == 20.0 and ident == "brave-fox-3a")
ident2, _ = provisioner._parse_cheapest_lium(ls_json, cap=10.0)
check("provisioner.cap_refuses_all", ident2 is None)
# Table scrape must never return Config labels like '8×H200'.
ls_table = """
 index  huid              gpu      price_gpu  price_total
 1      brave-fox-3a      8xH200   $2.50      $20.00
 2      calm-owl-9b       8xH200   $6.00      $48.00
"""
ident3, price3 = provisioner._parse_cheapest_lium(ls_table, cap=40.0)
check("provisioner.table_uses_huid",
      price3 == 20.0 and ident3 == "brave-fox-3a")

provisioner.register_secret("sekrit-token")
check("provisioner.redaction",
      "sekrit" not in provisioner._redact("Bearer sekrit-token here"))

# -- eval client error taxonomy -------------------------------------------------
from affine.eval_client import (INFRA_FAULT_CODES, EvalBusyError,  # noqa: E402
                                Fault, InfraFaultError, TransientEvalError)

check("eval_client.busy_is_transient", issubclass(EvalBusyError, TransientEvalError))
check("eval_client.infra_is_transient", issubclass(InfraFaultError, TransientEvalError))
check("eval_client.fault_codes_are_infra",
      {Fault.TEACHER, Fault.KING_LAUNCH, Fault.POD_CAPACITY,
       Fault.CHALLENGER_INFRA, Fault.CONTEXT_LIMIT} <= INFRA_FAULT_CODES)

# -- dueling helpers (pure parts) ----------------------------------------------
sys.modules.setdefault("httpx_missing_guard", None)
from evalsrv.dueling import duel_seed, sample_slice  # noqa: E402

rows = [{"traj_id": f"repoA.func_basic_{i}", "turn_idx": i, "prefix": []}
        for i in range(50)]
s1 = sample_slice(rows, 10, duel_seed("0xabc", "hk1"))
s2 = sample_slice(rows, 10, duel_seed("0xabc", "hk1"))
s3 = sample_slice(rows, 10, duel_seed("0xabc", "hk2"))
check("dueling.slice_deterministic",
      [r["turn_idx"] for r in s1] == [r["turn_idx"] for r in s2])
check("dueling.slice_hotkey_dependent",
      [r["turn_idx"] for r in s1] != [r["turn_idx"] for r in s3])

print(f"\nALL {PASS} CHECKS PASSED")
