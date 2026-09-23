"""Teacher swap Qwen3.8-27B -> GLM-5.3-Flash + 262k window: the contract and
ops edits, staged. NOTHING here runs without an explicit dated operator
directive naming weight_version_key (--apply DATE --wvk-to N).

What --apply changes (every anchor is asserted in its pre-flip state first):

  affine/affine.toml
    [teacher].repo                 Qwen/Qwen3.8-27B -> zai-org/GLM-5.3-Flash  (+ dated comment)
    [miner_serving].max_model_len  131072 -> 262144
    weight_version_key             N -> N+1  (+ one history paragraph)
    (asserted, not edited: score_mode = "sd_min_rga", max_thought_tokens 4096,
     ref_max_tokens 4864, thought_cap_ratio 1.25)
  ops/teacher-swarm/swarm.toml
    [swarm].model                  -> zai-org/GLM-5.3-Flash
    [swarm].vllm_version           -> VLLM_VERSION_GLM53 (0.29.x + FlashInfer >= 0.6.17)
    [swarm].max_model_len          -> 262144
    [types.*] tp / replicas        -> the GLM-5.3-Flash layout (306 GiB FP8 weights):
                                      8x180 GB (b200) / 8x141 GB (h200) / 8x96 GB (pro6000):
                                      tp 4, replicas 2; 8x80 GB (h100): tp 8, replicas 1;
                                      4-GPU boxes: tp 4, replicas 1; 1-GPU boxes: target 0
  affine/datagen/slicer.py
    MAX_PREFIX_CHARS               300_000 -> 1_000_000  (the fold's token guard binds)
  rollouts/rollouts/policies.toml
    every teacher_* endpoint       model qwen3.8-27b -> glm-5.3-flash, name engy -> engy-glm53
    [pricing.engy-glm53]           Engy list price for the ledger
  rollouts/rollouts/sources.toml
    [band_filter.defaults].teacher_models  [] -> ["qwen3.8-27b", "glm-5.3-flash"]
                                   ("both counted" transition; tighten to the new id
                                    with --band-new-only once the pre-pass covers the pools)

The website mirror affine/website/code/affine.toml is overwritten when it
exists. ops/corpus_build.py needs no edit: prefix_token_cap() reads the
window from the toml and guard_tokenizers() adds the genesis tokenizer
automatically once the teacher's vocabulary differs.

    python ops/v19/teacher_swap_toml_edits.py --preview          # writes ops/v19/teacher_swap.patch
    python ops/v19/teacher_swap_toml_edits.py --apply 2026-09-XX --wvk-to 24
    python ops/v19/teacher_swap_toml_edits.py --band-new-only     # later: teacher_models = ["glm-5.3-flash"]
"""

from __future__ import annotations

import argparse
import difflib
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOML = REPO / "affine" / "affine.toml"
MIRROR = REPO / "affine" / "website" / "code" / "affine.toml"
SWARM = REPO / "ops" / "teacher-swarm" / "swarm.toml"
SLICER = REPO / "affine" / "datagen" / "slicer.py"
POLICIES = REPO / "rollouts" / "rollouts" / "policies.toml"
SOURCES = REPO / "rollouts" / "rollouts" / "sources.toml"
PATCH = REPO / "ops" / "v19" / "teacher_swap.patch"

NEW_TEACHER = "zai-org/GLM-5.3-Flash"
NEW_ENGY_ID = "glm-5.3-flash"
OLD_ENGY_ID = "qwen3.8-27b"
VLLM_VERSION_GLM53 = "0.29.0"
WINDOW_NEW = 262144

# -- affine.toml -----------------------------------------------------------------
TEACHER_OLD = 'repo = "Qwen/Qwen3.8-27B"\nbase_url = "http://127.0.0.1:9100/v1"\n'
TEACHER_NEW = (
    "# {date} (weight_version_key {a}→{b}, explicit dated operator directive):\n"
    "# teacher Qwen/Qwen3.8-27B → zai-org/GLM-5.3-Flash (320B MoE, 18B active,\n"
    "# FP8, MIT; TB2.1 84.3 / DeepSWE 63.4 / NL2Repo 56.3 vs the 27B's 73.0 /\n"
    "# 42.2 / 42.3). Why: the teacher-vs-king control went negative under the\n"
    "# 27B (kings beat its held-out replies at z −5…−8 on the last 30 verdicts)\n"
    "# — no headroom left in the meter. Serving: vLLM ≥ 0.29 (glm53 build),\n"
    "# TP4 per replica on 8-GPU boxes, FP8 KV on Blackwell; vocab 154,880 so\n"
    "# the echo spike shrinks (4.7 GiB at chunk 8192). Chat template differs\n"
    "# from Qwen's: tool turns of D are re-derived (ops/corpus_build.py\n"
    "# --rederive) and the fold's prefix guard measures BOTH tokenizers. The\n"
    "# ranked rule (sd-meter) is unchanged; anchors are the new teacher's own\n"
    "# leave-one-out values, so no per-byte re-calibration is needed.\n"
    f'repo = "{NEW_TEACHER}"\n'
    'base_url = "http://127.0.0.1:9100/v1"\n')
WINDOW_OLD = "# (ops/teacher-swarm/swarm.toml). 2xH200 TP2 has ample KV for 128k.\nmax_model_len = 131072\n"
WINDOW_NEW_TXT = (
    "# (ops/teacher-swarm/swarm.toml). 2xH200 TP2 has ample KV for 128k.\n"
    "# 131072 → 262144 ({date}, with the GLM-5.3-Flash swap): the genesis\n"
    "# family's native window; a 262k sequence is ~5 GB of KV on the miner\n"
    "# slots (10 full-attention layers × 2 KV heads × 256 dims). The fold's\n"
    "# prefix cap follows (ops/corpus_build.py prefix_token_cap: window −\n"
    "# 5120 − 768 − 512 = 255,744 tokens, measured with the teacher AND the\n"
    "# genesis tokenizer) and the teacher swarm moves in lockstep.\n"
    f"max_model_len = {WINDOW_NEW}\n")
WVK_HIST_ANCHOR = "# stands, min_submission_block unchanged. ~3x duel cost accepted.\n#\n"
WVK_HIST_NEW = (
    "# stands, min_submission_block unchanged. ~3x duel cost accepted.\n"
    "# {a}→{b} teacher swap ({date}): Qwen/Qwen3.8-27B → zai-org/GLM-5.3-Flash and\n"
    "# the serving window 131072 → 262144 (see [teacher] / [miner_serving]).\n"
    "# The scoring rule is unchanged; every anchor (μ, σ) is the new teacher's\n"
    "# own leave-one-out statistic, so scores re-baseline but the formula does\n"
    "# not. Forward-only: {reign_clause}; min_submission_block {msb_clause}.\n"
    "#\n")

# -- swarm.toml -------------------------------------------------------------------
SWARM_EDITS = [
    ('model = "Qwen/Qwen3.8-27B"\n',
     f'model = "{NEW_TEACHER}"\n'),
    ('vllm_version = "0.28.0"\n',
     f'vllm_version = "{VLLM_VERSION_GLM53}"\n'),
    ("max_model_len = 131072\n", f"max_model_len = {WINDOW_NEW}\n"),
]
# (type header substring, old "tp/replicas" pair, new pair)
SWARM_TYPES = {
    "[types.pro6000-8x]": (("tp = 2\nreplicas = 4\n"), "tp = 4\nreplicas = 2\n"),
    "[types.pro6000-4x]": (("tp = 2\nreplicas = 2\n"), "tp = 4\nreplicas = 1\n"),
    "[types.b200-8x]": (("tp = 1\nreplicas = 8\n"), "tp = 4\nreplicas = 2\n"),
    "[types.eval-b200-8x]": (("tp = 1\nreplicas = 8\n"), "tp = 4\nreplicas = 2\n"),
    "[types.h100-8x]": (("tp = 2\nreplicas = 4\n"), "tp = 8\nreplicas = 1\n"),
    "[types.h200-8x]": (("tp = 2\nreplicas = 4\n"), "tp = 4\nreplicas = 2\n"),
    "[types.h200-4x]": (("tp = 2\nreplicas = 2\n"), "tp = 4\nreplicas = 1\n"),
}
SWARM_SINGLE_GPU = ("[types.b300-1x]", "[types.b200-1x]")   # 306 GiB does not fit: target 0

# -- slicer -----------------------------------------------------------------------
SLICER_OLD = "MAX_PREFIX_CHARS = 300_000\nMAX_PREFIX_CHARS_262K = 1_000_000\n"
SLICER_NEW = "MAX_PREFIX_CHARS = 1_000_000\nMAX_PREFIX_CHARS_262K = 1_000_000\n"

# -- policies / sources -------------------------------------------------------------
PRICING_OLD = "[pricing.engy]\nin_per_m = 0.045\nout_per_m = 0.32\ncached_in_per_m = 0.015\n"
PRICING_NEW = (PRICING_OLD +
               "\n# Engy list price for glm-5.3-flash (read 2026-09-21 from /v1/models).\n"
               "[pricing.engy-glm53]\nin_per_m = 0.135\nout_per_m = 0.45\ncached_in_per_m = 0.027\n")
BAND_OLD = "teacher_models = []\n"
BAND_BOTH = f'teacher_models = ["{OLD_ENGY_ID}", "{NEW_ENGY_ID}"]\n'
BAND_NEW = f'teacher_models = ["{NEW_ENGY_ID}"]\n'


def sub1(s: str, old: str, new: str, what: str) -> str:
    if s.count(old) != 1:
        raise SystemExit(f"anchor for {what} not unique/missing ({s.count(old)} hits): {old[:60]!r}")
    return s.replace(old, new)


def edit_contract(s: str, date: str, a: int, b: int, reign_clause: str, msb_clause: str) -> str:
    if not re.search(rf"^weight_version_key = {a}$", s, re.M):
        raise SystemExit(f"weight_version_key is not {a}")
    for must in ('score_mode = "sd_min_rga"\n', "max_thought_tokens = 4096\n",
                 "ref_max_tokens = 4864\n", "thought_cap_ratio = 1.25\n"):
        if must not in s:
            raise SystemExit(f"expected pre-flip anchor missing: {must!r}")
    s = sub1(s, TEACHER_OLD, TEACHER_NEW.format(date=date, a=a, b=b), "[teacher].repo")
    s = sub1(s, WINDOW_OLD, WINDOW_NEW_TXT.format(date=date), "[miner_serving].max_model_len")
    s = sub1(s, WVK_HIST_ANCHOR,
             WVK_HIST_NEW.format(a=a, b=b, date=date, reign_clause=reign_clause,
                                 msb_clause=msb_clause), "wvk history")
    s = re.sub(rf"^weight_version_key = {a}$", f"weight_version_key = {b}", s, flags=re.M)
    return s


def edit_swarm(s: str) -> str:
    for old, new in SWARM_EDITS:
        s = sub1(s, old, new, f"swarm {old.strip()}")
    for header, (old, new) in SWARM_TYPES.items():
        i = s.index(header)
        j = s.find("\n[types.", i + 1)
        block = s[i:j if j > 0 else None]
        if block.count(old) != 1:
            raise SystemExit(f"{header}: tp/replicas anchor not found")
        s = s[:i] + block.replace(old, new) + (s[j:] if j > 0 else "")
    for header in SWARM_SINGLE_GPU:
        i = s.index(header)
        j = s.find("\n[types.", i + 1)
        block = s[i:j if j > 0 else None]
        block2 = re.sub(r"^target = \d+", "target = 0", block, flags=re.M)
        s = s[:i] + block2 + (s[j:] if j > 0 else "")
    return s


def edit_policies(s: str) -> str:
    s = sub1(s, PRICING_OLD, PRICING_NEW, "pricing block")
    # Every teacher_* policy's Engy endpoint: model + ledger name.
    out, in_teacher = [], False
    for line in s.splitlines(keepends=True):
        m = re.match(r"^\[\[policy\.([A-Za-z0-9_]+)\.endpoints\]\]", line)
        if m:
            in_teacher = m.group(1).startswith("teacher_")
        elif line.startswith("[") and not line.startswith("[["):
            in_teacher = False
        if in_teacher and line.strip() == f'model = "{OLD_ENGY_ID}"':
            line = line.replace(OLD_ENGY_ID, NEW_ENGY_ID)
        elif in_teacher and line.strip() == 'name = "engy"':
            line = line.replace('"engy"', '"engy-glm53"')
        out.append(line)
    s2 = "".join(out)
    if s2.count(f'model = "{NEW_ENGY_ID}"') < 5:
        raise SystemExit("fewer than 5 teacher endpoints rewritten; check policies.toml layout")
    return s2


def build(date: str, a: int, b: int, reign_clause: str, msb_clause: str) -> dict[Path, str]:
    return {
        TOML: edit_contract(TOML.read_text(), date, a, b, reign_clause, msb_clause),
        SWARM: edit_swarm(SWARM.read_text()),
        SLICER: sub1(SLICER.read_text(), SLICER_OLD, SLICER_NEW, "slicer cap"),
        POLICIES: edit_policies(POLICIES.read_text()),
        SOURCES: sub1(SOURCES.read_text(), BAND_OLD, BAND_BOTH, "band teacher_models"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--apply", metavar="YYYY-MM-DD",
                    help="the dated operator directive; writes every file")
    ap.add_argument("--wvk-to", type=int, default=None)
    ap.add_argument("--reign-stands", default="reign 21 stands, no re-verdicts",
                    help="or e.g. 'throne reset: reign 0 re-seeded from the untouched genesis'")
    ap.add_argument("--min-submission-block", default="unchanged",
                    help="or 'bumped to the finney tip at the flip'")
    ap.add_argument("--band-new-only", action="store_true",
                    help="post-transition: teacher_models -> the new id alone")
    args = ap.parse_args()

    if args.band_new_only:
        s = SOURCES.read_text()
        SOURCES.write_text(sub1(s, BAND_BOTH, BAND_NEW, "band teacher_models (transition)"))
        print("sources.toml: teacher_models ->", BAND_NEW.strip())
        return

    cur = int(re.search(r"^weight_version_key = (\d+)$", TOML.read_text(), re.M).group(1))
    b = args.wvk_to or cur + 1
    date = args.apply or "YYYY-MM-DD"
    files = build(date, cur, b, args.reign_stands, args.min_submission_block)
    if args.preview or not args.apply:
        chunks = []
        for p, new in files.items():
            chunks += difflib.unified_diff(p.read_text().splitlines(True), new.splitlines(True),
                                           str(p.relative_to(REPO)), str(p.relative_to(REPO)))
        PATCH.write_text("".join(chunks))
        print(f"preview written: {PATCH.relative_to(REPO)} ({sum(1 for c in chunks if c.startswith('+') and not c.startswith('+++'))} added lines)")
        if not args.apply:
            return
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", args.apply):
        sys.exit("--apply needs the directive date as YYYY-MM-DD")
    for p, new in files.items():
        p.write_text(new)
        print("wrote", p.relative_to(REPO))
    if MIRROR.exists():
        shutil.copyfile(TOML, MIRROR)
        print("mirrored", MIRROR.relative_to(REPO))
    print(f"weight_version_key {cur} -> {b}; teacher -> {NEW_TEACHER}; window -> {WINDOW_NEW}")


if __name__ == "__main__":
    main()
