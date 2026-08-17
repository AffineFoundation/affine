#!/usr/bin/env python3
"""Smoke-check pod affine_pkg is on Reason v4 (wvk=7)."""
from pathlib import Path

from affine.config import load_config
from affine.score import DEFAULT_TEMPER_TAU, turn_reason
import evalsrv.dueling as d
import evalsrv.terms as t

cfg = load_config("/root/mining_src/affine_pkg/affine.toml")
print("cfg.wvk", cfg.weight_version_key)
print(
    "cfg.duel",
    cfg.duel.n_teacher_samples,
    cfg.duel.tau,
    cfg.duel.n_turns,
    cfg.duel.k_sigma,
    cfg.duel.min_margin,
)
print("DEFAULT_TEMPER_TAU", DEFAULT_TEMPER_TAU)
pair = {"lpC_yc_za": 1.0, "lpC_yc_e": 0.5}
print("k1", turn_reason([pair], tau=0.03))
pairs = [
    {"lpC_yc_za": 0.1, "lpC_yc_e": 0.0},
    {"lpC_yc_za": 1.0, "lpC_yc_e": 0.0},
    {"lpC_yc_za": 0.2, "lpC_yc_e": 0.0},
]
lme = turn_reason(pairs, tau=0.03)
mean = sum(p["lpC_yc_za"] - p["lpC_yc_e"] for p in pairs) / 3
print("LME", round(lme, 6), "mean", round(mean, 6), "LME>mean", lme > mean)
print("dueling", d.__file__)
print("terms", t.__file__)
print("ok", Path("/root/mining_src/affine_pkg/affine.toml").exists())
