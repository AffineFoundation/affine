#!/usr/bin/env python3
"""Generate affine_scicomp tasks with the teacher and verify them.

Runs on the datagen box (needs ENGY). For each (domain, method) topic the
teacher writes N problems as JSON: background, task, exact signature,
reference implementation, a pytest file (5-8 tests, numeric tolerances).
A task is kept only if ALL of:

  1. the reference passes its tests (executed here, in a scratch venv with
     numpy / scipy / sympy / pytest). Two recoveries before giving up, both
     found on the 17:15 probe (kept 5/20, `reference_fails` 11): (a) if the
     reference passes >= MIN_TESTS tests and fails others, the failing test
     functions are pruned — the tests are the teacher's guess at the
     numbers, the reference is the computation; (b) one repair round: the
     pytest output goes back to the teacher, which returns corrected
     reference_code + tests_code;
  2. a stub that returns None FAILS at least one test (the tests are not
     vacuous);
  3. the teacher, solving BLIND from the same prompt the env shows, passes
     the tests in 1-3 of 3 attempts (0/3 = too hard for a reference; the
     king side of the band is the fold's [band_filter]).

Throughput: items run in --workers threads (Engy handles the concurrency;
the spend ledger is locked). ~2.5 min per item sequential on the probe, so
8 workers ≈ 20 items/h... plan for ~1,200 items ≈ 6 h.

Budget: --budget-usd (default 30 at Engy's 0.045 / 0.32 per 1M rates the
runner exports; the count caps bind first). Output: data/e<epoch>/
tasks.jsonl.gz + spend.json + rejects.jsonl.gz (with the pytest tail of
every reference failure, for the next round of prompt fixes).

  python -m affine_scicomp_v1.generate --epoch 1 --per-topic 6 --workers 8 --budget-usd 30
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import json
import os
import random
import re
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from affine_gen_v1.store import GenTaskStore, gen_uid
from affine_gen_v1.teacher import BudgetExceeded, Spend, TeacherClient
from affine_gen_v1.verify import extract_code

from affine_scicomp_v1.taskset import PACKAGE_DIR, SOURCE, SYSTEM, build_prompt

MIN_TESTS = 4

# (domain, method) inventory — public method names from SciPy / NumPy docs and
# standard textbook chapters; the teacher writes the problem, the data, the tests.
TOPICS = [
    ("physics", m) for m in [
        "velocity Verlet integration of a 1D harmonic oscillator", "RK4 for the damped driven pendulum",
        "finite-difference Poisson solver on a 2D grid with Dirichlet boundaries", "1D heat equation explicit scheme with stability check",
        "projectile motion with quadratic drag by numerical integration", "Fourier series coefficients of a periodic signal",
        "power spectral density with Welch's method", "two-body Kepler orbit propagation", "Ising model energy and Metropolis step",
        "wave equation leapfrog scheme", "electric potential of a charge distribution on a grid", "Snell's law ray tracing through layered media",
        "Bragg diffraction peak positions", "blackbody spectrum integration", "N-body gravitational acceleration with softening",
        "Lorenz system trajectory and largest Lyapunov exponent estimate", "quantum harmonic oscillator eigenvalues by finite differences",
        "transfer-matrix transmission through a potential barrier",
        "RLC circuit transient response by ODE integration", "Fresnel reflection coefficients vs angle", "double pendulum energy drift check",
        "Crank-Nicolson scheme for the 1D diffusion equation", "shooting method for the Schrödinger eigenvalue of a finite well",
        "Boltzmann distribution sampling and mean energy", "Rutherford scattering angle distribution", "moment of inertia tensor of a point-cloud body",
        "relativistic Doppler shift and time dilation for a moving emitter", "Biot-Savart field of a current loop on axis",
        "coupled oscillators normal modes via eigen-decomposition", "Carnot and Otto cycle efficiency from state points",
        "Planck-weighted mean photon energy of a spectrum", "geometric optics thin-lens system ray transfer matrices",
        "Compton scattering wavelength shift vs angle", "damped wave packet dispersion by FFT propagation",
        "gravitational lensing deflection for a point mass", "1D Schrödinger time evolution by split-operator FFT"]
] + [
    ("chemistry", m) for m in [
        "equilibrium concentrations from an equilibrium constant", "Arrhenius fit of rate constants", "first-order reaction kinetics integration",
        "Michaelis-Menten parameter fit", "titration curve of a weak acid", "van der Waals gas pressure and compressibility",
        "Beer-Lambert concentration from absorbance spectra", "molecular weight from a formula string", "Nernst equation cell potential",
        "Debye-Hückel activity coefficients", "Lennard-Jones pair energy of a configuration", "radial distribution function from coordinates",
        "Henderson-Hasselbalch buffer pH", "Clausius-Clapeyron vapour pressure", "reaction stoichiometry balancing by linear algebra",
        "second-order kinetics half-life and concentration profile", "polyprotic acid speciation fractions vs pH", "ideal gas mixture partial pressures and mole fractions",
        "Gibbs free energy and equilibrium constant from formation data", "Raoult's law vapour composition of a binary mixture",
        "Eyring equation activation parameters from rate data", "isotope pattern of a molecular formula", "Hückel MO energies of a conjugated chain",
        "Langmuir isotherm fit of adsorption data", "BET surface area from adsorption isotherm", "electrochemical Butler-Volmer current density",
        "radioactive decay chain by matrix exponential", "solubility product and common-ion effect", "heat of reaction from bond enthalpies",
        "Morse potential vibrational levels", "reaction rate from collision theory", "chromatographic resolution from peak parameters"]
] + [
    ("biology", m) for m in [
        "logistic growth ODE fit", "Lotka-Volterra predator-prey simulation", "SIR epidemic model with RK4", "Hardy-Weinberg allele frequencies",
        "Needleman-Wunsch global alignment score", "Smith-Waterman local alignment", "GC content and skew along a genome window",
        "codon usage table from a coding sequence", "Hill equation dose-response fit", "phylogenetic distance matrix by Jukes-Cantor",
        "Michaelis-Menten with competitive inhibition", "population genetics Wright-Fisher drift simulation", "k-mer counting with reverse complements",
        "SEIR model with vaccination term", "Hodgkin-Huxley single-compartment spike", "gene expression Hill-type ODE oscillator",
        "Shannon diversity and evenness of a community", "Lineweaver-Burk kinetic parameter estimation", "open reading frame finder in six frames",
        "Kimura two-parameter distance", "Gillespie stochastic simulation of a birth-death process", "Leslie matrix population projection",
        "enzyme inhibition IC50 from a dose curve", "Fitzhugh-Nagumo excitable dynamics", "restriction enzyme digest fragment sizes",
        "pairwise identity matrix of aligned sequences", "Luria-Delbrück mutation rate estimate", "reaction-diffusion Turing pattern one step",
        "Nei's genetic distance between populations", "isoelectric point of a peptide from pKa values"]
] + [
    ("materials", m) for m in [
        "lattice constant from XRD peaks", "Miller index interplanar spacing", "elastic stress from a strain tensor with Hooke's law",
        "Arrhenius diffusion coefficient", "Fick's second law 1D diffusion profile", "Debye heat capacity integral", "band gap from Tauc plot fit",
        "Voigt profile peak fit", "grain size from Scherrer equation", "Young's modulus from a stress-strain curve",
        "von Mises equivalent stress of a tensor", "thermal expansion length change with temperature-dependent coefficient",
        "Hall-Petch yield strength vs grain size", "Kissinger analysis of activation energy from DSC peaks", "Paris law fatigue crack growth",
        "Weibull statistics of fracture strength", "Bragg-Williams order parameter", "Vegard's law alloy lattice constant",
        "creep strain by Norton power law", "Mott-Schottky carrier density from capacitance", "packing fraction of cubic lattices",
        "thermal conductivity from Wiedemann-Franz law", "phonon dispersion of a 1D diatomic chain", "Nernst-Einstein ionic conductivity"]
] + [
    ("numerics", m) for m in [
        "adaptive Simpson quadrature", "Newton's method with Jacobian for a nonlinear system", "cubic spline interpolation from scratch",
        "conjugate gradient for a sparse SPD system", "QR decomposition by Householder reflections", "least squares with Tikhonov regularisation",
        "FFT-based convolution", "Gauss-Legendre nodes and weights", "Richardson extrapolation of a derivative", "power iteration for the dominant eigenpair",
        "bisection with bracketing for roots", "Monte Carlo integration with variance estimate",
        "Romberg integration table", "LU decomposition with partial pivoting", "Jacobi and Gauss-Seidel iteration convergence",
        "Chebyshev polynomial approximation of a function", "Lagrange interpolation with error bound", "trapezoid rule with error estimate",
        "secant method with convergence history", "Gram-Schmidt orthonormalisation", "singular value truncation low-rank approximation",
        "Runge-Kutta-Fehlberg adaptive step ODE", "finite-difference Jacobian with step selection", "Savitzky-Golay smoothing filter",
        "bootstrap confidence interval of a statistic", "kernel density estimate with bandwidth rule", "Kalman filter for a 1D constant-velocity track",
        "Levenberg-Marquardt curve fit", "Simpson's rule on non-uniform grids", "Padé approximant coefficients"]
] + [
    ("earth_science", m) for m in [
        "barometric formula pressure vs altitude", "Coriolis deflection of a projectile", "tidal range from lunar and solar terms",
        "seismic travel time in a layered crust", "radiocarbon age with half-life and fractionation", "geostrophic wind from a pressure gradient",
        "hydraulic conductivity from Darcy's law", "river discharge from a rating curve fit", "Beer-Lambert attenuation of sunlight through the atmosphere",
        "adiabatic lapse rate temperature profile", "Stokes settling velocity of sediment grains", "Manning equation open-channel flow",
        "carbonate chemistry pH from alkalinity and DIC", "solar zenith angle and daily insolation", "isostatic rebound with a Maxwell relaxation time"]
]

VARIETY = ["a realistic dataset the function must process", "vectorised over a batch of inputs",
           "with a convergence tolerance argument", "returning a tuple of diagnostics",
           "with an edge case the naive formula gets wrong", "with a non-uniform grid or irregular sampling",
           "with physically motivated input validation (raise ValueError on impossible inputs)",
           "as the inner step of a larger simulation, returning the updated state"]

GEN_SYSTEM = (
    "You write SciCode-style scientific-computing problems for evaluating code models. Reply with ONE JSON "
    "object inside a ```json fenced block and nothing else. Keys: title, background (2-4 paragraphs of the scientific "
    "context and the equations, LaTeX allowed), task (what the function must compute, inputs/outputs, units, edge "
    "cases), signature (one Python def line with type hints), reference_code (a complete, correct implementation of "
    "that function plus any helpers, using only numpy/scipy/sympy/stdlib), tests_code (a pytest file that does "
    "`from solution import *`, 5-8 test functions with numeric tolerances via numpy.testing or math.isclose "
    "(rel_tol 1e-6 unless the method is approximate, then 1e-3), covering normal cases, an edge case and a "
    "shape/units check; every expected number in a test must be one you computed by carrying out the same "
    "calculation, not a guess; tests must be deterministic — seed any RNG — and run in under 10 s). The problem "
    "must require real numerical work (discretisation, iteration, fitting, linear algebra), not a one-line formula. "
    "In JSON strings escape backslashes and newlines correctly (\\\\n inside code)."
)

REPAIR_SYSTEM = (
    "You fix a scientific-computing problem whose reference implementation does not pass its own pytest file. Decide "
    "whether the tests' expected numbers or the reference are wrong (recompute the numbers), then reply with ONE JSON "
    "object inside a ```json fenced block: {\"reference_code\": str, \"tests_code\": str} — the corrected pair, same "
    "signature, same task; keep >= 5 tests."
)


def _venv() -> str:
    """Scratch venv with the runtime's packages; created once per box."""
    root = Path(os.environ.get("AFFINE_SCICOMP_VENV", "/tmp/scicomp-venv"))
    py = root / "bin" / "python"
    if not py.exists():
        subprocess.run(["uv", "venv", "-q", str(root), "--python", "3.12"], check=True)
        subprocess.run(["uv", "pip", "install", "-q", "--python", str(py), "numpy>=1.26", "scipy>=1.11",
                        "sympy>=1.12", "pytest>=8"], check=True)
    return str(py)


def run_tests(python: str, solution: str, tests: str, timeout: float = 120.0) -> dict:
    """{"ok", "n_pass", "n_total", "failed": [test names], "tail": pytest output tail}."""
    with tempfile.TemporaryDirectory() as d:
        Path(d, "solution.py").write_text(solution)
        Path(d, "test_solution.py").write_text(tests)
        try:
            p = subprocess.run([python, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--tb=short", "-rfE", "test_solution.py"],
                               cwd=d, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"ok": False, "n_pass": 0, "n_total": 0, "failed": [], "tail": "TIMEOUT"}
        out = p.stdout
        n_pass = int(m.group(1)) if (m := re.search(r"(\d+) passed", out)) else 0
        n_fail = (int(m.group(1)) if (m := re.search(r"(\d+) failed", out)) else 0) \
            + (int(m.group(1)) if (m := re.search(r"(\d+) error", out)) else 0)
        failed = re.findall(r"^(?:FAILED|ERROR) test_solution\.py::(\w+)", out, re.M)
        return {"ok": p.returncode == 0 and n_pass > 0 and n_fail == 0, "n_pass": n_pass, "n_total": n_pass + n_fail,
                "failed": failed, "tail": (out + "\n" + p.stderr)[-3000:]}


def parse_json(text: str, need: tuple[str, ...]) -> dict | None:
    """First JSON object that carries every key in `need`: fenced block first,
    then a raw_decode scan from every '{' (robust to prose before/after)."""
    text = text or ""
    cands = re.findall(r"```(?:json)?\s*\n(.*?)```", text, re.S) or []
    cands.append(text)
    dec = json.JSONDecoder()
    for c in cands:
        i = c.find("{")
        while i != -1:
            try:
                obj, _ = dec.raw_decode(c[i:])
            except json.JSONDecodeError:
                obj = None
            if isinstance(obj, dict) and all(isinstance(obj.get(k), str) and obj[k].strip() for k in need):
                return obj
            i = c.find("{", i + 1)
    return None


def prune_failing_tests(tests_code: str, failed: list[str]) -> str:
    """Remove the named test functions (top-level `def test_x` blocks)."""
    blocks = re.split(r"(?m)^(?=def test_)", tests_code)
    keep = [b for b in blocks if not any(re.match(rf"def {re.escape(f)}\b", b) for f in failed)]
    return "".join(keep)


def stub_of(signature: str) -> str:
    return signature.strip().rstrip(":") + ":\n    return None\n"


class Gen:
    def __init__(self, a, py: str) -> None:
        self.a, self.py = a, py
        self.spend = Spend(a.budget_usd)
        self.teacher = TeacherClient(self.spend)
        self.kept: list[dict] = []
        self.rejects: list[dict] = []
        self.lock = threading.Lock()
        self.stop = False

    def reject(self, domain: str, method: str, why: str, tail: str = "") -> None:
        with self.lock:
            self.rejects.append({"domain": domain, "method": method, "why": why, "tail": tail[-1500:]})

    def verify_reference(self, rec: dict, domain: str, method: str) -> bool:
        res = run_tests(self.py, rec["reference_code"], rec["tests_code"])
        if res["ok"] and res["n_total"] >= MIN_TESTS:
            return True
        # (a) prune the failing tests when enough pass
        if res["n_pass"] >= MIN_TESTS and res["failed"]:
            pruned = prune_failing_tests(rec["tests_code"], res["failed"])
            res2 = run_tests(self.py, rec["reference_code"], pruned)
            if res2["ok"] and res2["n_total"] >= MIN_TESTS:
                rec["tests_code"] = pruned
                rec["pruned_tests"] = res["failed"]
                return True
        # (b) one repair round with the pytest output
        fix = parse_json(self.teacher.complete(
            "repair", REPAIR_SYSTEM,
            f"Signature:\n{rec['signature']}\n\nTask:\n{rec['task']}\n\nreference_code:\n```python\n{rec['reference_code']}\n```\n\n"
            f"tests_code:\n```python\n{rec['tests_code']}\n```\n\npytest output:\n{res['tail']}\n\nReturn the corrected JSON.",
            max_tokens=8000, temperature=0.4), ("reference_code", "tests_code"))
        if fix:
            res3 = run_tests(self.py, fix["reference_code"], fix["tests_code"])
            if res3["ok"] and res3["n_total"] >= MIN_TESTS:
                rec["reference_code"], rec["tests_code"], rec["repaired"] = fix["reference_code"], fix["tests_code"], True
                return True
            self.reject(domain, method, f"reference_fails_after_repair {res3['n_pass']}/{res3['n_total']}", res3["tail"])
            return False
        self.reject(domain, method, f"reference_fails {res['n_pass']}/{res['n_total']}", res["tail"])
        return False

    def item(self, domain: str, method: str, k: int, rng: random.Random) -> None:
        if self.stop:
            return
        variety = rng.choice(VARIETY)
        try:
            raw = self.teacher.complete("generate", GEN_SYSTEM,
                                        f"Domain: {domain}. Method: {method}. Variant {k + 1}: {variety}. Write the JSON object.",
                                        max_tokens=8000, temperature=0.9)
            need = ("title", "background", "task", "signature", "reference_code", "tests_code")
            rec = parse_json(raw, need)
            if not rec:
                self.reject(domain, method, "bad_json", raw[-800:]); return
            if not rec["signature"].lstrip().startswith("def "):
                self.reject(domain, method, "bad_signature", rec["signature"]); return
            if not self.verify_reference(rec, domain, method):
                return
            stub = run_tests(self.py, stub_of(rec["signature"]), rec["tests_code"])
            if stub["ok"]:
                self.reject(domain, method, "vacuous_tests"); return
            n_total = run_tests(self.py, rec["reference_code"], rec["tests_code"])["n_total"]
            rec.update({"domain": domain, "topic": method, "n_tests": n_total})
            prompt = build_prompt(rec)
            passes = 0
            for _ in range(3):
                reply = self.teacher.complete("blind_solve", SYSTEM, prompt, max_tokens=12000, temperature=0.8)
                code = extract_code(reply)
                if code and run_tests(self.py, code, rec["tests_code"])["ok"]:
                    passes += 1
            if passes == 0:
                self.reject(domain, method, "teacher_0_of_3"); return
            rec["teacher_pass"] = passes
            rec["uid"] = gen_uid("scicomp", self.a.epoch, rec["signature"] + rec["task"])
            with self.lock:
                self.kept.append(rec)
                n = len(self.kept)
            print(f"kept {n} ({domain}/{method}, teacher {passes}/3) spend USD {self.spend.usd:.2f}", file=sys.stderr)
        except BudgetExceeded as e:
            self.stop = True
            print(f"STOP: {e}", file=sys.stderr)
        except Exception as e:  # noqa: BLE001 - one bad item must not kill the run
            self.reject(domain, method, f"error {type(e).__name__}", str(e))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=1)
    ap.add_argument("--per-topic", type=int, default=6)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--budget-usd", type=float, default=30.0)
    ap.add_argument("--topics", type=int, default=0, help="limit to the first N topics (pilot)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    store = GenTaskStore(SOURCE, PACKAGE_DIR, a.epoch)
    g = Gen(a, _venv())
    topics = TOPICS[: a.topics] if a.topics else TOPICS
    items = [(d, m, k) for d, m in topics for k in range(a.per_topic)]
    print(f"{len(topics)} topics × {a.per_topic} = {len(items)} items, {a.workers} workers, budget USD {a.budget_usd}", file=sys.stderr)
    try:
        with cf.ThreadPoolExecutor(a.workers) as ex:
            list(ex.map(lambda t: g.item(t[0], t[1], t[2], random.Random(hash((a.seed, t[0], t[1], t[2])) & 0xFFFFFFFF)), items))
    finally:
        # resume-friendly: merge with an existing epoch file rather than overwrite
        existing = []
        if (store.local_dir / "tasks.jsonl.gz").exists():
            existing = [r for r in store.tasks()]
        seen = {r["uid"] for r in existing}
        merged = existing + [r for r in g.kept if r["uid"] not in seen]
        store.write_tasks(merged)
        with gzip.open(store.local_dir / "rejects.jsonl.gz", "at") as f:
            for r in g.rejects:
                f.write(json.dumps(r) + "\n")
        g.spend.write(store.local_dir / "spend.json")
        by_why = {}
        for r in g.rejects:
            by_why[r["why"].split(" ")[0]] = by_why.get(r["why"].split(" ")[0], 0) + 1
        print(json.dumps({"items": len(items), "kept_this_run": len(g.kept), "kept_total": len(merged),
                          "rejected": len(g.rejects), "rejects_by_reason": by_why, "spend": g.spend.to_dict()}, indent=1))


if __name__ == "__main__":
    main()
