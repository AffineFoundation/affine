#!/usr/bin/env python3
"""Generate affine_scicomp tasks with the teacher and verify them.

Runs on the datagen box (needs ENGY). For each (domain, method) topic the
teacher writes N problems as JSON: background, task, exact signature,
reference implementation, a pytest file (5-8 tests, numeric tolerances).
A task is kept only if ALL of:

  1. the reference passes its own tests (executed here, in a scratch venv
     with numpy / scipy / sympy / pytest);
  2. a stub that returns None FAILS at least one test (the tests are not
     vacuous);
  3. the teacher, solving BLIND from the same prompt the env shows, passes
     the tests in 1-3 of 3 attempts (0/3 = too hard for a reference; the
     king side of the band is the fold's [band_filter]).

Budget: --budget-usd (default 60; plan says USD 30 for 3k problems).
Output: data/e<epoch>/tasks.jsonl.gz + spend.json + rejects.jsonl.gz.

  python -m affine_scicomp_v1.generate --epoch 1 --per-topic 3 --budget-usd 60
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

from affine_gen_v1.store import GenTaskStore, gen_uid
from affine_gen_v1.teacher import BudgetExceeded, Spend, TeacherClient
from affine_gen_v1.verify import extract_code

from affine_scicomp_v1.taskset import PACKAGE_DIR, SOURCE, SYSTEM, build_prompt

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
        "transfer-matrix transmission through a potential barrier"]
] + [
    ("chemistry", m) for m in [
        "equilibrium concentrations from an equilibrium constant", "Arrhenius fit of rate constants", "first-order reaction kinetics integration",
        "Michaelis-Menten parameter fit", "titration curve of a weak acid", "van der Waals gas pressure and compressibility",
        "Beer-Lambert concentration from absorbance spectra", "molecular weight from a formula string", "Nernst equation cell potential",
        "Debye-Hückel activity coefficients", "Lennard-Jones pair energy of a configuration", "radial distribution function from coordinates",
        "Henderson-Hasselbalch buffer pH", "Clausius-Clapeyron vapour pressure", "reaction stoichiometry balancing by linear algebra"]
] + [
    ("biology", m) for m in [
        "logistic growth ODE fit", "Lotka-Volterra predator-prey simulation", "SIR epidemic model with RK4", "Hardy-Weinberg allele frequencies",
        "Needleman-Wunsch global alignment score", "Smith-Waterman local alignment", "GC content and skew along a genome window",
        "codon usage table from a coding sequence", "Hill equation dose-response fit", "phylogenetic distance matrix by Jukes-Cantor",
        "Michaelis-Menten with competitive inhibition", "population genetics Wright-Fisher drift simulation", "k-mer counting with reverse complements"]
] + [
    ("materials", m) for m in [
        "lattice constant from XRD peaks", "Miller index interplanar spacing", "elastic stress from a strain tensor with Hooke's law",
        "Arrhenius diffusion coefficient", "Fick's second law 1D diffusion profile", "Debye heat capacity integral", "band gap from Tauc plot fit",
        "Voigt profile peak fit", "grain size from Scherrer equation", "Young's modulus from a stress-strain curve"]
] + [
    ("numerics", m) for m in [
        "adaptive Simpson quadrature", "Newton's method with Jacobian for a nonlinear system", "cubic spline interpolation from scratch",
        "conjugate gradient for a sparse SPD system", "QR decomposition by Householder reflections", "least squares with Tikhonov regularisation",
        "FFT-based convolution", "Gauss-Legendre nodes and weights", "Richardson extrapolation of a derivative", "power iteration for the dominant eigenpair",
        "bisection with bracketing for roots", "Monte Carlo integration with variance estimate"]
]

GEN_SYSTEM = (
    "You write SciCode-style scientific-computing problems for evaluating code models. Reply with ONE JSON "
    "object and nothing else, with keys: title, background (2-4 paragraphs of the scientific context and the "
    "equations, LaTeX allowed), task (what the function must compute, inputs/outputs, units, edge cases), "
    "signature (one Python def line with type hints), reference_code (a complete, correct implementation of "
    "that function plus any helpers, using only numpy/scipy/sympy/stdlib), tests_code (a pytest file that does "
    "`from solution import *`, 5-8 test functions with numeric tolerances via numpy.testing or math.isclose, "
    "covering normal cases, an edge case and a shape/units check; tests must be deterministic and run in "
    "under 10 s). The problem must require real numerical work (discretisation, iteration, fitting, linear "
    "algebra), not a one-line formula."
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


def run_tests(python: str, solution: str, tests: str, timeout: float = 120.0) -> tuple[bool, int, int]:
    with tempfile.TemporaryDirectory() as d:
        Path(d, "solution.py").write_text(solution)
        Path(d, "test_solution.py").write_text(tests)
        try:
            p = subprocess.run([python, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--tb=no", "test_solution.py"],
                               cwd=d, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, 0, 0
        import re
        n_pass = int(m.group(1)) if (m := re.search(r"(\d+) passed", p.stdout)) else 0
        n_fail = (int(m.group(1)) if (m := re.search(r"(\d+) failed", p.stdout)) else 0) \
            + (int(m.group(1)) if (m := re.search(r"(\d+) error", p.stdout)) else 0)
        return p.returncode == 0 and n_pass > 0 and n_fail == 0, n_pass, n_pass + n_fail


def parse_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1].rsplit("```", 1)[0]
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        i, j = text.find("{"), text.rfind("}")
        if i >= 0 and j > i:
            try:
                return json.loads(text[i:j + 1])
            except json.JSONDecodeError:
                return None
    return None


def stub_of(signature: str) -> str:
    return signature.strip().rstrip(":") + ":\n    return None\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=1)
    ap.add_argument("--per-topic", type=int, default=3)
    ap.add_argument("--budget-usd", type=float, default=60.0)
    ap.add_argument("--topics", type=int, default=0, help="limit to the first N topics (pilot)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    store = GenTaskStore(SOURCE, PACKAGE_DIR, a.epoch)
    spend = Spend(a.budget_usd)
    teacher = TeacherClient(spend)
    py = _venv()
    rng = random.Random(a.seed)
    topics = TOPICS[: a.topics] if a.topics else TOPICS
    kept, rejects = [], []
    try:
        for domain, method in topics:
            for k in range(a.per_topic):
                variety = rng.choice(["a realistic dataset the function must process", "vectorised over a batch of inputs",
                                      "with a convergence tolerance argument", "returning a tuple of diagnostics",
                                      "with an edge case the naive formula gets wrong"])
                raw = teacher.complete("generate", GEN_SYSTEM,
                                       f"Domain: {domain}. Method: {method}. Variant {k + 1}: {variety}. Write the JSON object.",
                                       max_tokens=6000, temperature=0.9)
                rec = parse_json(raw)
                need = ("title", "background", "task", "signature", "reference_code", "tests_code")
                if not rec or any(not isinstance(rec.get(x), str) or not rec[x].strip() for x in need):
                    rejects.append({"domain": domain, "method": method, "why": "bad_json"}); continue
                ok, n_pass, n_total = run_tests(py, rec["reference_code"], rec["tests_code"])
                if not ok or n_total < 4:
                    rejects.append({"domain": domain, "method": method, "why": f"reference_fails {n_pass}/{n_total}"}); continue
                stub_ok, _, _ = run_tests(py, stub_of(rec["signature"]), rec["tests_code"])
                if stub_ok:
                    rejects.append({"domain": domain, "method": method, "why": "vacuous_tests"}); continue
                rec.update({"domain": domain, "topic": method, "n_tests": n_total})
                prompt = build_prompt(rec)
                passes = 0
                for _ in range(3):
                    reply = teacher.complete("blind_solve", SYSTEM, prompt, max_tokens=8192, temperature=0.8)
                    code = extract_code(reply)
                    if code and run_tests(py, code, rec["tests_code"])[0]:
                        passes += 1
                if passes == 0:
                    rejects.append({"domain": domain, "method": method, "why": "teacher_0_of_3"}); continue
                rec["teacher_pass"] = passes
                rec["uid"] = gen_uid("scicomp", a.epoch, rec["signature"] + rec["task"])
                kept.append(rec)
                print(f"kept {len(kept)} ({domain}/{method}, teacher {passes}/3) spend USD {spend.usd:.2f}", file=sys.stderr)
    except BudgetExceeded as e:
        print(f"STOP: {e}", file=sys.stderr)
    finally:
        store.write_tasks(kept)
        with gzip.open(store.local_dir / "rejects.jsonl.gz", "wt") as f:
            for r in rejects:
                f.write(json.dumps(r) + "\n")
        spend.write(store.local_dir / "spend.json")
        print(json.dumps({"kept": len(kept), "rejected": len(rejects), "spend": spend.to_dict()}, indent=1))


if __name__ == "__main__":
    main()
