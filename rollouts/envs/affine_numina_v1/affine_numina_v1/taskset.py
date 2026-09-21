"""affine-numina-v1: Lean 4 theorem proving on NuminaMath-LEAN (any shell harness).

Wrapper over research-environments' `numina_v1` (104k statements; the shared
`verifiers.v1.tasksets.lean` base plants a `sorry` starter file in a Mathlib
sandbox, the agent edits and compiles with `lake env lean`, the reward is a
clean compile with the theorem statement intact). Changes for the duel corpus:

  * `tasks: list[str]` selector by `uuid` (the base name column), so the
    scheduler addresses rows by name (rollouts/catalog.py `_numina_meta`);
  * the grade lands under `solved` (fold key) and drops the base's
    `if trace.has_error: return 0.0` guard: on the pods' verifiers
    (a298bcf) `Trace.ok` is False throughout scoring, so the base reward
    grades every rollout 0.0 without compiling (the prolog lesson, env
    wave 1 §2.6). Recorded errors are checked directly instead;
  * `docker_image` stays configurable. The upstream default
    `team-clyvldofb0000gg1kx39rgzjq/lean-tactic:mathlib-v4.15.0-v1` is a
    Prime TEAM image (id-form reference) that resolves only inside Prime
    sandboxes - not pullable from Docker Hub, with or without a Prime API
    key (checked 2026-09-12). sources.toml must point `--env.taskset.
    docker-image` at a Mathlib image the pod can pull or build, and
    `--env.taskset.task.lean-project-path` at the Mathlib project inside it.

The system prompt is the base's ("You are an expert Lean 4 theorem prover
working with Mathlib.") - no dialect word, the shell harness supplies it.
"""

from __future__ import annotations

import verifiers.v1 as vf
from numina_v1.taskset import NuminaConfig, NuminaTaskset as BaseNuminaTaskset
from verifiers.v1.tasksets.lean import LeanTask
from verifiers.v1.tasksets.lean.scoring import (
    expected_protected_signature,
    protected_signature_substring_present,
)


class AffineLeanTask(LeanTask):
    @vf.reward(weight=1.0)
    async def solved(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        if trace.errors:
            return 0.0
        current = (await runtime.read(self.config.proof_file_path)).decode("utf-8", "replace")
        expected_sig = self.data.protected_signature or expected_protected_signature(
            self.data.formal_statement)
        if expected_sig and not protected_signature_substring_present(current, expected_sig):
            trace.info["lean_tampered"] = True
            trace.info["compile_output"] = "signature rewritten or hidden in a comment"
            return 0.0
        trace.info["lean_tampered"] = False
        compiled, output, exit_code = await self._compile(runtime)
        trace.info["lean_compiled"] = compiled
        trace.info["compile_exit_code"] = exit_code
        trace.info["compile_output"] = output[-4000:]
        return 1.0 if compiled else 0.0

    async def lean_compiled(self, trace: vf.Trace, runtime: vf.Runtime) -> float:
        # Undecorated override: the grade is counted once, under `solved`.
        return await self.solved(trace, runtime)


class AffineNuminaConfig(NuminaConfig):
    tasks: list[str] = []
    """Row uuids to load (empty = the whole split; 104k rows - always pass tasks)."""


class AffineNuminaTaskset(BaseNuminaTaskset, vf.Taskset[AffineLeanTask, AffineNuminaConfig]):
    def load(self):
        want = set(self.config.tasks)
        for task in super().load():
            if want and task.data.name not in want:
                continue
            yield AffineLeanTask(task.data, self.config.task)
