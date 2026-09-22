"""Shared pieces for the teacher-generated D sources (aa-gap-fill-plan, 2026-09-22).

Three envs synthesize their tasks with the teacher instead of loading a
public dataset: affine_scicomp (scientific-computing functions + hidden
tests), affine_scitext (textbook-science variants with a code-checked
answer) and affine_docqa (questions over bundles of public documents). They
share:

  * `store`   — where generated tasks live: `data/e<epoch>/tasks.jsonl.gz`
                inside the env package, or the same file under
                `https://data.affine.io/envs/<source>/e<epoch>/` (docqa's
                bundles are too big to commit). `gen_uid` builds the task
                name WITH the `[GEN:<epoch>]` marker the fold's
                decontamination rule requires (no marker -> never folded).
  * `teacher` — an OpenAI-compatible client for Qwen3.8-27B (Engy) with a
                USD budget that stops generation when spent, and a spend
                ledger written next to the tasks.
  * `verify`  — run generated code with a timeout, extract `\\boxed{}`
                answers, compare answers (math-verify when present).

Nothing here is imported by the fold or the duel; only the env packages and
their `generate.py` scripts use it.
"""

# Only the store is imported eagerly: tasksets need it at duel time and must
# not pull the generator-side deps (openai). Generators import
# `affine_gen_v1.teacher` / `affine_gen_v1.verify` directly.
from affine_gen_v1.store import GenTaskStore, gen_uid, has_gen_marker

__all__ = ["GenTaskStore", "gen_uid", "has_gen_marker"]
