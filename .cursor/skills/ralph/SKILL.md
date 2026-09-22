---
name: ralph
description: >-
  Install and run a Ralph loop: a fresh cursor-agent on the same GOAL.md until
  the goal is done. Writes ~/.ralph/ralph.sh from this skill's exact runner,
  writes GOAL.md, starts the loop. Use when the user says /ralph, ralph loop,
  grind on a goal overnight, keep working continuously, or asks to loop a
  cursor-agent on a goal.
disable-model-invocation: true
---

# Ralph loop

A ralph loop hands the same goal to a **fresh** `cursor-agent` over and over.
Each pass remembers nothing. The working directory is the only shared state.
Work compounds through files, not chat context.

`/ralph <goal>` → write the runner if needed, write `GOAL.md`, start looping.

This is **not** Cursor's `/loop` skill (in-chat timers). This is **not** the
SN120 `ralphctl.sh` ops loops (`prompt.md` + interval). Those stay in the
subnet120 repo.

## Parse

| user says | do |
|---|---|
| `/ralph <goal text>` | install runner, write/update `GOAL.md`, start `-b` |
| `/ralph` with a goal already in `GOAL.md` | install runner, start on that file |
| `/ralph status` / `stop` / `ensure` | that action only; do not rewrite the goal |
| `/ralph` empty, no `GOAL.md` | ask what the goal is |

Default working directory: the **project root**. Use a subdirectory only when
the user names one, or the goal must not touch the rest of the project.

## 1. Create the runner

The runner is **machine-wide**. Never write `ralph.sh` into a project's source tree.

Find `scripts/install.sh` next to this `SKILL.md` and run it. Typical paths:

```bash
# this repo / Cloud Agent / any clone
bash .cursor/skills/ralph/scripts/install.sh

# this user account, all local projects
bash "$HOME/.cursor/skills/ralph/scripts/install.sh"
```

If both are missing:

1. Read `scripts/ralph.sh` next to this `SKILL.md`.
2. Write those bytes to `~/.ralph/ralph.sh`.
3. `chmod +x ~/.ralph/ralph.sh`.

Copy **byte-for-byte**. Do not invent a shorter loop, a Python wrapper, a
cron job, or an in-chat `/loop`. If you cannot read `scripts/ralph.sh`, stop
and say so — a homemade runner is worse than no runner.

Do not overwrite `~/.ralph/ralph.sh` if it already exists (a live loop may be
using it). `install.sh` already follows this rule.

Need `cursor-agent` on `PATH` and `cursor-agent login`. If either fails, stop
and say so before writing a goal.

## 2. Write the goal

The goal is the whole product. Expand a one-line request into the shape in
[GOAL.template.md](GOAL.template.md). A memoryless stranger will read only
what the directory tells them.

Must include:

- **Objective** — success is checkable (yes or no).
- **Read this first** — exact files, in order. Name them.
- **One increment per pass**, then stop. Say it.
- **Memory files** with caps. Default: `STATE.md` (rewrite every pass: stage,
  running, **one** next action) and `NOTES.md` (append-only; negative results
  required). Add more when the work needs them (`LESSONS.md`, `LEDGER.md`, …).
- **Hard rules** — off-limits paths, money, git, secrets. Repeated every pass;
  that is the only enforcement.
- **Stage gates** on long work, so it does not skip to the expensive part.
- **Whether it ever completes.** Ongoing work: "never create `.ralph/DONE`".
  Finite work: what finished looks like.

`/ralph every step train a new model` is not a goal. Name where checkpoints
go, how to compare a run to the last one, where results are recorded, and
what "better" means.

If `GOAL.md` already exists and the user gave new text, update it — it is the
operator file; the next pass will read it. Do not revert operator edits.

## 3. Project plumbing

Add `.ralph/` to `.gitignore` if a git repo is present and it is not already
ignored. Do not gitignore `GOAL.md` or the memory files.

Seed empty `STATE.md` / `NOTES.md` only when they do not exist.

## 4. Start, then confirm

```bash
~/.ralph/ralph.sh -b -m <model> <dir>     # prefer a pinned model
~/.ralph/ralph.sh --status <dir>
```

Pin a model with `-m`. The default is Auto, which varies per pass and is not
recorded. For a long grind, pass a comma list to rotate on stalls:
`-m composer-2,gpt-5.4-high`.

Wait until the first pass finishes (or a minute or two), then `--status`.
Do not tell the user it is working until a pass has outcome `ok` or you can
show a real `status.log` / `health.log` line.

Report: pid, goal path, log path, first-pass outcome.

## Commands

```bash
~/.ralph/ralph.sh <dir>                    # foreground
~/.ralph/ralph.sh -b <dir>                 # background
~/.ralph/ralph.sh -b -i 0 -t 3600 <dir>    # no sleep, 1h per pass
~/.ralph/ralph.sh -b -m <model> <dir>
~/.ralph/ralph.sh --status <dir>
~/.ralph/ralph.sh --stop <dir>
~/.ralph/ralph.sh --ensure <dir>           # start only if not running
```

Flags: `-i` interval seconds (default 10), `-t` per-pass timeout (default
1800), `-n` max passes this run (0 = forever), `-m` model or comma list,
`--max-stalls N` give up after N unproductive passes.

## How a pass is judged

By whether it **changed the tree**, not by exit code. A refusal, rate limit,
or apology still exits 0. Those are classified (`refusal`, `apierror`,
`stall`, `timeout`, `fail-rc*`), written to `.ralph/health.log`, and answered
with exponential backoff plus model rotation. The loop stays up.

`--status` showing `ATTENTION` means passes stopped making progress.

## State on disk

```
<dir>/GOAL.md                 standing brief (operator-owned)
<dir>/STATE.md                rewrite every pass
<dir>/NOTES.md                append-only journal
<dir>/.ralph/status.log       one line per pass (agent writes)
<dir>/.ralph/health.log       outcome, duration, model
<dir>/.ralph/passes/NNNN.log  full transcript
<dir>/.ralph/loop.log         runner output
<dir>/.ralph/ATTENTION        present when stalled
<dir>/.ralph/DONE             agent creates this to stop forever
```

Pass numbers resume across restarts. All state is on disk.

## Gotchas

- Pass logs stay empty until a pass ends. Watch file mtimes for live progress.
- `--stop` kills the in-flight agent. Goals that do non-atomic work must
  tolerate that.
- Backoff is a multiple of the interval, so `-i 0` disables it.
- Editing `GOAL.md` takes effect on the next pass. No restart.
- Secrets stay out of `GOAL.md`. Point at a gitignored `.env`.
- In this repo the skill lives at `.cursor/skills/ralph/`. That is what Cloud
  Agents and other machines see after a pull. The same folder also lives at
  `~/.cursor/skills/ralph/` on a machine that has it installed for all
  projects. Cursor does **not** sync `~/.cursor/skills/` across devices.
