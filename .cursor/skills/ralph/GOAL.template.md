# GOAL — <one sentence, checkable>

## Objective

<What done looks like. A stranger must be able to say yes or no.>

## HARD RULES — breaking any one is a total failure of the run

- <What is off limits: paths, processes, money, git, secrets.>
- `GOAL.md` is the operator's file. Never revert it. If it changed mid-pass, that is a new instruction — read and follow it.
- Never commit secrets. `.ralph/` is loop plumbing — keep it gitignored.

## Read these first, every pass

You are a fresh agent with no memory. Before doing anything else:

1. `STATE.md` — where the run is, single next action.
2. `NOTES.md` — journal, including negative results.
3. <other memory files this goal needs, in order>

Then do the work. Do not guess state from the tree if these files exist.

## How one pass works

Do **one** useful increment, then stop.

1. Read the files above.
2. Do the highest-value next action from `STATE.md`.
3. Record results (including failures).
4. Rewrite `STATE.md`.
5. Append exactly one line to `.ralph/status.log`:
   `<UTC ISO8601> | pass <n> | <what you did>`

Long jobs: start them in the background, write how to check them in `STATE.md`, end the pass. Never block a pass on something that will not finish inside the timeout.

## Memory files

| file | cap | contents |
|---|---|---|
| `STATE.md` | 60 lines | stage, live facts, running, blocked, **one** next action |
| `NOTES.md` | 150 lines | append-only journal; negative results required |
| <add more> | | |

Overflow → `archive/`. Numbers, not vibes. Pre-register the decision rule before a costly step.

## Stages

**Stage 0 — …**
*Gate: …*

**Stage 1 — …**
*Gate: …*

Do not skip to the expensive stage. A gate that is not met stays the next action.

## Done

<Either:> Create `.ralph/DONE` (one line saying why) only when the objective is permanently and completely satisfied.

<Or, for ongoing work:> Never create `.ralph/DONE`. This goal does not finish.
