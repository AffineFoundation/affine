# Green-only reveal (Jacob, 2026-10-31 13:07 CR)

Rule: a held-out round is revealed only if a GREEN closed it, and then only the GREEN turn (seed, task ids, head
baseline, the GREEN candidate's per-task grades). RED turns and their probe results are never published, including
RED turns inside a round that a GREEN later closed. A round that closes on a head change or after 8 RED turns is
never revealed. Gate decisions, scoring and the trainer are unchanged: only the publish/reveal step changed.

Changes (live copies under `services/`; `gate_service.py` is the sanitized copy, the same file as
`miner/validator/gate/gate_service.py`):

- `gate_service.py` `heldout_publish()` only: new `_green_turn(closed, turns)`; `/gate/heldout/<round>.json` is
  written only for a GREEN-closed round, with only the GREEN turn; the index/current rows carry `reveal` and the
  baseline numbers (`head_pass_on_kept`, `kept_tasks`) only for GREEN-closed rounds; public rule text updated
  (`HO_PUBLIC_RULE`, gate/index.json rule). `process_heldout_epoch`, `ensure_heldout_sample` and `heldout.py` are untouched.
- `k1_transparency.py`: `publish_reveal()` writes `/gate/reveal/<e>.json` only for a GREEN decision, and for a
  held-out epoch only when that epoch closed its round; `turn.json` and `/k1/turns/index.json` link `gate_reveal`
  only for GREEN epochs.
- `gate_ledger.py`, `training_params.py`: rule text only (gate.json, training.json).

Cleanup (`delete_red.py`): backed up (0600, private) then deleted 136 RED `/gate/reveal/<e>.json` files and 11
`/gate/heldout/<round>.json` files of rounds no GREEN closed; stripped the RED rows from the 3 GREEN round files
that had them (rounds of E292, E354, E368); set `gate_reveal` to null in 147 RED `turn.json` files; rebuilt
`/gate/reveal/index.json` and `/k1/turns/index.json`. `revert.sh` restores the code and every backed-up file.

Not changed: git history of this repo is not rewritten. `k1.json` still shows the aggregate pass rates (h_pass, c_pass, n)
of closed-round turns (no per-task data).
