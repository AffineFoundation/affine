# Hourly owner-weight heartbeat

Sets SN120 Finney mechanism-0 weights to its registered owner hotkey, resolved from `SubnetOwnerHotkey` and `Uids` on chain each invocation. Owner expected at deployment: `5HmYnmUYT6qe3yFMg1Ad8WLLqDvnwtjYakXBpDvoRW1Qqzb8`, currently UID 0. Refuses to run if the owner changes or the wallet mismatches. Uses wallet `default/default` at `~/.bittensor/wallets` without copying its private key.

Bittensor 11.0.2 `SetWeights` chooses direct or timelocked commit/reveal from the subnet configuration. Commit/reveal is enabled on SN120 at deployment. An accepted commit precedes automatic chain reveal; accepted submission is not proof of an immediate reward change. This implements the requested owner-only allocation; it does not independently establish economics or emission amounts.

The SDK read-only `plan` runs before each submission. The subnet currently limits submissions to once per 100 blocks. If the live legacy validator set weights recently, this job records a deferral and exits successfully; it retries at the next hourly tick. An SDK/chain rate-limit race is handled similarly. Other failures return nonzero and are visible in the journal. There is no unbounded retry loop. The legacy validator is intentionally left running, so it may overwrite these weights and prevent the hourly job from submitting. Exclusive burn operation requires a separate operator decision about that validator.

Source: `ops/hourly_burn.py`; runtime copy: `~/.local/share/affine-burn/hourly_burn.py`. Dedicated interpreter and pinned dependencies: `~/.local/share/affine-burn/venv` and `ops/hourly_burn.requirements.txt`. Neither legacy checkout nor archive is a runtime dependency. Invocation lock and last status: `~/.local/state/affine-burn/`. Runtime and state are private to the operator account.

User-systemd units are installed in `~/.config/systemd/user/`. `affine-hourly-burn.timer` runs at the top of every hour with one missed-run catch-up after downtime. User lingering is already enabled, so it survives logout. Service timeout is 12 minutes; file lock and systemd prevent overlapping runs. Wallet remains outside the repository.

Commands:

```sh
# Read-only chain validation/plan (no weights sent)
~/.local/share/affine-burn/venv/bin/python ~/.local/share/affine-burn/hourly_burn.py
# Inspect schedule and latest run
systemctl --user status affine-hourly-burn.timer affine-hourly-burn.service
journalctl --user -u affine-hourly-burn.service --no-pager -n 40
cat ~/.local/state/affine-burn/status.json
# Stop hourly submissions
systemctl --user disable --now affine-hourly-burn.timer
# Reinstall runtime after checkout changes
install -m 700 ops/hourly_burn.py ~/.local/share/affine-burn/hourly_burn.py
```

Restore dependencies into a fresh durable runtime with `uv venv ~/.local/share/affine-burn/venv --python /usr/bin/python3.12` then `uv pip install --python ~/.local/share/affine-burn/venv/bin/python -r ops/hourly_burn.requirements.txt`. Install the two tracked `systemd/` units into `~/.config/systemd/user/`, reload the user daemon, and enable the timer. Do not start multiple timers/services for the same wallet.

Validation: read-only live chain destination/config queries; safety checks exercised with mocked chain responses for changed owner, nonzero owner UID, rate limit, denied plan, and exact owner-only submission; systemd unit verification. Initial manual service invocation was deferred by the live validator's recent weight update.

Official reference: https://www.bittensor.com/docs/hyperparameters/weights-rate-limit and installed SDK 11.0.2 `SetWeights`/`plan` source (official web SDK reference currently documents an older API).
