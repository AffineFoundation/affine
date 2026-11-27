# TODO — ROOT-sign host-v834f (ZeRO-3 / FSDP)

**Do not bootstrap or override ROOT.** Jacob signs with the existing ROOT
lifecycle seed process (same pattern as `ops/ff_train_params/sign_v833j.py`).

## Why

host-v834e remains the live admitted package (B300/B200/H200). The 8×RTX5090
sharded path needs a **separate** ROOT admission over host-v834f so
`train_pit.py` can keep calling `host.checked(…ROOT-ADMITTED.private.json)`
on every path.

## Artifact in this directory

| Path | Role |
| --- | --- |
| `stage/` | Unsigned trainer-host-v834f tree (admitted v834e + `v835_zero3.py` + DP/launcher hooks) |
| `PROPOSAL.unsigned.json` | Host-file sha256 inventory + change list |
| `SIGNED-MANIFEST.sha256` | `sha256sum` listing with `./trainer-host-v834f/…` paths (no `.pyc`) |
| `sign_host_v834f.py` | Dry-run / operator signing helper (reads ROOT seed only when not `--dry-run`) |

## Operator steps

1. Verify `sha256sum --quiet -c SIGNED-MANIFEST.sha256` from this directory's
   parent after linking `trainer-host-v834f -> stage` (or run the helper).
2. Point the helper at the current ROOT-admitted host-v834e private json as
   the base admission (engine / authority / original_methods unchanged except
   host_files digests + `v835_zero3.py`).
3. Run `python3 sign_host_v834f.py --dry-run` first, then without `--dry-run`
   on the operator host that holds the ROOT seed.
4. Publish the signed release as `trainer-host-v834f/` + private json; set
   `PIT_TRAINER_HOST` / release path accordingly. pe_box defaults stay on
   signed host-v834e until that cutover.
5. Flip `pit.allow_sharded_gpus` only after a first live test.

## Non-goals

- No `host.private_json()` bypass of `checked()`.
- No public `trainer-config.public.json` engine overlay of signed settings.
- No implicit in-repo trainer ship (`PIT_USE_INREPO_TRAINER` / `PIT_TRAINER_HOST` only).
