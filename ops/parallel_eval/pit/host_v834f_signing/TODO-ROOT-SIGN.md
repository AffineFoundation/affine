# TODO — ROOT-sign host-v834f (sharded / ZeRO-3 trainer)

**Do not bootstrap or override ROOT.** Jacob signs with the existing ROOT
lifecycle seed process (same pattern as `ops/ff_train_params/sign_v833j.py`).

## Why

host-v834e stays the live admitted package (B300/B200/H200). Eight RTX5090
(32 GiB each) cannot hold its update. host-v834f is host-v834e plus a sharded
execution of the same update. It needs its **own** ROOT admission so
`train_pit.py` keeps calling `host.checked(…ROOT-ADMITTED.private.json)` on
every path.

## What changes against host-v834e

| Host file | Change |
| --- | --- |
| `v835_zero3.py` | new: the sharded update (torch FSDP), helper entry, memory checks |
| `mimo_host_trainer_v83.py` | `HOST_FILES` + `zero3()` + the `install()` hook; taken only when `PIT_ZERO3` is on |
| the other 14 host files | byte-identical to the admitted host-v834e |

The engine block, the source root, the authority and the original-method pins
are copied from the host-v834e admission. `sign_host_v834f.py` refuses anything
else.

## Where the fp32 master is, and who saves it

- The fp32 master is **not** sharded. It stays whole in the trainer process
  (rank 0), in host memory, as in host-v834e.
- Ranks 1 to 7 hold shards of the bf16 actor and their fp32 gradient shards.
  They hold no fp32 master, so they cannot save one. The fp32 parameter shards
  FSDP keeps are the bf16 weights written as fp32. They are not the master and
  must never be saved as one.
- Rank 0 runs the unchanged host-v834e code: `_fp32_prepare`, `_fp32_load`,
  the optimizer step, `_fp32_finish`, `_fp32_save`. The entry on disk is
  `<FP32_STORE>/<sha256 of the candidate weights>/` with
  `fp32-master-NNNNN-of-MMMMM.safetensors` and `meta.json`, the same names and
  keys as host-v834e. The training report carries the same
  `optimizer.sgd.fp32_master` block.
- A run makes one optimizer step. The store holds the head's master until that
  step and gets the candidate's master once, after it.

## Files in this directory

| Path | Role |
| --- | --- |
| `stage/` | the unsigned trainer-host-v834f tree |
| `SIGNED-MANIFEST.sha256` | every file of `stage/` with its sha256 (`./trainer-host-v834f/…`) |
| `PROPOSAL.unsigned.json` | host-file digests, change list, sha256 of the manifest |
| `sign_host_v834f.py` | check / dry run / ROOT signing / regenerate the two files above |
| `harness/` | CPU test fixtures and drivers. Not part of the release. |

## Operator steps

1. Review the PR. Note `manifest_sha256` from `PROPOSAL.unsigned.json` as reviewed.
2. On a clean checkout: `python3 sign_host_v834f.py` (stage, manifest and proposal must agree; any extra file, also `__pycache__`, is refused).
3. Dry run, no seed is opened:
   `python3 sign_host_v834f.py --dry-run --base-private <host-v834e ROOT-ADMITTED private json> --source-root <admitted source closure>`
4. Sign on the host that holds the ROOT seed:
   `python3 sign_host_v834f.py --sign --base-private <…> --lifecycle <lifecycle private json> --source-root <…> --manifest-sha256 <reviewed value> --out <new release dir outside the checkout>`
   The tool verifies the written release with the launcher's own `checked()`.
5. Publish `<out>/trainer-host-v834f/` as `host-v834f` on the box. pe_box ships only host-v834e today; shipping host-v834f and setting `PIT_ALLOW_SHARDED_GPUS` is a separate change.
6. Flip `pit.allow_sharded_gpus` only after a first live test on an 8×RTX5090 box.

## Not verified yet

The sharded path has run only on CPU (torch FSDP over gloo, world 2 and 8, tiny
model). It has **not** run on CUDA/NCCL, on the 9B model, or with the fla
training kernels.

- GPU memory: the per-GPU numbers are estimates (hand counts, see
  `v835_zero3.estimate_peak_gib`). The update measures one zero-weight step on
  every rank before it produces a gradient. A rank over the limit moves its
  checkpointed layer inputs to host memory and all ranks measure again. What
  is still over the limit is refused. The first live test gives the real numbers.
- Host memory: about 129 GiB for the 9B model by count
  (`v835_zero3.host_ram_gib`), plus up to about 70 GiB while the layer inputs
  of a full-context micro-batch are on the host. Not measured.
- fp32 master with bf16 weights: the CPU launcher runs load fp32 weights (the
  real runtime refuses bf16 on a CPU), so there the master equals the weights.
  The bf16 case is tested without the DP world only.

## Non-goals

- No `host.private_json()` bypass of `checked()`.
- No engine overlay: `stage/trainer-config.public.json` restates no signed setting.
- No implicit in-repo trainer ship (`PIT_USE_INREPO_TRAINER` / `PIT_TRAINER_HOST` only).
