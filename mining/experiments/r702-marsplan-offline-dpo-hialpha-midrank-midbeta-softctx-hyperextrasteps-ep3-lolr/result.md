# R702 p3744 result

**Status:** CHALL_READY → **N80 LIVE** on zesty-comet-da GPUs 6,7/:8003 vs reign34 (wvk=7)

| knob | value |
|---|---|
| base | marsplan0624/affine-5gedzafcvg-queen@556d02a2 |
| β / r / α / lr | 0.1 / 32 / 128 / 1e-6 |
| max_len / steps / ep | 12288 / 10800 / 3 |
| merge | `/tmp/r702_merged` 16sh/66G |
| chall | vllm **833430** · Triton seed chall_r691 n_star=26 |
| n80 | sim **836081** · `r702_*_reign34_wvk7.json` |

**Also this pass:** R703 TRAIN→MERGE_DONE (16sh/66G) on zesty 4,5 — slot idle for next CHALL. R696 SCP still mid-pipe (~4.6G/2sh). B300 stock empty.
