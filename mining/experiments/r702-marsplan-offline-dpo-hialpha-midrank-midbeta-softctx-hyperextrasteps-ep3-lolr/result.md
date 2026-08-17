# R702 p3734 result

**Status:** TRAIN live on zesty-comet-da GPUs 6,7

| knob | value |
|---|---|
| base | marsplan0624/affine-5gedzafcvg-queen@556d02a2 |
| β / r / α / lr | 0.1 / 32 / 128 / 1e-6 |
| max_len / steps / ep | 12288 / 10800 / 3 |
| pid | 828063 |
| wait→merge | armed |

**Note:** first launch failed `FATAL bad BASE` (mine.env pinned r252). Fixed by hard-pinning marsplan BASE in `start_r702.sh` after mine.env source.

**Also this pass:** R694 v4 n80 REFUTE m=−0.001579 (~−0.23×) thought✓161 B✓0.494; chall reaped; freed `/tmp/r694_merged`.
