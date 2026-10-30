# Standing rules (SN120 / Affine)

## No encrypted disk, ever (Jacob, 2026-10-30, permanent)
SN120/Affine never uses the encrypted disk (Lium pod /root on fuse.gocryptfs, or any gocryptfs mount) for anything: code, venvs,
site-packages, model weights/checkpoints/deltas, compile caches, prompt caches, rollouts, grades, logs or temp files. Every pod
(probe, trainer, evaluator, verifiers, any new pod) uses plain local disk (e.g. /opt, overlay /, NVMe/volumes). New pod setup
scripts must place everything on plain disk from the start; reviews must reject any path on gocryptfs. Reason: gocryptfs is a
single-daemon bottleneck (~50x slower writes, grading halved) and no security/compliance requirement of ours depends on it.
