"""Pre-build terminal-lego (and other local_docker_build) images for the replay shard so the
driver's batches skip the per-batch build step. Same shard hash as rollouts.prepass."""
import hashlib, json, subprocess, sys, shutil, time, os
from concurrent.futures import ThreadPoolExecutor
src, uids_file, shard, par, min_free_gb = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5])
i, n = (int(x) for x in shard.split("/"))
want = {l.strip() for l in open(uids_file) if l.strip()}
rows = [json.loads(l) for l in open(f"/root/rollouts-data/catalogs/{src}.jsonl")]
rows = [r for r in rows if r["uid"] in want and int(hashlib.sha256(r["uid"].encode()).hexdigest()[:8], 16) % n == i]
env = dict(os.environ); env["PATH"] = "/root/rollouts/rollouts/dockerwrap:" + env["PATH"]
def have(img): return subprocess.run(["docker", "image", "inspect", img], capture_output=True).returncode == 0
def build(r):
    img, d = r["image"], r["task_dir"] + "/environment"
    if have(img): return "cached"
    if shutil.disk_usage("/var/lib/docker").free / 1e9 < min_free_gb: return "lowdisk"
    p = subprocess.run(["docker", "build", "-q", "-t", img, "-f", d + "/Dockerfile", d], capture_output=True, text=True, timeout=1800)
    return "built" if p.returncode == 0 else "failed"
print(f"prebuild {src} shard {shard}: {len(rows)} tasks, parallel {par}", flush=True)
t0 = time.time(); stats = {}
with ThreadPoolExecutor(par) as ex:
    for k, res in enumerate(ex.map(build, rows), 1):
        stats[res] = stats.get(res, 0) + 1
        if k % 50 == 0: print(f"{k}/{len(rows)} {stats} {time.time()-t0:.0f}s", flush=True)
print(f"done {stats} {time.time()-t0:.0f}s", flush=True)
