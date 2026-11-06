# Loop: claim /root/kl/evalq/*.todo (lowest round first), run eval_ckpt.py in subprocess.
import glob, os, sys, time, subprocess, re
gpu = sys.argv[1]; mem = sys.argv[2] if len(sys.argv) > 2 else "0.85"
os.makedirs("/root/kl/evals", exist_ok=True)
while True:
    todos = glob.glob("/root/kl/evalq/*.todo")
    if not todos:
        if os.path.exists("/root/kl/evalq/STOP"): break
        time.sleep(10); continue
    todos.sort(key=lambda f: (int(re.search(r"__r(\d+)", f).group(1)) if "__r" in f else -1, f))
    f = todos[0]; run = f.replace(".todo", f".running{gpu}")
    try: os.rename(f, run)
    except OSError: continue
    path = open(run).read().strip(); name = os.path.basename(f)[:-5]
    t = time.time()
    r = subprocess.run(f"CUDA_VISIBLE_DEVICES={gpu} setsid python /root/kl/eval_ckpt.py {path} /root/kl/evals/{name}.json {mem} > /root/kl/logs/eval_{name}.log 2>&1", shell=True)
    subprocess.run(f"pkill -9 -f 'eval_ckpt.py {path} '", shell=True)
    os.rename(run, run.replace(f".running{gpu}", ".done" if r.returncode == 0 else ".failed"))
    print(name, r.returncode, time.time() - t, flush=True); time.sleep(15)
