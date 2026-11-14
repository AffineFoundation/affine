"""Bench node: make a king's base from the newest checkpoint and the LoRAs of the steps after it (the head chain).
usage: chain_rebuild.py DEST_ROOT WORK --python PY --repo-url URL --ref COMMIT --clone DIR --main-clone DIR --deps DIR [--device cpu|cuda:N]
stdin: {"descriptor": {"id": base hash, "files": {name: sha256}}, "head": version id, "chain": chain record, "links": links file}

Same on-disk result as gate_fetch.py: DEST_ROOT/<id>/ with the mark .verified. The mark is written only when
  1. the checkout of the chain code is exactly at COMMIT, with no changed file (else the tool does not run),
  2. the rebuild tool of that checkout (python -m sn120_v2.headfold rebuild) ended with code 0: it compares the files
     of each step with the chain record, and
  3. this script has made the sha256 of each file of the result and each one is the sha256 of the descriptor (the
     king's head manifest), and the result has no other file.
In every other case DEST_ROOT/<id>/ is not made and the last line of the output says why (status refused or failed).
The links file is on the disk (mode 0600) only while the tool runs. WORK keeps the tool's files after a cut, so the
next call goes on from the last whole step; after a good end they are deleted. One call at a time works in WORK: a
second one waits for the first."""
import argparse, fcntl, hashlib, json, os, re, shutil, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor

NEED = ("boto3", "botocore", "s3transfer", "python-dateutil")  # the tool's module imports the store client; the vLLM environment does not have it
PINS = "v2/bench/deploy/requirements-harness.txt"  # in the main checkout: the pinned versions of NEED


class Refused(Exception):
    """A check failed: the weights of this call are not used."""


def git(d, *a):
    p = subprocess.run(["git", "-C", d, *a], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    return p.returncode, p.stdout.strip()


def exact(d, ref):
    """True when the checkout in `d` is at commit `ref` and no tracked file differs from it."""
    (rc, head), (rc2, dirty) = git(d, "rev-parse", "HEAD"), git(d, "status", "--porcelain", "--untracked-files=no")
    return rc == 0 and rc2 == 0 and head == ref and not dirty


def checkout(a):
    """The directory of a checkout that is exactly at a.ref: the main checkout when it is that commit, else a second one."""
    if exact(a.main_clone, a.ref):
        return a.main_clone
    if not os.path.isdir(a.clone + "/.git"):
        shutil.rmtree(a.clone, ignore_errors=True)
        subprocess.run(["git", "clone", "-q", "--no-checkout", a.repo_url, a.clone], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if git(a.clone, "cat-file", "-e", a.ref + "^{commit}")[0] != 0:
        git(a.clone, "fetch", "-q", "origin")
    if git(a.clone, "cat-file", "-e", a.ref + "^{commit}")[0] != 0:
        git(a.clone, "fetch", "-q", "origin", a.ref)
    git(a.clone, "checkout", "-q", "--detach", "--force", a.ref)
    if not exact(a.clone, a.ref):
        raise Refused("the checkout of the chain code is not at commit %s (it is at %s): the tool was not run" % (a.ref, git(a.clone, "rev-parse", "HEAD")[1] or "no commit"))
    return a.clone


def tool_env(a):
    """The environment of the tool: the packages of NEED, at the versions of the pinned list, in a directory of their own."""
    env = dict(os.environ, PYTHONPATH=a.deps)
    if subprocess.run([a.python, "-c", "import boto3"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0:
        return env
    want = {n: None for n in NEED}
    for line in open(os.path.join(a.main_clone, PINS)):
        m = re.fullmatch(r"([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+-]+)", line.strip())
        if m and m.group(1).lower() in want:
            want[m.group(1).lower()] = line.strip()
    if None in want.values():
        raise Refused("the pinned list has no version for " + ", ".join(n for n, v in want.items() if v is None))
    p = subprocess.run(["uv", "pip", "install", "-q", "--python", a.python, "--target", a.deps, "--no-deps", *want.values()],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
    if p.returncode != 0:
        raise RuntimeError("the install of %s failed: %s" % (", ".join(NEED), p.stdout.strip()[-200:]))
    return env


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def last_refusal(log):
    with open(log, "rb") as f:
        f.seek(max(0, os.path.getsize(log) - 4000))
        lines = f.read().decode("utf-8", "replace").splitlines()
    return next((line for line in reversed(lines) if line.startswith("refused: ")), "")


def rebuild(a, blob):
    desc, head = blob["descriptor"], blob["head"]
    cid, files = desc["id"], desc["files"]
    hex64 = re.compile(r"[0-9a-f]{64}")
    if not (hex64.fullmatch(cid) and hex64.fullmatch(head) and re.fullmatch(r"[0-9a-f]{40}", a.ref) and re.fullmatch(r"cpu|cuda:\d+", a.device)):
        raise Refused("an id, the commit or the device of this call is not well formed")
    if not files or any(not n or "/" in n or n.startswith(".") for n in files):
        raise Refused("the descriptor has an unsafe file name")
    dest = os.path.join(a.dest_root, cid)
    if os.path.exists(os.path.join(dest, ".verified")):
        return dict(ckpt=cid, status="present")
    t0 = time.time()
    clone = checkout(a); env = tool_env(a)
    chain, links, log = (os.path.join(a.work, n) for n in ("chain.json", "links.private.json", "rebuild.log"))
    with open(chain, "w") as f:
        json.dump(blob["chain"], f)
    try:
        with os.fdopen(os.open(links, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
            json.dump(blob["links"], f)
        with open(log, "ab") as err:
            p = subprocess.run([a.python, "-m", "sn120_v2.headfold", "rebuild", "--chain", chain, "--work", a.work, "--links", links, "--head", head,
                                "--device", a.device], cwd=os.path.join(clone, "v2"), env=env, stdout=subprocess.PIPE, stderr=err, universal_newlines=True)
    finally:
        if os.path.exists(links):
            os.remove(links)
    if p.returncode != 0:
        why = last_refusal(log) or "the rebuild tool ended with code %d" % p.returncode
        if p.returncode == 2:
            raise Refused(why)
        raise RuntimeError(why)
    try:
        result = json.loads(p.stdout.strip().splitlines()[-1]); base = result["base_dir"]
    except (ValueError, IndexError, KeyError, TypeError):
        raise Refused("the rebuild tool gave no result line")
    if (result.get("version_id"), result.get("base_hash")) != (head, cid) or not os.path.isdir(base):
        raise Refused("the rebuild tool made head %s, not the head of this row" % str(result.get("version_id"))[:12])
    have = sorted(os.path.relpath(os.path.join(d, n), base) for d, _, names in os.walk(base) for n in names)
    with ThreadPoolExecutor(8) as ex:
        got = dict(zip(have, ex.map(sha256, [os.path.join(base, n) for n in have])))
    if got != dict(files):
        bad = sorted(set(got) ^ set(files)) + sorted(n for n in set(got) & set(files) if got[n] != files[n])
        shutil.rmtree(base)
        raise Refused("the rebuilt base is not the base of the head manifest (sha256 mismatch: %s); it was deleted" % ", ".join(bad)[:160])
    with open(os.path.join(base, ".verified"), "w") as f:
        json.dump(dict(files=files, at=time.time(), by="chain-rebuild", chain_ref=a.ref, started_from=result.get("started_from"), steps=result.get("steps")), f)
    if os.path.exists(dest):
        shutil.rmtree(dest)
    os.replace(base, dest)
    for d in ("cache", "fetch", "fold"):
        shutil.rmtree(os.path.join(a.work, d), ignore_errors=True)
    return dict(ckpt=cid, status="rebuilt", seconds=round(time.time() - t0, 1), steps=result.get("steps"), started_from=result.get("started_from"),
                fetched_bytes=result.get("fetched_bytes"), chain_ref=a.ref, device=a.device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dest_root"); ap.add_argument("work")
    for name in ("--python", "--repo-url", "--ref", "--clone", "--main-clone", "--deps"):
        ap.add_argument(name, required=True)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    blob = json.load(sys.stdin)
    os.makedirs(a.work, exist_ok=True)
    lock = open(os.path.join(a.work, ".lock"), "w")
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        out, rc = rebuild(a, blob), 0
    except Refused as e:
        out, rc = dict(status="refused", why=str(e)[:400]), 2
    except Exception as e:
        out, rc = dict(status="failed", why=(type(e).__name__ + ": " + str(e))[:400]), 1
    print(json.dumps(out))
    return rc


if __name__ == "__main__":
    sys.exit(main())
