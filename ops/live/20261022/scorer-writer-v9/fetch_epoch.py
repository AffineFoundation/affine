"""READ-ONLY R2 fetch of one sealed epoch's miner training documents (LIST + GET only; never PUT/DELETE).
Writes a compact private jsonl.gz (0600): full miner public key, uid (epoch registrations), slot, task index,
checkpoint built on, trained flag, prompt tokens, rollouts (output tokens, reward/classification).
v9.1: rewards are the lifecycle's NATIVE grades (signed epoch dispatch in AUDIT-POPULATION.json: selected + excluded rows that
carry native_grades), bound to the exact batch (sha256) and each output (output_sha256). Documents the host did not natively grade
(e.g. native-claim-invalid, i.e. miner-claimed outcome disagreed) are dropped; the miner-claimed fields are never used as labels.
Credentials: existing lifecycle R2 env file read in-process; nothing secret is printed or written."""
import gzip, hashlib, json, os, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import boto3
canon = lambda v: json.dumps(v, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
from botocore.config import Config
ENVFS = ["/home/const/subnet120-rewrite/state/private/r2-readonly.env",   # preferred: read-only token
         "/home/const/subnet120-rewrite/state/private/r2-direct.env"]     # fallback: lifecycle key, used READ-ONLY (authorized)
B = "affine-verification-mock-20260930"; EP = "https://00523074f51300584834607253cae0fa.r2.cloudflarestorage.com"
S = Path("/home/const/subnet120-rewrite/state/mimo-mainnet-v1/epochs")
def client(envf):
    env = {}
    for l in open(envf):
        l = l.strip()
        if "=" in l and not l.startswith("#"):
            k, v = l.split("=", 1); env[k.replace("export ", "").strip()] = v.strip().strip("\"").strip("'")
    return boto3.client("s3", endpoint_url=EP, aws_access_key_id=env["R2_ACCESS_KEY_ID"], aws_secret_access_key=env["R2_SECRET_ACCESS_KEY"],
                        region_name="auto", config=Config(retries={"max_attempts": 5}))
def fetch(E, out):
    regE = E if (S / str(E) / "AUDIT-POPULATION.json").exists() else E - 1
    pop = json.loads((S / str(regE) / "AUDIT-POPULATION.json").read_bytes()); pop = pop.get("payload", pop)
    uid_of = {v["public_key"]: v["uid"] for v in pop["registrations_document"]["payload"]["registrations"].values()}
    jp = S / str(E) / "TRAINING-JOB.json"
    trained = set(json.loads(jp.read_bytes())["payload"]["training"]["documents"]) if jp.exists() else set()
    own = json.loads((S / str(E) / "AUDIT-POPULATION.json").read_bytes()); own = own.get("payload", own)   # fail closed: own epoch only
    dd = own["dispatch_document"]; dd = dd.get("payload", dd)
    native = {r["batch_sha256"]: r["native_grades"] for r in dd["selected"] + dd["excluded"] if r.get("native_grades")}
    c = None
    for envf in ENVFS:
        try:
            c = client(envf); c.list_objects_v2(Bucket=B, Prefix=f"private/affine-mimo9b-grpo-20261013-e{E:06d}/training/", MaxKeys=1); cred = Path(envf).stem; break
        except Exception: c = None
    if c is None: raise RuntimeError("no working read credential")
    pre = f"private/affine-mimo9b-grpo-20261013-e{E:06d}/training/"
    keys = [o["Key"] for p in c.get_paginator("list_objects_v2").paginate(Bucket=B, Prefix=pre) for o in p.get("Contents", [])]
    def one(k):
        b = c.get_object(Bucket=B, Key=k)["Body"].read()   # GET only
        d = json.loads(b); bt = d["batch"]; ro = bt["rollouts"]
        g = native.get(hashlib.sha256(canon(bt)).hexdigest())
        if g is None or g["miner"] != d["miner"] or len(g["results"]) != len(ro): return None
        for row, r in zip(g["results"], ro):
            if row["output_sha256"] != hashlib.sha256(canon(r["turns"][0]["output"])).hexdigest(): return None
        return dict(uid=uid_of.get(d["miner"]), miner=d["miner"][:16], miner_full=d["miner"], slot=d["slot"], index=bt["index"],
                    built_on=bt["checkpoint"][:16], trained=hashlib.sha256(b).hexdigest() in trained, prompt=ro[0]["turns"][0]["prompt"],
                    rollouts=[dict(out=r["turns"][0]["output"], reward=row["reward"], cls=row["classification"],
                                  claimed_matches=(r["reward"] == row["reward"] and r["classification"] == row["classification"]))
                             for row, r in zip(g["results"], ro)], labels="native")
    with ThreadPoolExecutor(32) as ex: allr = list(ex.map(one, keys))
    recs = [r for r in allr if r is not None]
    tmp = Path(str(out) + ".tmp"); fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with gzip.open(os.fdopen(fd, "wb"), "wt") as f:
        for r in recs: f.write(json.dumps(r) + "\n")
    os.replace(tmp, out)
    return dict(credential=cred, epoch=E, docs=len(recs), trained=sum(r["trained"] for r in recs), uids=len({r["uid"] for r in recs}), regs_from=regE,
                listed=len(allr), dropped_not_natively_graded=len(allr) - len(recs), native_grades_in_dispatch=len(native),
                claimed_disagreements=sum(not x["claimed_matches"] for r in recs for x in r["rollouts"]))
if __name__ == "__main__":
    print(json.dumps(fetch(int(sys.argv[1]), Path(sys.argv[2]))))
