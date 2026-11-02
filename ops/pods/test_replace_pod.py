"""Offline tests for replace_pod.py (no Lium, no ssh): endpoint substitution on every config format the roles repoint,
excluded-pod guard, current-pod parsing, node choice + rent caps with a fake lium.  python3 -m pytest -q ops/pods/test_replace_pod.py"""
import importlib.util, json, os, stat, sys, tempfile, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TMP = Path(tempfile.mkdtemp())
FAKE = TMP / "lium"
FAKE.write_text("""#!/usr/bin/env python3
import json, sys
a = sys.argv[1:]
if a[:1] == ['ls']:
    g = a[a.index('--gpu') + 1]
    rows = [dict(huid='cheap-spot-1', gpu_type=g, gpu_count=8, available_gpu_count=8, price_per_gpu_hour=1.0, price_per_hour=8.0, tier='spot', disk_gb=2000),
            dict(huid='h-sec-2', gpu_type=g, gpu_count=8, available_gpu_count=8, price_per_gpu_hour=1.9, price_per_hour=15.2, tier='secure', disk_gb=2000),
            dict(huid='h-sec-1', gpu_type=g, gpu_count=8, available_gpu_count=8, price_per_gpu_hour=1.8, price_per_hour=14.4, tier='secure', disk_gb=2000),
            dict(huid='split-4', gpu_type=g, gpu_count=8, available_gpu_count=4, price_per_gpu_hour=1.5, price_per_hour=12.0, tier='secure', disk_gb=2000),
            dict(huid='tiny-disk', gpu_type=g, gpu_count=8, available_gpu_count=8, price_per_gpu_hour=1.0, price_per_hour=8.0, tier='secure', disk_gb=50),
            dict(huid='pricey', gpu_type=g, gpu_count=8, available_gpu_count=8, price_per_gpu_hour=9.0, price_per_hour=72.0, tier='secure', disk_gb=2000)]
    print(json.dumps(rows if g in ('H100', 'H200') else []))
elif a[:1] == ['ps']:
    print('[]')
else:
    sys.exit(1)
""")
FAKE.chmod(FAKE.stat().st_mode | stat.S_IEXEC)
os.environ.update(LIUM=str(FAKE), PODS_STATE=str(TMP / "state"))
spec = importlib.util.spec_from_file_location("rp", HERE / "replace_pod.py"); rp = importlib.util.module_from_spec(spec); spec.loader.exec_module(rp)

OLD = dict(id="aaaaaaaa-0000-4000-8000-000000000001", ip="198.51.100.7", port=40111)
NEW = dict(id="11111111-2222-3333-4444-555555555555", ip="1.2.3.4", port=20001, gpu="8xB200", price_h=70.0)


def test_gate_service_argv_list():
    t = ("PROBE_SSH = ['ssh', '-T', '-o', 'ConnectTimeout=15',\n"
         "             '-o', 'UserKnownHostsFile=/home/const/.lium/known_hosts/aaaaaaaa-0000-4000-8000-000000000001', '-p', '40111', 'root@198.51.100.7']\n"
         "MARGIN = 40111 + 0\n")
    out, n = rp.substitute(t, OLD, NEW)
    assert n == 1 and "known_hosts/11111111-2222-3333-4444-555555555555', '-p', '20001', 'root@1.2.3.4'" in out
    assert "MARGIN = 40111" in out           # a line that does not name the pod is never touched


def test_dropin_and_tuple_and_pp():
    old = dict(id="aaaaaaaa-0000-4000-8000-000000000002", ip="198.51.100.20", port=20222)
    t = ('ExecStart="/usr/bin/ssh" "-o" "UserKnownHostsFile=/home/const/.lium/known_hosts/aaaaaaaa-0000-4000-8000-000000000002" "-p" "20222" "root@198.51.100.20" "x"\n'
         "    'eval': ('198.51.100.20', 20222, 'aaaaaaaa-0000-4000-8000-000000000002', 40 * GB,\n"
         "exec ssh -o UserKnownHostsFile=/home/const/.lium/known_hosts/aaaaaaaa-0000-4000-8000-000000000003 -p 20333 root@198.51.100.20 \"$@\"\n")
    out, n = rp.substitute(t, old, NEW)
    assert n == 2
    assert '"-p" "20001" "root@1.2.3.4"' in out and "('1.2.3.4', 20001, '11111111-2222-3333-4444-555555555555'" in out
    assert "-p 20333 root@198.51.100.20" in out   # same node IP, other pod: untouched


def test_perf_line_subs_keep_slot_key():
    old = dict(id="aaaaaaaa-0000-4000-8000-000000000003", ip="198.51.100.20", port=20333)
    t = "PODS = {'calm-orbit-ef': dict(pp=RA + '/pp.sh', gpu='2xH200', cost=9.52, lium_id='aaaaaaaa-0000-4000-8000-000000000003', role='primary'),\n"
    subs = [["gpu='[^']*'", "gpu='{gpu}'"], ["cost=[0-9.]+", "cost={price}"]]
    out, n = rp.substitute(t, old, dict(NEW, gpu="2xB200", price_h=11.5), subs)
    assert n == 1 and "'calm-orbit-ef': dict(" in out and "gpu='2xB200', cost=11.5, lium_id='11111111-" in out


def test_excluded():
    assert rp.excluded(dict(name="casper-h100-ask-before-delete", huid="x"))
    assert rp.excluded(dict(name="affine-tobias-4xh200", huid="x"))
    assert rp.excluded(dict(name="whatever", huid="swift-wolf-bd"))
    assert rp.excluded(dict(name="mech-1", huid="q")) and rp.excluded(dict(name="tobias-mech1", huid="q"))
    assert not rp.excluded(dict(name="affine-probe-8xb200-1", huid="noble-shark-d6"))


def test_current_ref_formats():
    f = TMP / "pp.sh"
    f.write_text("exec ssh -T -o UserKnownHostsFile=$HOME/.lium/known_hosts/aaaaaaaa-0000-4000-8000-000000000004 -p 20444 root@198.51.100.79 \"$@\"\n")
    rp.ROLES["_t"] = dict(current_from=dict(file=str(f)))
    r = rp.current_ref("_t"); assert (r["id"], r["ip"], r["port"]) == ("aaaaaaaa-0000-4000-8000-000000000004", "198.51.100.79", 20444)
    f.write_text("PROBE_SSH = ['ssh', '-o', 'UserKnownHostsFile=/home/const/.lium/known_hosts/aaaaaaaa-0000-4000-8000-000000000001', '-p', '40111', 'root@198.51.100.7']\n")
    r = rp.current_ref("_t"); assert (r["id"], r["ip"], r["port"]) == (OLD["id"], OLD["ip"], OLD["port"])
    del rp.ROLES["_t"]


def test_candidates_cheapest_secure_whole_node():
    c = rp.candidates("prior")                     # H100 x8 whole node, secure+spot allowed, <= $32/h, disk >= 400
    assert [x["huid"] for x in c][:3] == ["cheap-spot-1", "h-sec-1", "h-sec-2"]
    assert all(x["huid"] not in ("split-4", "tiny-disk", "pricey") for x in c)
    c = rp.candidates("trainer")                   # secure only
    assert c[0]["huid"] == "h-sec-1" and all(x["tier"] == "secure" for x in c)
    assert rp.candidates("probe") == []           # 8xB200 only: no fallback to another GPU type


def test_rent_caps():
    st = TMP / "state"; st.mkdir(exist_ok=True)
    with open(st / "rents.jsonl", "w") as f:
        for i in range(3): f.write(json.dumps(dict(at=time.time() - 60, role="prior", event="rent")) + "\n")
        f.write(json.dumps(dict(at=time.time() - 90000, role="prior", event="rent")) + "\n")
    assert rp.rents_24h("prior") == 3 and rp.rents_24h("perf") == 0 and rp.rents_24h() == 3


def test_every_role_is_well_formed():
    for role, R in rp.ROLES.items():
        assert R["name"] in R["names"]
        if R.get("delegate"): continue
        assert R["gpu_specs"] and R["max_price_h"] > 0 and R["health"]
        for s in R["provision"]: assert s["type"] in ("clone", "script", "push_file", "push_dir", "pod_cmd", "golden_restore", "fetch_head")
        g = rp.gspec(role)
        if any(s["type"] == "golden_restore" for s in R["provision"]): assert g.get("paths"), role
        assert not rp.excluded(dict(name=R["name"], huid=""))


def test_probe_gpu_change_only_with_flag():
    assert rp.candidates("probe") == []
    c = rp.candidates("probe", allow_gpu_change=True)
    assert c and c[0]["gpu"] == "H200" and c[0]["tier"] == "secure"   # fake lium offers no B200/B300; H200 fallback, secure only
