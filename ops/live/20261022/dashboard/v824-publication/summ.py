import json,datetime,sys
d=json.load(open(sys.argv[1]))
ts=lambda t: (datetime.datetime.utcfromtimestamp(t)-datetime.timedelta(hours=5)).strftime("%m-%d %H:%M CDT")
s=d["summary"]; print({k:s.get(k) for k in ("updated_at","epochs","evaluation_status","evidence_status","audit_evidence_status","audit_epochs_available","audit_epochs_expected","audit_evidence_refused","training_steps")})
print("updated", ts(s["updated_at"]))
print("training_run",{k:(v[:8] if isinstance(v,str) and len(v)==64 else v) for k,v in d["training_run"].items()})
for e in d["epochs"][:int(sys.argv[2]) if len(sys.argv)>2 else 13]:
    print(e["id"][-7:],"disp",e["display_epoch"],e["phase"],"fin",e["finalized"],ts(e["start"]),e["checkpoint"][:8],"audit",e.get("audit_breakdown_available"),"steps",(e["training"] or {}).get("steps"))
print("evaluations",len(d["evaluations"]))
for r in d["evaluations"][-12:]:
    print(r["epoch_id"][-7:],r["status"],r["env_id"],ts(r["timestamp"]),"%s/%s"%(r["successes"],r["count"]),r["mean_reward"],r["checkpoint"][:8])
print("incentive", ts(d["incentive"]["timestamp"]) if d.get("incentive") else None)
