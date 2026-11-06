import pickle, requests, numpy as np, json
recs = pickle.load(open("/root/kl/data/pool/shard3.pkl", "rb"))
al = [r for r in recs if r["src"] == "alpaca"][:6] + [r for r in recs if r["src"] == "writing"][:3]
for r in al:
    ids = r["pids"].tolist() + r["rids"].tolist()
    j = requests.post("http://localhost:8001/v1/completions", json=dict(model="teacher", prompt=ids, max_tokens=1, prompt_logprobs=0, temperature=1.0)).json()
    pl = j["choices"][0]["prompt_logprobs"]
    n0 = len(r["pids"]); srv = []
    for t, d in zip(r["rids"].tolist(), pl[n0:]):
        srv.append(d[str(t)]["logprob"])
    srv = np.array(srv)
    print(r["src"], len(srv), "offline mean lp %.4f server-prompt mean lp %.4f  mean|diff| %.4f" % (r["lp"].mean(), srv.mean(), np.abs(srv - r["lp"]).mean()))
