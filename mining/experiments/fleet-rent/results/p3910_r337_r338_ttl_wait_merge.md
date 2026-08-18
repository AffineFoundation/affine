# p3910 — R337/R338 TTL + wait→merge for idle fills

- Contract: wvk=7, king reign35 tammy
- TTL: POST `/pods/{id}/schedule-removal` +24h
  - R337 `7beb566e-…` → Removal **2026-08-19T17:35:23Z**
  - R338 `78c72c7f-…` → Removal **2026-08-19T17:35:23Z**
- Armed wait→merge (no local king; stamp `*_scp_needed.p3910` after MERGE_DONE):
  - R796 GPUs 2,3 · R830 GPUs 4,5 on gentle-shark-35
  - R831 GPUs 2,3 · R832 GPUs 4,5 on calm-lion-9f
- Progress at arm: R796/R830/R831 ~75 · R832 ~80 · R337 online ~147/300 · R338 ~77/300 · R818 ~780/19k
- B300×8 stock 0; burn ~$366.50/h; bal ~$86454
