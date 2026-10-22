# Audit verify workers (roles: verify), recorded 2026-10-21 22:20 UTC BEFORE any action
All three are Lium pods (1x H200, template "Pytorch (Cuda + DinD)"), driven from Arbos by user systemd units
(ssh "detached owned worker watcher" + reverse queue tunnel to 127.0.0.1:19183). The machines are NOT to be deleted.

| unit (Arbos)                      | Lium pod (huid / id)                                   | host:port          | $/h  | spent (lium ps 22:12Z) | uptime |
|-----------------------------------|---------------------------------------------------------|--------------------|------|------------------------|--------|
| affine-mimo-verifier-1-supervised | brave-matrix-b5 / 0457d905-9863-43b6-a392-c204e3fe4468 | 38.255.28.21:20098 | 5.38 | $2113.57               | 16.4 d |
| affine-mimo-verifier-2-supervised | brave-wolf-e3 / 47d94d93-c2f7-4507-a677-ec11a568a3a3   | 89.169.125.182     | 5.76 | $2262.85               | 16.4 d |
| affine-mimo-verifier-3-supervised | lunar-lion-21 / 2a9730c8-5605-47e0-a6aa-7b5185562cb2   | 89.169.102.43      | 5.76 | $2262.85               | 16.4 d |

Total $16.90/h = ~$405.6/day (~$12.2k/30 days); spent so far ~$6,639. Pod names affine-verifier-capacity-{5,6,8}-new-architecture-20261005.
Worker identities (lifecycle config, role verify): 5091a0..., 9d39df..., ea2d5e...
Plan: after v8.2.8 is verified on its first epoch, stop ONLY the Arbos units affine-mimo-verifier-{1,2,3}-supervised and -tunnel
(no login to the pods, nothing deleted). Stopping the units does not stop Lium billing; stopping/deleting the pods is Jacob's call.

## 2026-10-22 01:37Z: DELETED (Jacob approved 16:31 CR)
brave-matrix-b5 0457d905... final $2132.10; brave-wolf-e3 47d94d93... final $2282.69; lunar-lion-21 2a9730c8... final $2282.69. Units affine-mimo-verifier-{1,2,3}-{supervised,tunnel} stopped (still enabled). Rollback section A restart of these units will NOT work now: pods are gone.
