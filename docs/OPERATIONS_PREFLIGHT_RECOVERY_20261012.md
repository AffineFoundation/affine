# Historical training dispatch recovery

Epoch 157 reached the training phase but stopped before remote dispatch. The
controller's new routing wrapper passed the original signed H10 policy directly
to its preflight checker. That document preserves an older top-level checker
configuration: H10 normally replaces that field with its signed independent
dispatch configuration when constructing the running service. The wrapper had
omitted that projection, so the original checker correctly refused the mismatch.

The recovery restores that exact effective-policy projection after authenticating
the historical source, authority, transition helper and nested checker declaration.
It then runs the unchanged original checker. Earlier historical manifests retain
their original policy; the future source route is unchanged. The issued manifest,
prepared training job, checkpoint and optimizer lineage are preserved.

The original prepared epoch-157 job reproduced the failure, then passed the
unchanged CPU preflight with the recovery. That check reached the model-factory
sentinel and rejected a deliberately foreign runtime without loading a model.
The recovery has been installed, but this validation does not establish that
epoch 157 has completed numerical training. Portable tests cover the lost
projection, unchanged earlier routing and refusal of altered policy dependencies.

This is an operator runtime fix. It changes no sampling rules, batch quotas,
miner client requirements or website appearance. The fixed epoch 158–160 coverage
trial and its original comparator remain unchanged; record this operational
interruption separately when reporting epoch timings.
