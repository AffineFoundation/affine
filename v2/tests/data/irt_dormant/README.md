# IRT dormant-row fixtures

- `r000066.json`, `r000069.json`: reduced projections of the public v1 rotation
  artifacts downloaded from `https://affine.io/public/pool/rotation/`. Retired
  rows, rule parameters, IDs and admitted rows are exact; full active/queue
  collections are reduced for fixture size. They are not full replay inputs.
- `irt-r000066-source.json`: the exact `dapo:10781` row from the public source
  snapshot of r000066, `irt-c2afc207598fe7734c3bf33cd7034784f99b81f5f6a8c69a0cb509b68188197f`.
- `irt-r000072-measured-admits.json`: exact measured `nmv2` rows from the
  public r000072 IRT snapshot. Nine belong to the r000066 admissions.
- `v2_public/`: a complete small v2 public rotation lineage, with its source
  IRT snapshot, active sets, decision, state and ranking. It follows the rule
  introduced at `6c3d5b64` and is replayed strictly by `replay_rotation`.

The tests require no network. During development, the full public r000066,
r000069, r000070, r000071 and r000072 decisions were also replayed against
their downloaded full public source snapshots and state artifacts.
