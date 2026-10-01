The new `ops/materialize_native_tau2_disjoint.py` constructs an operator-private inventory from the pinned original telecom `full` pool, excluding the original `base` tasks. It preserves each original task body and groups tasks by the complete `user_scenario.instructions` object, excluding persona. A group belongs entirely to mining or heldout evaluation.

The measured original pool contains 2,171 eligible task IDs but only five instruction groups. The new 32-task pilot uses 16 untouched variants from two mining groups and 16 from three heldout groups. Variants are selected round-robin within each partition. This is a small, correlated scenario-disjoint pilot, not 32 independent scenarios. The earlier first-32 inventory remains historical control evidence and must not be relabeled as this dataset.

Run from a clean checkout with the pinned Tau2 provider installed:

```bash
python -m ops.materialize_native_tau2_disjoint \
  --data /path/to/pinned/tau2/data \
  --out /path/to/new/private/taskset \
  --count 32
```

The destination must be new. Its mode-600 private collection includes original user, database, and grading fields. The public collection contains only task identifiers, task hashes, scenario-group hashes, and split metadata. Keep the private collection out of miner source archives, public buckets, and logs. Taskset membership and split hashes must be authenticated by a new epoch manifest before use.

Five selection controls cover correlated persona/state variants, exact body preservation, disjoint partitions, insufficient groups, and duplicate/malformed inputs. Materialization executes no simulation, inference proof, or training update. Common fixed-auxiliary role admission still requires separate model and native verification and an unchanged auxiliary model and policy across checkpoint comparisons.
