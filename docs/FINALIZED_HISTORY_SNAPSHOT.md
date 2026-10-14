# Finalized history publication snapshot

The evidence publisher freezes an inventory of signed MiMo finalizations and evaluation receipts before fetching historical inputs. It records the inventory hashes in its private prepared plan, then captures the publication cutoff. Preparation and pre-publication validation use the same inventory.

This prevents a training job that finishes during a long export from being checked against a timestamp from before it completed. It also prevents newly arriving evaluations from changing the prepared artifact set. New records are included in the next publication cycle. Existing records must still match their frozen hashes, original signatures, approved worker identity, checkpoint and dispatch bindings, and the original signed job deadline. An explicit historical cutoff remains strict; it does not grant extra report lifetime.

Only finalized epochs can enter the public catalog. The existing publisher remains the sole writer, with its original lock, conditional history update, GET-only links, and pre-publication reconstruction. Mining, grading, training, rewards and signed worker code are unchanged.

The former trainer-log collector targets the archived controller format. Its retirement does not remove historical logs, and it should not be pointed at a MiMo trainer as if the formats were interchangeable. MiMo's signed training metrics remain available through the finalized catalog; raw logs are explicitly marked unavailable when they lack an authenticated public evidence record.
