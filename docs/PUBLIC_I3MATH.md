# Public i3math proposal controls

`subnet.public_i3math.candidates(messages)` uses only reset user messages. It
recognizes two original problem forms: a decimal digit-sum difference and a
four-move consecutive-card game. The first uses the visible integer's congruence
modulo nine; the second enumerates the legal moves and adversarial choices. It
returns the computed answer plus a same-format alternative. Unknown problems
return no proposals. It never accepts expected-answer fields, file paths, native
grader state, or held-out snapshots.

Actual original native grader controls for training indices 0 and 1 are stored
in `state/wide-extra-tasksets/i3math-public-generator-native-controls.json`.
Both produced one positive and one negative. These controls alone establish
neither model generation nor a scored/trained batch. The prospective extension
uses all 32 original materialized tasks, searches only the two recognized training
indices, and reserves indices 16–31 for fixed autoregressive evaluation. Other
training problem forms are not claimed solved by this generator.

This is a curated public-input proposal policy, not autonomous mathematical
solving or an unbiased sample of the model's unrestricted policy. The model's
sequence probabilities and seed choose among approved proposals; subsequent
proof verification and original environment replay remain mandatory.
