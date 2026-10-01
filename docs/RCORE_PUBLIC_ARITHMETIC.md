# Original RCore public arithmetic prerequisite

This is an operator native-grader qualification, not a miner epoch or training
result. The public policy reads only the original user prompt and parses a
bounded arithmetic expression with `ast` and `Fraction`; it does not read the
reference answer, task metadata or held-out answers. It qualifies one original
training fixture, index 0. Its public-derived answer `0.5` earns original reward
1; the wrong answer `1.5` earns approximately 0.0012726 and is classified as
negative. The original partial-reward function is preserved.

The policy bounds prompts to 4,096 characters, syntax to 96 AST nodes, numeric
literals to 64 characters and exponents to ±128. It bounds intermediate rational
sizes and magnitudes before rendering, uses 17 significant digits, and rejects
identical candidate strings. Five controls cover the original expression,
malicious syntax, extreme exponents, oversized prompts and the distinct
1,000,000,000,000 versus 1,000,000,000,001 case. These checks do not guarantee
positive/negative classification for arbitrary arithmetic tasks; every proposed
fixture still needs its original grader.

The private original snapshot has 64 tasks from pinned dataset revision
`9314cc59dbcbd32f367716141ff660d02cd9731a`, with indices 0–31 reserved for
mining and 32–63 for held-out evaluation. Snapshot SHA-256 is
`aed8f5dce6689a831cbec75d6d6e39835c119bcf8cd963d1c353d8e38846dd38`.
All 64 normalized public prompts are distinct across the split. There are 38
generator kinds overall, rather than 32 equally represented kinds in both
partitions; semantic independence has not been established. The arithmetic
policy was checked only on training index 0.

Materialization wrote the complete snapshot, then exited 134 during Hugging
Face streaming shutdown (`PyGILState_Release`). That failed process record is
preserved. Fresh separate processes reloaded every task and invoked all 64
original graders successfully after scoped dependency installation. Earlier
36-, 10-, 8- and 5-error reports remain separate evidence. Successful grader
execution is distinct from solving: constant wrong controls retained their real
zero, partial or positive original rewards without rewriting them.

Dependencies were installed only into an owned isolated directory; the shared
Python environment was unchanged. The original grader additionally uses these
resource bytes, which require explicit qualification on another machine:

| Resource | SHA-256 |
| --- | --- |
| Vampire v4.9casc2024 binary | `ce3b39047565a3980e3d490315ff129f0eca5f8fbd40eeb0d6850122eb295b54` |
| NLTK WordNet corpus ZIP | `cbda5ea6eef7f36a97a43d4a75f85e07fccbb4f23657d27b4ccbc93e2646ab59` |

The private dependency profile records every installed file hash, original
snapshot/source pins, resource URLs and observed operator locations. Root
independently repeated all 64 original grades and both arithmetic controls,
checked the dependency and resource bytes, and recorded
`state/rcore-tasksets/original64/root-full-native-grader-qualification-check.json`.
No model computation, TOPLOC proof, sampled K/L batch, remote resource
qualification, common training, held-out learning result or chain write is
claimed. A future miner admission must bind the grader/resource profile and
qualify isolation and full numerical/native replay in a new signed source.

Run the bounded policy controls from a checkout with:

```sh
python3 -m unittest discover -s tests -p test_public_rcore.py -q
```
