# DeepMath-103K and NuminaMath-CoT

These are separate prose-math datasets. NuminaMath-CoT is not the existing Affine Numina Lean/tactic environment.

The pinned upstream revisions are `zwhe99/DeepMath-103K@5cf055d1fe3d7a2eb19719ac020211469736ae44` and `AI-MO/NuminaMath-CoT@9d8d210c9f6a36c8f3cd84045668c9b7800ef517`. Their dataset cards declare MIT and Apache-2.0 respectively; individual source-problem licensing is not independently certified. The qualification reads projected Parquet columns with bounded HTTP ranges. DeepMath solution columns are never loaded. Numina solution text is streamed to extract its last complete boxed reference and discarded; its chain-of-thought and messages are not retained.

| Corpus | Actual upstream rows | Retained train | New heldout | Usable official test |
|---|---:|---:|---:|---:|
| DeepMath-103K | 103,022 | 93,760 | 927 | — |
| NuminaMath-CoT | 859,494 train + 100 test | 738,575 | 7,500 | 91 |

The retained population excludes all 7,496 current MATH questions, including its 750 reserved questions. Conservative normalized-question hashes remove within/between-corpus duplicates and conflicting references. Proof-only prompts, missing/oversized references, failed bounded reference self-checks and context overflows are excluded. These checks establish extractability, context fit and reference self-equivalence, not correctness of every upstream answer or semantic independence of every question. Some references use the original grader's string fallback; that scope is recorded separately. No question or generated answer is truncated to force context fit.

`ops/qualify_math_corpora.py` inventories pinned projected data and token budgets. `ops/finalize_math_corpus_catalog.py` protects the complete official test population and checks references. `ops/prepare_math_corpus_shards.py` exports original MathTask snapshots; `ops/check_math_corpus_shards.py` independently rereads every asset against its exact catalog. The exports contain question and final reference, never CoT. Reset messages contain only the original MATH system instruction and exact question.

The 840,853 retained tasks produce 106 assets: 103 training shards and three separate heldout/test shards, each at most 8,192 rows. Together they compress to 128,090,377 bytes; the largest asset is 1,279,788 compressed bytes and 9,945,185 raw bytes. Assets are hydrated separately from executable source. A signed rotating manifest can select at most 64 environment rows without raising existing global index limits. Full shard pools remain public-authorized; an owned miner's smaller startup subset does not restrict external miners.

`math_corpus_provider.py` defines distinct versioned corpus/shard IDs and a snapshot-only alias to the unchanged original `affine_math_v1.MathTaskset`. Its environment hook is present in the prospective public checkout; deployed sources remain unchanged. `math_corpus_assets.py` checks admitted capability URLs, exact compressed/raw size and SHA, bounded decompression, native task class and question-only prompt structure, and refuses cache/symlink mutations. It does not authenticate signatures itself: job/manifest/source authentication must precede hydration. Corpus resources resolve through an absolute, unaliased `AFFINE_MATH_CORPUS_ASSET_ROOT` outside the readonly executable cache. Signed snapshot paths remain canonical and exact resource bytes are checked during native/source admission. Backend/bootstrap orchestration authenticates a separate task-asset registry before hydration; that integration requires reviewed qualification before deployment.

Native v2 controls exercised eight original tasks per corpus: 32 honest/wrong terminal grades, 32 fresh native replays and 80 rejected trace mutations. Prospective common alias controls exercised two tasks per corpus with eight additional fresh replays. These controls are a subset, not full native grading of the catalog. No new GPU proof, optimizer consumption or heldout improvement is claimed. Prospective plans remain unsigned until independent source/resource review and actual role-model qualification. Future qualification must bind grader dependency versions, actual model/runtime profile, full-vocabulary LP/TOPLOC, original native verification, R2 freeze/scoring, full optimizer attribution, fixed heldouts and successor publication.

## Pinned Qwen context audit

An operator-only CPU audit checked all 840,853 retained catalog rows, including 832,335 training rows, against `Qwen/Qwen2.5-Math-7B-Instruct@ef9926d75ab1d54532f6a30dd5e760355eb9aa4d`. It used the actual `text-tools-long-v2` question-only prompt builder with autoregressive temperature 0.8, top-p 1.0 and a full 1,024-token output reserve. Every row fit the 8,192-token runtime context limit: the maximum prompt was 3,347 tokens, or 4,371 including the reserve. The full audit took 233.65 seconds; 860 representative original MATH, DeepMath and Numina questions also passed. Every compressed and raw asset was checked against its published size and SHA before reading. No question was truncated, reindexed or moved between folds, and no published asset was changed.

The audit downloaded only these small pinned tokenizer/config files; it loaded no model weights and ran no GPU job. Their SHA-256 identities are:

| File | SHA-256 |
|---|---|
| `config.json` | `3a6dd7a4b3e6dd81c05ec943d837166508ceb953e1ce2baa57c84611d4765ef0` |
| `tokenizer.json` | `c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539` |
| `tokenizer_config.json` | `d743761ff02297cc5cdf38a612b0f78d5cdf5ceb3f251ab1d25ad353d24b1887` |
| `vocab.json` | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |
| `merges.txt` | `599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3` |

The catalog-context report SHA-256 is `b7d903bbe855e706f62b9ddb4d417b5c7064849cf08f069294a5aec9cf92a11d`; the prospective per-row eligibility stream SHA-256 is `5e5084ef73eeb7172b61b510dddf1d658928f6ec0f17c0d103cbf26f79d8347d`. These bind CPU context eligibility only. They do not establish native answer correctness, GPU proof qualification, a mined positive/negative pair, training or improvement.

Frozen source `a37b707462d911daa034a974cbdadab98df5513a935b2f530604786c44f36b3d` includes the matching provider, authenticated asset hydration and external-cache hooks. Future corpus manifests still need reviewed, signed 1,024-token environment specs and harness bindings. The current controller emits every configured environment row even when training groups deselect it, so the 93-row Numina plan cannot be passed wholesale through the 64-row manifest limit. DeepMath's 12 training shards plus one heldout fit 13 rows. Numina must be staged as separate bounded signed configurations: at most 16 training shards plus its heldout and official-test rows (18 rows), with the final 11 training shards plus those two reserved rows (13 rows). Group selection alone does not shrink the registry. Each manifest's signed task-asset registry must exactly match all its environment bindings, including empty-index reserved rows.

The owned miner may start with 16 indices from one training shard without reducing external miners' authorized indices in the current signed public pool. Rotating bounded Numina pools preserves the complete 738,575-task training population over time; it does not expose all 91 training shards in one manifest. All 927 DeepMath heldouts, 7,500 Numina heldouts and 91 usable official-test tasks remain reserved. Corpus-specific genuine GPU K1/L1 generation, both independent native/full-vocabulary/TOPLOC audits, ordinary deadline/freeze/scoring, a full-model gradient and optimizer update with exact pair attribution, immutable successor publication and paired fixed-heldout evaluations remain pending.
