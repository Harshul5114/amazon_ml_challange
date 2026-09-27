# Project status

## Final submission — complete (27 September 2026)

- User requested a valid submission immediately, so the long Stage 4/TF-IDF/model test runs were stopped and a conservative exact-key fallback was generated with `src/generate_quick_submission.py`. This is the submitted method; earlier prototype validation scores do not apply to it.
- Final files: `output/matching_results.tsv` and `output/candidate_pairs.tsv`. Both have 1,732,544 test S1 rows. There are 84,110 links on 77,328 non-empty S1 rows; 1,655,216 rows are empty.
- `python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test --check-ids` passed with no blocking issues. It verified all 9,969,589 test S2/S3 IDs for existence checks.
- Final package: `output/albertnewtonepsteinnetanyahu_submission.zip` (19,559,809 bytes), containing both TSVs, runnable code, instructions, requirements, and a filled methodology document. `utils/package_submission.py` tested ZIP integrity and wrote `output/package_manifest.json` with checksums.
- Warning: no labeled validation score was measured for the fallback; it is deliberately strict and may have low recall. The portal score is unknown.
- Next step: upload `output/matching_results.tsv` to the leaderboard; submit the ZIP for the final package requirement.

## Stage 1 — Dataset profiling: complete

- Raw TSVs were profiled without modification: `reports/data_profile.md`.

## Stage 2 — Validation and evaluation: complete

- `src/evaluate.py` implements the challenge's per-S1 F0.5 and macro average, with explicit singleton handling and set comparison. Its CLI can score a saved split partition with `--split-file splits/s1_train_validation.tsv --partition validation`.
- `splits/s1_train_validation.tsv` stores one train/validation assignment per training S1 ID. `splits/s1_split_manifest.json` records the fixed seed (2026) and SHA-256 assignment rule. The split is 1,766,299 train S1 entities (80.038%) and 440,522 validation S1 entities (19.962%). Ground truth covers exactly these S1 IDs.
- Balance details are in `reports/validation_split.md`. Singleton rates are 5.585% train and 5.584% validation. The largest difference in matches-per-S1 shares is 0.076 percentage points; the largest source-category difference is 0.022 points; country shares differ by 0.118 points.
- Validation sanity scores: perfect prediction 1.000000000000; all-empty prediction 0.055836030891; one extra false ID per S1 0.751672522042.
- Unit tests: 12 passed with `python -m unittest discover -s tests -v`.

## Stage 3 — Conservative text normalization: complete

- `src/normalize.py` adds six derived fields beside the raw name and address: normalized text, ordered tokens, and numeric tokens for each. It applies NFKC, casefolding, punctuation separation, and whitespace collapse without transliteration, legal suffix removal, or address abbreviation rules.
- `reports/normalization_audit.md` records a fixed 5% sample across all training sources (625,563 records), examples, collision counts, and observed risks. Among sampled records, 16,024 normalized names represent multiple distinct raw names; 52,489 records (8.391%) belong to these collision groups. Punctuation contributes to 7,951 groups covering 30,818 records (4.926%).
- Unit tests: 22 passed with `python -m unittest discover -s tests -v`.

## Stage 4 — First blocking baseline: complete

- `src/blocking.py` provides modular reverse inverted-index retrieval over the fixed validation S1 set. Rules: exact normalized name, sorted name token signature, exact numeric address, and shared address number plus rare name token. No target pairwise scan or model is used.
- `reports/blocking_baseline.md` contains rule-by-rule and union results, source/country/match-count breakdowns, singleton candidate behavior, runtime/memory, and 25 reproducibly sampled missed true links. Raw measured results are in `reports/blocking_results.json`; the run is reproducible with `python -m reports.evaluate_blocking` and the report with `python -m reports.render_blocking_report`.
- Union validation candidate recall: **60.609%** (924,782 / 1,525,811 true links). S2 recall **59.692%**; S3 **61.468%**. India **43.948%** versus US **71.709%**. Only **31.730%** of non-singleton S1 entities have every true link retrieved.
- Candidate volume: **10,707,669** pairs, mean **24.31**, median **6**, P95 **118**, P99 **241**, max **1,075** per S1. True singleton entities receive mean **22.46** candidates. Runtime **1,214.5 s**; approximate peak RSS **0.90 GiB**.
- Warning: this baseline falls well short of the high-recall objective. The fixed miss sample points to legal suffix/name truncation changes, Latin ↔ Indian-script transliteration, short or altered address numbers, and address-only evidence. The high candidate tail needs attention before downstream scoring.
- Unit tests: 26 passed with `python -m unittest discover -s tests -v`.
## Stage 5 — Character TF-IDF name retrieval: complete

- `src/tfidf_retrieval.py` adds country-aware, sparse character 4–5 gram TF-IDF retrieval on `name_norm`. It streams S2/S3 in 100,000-row batches and retains the best 20 targets per S1 and target source. Countries are handled dynamically, including unseen France and missing values. It forms no dense all-pairs similarity matrix.
- The project `.venv` contains the dependencies pinned in `requirements-stage5.txt`. The full run is reproducible with `.venv\Scripts\python.exe -m reports.run_tfidf_retrieval`, then `.venv\Scripts\python.exe -m reports.evaluate_tfidf_union`, then `.venv\Scripts\python.exe -m reports.render_tfidf_report`. Full results and 25 reviewed misses are in `reports/blocking_tfidf.md`; numeric results are in `reports/blocking_tfidf_results.json`. Large reusable arrays are excluded from version control under `artifacts/`.
- At selected K=10 per target source, TF-IDF alone reaches **59.687%** true-link recall; the exact union with Stage 4 reaches **77.085%** (1,176,168 / 1,525,811). This adds **251,386** true links, or **16.476 percentage points**, beyond Stage 4. Union S2 recall is **76.728%**, S3 **77.419%**, India **64.224%**, and US **85.653%**.
- The selected union has **18,043,178** candidate pairs, mean **40.96** per S1, P95 **132**, P99 **255**, and maximum **1,080**. K=20 raises union recall to **79.063%** at **26,242,184** pairs. The report includes K=5, 10, and 20 tradeoffs.
- TF-IDF fitting and retrieval took **2,442.4 s** (40.71 min), sampled peak RSS **1.06 GiB**. Exact Stage 4 pair replay took **214.3 s** concurrently; union evaluation sampled peak RSS **2.00 GiB**.
- The fixed miss sample is led by cross-script names (7/25), name truncation or addition (4/25), text noise (3/25), and concatenated web/hashtag names (3/25). Links with unrelated or opaque names but strong addresses remain missed. The selected union still misses **22.915%** of true links, especially in India.
- Unit tests: 29 passed with `.venv\Scripts\python.exe -m unittest discover -s tests -v`.
- Recommended next retrieval investigation: address evidence and cross-script names, while controlling candidate volume. No address TF-IDF, transliteration retrieval, embeddings, pairwise features, or ML matcher has been implemented.

## Stage 6A — Address TF-IDF subset experiment: complete

- `splits/stage6a_validation_subset.tsv` fixes a seed-2061 sample of **50,000** IDs from the existing validation split. Proportional country × exact match-count strata preserve India/US, singleton, and match-count distributions within 0.002 percentage points per displayed category.
- `src/tfidf_retrieval.py` now accepts an address text field while keeping the Stage 5 name default. `reports/run_address_tfidf_subset.py` performs one sparse address top-10 pass over training S2/S3 targets and evaluates K=5 and K=10 prefixes on the subset. It reuses saved Stage 4 pairs and Stage 5 name results; no name TF-IDF was recomputed. Full metrics and timing are in `reports/address_tfidf_subset.md` and `reports/address_tfidf_subset_results.json`.
- On the same subset, Stage 4 recall is **60.663%**, Stage 5 **77.061%**, Stage 5 + address K=5 **93.304%**, and Stage 5 + address K=10 **94.398%**. K=5 adds **28,129** true links and **374,990** candidates beyond Stage 5: **75.013 true links per 1,000 added candidates**. Moving K=5 → K=10 adds only **1,894** links among **490,332** further candidates (**3.863 per 1,000**). India rises from **64.290%** to **89.624%** at K=5.
- Subset address fitting and retrieval took **981.3 s** (16.36 min); end-to-end subset run took **1,029.0 s** and sampled peak RSS **0.62 GiB**. Approximate incremental address runtime projection: **0.67 h** for full validation or **1.91 h** for test, with substantial uncertainty from scale and France. K=5 is the preferred subset configuration; no full Stage 6 run was started.
- Next decision: whether to authorize full validation address retrieval at K=5 and assess remaining misses and candidate quality. No matcher or downstream model has been trained.

## Stage 6B — Detailed subset miss analysis: complete

- `reports/analyze_stage6b.py` exactly reconstructed Stage 6A K=5/K=10 candidate hits from cached Stage 4, name TF-IDF, and address TF-IDF arrays. It read raw records only for the **11,596** missed true links and computed field presence, dominant name script, token-overlap, and shared address-number diagnostics. Full results are in `reports/stage6b_miss_analysis.md` and `reports/stage6b_miss_summary.json`.
- Stage 5 + address K=5 misses **11,596 / 173,180** subset true links (**6.696%**). India misses **7,185 / 69,246** (**10.376%**); US **4,411 / 103,934** (**4.244%**). S2 misses **5,047 / 83,614** (**6.036%**); S3 **6,549 / 89,566** (**7.312%**).
- All missed pairs have names on both sides. **2,115** targets lack an address. **3,373** names use detectably different scripts, all in India. **5,472** pairs have zero exact normalized name-token overlap; **7,123** share at least one address number, but only **2,602** share a number of at least three digits.
- The cached address K=10 list contains **1,894 / 11,596** K=5 misses (**16.333%**) at Stage 6A's cost of 490,332 extra pairs. Cached name top-20 contains **1,027** K=5 misses, with no detectably different-script pairs.
- A seed-2062 SHA-256 sample of **150** missed links was manually assigned one primary category each; `reports/stage6b_miss_sample_labeled.tsv` retains every raw pair and diagnostics. Approximate sample frequencies: cross-script **31.3%**, shortened/alternate/web names **23.3%**, spelling variation **18.0%**, different/missing address **16.0%**, opaque or possibly corrupted names **7.3%**; other categories are smaller. Labels are subjective and may hide secondary issues.
- Recommended next retrieval investigation: script-aware transliteration as an added name representation, scoped and evaluated for incremental recall and candidate volume. No new retrieval channel, full validation/test address run, or matching model was started.

## Stage 7A — Transliteration-assisted name retrieval on the 50k subset: complete

- `src/transliterate.py` adds an optional `name_translit` representation using offline AnyAscii 0.3.3; raw fields and `name_norm` are preserved. AnyAscii uses the permissive ISC license, with attribution in `THIRD_PARTY_NOTICES.md` and a pin in `requirements-stage7a.txt`. It is a text utility, not a final ML model, and performs no external entity lookup.
- `src/transliteration_retrieval.py` scans only target names with predominantly non-Latin letters; all 50,000 sampled S1 names are Latin-script. The country-aware sparse TF-IDF pass evaluated K=5 and K=10 from one run. Existing Stage 4/5/address candidates were reused. Detailed results are in `reports/transliteration_subset.md` and `reports/transliteration_subset_results.json`.
- Current Stage 6A best recall **93.304%** rises only to **93.319%** at transliteration K=5 (+25 true links, +156,900 pairs) or **93.328%** at K=10 (+42 links, +309,998 pairs). India recall changes **89.624% → 89.660%** at K=5. Only **25/3,373 (0.741%)** previously missed cross-script links are recovered at K=5. Incremental yield is **0.159 true links per 1,000 added pairs**.
- The targeted retrieval took **255.9 s**; complete subset evaluation took **312.9 s**, peak RSS **0.65 GiB**. A direct-score sample showed median true-pair cosine **0.070** versus median top-10 cutoff **0.305**. Transliteration also merges multiple distinct normalized names into **14,309** output strings; many are cross-script equivalents, but the low candidate yield warns of confusability.
- Decision: do not include this transliteration channel in the current candidate baseline. **11,571** links remain missed after K=5, including **3,348** detected cross-script links. A future phonetic/language-aware approach would need a separate subset experiment. No full validation/test retrieval or ML matching model was run.

## Stage 8 — Full Validation Candidate Generation: complete

- `reports/run_full_validation_retrieval.py` executes self-sufficient, end-to-end candidate generation over the entire validation split (**440,522** S1 entities). It dynamically regenerates missing cache components (Stage 4 baseline rules, Name TF-IDF K=10, Address TF-IDF K=5) and computes the exact multi-channel union.
- Optimized implementation with fast line-based TSV streaming and 12-thread sparse matrix multiplications (`sp_matmul_topn`).
- Full validation candidate results:
  - **Union candidate recall: 93.342%** (1,424,219 / 1,525,811 true validation links retrieved, adding **499,437** true links or **+32.733 percentage points** over Stage 4 baseline).
  - **Total candidate volume: 21,344,622** pairs across 440,522 S1 entities.
  - **Candidate statistics per S1**: mean **48.45**, median **30**, P90 **93**, P95 **140**, P99 **262**, max **1,084**, zero-candidate entities: **0**.
  - Total runtime: **1,756.8 s** (~29.3 minutes); peak RSS: **2.59 GiB**.
- Reusable serialized artifacts created under `artifacts/`:
  - `artifacts/stage4_pairs_packed.npy` (10,707,669 packed uint64 pairs)
  - `artifacts/tfidf_top20.npz` (Top-10 name TF-IDF scores and IDs)
  - `artifacts/address_tfidf_top10.npz` (Top-10 address TF-IDF scores and IDs)
  - `artifacts/full_validation_candidates_packed.npy` (21,344,622 packed candidate pairs)
  - `artifacts/validation_candidate_pairs.tsv` (Standard TSV candidate pairs format exported in 18.8 s)
  - `reports/full_validation_retrieval_results.json` (Full numerical metrics)

## Stage 9 — Pairwise Similarity Feature Engineering: complete

- `src/similarity_features.py` implements 15 pairwise similarity features covering exact matches, token overlap, containment, character n-gram similarity, numeric agreement & conflict, same-country agreement, missingness indicators, and source indicator (`FEATURE_NAMES`):
  1. `exact_name_match` (boolean indicator)
  2. `name_jaccard` (token Jaccard similarity)
  3. `name_containment` (token containment ratio)
  4. `name_len_diff_ratio` (length difference normalized by max length)
  5. `name_char_ngram_jaccard` (character 3-gram Jaccard similarity)
  6. `exact_address_match` (boolean indicator)
  7. `address_jaccard` (address token Jaccard similarity)
  8. `address_containment` (address token containment ratio)
  9. `numeric_overlap_count` (shared house/postal numeric token count)
  10. `numeric_jaccard` (numeric token Jaccard similarity)
  11. `numeric_conflict` (boolean indicator for non-overlapping significant numbers)
  12. `same_country` (boolean country match)
  13. `missing_s1_address` (boolean missingness indicator)
  14. `missing_target_address` (boolean missingness indicator)
  15. `target_is_s2` (boolean target source indicator)
- `tests/test_similarity_features.py` verifies all edge cases: exact matches, complete disjoints, numeric conflicts, missing addresses, empty inputs, and matrix dimensions (8 unit tests passing).
- `reports/generate_features.py` sampled 20,000 training S1 entities and generated 1,045,766 training pairs (**69,025** true match links and **976,741** hard negative candidate pairs retrieved from inverted blocking indices).
- `reports/augment_training_features.py` addressed candidate distribution shift by augmenting `artifacts/train_features.npz` with 150,000 diverse negative pairs (75,000 random background pairs and 75,000 address-sharing / name-mismatch pairs), bringing the final training set to **1,195,766** pairs (10.7 MB).

## Stage 10 — Tree-Based Match Model: complete

- `src/train_match_model.py` implements a histogram-based gradient-boosted decision tree matcher (`HistGradientBoostingClassifier`, conforming to user constraint "go for any tree based model rather than linear regression") optimized for entity-level macro F0.5.
- Model architecture: `HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=20, early_stopping=True, validation_fraction=0.1)`. The current code does **not** apply sample weighting; the previous status claim about square-root inverse-frequency weights was incorrect.
- Training performance: fitted on **1,195,766** pairwise candidate vectors with 15 features in **9.8 s** utilizing all CPU cores. Training ROC-AUC: **0.9995**. Serialized to `artifacts/tree_match_model.joblib`.
- Validation evaluation: evaluated on the stratified **50,000** validation subset against all **2,423,980** multi-channel candidate pairs generated from Stage 8 (Stage 4 baseline + Name TF-IDF K=10 + Address TF-IDF K=5). Scored all pairs in **47.1 s**.
- Threshold search: fine-grained sweep across high confidence thresholds [0.85, 0.9995] with top-1 prediction per target source (S2 and S3) to strictly control precision for the F0.5 metric.
- Primary metrics at optimal threshold (**T = 0.97**):
  - **Entity Macro F0.5: 0.454827**
  - **Pair Precision: 54.282%** (52,541 true positives vs 44,252 false positives)
  - **Pair Recall: 30.339%** (52,541 / 173,180 true links)
  - **Average predicted matches per S1: 1.936**
- Full metrics and threshold curve across all examined thresholds are recorded in `reports/model_evaluation_results.json`.
- Unit tests: `tests/test_train_match_model.py` passes all training, probability calibration, and thresholding assertions (39 total unit tests passing in repository).

## Stage 11 — Singleton false-positive audit and numeric conflict gate: complete

- The original Stage 10 model and feature artifacts were absent locally. A fresh model was trained from 5,000 fixed-seed development S1 entities using a deterministic cap of 50 retrieved candidates per S1/source plus the existing 150,000 augmented negatives. This is a **new model**, so its score must not be read as a direct reproduction of the Stage 10 score.
- The original feature generator produced 5,183,834 candidates from only 2,000 S1 and pushed free system memory below 1 GiB during a 20,000-S1 run. `reports/generate_features.py` now bounds retrieved training candidates and caches only selected target records. The completed 5,000-S1 run produced 535,520 training pairs after augmentation, including 17,244 positives.
- On the fixed 50,000-S1 subset, the cached Stage 4 + name K=10 + address K=5 union contains 2,424,163 pairs and 161,584 / 173,180 true links (**93.304% candidate recall**), exactly matching the earlier Stage 6A subset result. The 5,000 training S1 IDs and 50,000 validation IDs have zero overlap.
- At threshold 0.97 with top one per target source, the fresh model scores **0.657888 macro F0.5**, **81.843% pair precision**, **41.073% pair recall**, and **19.305% singleton accuracy**. There are 3,106 false links on 2,253 of 2,792 true singleton entities; 1,710 of those links have conflicting significant address numbers.
- A single tested rule rejecting candidates with `numeric_conflict = 1` improves macro F0.5 to **0.694588**, pair precision to **88.665%**, and singleton accuracy to **52.185%**. Pair recall falls to **40.617%**; 789 true links are lost, while 6,788 false links are removed. Non-singleton macro F0.5 rises from 0.685379 to 0.704805.
- `reports/singleton_false_positive_audit.json` contains sampled error cases; `reports/numeric_conflict_gate_results.json` contains exact metrics. The gate remains experimental and is not yet applied to the production prediction path. Tests now save toy models to temporary files instead of overwriting `artifacts/tree_match_model.joblib`.
- Warning: the large improvement over the earlier reported Stage 10 score primarily reflects a different bounded training sample and model. The numeric gate's within-model gain is the valid comparison. Next: verify the bounded model and gate on the full fixed validation set after generating its address candidate cache, then inspect the remaining singleton false positives and false negatives before adoption.



