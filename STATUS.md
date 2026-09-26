# Project status

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
