# Stage 6A: address TF-IDF on a validation subset

This experiment uses **50,000 S1 entities** and **173,180 true links** drawn only from the saved 440,522-entity validation split. The subset is saved in `splits\stage6a_validation_subset.tsv`. It uses seed **2061** and proportionally allocates each S1 country × exact true-match-count stratum using largest remainders, then samples without replacement within each stratum. This is an experimental subset, not a new train/validation split.

Stage 4 candidates come from the saved packed pair artifact. Stage 5 name candidates come from the saved top-20 results at its selected K=10 per source. Neither was recomputed. All rows below use exactly these same sampled S1 entities and their ground-truth links.

## Sample balance

| Distribution | Category | Full validation | 50k sample | Difference |
| --- | --- | ---: | ---: | ---: |
| country | India | 39.927% (175,886) | 39.926% (19,963) | -0.001 pp |
| country | US | 60.073% (264,636) | 60.074% (30,037) | +0.001 pp |
| singleton | False | 94.416% (415,925) | 94.416% (47,208) | -0.000 pp |
| singleton | True | 5.584% (24,597) | 5.584% (2,792) | +0.000 pp |
| match_count | 0 | 5.584% (24,597) | 5.584% (2,792) | +0.000 pp |
| match_count | 1 | 5.381% (23,704) | 5.380% (2,690) | -0.001 pp |
| match_count | 2 | 16.961% (74,716) | 16.962% (8,481) | +0.001 pp |
| match_count | 3 | 24.084% (106,096) | 24.084% (12,042) | -0.000 pp |
| match_count | 4 | 21.911% (96,522) | 21.912% (10,956) | +0.001 pp |
| match_count | 5 | 14.578% (64,221) | 14.578% (7,289) | -0.000 pp |
| match_count | 6 | 7.532% (33,178) | 7.530% (3,765) | -0.002 pp |
| match_count | 7 | 2.915% (12,840) | 2.914% (1,457) | -0.001 pp |
| match_count | 8 | 0.839% (3,698) | 0.840% (420) | +0.001 pp |
| match_count | 9 | 0.191% (842) | 0.192% (96) | +0.001 pp |
| match_count | 10 | 0.024% (104) | 0.024% (12) | +0.000 pp |
| match_count | 11 | 0.001% (4) | 0.000% (0) | -0.001 pp |

## Candidate recall and volume

| Retrieval | True-link recall | India | US | S2 | S3 | Candidate pairs | Mean/S1 | P95 | P99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage 4 baseline | 60.663% | 44.122% | 71.684% | 59.799% | 61.471% | 1,216,802 | 24.34 | 120 | 238 |
| Stage 5 current best | 77.061% | 64.290% | 85.571% | 76.820% | 77.287% | 2,049,173 | 40.98 | 133 | 252 |
| Stage 5 + address K=5 | 93.304% | 89.624% | 95.756% | 93.964% | 92.688% | 2,424,163 | 48.48 | 141 | 260 |
| Stage 5 + address K=10 | 94.398% | 91.058% | 96.623% | 94.989% | 93.846% | 2,914,495 | 58.29 | 151 | 269 |

## Marginal value of address retrieval

Incremental links and candidate pairs are measured against the **same Stage 5 subset candidate set**, with exact duplicate pairs removed. These figures are candidate recall and candidate purity, not final matching precision.

| Address K per source | New true links | Added candidates | True links per 1,000 added candidates | Added candidates per new true link | Recall gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 28,129 | 374,990 | 75.013 | 13.3 | +16.243 pp |
| 10 | 30,023 | 865,322 | 34.696 | 28.8 | +17.336 pp |

Moving from address K=5 to K=10 adds **1,894** more true links and **490,332** more candidates, or **3.863 true links per 1,000 additional candidates**.

**Subset choice: address K=5 per source.** It captures most of the observed address recall gain with much better marginal yield than K=10. This is an experimental choice only; no full run was started.

## Configuration, runtime, and memory

The address channel uses normalized address text, `char_wb` 4–5 grams, `min_df=2`, `max_df=0.02`, sublinear term frequency, L2-normalized cosine similarity, a 0.05 score floor, 50,000-target sparse batches, and top 10 per source. Its vocabulary has **90,252** grams and its subset query matrix has **1,841,267** nonzeros. K=5 is a prefix of the same retrieval result; no second target scan was run. Country partitions and missing-country behavior follow Stage 5.

The subset address fit took **1.2 s**; the S2/S3 scan took **980.1 s**; total address fitting and retrieval took **981.3 s** (16.36 min). Of the scan, target vectorization used 576.9 s and sparse products used 181.1 s. End-to-end sampling, retrieval, and evaluation took **1029.0 s**. Approximate sampled peak process RSS was **0.62 GiB**.

For the selected K=5 configuration, a rough **incremental address retrieval** projection is **0.67 h** for all 440,522 validation queries and the same 10,320,219 training targets, or **1.91 h** for 1,732,544 test queries and 9,969,589 test targets. These use the measured K=10 scan as a conservative proxy for K=5, hold target parsing/vectorization approximately fixed per target, and scale sparse products and query fitting with query count. If Stage 4 and Stage 5 also had to be rerun sequentially, a coarse total is **1.68 h** for validation and **5.77 h** for test; the test total extrapolates their measured validation runtimes by query and target counts. Reusing the saved validation Stage 4/5 candidates requires only the incremental address work. Vocabulary size, country mix (including unseen France), memory pressure, and sparse-product density can change runtime materially. These are order-of-magnitude estimates, not measured full runs; candidate assembly time is excluded.

The experiment stops at this subset. No full-validation or test address retrieval was run, and no matching model was trained.
