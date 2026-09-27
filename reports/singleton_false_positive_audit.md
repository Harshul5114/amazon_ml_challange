# Singleton false-positive audit — fixed validation subset

## Question and setup

The merged Stage 10 report selected threshold 0.97 but reported only 2.1% singleton accuracy. Its saved model, training feature matrix, and full validation candidate artifact were absent from this workspace, so that result could not be replayed exactly. The local unit tests had also been writing toy models to the production artifact path; those tests now use temporary paths.

This experiment uses the **existing** 50,000-S1 validation subset. Training S1 IDs were sampled only from the fixed development partition (seed 2026); the 5,000 selected training IDs have zero overlap with the 50,000 validation IDs. The candidate set is the exact Stage 6A cache union: Stage 4 rules, name TF-IDF K=10/source, and address TF-IDF K=5/source. It has 2,424,163 distinct pairs and contains 161,584 of 173,180 true links (93.304% recall), matching `reports/address_tfidf_subset_results.json`.

The original unbounded feature generator attempted to hold 5,183,834 candidate pairs from just 2,000 S1 entities, and a 20,000-S1 run reduced available physical memory below 1 GiB. `reports/generate_features.py` now keeps at most 50 retrieved candidates per S1 and source by deterministic pair priority. It loads selected targets in a second streaming pass. A 5,000-S1 sample produced 385,520 pairs before augmentation and 535,520 after the existing 150,000 negative examples; 17,244 pairs are positive. A histogram gradient boosting model with the Stage 10 hyperparameters was fitted on that matrix. No sample weights are used in the current implementation.

The new training distribution means this model is **not** the Stage 10 model. Its baseline and gated decisions are compared to each other on the same candidates, scores, entities, and threshold.

## Error findings

At threshold 0.97 and top one candidate per target source, 2,253 of 2,792 true singleton entities receive at least one false match (3,106 false links). Of those false links, 1,710 (55.1%) have `numeric_conflict = 1`: both addresses contain significant numbers but share none. Only four have an exact normalized name and six an exact normalized address. The median false-link score is 0.9917, so the model assigns high scores to many conflicting pairs.

The 20 highest-scoring examples in `singleton_false_positive_audit.json` frequently share most name tokens and much of an address while differing in a leading name or owner token, an added business descriptor, or a street number. These are hard negatives under the provided ground truth. Numeric conflict explains many, but not all, singleton errors.

## One decision experiment

Reject any candidate with `numeric_conflict = 1` before choosing the top one per source; keep threshold 0.97. This changes only final filtering, so candidate recall is unchanged.

| Metric | Baseline | Numeric conflict gate | Change |
| --- | ---: | ---: | ---: |
| Macro F0.5 | 0.657888 | **0.694588** | +0.036701 |
| Pair precision | 81.843% | **88.665%** | +6.822 points |
| Pair recall | 41.073% | 40.617% | −0.456 points |
| True positive links | 71,130 | 70,341 | −789 |
| False positive links | 15,780 | 8,992 | −6,788 |
| Singleton accuracy | 19.305% | **52.185%** | +32.880 points |
| Singleton entities with false matches | 2,253 | 1,335 | −918 |
| Non-singleton macro F0.5 | 0.685379 | **0.704805** | +0.019426 |
| Average predicted matches per S1 | 1.738 | 1.587 | −0.152 |

The rule improves macro F0.5 in both countries represented in training and validation: India **0.626329 → 0.640318** (19,963 S1) and US **0.678862 → 0.730657** (30,037 S1). Pair recall falls in both countries. France is absent from training, so this experiment does not establish the rule's behavior there.

Candidate scoring took 384.9 seconds. The final decision replay, country breakdown, and exact candidate recall check took 24.5 seconds. The 5,000-S1 bounded training feature build took several minutes across two target scans; precise peak RSS was not captured. Process memory stayed far below the aborted unbounded run. No raw dataset file was changed.

## Limits and next step

This is a single exploratory experiment on the fixed subset. The full validation address candidate cache is absent locally, and the model differs from the branch's reported model. The 0.694588 score therefore should not be compared as an isolated gate gain against the earlier 0.454827 result. The within-model gain of 0.036701 is the measured effect of the gate.

The next step is a full fixed-validation replay of the bounded model and gate after generating the full address candidate cache, followed by inspection of the remaining singleton false positives and the 789 lost true links. The gate has not been enabled in the production prediction path.

## Reproduce

```powershell
.venv\Scripts\python.exe -c "from reports.generate_features import build_training_pairs; build_training_pairs(sample_size=5000,seed=2026,max_candidates_per_s1_source=50)"
.venv\Scripts\python.exe -m reports.augment_training_features
.venv\Scripts\python.exe -c "import numpy as np; from src.train_match_model import train_tree_model; d=np.load('artifacts/train_features.npz'); train_tree_model(d['X'],d['y'])"
.venv\Scripts\python.exe -m reports.analyze_singleton_false_positives
.venv\Scripts\python.exe -m reports.evaluate_numeric_conflict_gate
```

The scripts write reusable local arrays under ignored `artifacts/` and metrics under `reports/`.
