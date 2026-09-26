# Amazon ML Challenge 2026 — Agent Instructions

## Objective

Build a high-quality, reproducible solution for the Amazon ML Challenge 2026 Business Entity Resolution task.

For every Source 1 (S1) business entity, identify all corresponding records from Source 2 (S2) and Source 3 (S3) representing the same real-world business.

An S1 entity may have zero, one, or multiple matches.

Primary optimization target:

**Entity-level macro F0.5**

Precision is weighted more heavily than recall. False matches, particularly on true singleton entities, are costly.

---

## Data

Expected structure:

```text
dataset/
├── train/
│   ├── train_source1.tsv
│   ├── train_source2.tsv
│   ├── train_source3.tsv
│   └── train_ground_truth.tsv
└── test/
    ├── test_source1.tsv
    ├── test_source2.tsv
    └── test_source3.tsv
```

All input files are TSV. Use tab separation.

Never modify raw dataset files.

The dataset is large (~1 GB). Prefer memory-efficient/streaming/vectorized processing where appropriate. Do not make unnecessary copies of large datasets.

---

## Important Dataset Facts

Current profiling found:

- Training S1 entities: 2,206,821
- Training singleton rate: ~5.58%
- ~80.48% of training S1 entities have matches represented in both S2 and S3
- Maximum observed matches for one S1: 11
- No duplicate entity IDs were found
- No S2/S3 entity ID maps to multiple S1 entities
- France is absent from training but represents roughly 14–15% of each test source
- Test addresses are somewhat longer on average

Therefore:

- Do not hard-code only US/India behavior.
- Treat country handling as open-set.
- Avoid language-specific transformations that damage unseen French records.
- Candidate generation must scale to millions of entities.

---

## Validation

A fixed reproducible 80/20 S1-level train/validation split already exists.

Current split:

- Development/train: 1,766,299 S1 entities
- Validation: 440,522 S1 entities
- Train singleton rate: 5.585%
- Validation singleton rate: 5.584%

Do not create a new validation split for individual experiments.

Always compare experiments on the same validation entities.

The evaluator has been unit-tested:

- Perfect prediction macro F0.5 = 1.000000
- All-empty prediction macro F0.5 = 0.055836
- 12 evaluator tests pass

Never optimize based only on pairwise accuracy or public leaderboard score.

---

## Intended Pipeline

```text
Raw records
    ↓
Conservative normalization
    ↓
Candidate generation / blocking
    ↓
Candidate S1 ↔ S2/S3 pairs
    ↓
Pairwise similarity features
    ↓
Match model
    ↓
Pair scores
    ↓
Threshold / decision logic
    ↓
Entity-level match sets
    ↓
Macro F0.5 evaluation
```

Brute-force S1 × S2/S3 comparison is not acceptable.

Candidate generation determines the maximum achievable downstream recall and must be evaluated explicitly.

---

## Development Strategy

Work incrementally.

Preferred workflow:

```text
current evidence
→ identify bottleneck
→ implement one justified experiment
→ evaluate on fixed validation
→ report results
→ decide next experiment
```

Do not implement many speculative approaches simultaneously.

Do not rewrite working infrastructure unless necessary.

Do not proceed automatically to later stages when a task explicitly asks you to stop.

Maintain reproducibility with fixed random seeds and deterministic processing where practical.

---

## Normalization Rules

Always preserve raw fields.

Create normalized/derived fields separately.

Conservative normalization may include:

- Unicode normalization
- casefold/lowercase
- punctuation normalization
- whitespace normalization
- tokenization
- numeric-token extraction

Be cautious with:

- transliteration
- legal-suffix removal
- address abbreviation replacement
- language/country-specific rules

Avoid destructive over-normalization.

France is unseen during training, so transformations must generalize beyond US and India.

---

## Candidate Generation

Candidate generation must balance:

- candidate recall
- candidates per S1
- reduction ratio
- runtime
- memory

Prefer the union of complementary retrieval/blocking signals rather than one overly aggressive rule.

Potential signals include:

- exact normalized name
- token signatures
- token overlap
- character n-grams
- TF-IDF retrieval
- sparse cosine similarity
- address evidence
- numeric/postal components

Every final predicted match must occur in the generated candidate set.

---

## Modeling

Do not jump directly to deep learning.

Establish strong interpretable baselines first.

Reasonable progression:

1. deterministic similarity baseline
2. logistic regression
3. gradient-boosted trees
4. more sophisticated approaches only when justified by validation/error analysis

Potential pairwise evidence includes name similarity, address similarity, numeric agreement, token overlap, edit similarity, character n-grams, TF-IDF similarity, missingness and interactions between signals.

Optimize final decisions for **entity-level macro F0.5**, not ordinary classification accuracy.

---

## Thresholding

Threshold selection is part of the model.

Evaluate thresholds using complete predicted match sets for each validation S1 entity.

Track at minimum:

- macro F0.5
- precision
- recall
- singleton performance
- non-singleton performance
- false matches on singletons
- average predicted matches per S1

Do not select thresholds from leaderboard behavior.

---

## Error Analysis

Before adding complexity, inspect false positives and false negatives.

Useful categories include:

- spelling variations
- abbreviations
- legal suffix differences
- reordered names
- transliteration
- missing address components
- difficult/landmark addresses
- numeric conflicts
- similar businesses
- singleton false positives
- multiple close-scoring candidates
- country-specific behavior

Prefer changes supported by observed error patterns.

---

## Competition Restrictions

### No external business data

Do not use:

- internet business lookup
- geocoding APIs
- business-registration databases
- commercial entity-resolution APIs
- external business databases
- external data augmentation that identifies businesses

Only competition-provided business information may be used for entity matching.

Do not add external entity information to the dataset.

Any pretrained model proposed for final use must satisfy the competition's licensing/model-size requirements. Verify compliance before integrating one.

---

## Output Requirements

Final submission requires:

### `matching_results.tsv`

Columns:

```text
source1_entity_id
matched_entity_ids
```

Every test S1 entity must appear exactly once.

### `candidate_pairs.tsv`

Columns:

```text
source1_entity_id
candidate_entity_ids
```

The candidate set is generated before final match filtering.

Every predicted match must also appear in the corresponding candidate set.

Before submission verify:

- every S1 appears exactly once
- IDs exist
- no duplicate IDs
- only S2/S3 IDs appear as targets
- no S1 IDs appear as matches
- final matches are a subset of candidate pairs
- empty match lists work
- multiple matches work
- France is processed
- Unicode/null values do not crash the pipeline

Run:

```bash
python utils/validate_submission.py
```

before packaging the final submission.

---

## Repository Organization

Prefer modular code such as:

```text
src/
├── load_data.py
├── normalize.py
├── blocking.py
├── similarity_features.py
├── train_match_model.py
├── predict.py
├── evaluate.py
└── generate_submission.py
```

Avoid unnecessary monolithic scripts.

---

## Experiment Tracking

Maintain `STATUS.md`.

It should contain:

- current stage
- implemented components
- latest validation results
- warnings/problems
- next recommended step

Maintain an experiment log once modeling/blocking experiments begin.

Record approximately:

- experiment ID
- blocking version
- feature version
- model
- hyperparameters
- threshold
- candidate recall
- pair precision/recall
- entity precision/recall
- macro F0.5
- singleton performance
- runtime
- memory
- notes


---

## Agent Reporting

After each requested experiment or stage, provide a concise report containing:

1. what changed
2. what was executed
3. important numerical results
4. runtime/memory issues if relevant
5. unexpected findings
6. files created/modified
7. recommended next step

If results look suspicious—for example impossible metrics, leakage, unexpectedly low candidate recall, memory explosion, or validation inconsistencies—stop and investigate rather than continuing automatically.