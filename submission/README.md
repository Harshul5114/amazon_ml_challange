# Reproduce this submission

Use Python 3.10 or newer. Place the competition TSV files in
`dataset/test/test_source1.tsv`, `test_source2.tsv`, and `test_source3.tsv` under
this directory. The raw dataset is not included in the zip.

From `code/business_entity_resolution/`, run:

```bash
python -m src.generate_quick_submission
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test
```

The generator uses only the Python standard library and `src/normalize.py`.
It indexes test S1 records by country, normalized business name, and normalized
business address. A target from S2 or S3 is a candidate and a final match when
that key identifies exactly one S1 record. Ambiguous S1 keys are excluded.
Normalization applies NFKC, casefolding, punctuation separation, and whitespace
collapse; it does not use external business information. The program writes
every S1 record exactly once, including empty lists.

`output/matching_results.tsv` is the leaderboard upload. The final zip also
contains `output/candidate_pairs.tsv`, this code, and the methodology document.
