# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** albertnewtonepsteinnetanyahu  
**Team Members:** Harshul Anand, Nigam Jee, Kahaan Parikh, Vedant Patel  
**Submission Date:** 27 September 2026

## 1. Executive Summary

This time-constrained submission uses a conservative deterministic exact-match
rule. It links a Source 2 or Source 3 record to a Source 1 record only when the
country, normalized business name, and normalized address identify one unique
Source 1 entity. It uses only competition-provided records.

## 2. Methodology

### 2.1 Problem Analysis

The training set contains 2,206,821 Source 1 entities, including about 5.58%
singletons. The test set includes France, which is absent from labeled training
data. A country-independent text normalization avoids language-specific rules.

### 2.2 Solution Strategy

**Approach Type:** Deterministic blocking and matching.  
**Core decision:** Exact agreement on country, normalized name, and normalized
address, with ambiguous Source 1 keys excluded.

## 3. Candidate Generation (Blocking)

The blocking key is `(country, normalized business name, normalized business
address)`. Normalization uses Unicode NFKC, casefolding, punctuation separation,
and whitespace collapse. Both name and address must be present. A key shared by
multiple S1 records is discarded to avoid ambiguous links. S2/S3 targets with
unique S1 keys become candidates. The candidate TSV is produced before writing
the final match TSV; this baseline accepts all its candidates as final matches.
The test run produced 84,110 candidate links for 1,732,544 S1 records.

## 4. Matching Model

No machine-learning model is used in this final fast baseline. The exact-key
decision favors precision over recall. Multiple S2/S3 targets may match the
same S1. The generator does not use external business data or pretrained models.

## 5. Results & Error Analysis

No labeled validation score was computed for this exact-key fallback. The
previous, more complex prototype achieved macro F0.5 of 0.694588 on a fixed
50,000-S1 validation subset, but that number **does not describe this final
submission**. The exact-key rule will miss legitimate spelling and address
variants; ambiguous S1 keys are intentionally left unmatched. The challenge
portal determines the score of these final predictions.

The supplied validator passed both final TSVs with ID-existence checks:
1,732,544 rows each, 77,328 non-empty S1 rows, and 1,655,216 empty rows.

## 6. Conclusion

This package prioritizes complete, reproducible, format-valid output under the
available time. It covers every test S1 record and handles France and Unicode
with the same normalization rule.

## Appendix: Code Artefacts

`code/business_entity_resolution/src/generate_quick_submission.py` is the
end-to-end entry point, with normalization in `src/normalize.py`. The supplied
validation script is under `utils/`. Exact execution steps are in the packaged
`README.md`.
