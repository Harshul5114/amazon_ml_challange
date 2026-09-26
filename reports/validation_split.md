# Fixed S1 train/validation split

Explicit membership: `splits/s1_train_validation.tsv`. Reproduction rule and seed: `splits/s1_split_manifest.json`. Each S1 ID is assigned once by SHA-256 with seed 2026 and a 20% validation threshold. All matches for an S1 remain with that S1.

Train: **1,766,299** S1 entities (80.038%). Validation: **440,522** S1 entities (19.962%). The ground truth S1 IDs exactly cover the source1 IDs.

## Balance checks

### Singleton status

| Category | Train count | Train share | Validation count | Validation share | Difference (validation − train) |
| --- | ---: | ---: | ---: | ---: | ---: |
| singleton | 98,650 | 5.585% | 24,597 | 5.584% | -0.002 pp |
| non-singleton | 1,667,649 | 94.415% | 415,925 | 94.416% | +0.002 pp |

### Matches per S1

| Category | Train count | Train share | Validation count | Validation share | Difference (validation − train) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0 | 98,650 | 5.585% | 24,597 | 5.584% | -0.002 pp |
| 1 | 95,453 | 5.404% | 23,704 | 5.381% | -0.023 pp |
| 2 | 300,496 | 17.013% | 74,716 | 16.961% | -0.052 pp |
| 3 | 424,745 | 24.047% | 106,096 | 24.084% | +0.037 pp |
| 4 | 387,593 | 21.944% | 96,522 | 21.911% | -0.033 pp |
| 5 | 257,736 | 14.592% | 64,221 | 14.578% | -0.013 pp |
| 6 | 131,690 | 7.456% | 33,178 | 7.532% | +0.076 pp |
| 7 | 51,128 | 2.895% | 12,840 | 2.915% | +0.020 pp |
| 8 | 14,982 | 0.848% | 3,698 | 0.839% | -0.009 pp |
| 9 | 3,363 | 0.190% | 842 | 0.191% | +0.001 pp |
| 10 | 430 | 0.024% | 104 | 0.024% | -0.001 pp |
| 11 | 33 | 0.002% | 4 | 0.001% | -0.001 pp |

### Matched source category

| Category | Train count | Train share | Validation count | Validation share | Difference (validation − train) |
| --- | ---: | ---: | ---: | ---: | ---: |
| zero | 98,650 | 5.585% | 24,597 | 5.584% | -0.002 pp |
| S2 only | 114,400 | 6.477% | 28,629 | 6.499% | +0.022 pp |
| S3 only | 131,703 | 7.456% | 32,795 | 7.445% | -0.012 pp |
| both | 1,421,546 | 80.482% | 354,501 | 80.473% | -0.009 pp |
| other prefix | 0 | 0.000% | 0 | 0.000% | +0.000 pp |

### S1 country

| Category | Train count | Train share | Validation count | Validation share | Difference (validation − train) |
| --- | ---: | ---: | ---: | ---: | ---: |
| India | 707,302 | 40.044% | 175,886 | 39.927% | -0.118 pp |
| US | 1,058,997 | 59.956% | 264,636 | 60.073% | +0.118 pp |

## Evaluation sanity checks on validation

| Prediction | Macro F0.5 |
| --- | ---: |
| Perfect ground truth copy | 1.000000000000 |
| Empty for every S1 | 0.055836030891 |
| One synthetic false ID added to every S1 | 0.751672522042 |

The all-empty score equals the validation singleton fraction. Adding a false ID to every S1 lowers the score. The synthetic ID is used only for this metric check and is not a challenge prediction.
