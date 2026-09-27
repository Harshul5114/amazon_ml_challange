# Shared generated artifacts

The binary files in this directory are stored with Git LFS. After cloning or pulling:

```powershell
git lfs install
git lfs pull
```

`MANIFEST.json` lists each file's byte size and SHA-256 digest. Verify a download on Windows with `Get-FileHash artifacts\<filename> -Algorithm SHA256`.

| File | Purpose |
| --- | --- |
| `stage4_pairs_packed.npy` | Stage 4 candidate pairs for all 440,522 validation S1 entities. |
| `tfidf_top20.npz` | Name TF-IDF retrieval for all validation S1 entities, top 20 per target source. |
| `tfidf_run_metadata.json` | Parameters and timings for the name retrieval cache. |
| `address_tfidf_subset_top10.npz` | Address retrieval for the fixed 50,000-S1 validation subset, top 10 per source. |
| `translit_top10_subset.npz` | Experimental transliteration retrieval on that subset; not part of the selected candidate baseline. |
| `translit_broad_gate_results.json` | Results from the transliteration gating investigation. |
| `train_features.npz` | Bounded 5,000-S1 training sample with 150,000 augmented negatives (535,520 pairs). |
| `tree_match_model.joblib` | Exploratory tree model fitted to `train_features.npz`; not a final submission model. |
| `singleton_subset_scored.npz` | Scores and features for the fixed subset's 2,424,163 candidates, from that exploratory model. |

The raw `dataset/` is not included. Scripts that rescan raw records still require the competition-provided TSV files. The full validation address cache and final submission files have not been generated. See `reports/singleton_false_positive_audit.md` and `STATUS.md` before using the model scores. Generated model files should be loaded only from a trusted source because `joblib` uses Python object serialization.

New artifacts remain ignored by default. Update `.gitignore` and `MANIFEST.json` deliberately before sharing additional files.
