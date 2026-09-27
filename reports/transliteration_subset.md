# Stage 7A: transliteration-assisted name retrieval on the fixed subset

## Scope and dependency

The experiment uses the exact saved 50,000-S1 Stage 6A validation subset and 173,180 truth links. Stage 4 pairs, Stage 5 name results, and Stage 6A address results were reused from their cached arrays. No full-validation or test retrieval was run.

`src/transliterate.py` creates a separate `name_translit` value from `name_norm`; neither `name_norm` nor any raw field is changed. [AnyAscii 0.3.3](https://pypi.org/project/anyascii/0.3.3/) is a deterministic, offline, character-based transliteration utility under the [ISC license](https://github.com/anyascii/anyascii/blob/master/LICENSE). Its copyright and permission notice are kept in `THIRD_PARTY_NOTICES.md`, and the dependency is pinned in `requirements-stage7a.txt`. The local competition rules in `README.md` restrict the **final model** to MIT/Apache 2.0; AnyAscii is a text utility, not a model, and performs no external business lookup. This is compatible with the stated restrictions for this experiment. If it is packaged with a final submission, retain the ISC notice.

All 50,000 sampled S1 names are Latin-script. Retrieval therefore compared their existing `name_norm` against `name_translit` for S2/S3 target names with at least 80% non-Latin alphabetic characters. Country partitioning remains dynamic. Of 10,320,219 training targets scanned, 456,369 S2 and 244,741 S3 names were transliterated. The channel used sparse `char_wb` TF-IDF 4–5 grams, `min_df=2`, `max_df=0.02`, a cosine floor of 0.05, 50,000-target batches, and top 10 per source. K=5 uses the first five results from that same run. No dense all-pairs matrix was formed.

## Results on the same 50,000 S1 entities

| Candidate set | True links retrieved | Recall | India recall | Candidate pairs | Mean/S1 | P95 | P99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage 6A best: Stage 5 + address K=5 | 161,584 | 93.304% | 89.624% | 2,424,163 | 48.48 | 141 | 260 |
| Transliteration alone K=5 | 101 | 0.058% overall | 0.146% | 157,056 | 3.14 | 10 | 10 |
| Stage 6A + transliteration K=5 | 161,609 | **93.319%** | **89.660%** | 2,581,063 | 51.62 | 144 | 263 |
| Transliteration alone K=10 | 153 | 0.088% overall | 0.221% | 310,230 | 6.20 | 20 | 20 |
| Stage 6A + transliteration K=10 | 161,626 | **93.328%** | **89.685%** | 2,734,161 | 54.68 | 149 | 266 |

Transliteration is only applicable to the 11,463 truth links with a detectably different target name script. It retrieves **101/11,463 (0.881%)** alone at K=5 and **153/11,463 (1.335%)** at K=10. All retrieved truth links are in India; US recall stays at 95.756%.

| K per source | New true links over Stage 6A | Of 3,373 previously missed cross-script links recovered | Added candidate pairs | New true links per 1,000 added pairs |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 25 | 25 (0.741%) | 156,900 | 0.159 |
| 10 | 42 | 42 (1.245%) | 309,998 | 0.135 |

The K=5 union still misses **11,571** true links, including **3,348** detectably cross-script links. Among the previous fixed 150-link reviewed miss sample, none was recovered by this targeted channel; its primary categories remain 47 cross-script, 35 shortened/alternate/web-style names, 27 severe spelling variants, 24 different/missing addresses, and smaller groups. The sample is a qualitative guide rather than a fresh random sample of the remaining misses. Cross-script names have **not** ceased to be the largest visible issue.

## Runtime, quality diagnosis, and collision risk

The targeted K=10 retrieval pass took **255.9 seconds** including fitting and target scanning. The entire run, including exact union evaluation, took **312.9 seconds**; sampled peak process RSS was **0.65 GiB**. This is a subset-only measurement.

An initial broader gate admitted any target name containing a non-Latin letter. It added 31 links and 172,732 pairs at K=5; its measurements are saved in `reports/transliteration_broad_gate_results.json`. Inspection showed that mixed-script names retaining common English words occupied many top ranks. The stricter predominantly non-Latin gate removed those competitors but still recovered only 25 links at K=5. This supports a genuine representation/ranking limitation rather than a candidate-union counting error.

In a direct-score check of the prior cross-script sample, the median cosine for a true pair was **0.070**, versus a median K=10 cutoff of **0.305**; 17 of 46 direct scores were zero and none of those 46 true targets reached top 10. AnyAscii's character-based romanization often differs substantially from the Latin name (for example, `vijay food` versus `vijy phud`). The score check is saved in `reports/transliteration_score_diagnostic.json`.

The collision audit found **213,611 distinct transliterated strings** among the 701,110 selected target records. **14,309** transliterated strings represented multiple distinct `name_norm` values; **220,900 records (31.51%)** belong to those groups. Many examples are equivalent words written in different Indian scripts, such as variants mapping to `om phud praivet limited`; this count does **not** prove that those records are unrelated. The very low incremental yield (0.159 true links per 1,000 added candidates at K=5) is the practical warning: the channel produces many unrelated candidate pairs. Distinctive-name collisions and false merges would need careful filtering before adoption.

## Decision

**Do not add this AnyAscii + character 4–5 gram top-K channel to the current candidate baseline.** Its incremental recall is negligible for its candidate cost. A later experiment could test a more phonetic or language-aware representation, but should first target the direct-score mismatch and measure candidate volume on the same subset. Shortened/web-style names and spelling noise are also substantial remaining groups. Stage 7A stops here; no matcher was trained.
