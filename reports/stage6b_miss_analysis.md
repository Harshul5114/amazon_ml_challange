# Stage 6B: misses after Stage 5 + address TF-IDF K=5

This analyzes the exact saved Stage 6A subset of **50,000 S1 entities** and **173,180 true links**. The Stage 4 packed pairs, Stage 5 name top-20 arrays, and Stage 6A address top-10 arrays are reused. No retrieval is run here. The reconstruction exactly matches both Stage 6A K=5 and K=10 truth-hit totals.

**Remaining missed links: 11,596 / 173,180 (6.696%).** These are candidate-generation misses, not model false negatives. The analysis script is `reports/analyze_stage6b.py`; machine-readable counts and the fixed 150-link sample are in `reports/stage6b_miss_summary.json` and `reports/stage6b_miss_sample_labeled.tsv`.

## Where misses occur

| Country | True links | Missed links | Miss rate | Share of all misses |
| --- | ---: | ---: | ---: | ---: |
| India | 69,246 | 7,185 | 10.376% | 61.961% |
| US | 103,934 | 4,411 | 4.244% | 38.039% |

| Source | True links | Missed links | Miss rate | Share of all misses |
| --- | ---: | ---: | ---: | ---: |
| S2 | 83,614 | 5,047 | 6.036% | 43.524% |
| S3 | 89,566 | 6,549 | 7.312% | 56.476% |

## Field and text evidence across all 11,596 misses

A field is present when its conservative normalized value is nonempty. Script is the dominant Unicode script of alphabetic **name** characters, requiring at least 80% agreement within each name. Mixed or empty cases are uncertain. Token overlap is set Jaccard on normalized whitespace tokens; it includes common and legal words, so a high value alone need not identify a business.

| Dimension | Category | Misses | Share of misses |
| --- | --- | ---: | ---: |
| Name presence | both present | 11,596 | 100.000% |
| Address presence | both present | 9,481 | 81.761% |
| Address presence | S1 only | 2,115 | 18.239% |
| Name scripts | same | 8,100 | 69.852% |
| Name scripts | different | 3,373 | 29.088% |
| Name scripts | uncertain | 123 | 1.061% |
| Name token Jaccard | zero | 5,472 | 47.189% |
| Name token Jaccard | high (>=0.5) | 4,728 | 40.773% |
| Name token Jaccard | medium (0.25–<0.5) | 1,232 | 10.624% |
| Name token Jaccard | low (<0.25) | 164 | 1.414% |
| Address token Jaccard | medium (0.25–<0.5) | 4,819 | 41.557% |
| Address token Jaccard | high (>=0.5) | 2,734 | 23.577% |
| Address token Jaccard | not comparable: empty field | 2,115 | 18.239% |
| Address token Jaccard | low (<0.25) | 1,919 | 16.549% |
| Address token Jaccard | zero | 9 | 0.078% |
| Any shared address number | yes | 7,123 | 61.426% |
| Any shared address number | no | 4,473 | 38.574% |
| Shared address number ≥3 digits | no | 8,994 | 77.561% |
| Shared address number ≥3 digits | yes | 2,602 | 22.439% |

Any shared number counts even one-digit fragments and can be weak evidence. The ≥3-digit row is a stricter diagnostic. Address Jaccard is not comparable when a target address is empty.

## What cached deeper lists recover

The already-computed address K=10 list contains **1,894** of the 11,596 K=5 misses (**16.333%**). It would reduce the miss count to **9,702** and subset miss rate to **5.602%**. Stage 6A measured its marginal cost at **490,332** further candidate pairs, or only **3.863** true links per 1,000 additional candidates.

The cached name top-20 list contains **1,027** K=5 misses (**8.857%**), including **1,024** same-script and **0** detectably different-script pairs. This is a diagnostic membership check, not a new retrieval run; its candidate-volume tradeoff was not reevaluated for this subset.

| Name-script relation | K=5 misses | Recovered by address K=10 | Recovery within group |
| --- | ---: | ---: | ---: |
| different | 3,373 | 490 | 14.527% |
| same | 8,100 | 1,388 | 17.136% |
| uncertain | 123 | 16 | 13.008% |

## Reproducible 150-link review

The sample is the 150 smallest SHA-256 hashes of `2062:<S1 ID>:<target ID>` among all K=5 missed links. Each pair was assigned one manually reviewed primary category using its raw name, address, and diagnostics. Categories are exclusive for counting, although a link can have several failure modes. Percentages are approximate sample frequencies, not exact population rates.

| Primary failure category | Sample links | Sample share |
| --- | ---: | ---: |
| cross-script / transliteration | 47 | 31.333% |
| shortened / alternate / web-style name | 35 | 23.333% |
| severe spelling variation | 27 | 18.000% |
| different or missing address | 24 | 16.000% |
| opaque / possibly corrupted name | 11 | 7.333% |
| substantially different address formatting | 3 | 2.000% |
| legal suffix / abbreviation | 2 | 1.333% |
| weak information in both fields | 1 | 0.667% |
| other | 0 | 0.000% |

The opaque-name category includes possible encoding or generated-text corruption, but the raw text alone cannot distinguish corruption from a genuine alternate name. Address formatting and legal suffix variation also appear as secondary issues in many links assigned to larger categories.

### Representative reviewed links

The labeled TSV contains all 150 sampled links with complete raw names and addresses, source, country, overlap diagnostics, and K=10 recovery status. Examples below show why a single primary category was assigned.

| Category | S1 → target | Raw names | Raw addresses |
| --- | --- | --- | --- |
| cross-script / transliteration | `S1-603534940` → `S3-562338812` | `Sai Marketing Private Limited` → `ಸಾಯಿ ಮಾರ್ಕೆಟಿಂಗ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್` | `No.4/F2, Gajanana Apartments, Nhcs Layout 4Th Cross, Marenahalli, Vijayanagar, Bangal…` → `No.d/4/f2, Bangalore, ಕರ್ನಾಟಕ` |
| shortened / alternate / web-style name | `S1-646482399` → `S3-749194499` | `Womens Health Partners` → `womenshealthpartners.com` | `13 Mug Hollow, Branchland, WV` → `13 Mug Hollow, Braanchland, West Virginia` |
| severe spelling variation | `S1-799580041` → `S3-805558791` | `Lion Services Private Limited` → `Li0n Services Private  Limited` | `S-093, Forest County, Sr. No. 40, 41, 59, Opp Dhole Patil Farms Road, . Eon It Park, …` → `S-093, Pune, महाराष्ट्र` |
| different or missing address | `S1-533995455` → `S3-537724448` | `Orthopedic Physicians` → `Orthopedic  Physicians Authority` | `10102 Amblewood Drive, Houston, TX` → `*(empty)*` |
| opaque / possibly corrupted name | `S1-890632465` → `S2-512856192` | `XZ Multitrades-Nagpur` → `Nexhalonyla` | `Plot No 29, Rathor Layout, Nagpur, Maharashtra` → `PLOT NO G-29, NAGPUR, महाराष्ट्र` |
| substantially different address formatting | `S1-194519331` → `S3-401757251` | `Federal Worldwide PLLC` → `Federal PLLC (Services)` | `OR, 445 Currin Street, Estacada` → `00445 Currin Street, Estacaa, Oregon` |
| legal suffix / abbreviation | `S1-498501674` → `S2-581742533` | `NT Enterprises Company` → `NT Enterprises` | `62, Main Street, Mhow, Indore, Madhya Pradesh` → `62, MAIN STREET, MHOW, Madhya Pradesh` |
| weak information in both fields | `S1-158868631` → `S2-288991134` | `Hester, Johnson and Caccavale Banking LLC` → `LUMKOR` | `209 W Herron Boulevard, Lakebay, WA` → `0480 WEST HERRON BLVD, LAKEBAY, WA` |

## Next retrieval mechanism suggested by this evidence

**Investigate script-aware transliteration of business names followed by conservative name retrieval on the same-country partition.** Detectably different name scripts account for **3,373** misses (**29.088%**); **3,373** are in India and **0** in the US. Cached name top-20 finds none of these, while address K=10 finds only **490**. This gives a targeted pool that neither current name retrieval nor a modest address-K increase handles well. It should remain an added representation beside raw and conservative normalized text, and be measured for incremental recall and candidate volume before any full run.

A second promising investigation is a precise address or locality anchor for pairs with same-script names but altered or missing address text. Shared numeric fragments are common, but only 22.4% of misses share a number of at least three digits, so an unrestricted numeric block would likely generate many false candidates. The current evidence favors targeted transliteration first; it does not establish downstream match precision or F0.5.

No new retrieval channel, full-validation run, test run, production pairwise features, or matching model was implemented in Stage 6B.
