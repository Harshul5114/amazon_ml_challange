# Dataset profile

All seven TSVs were read with tab separation as UTF-8 text. Counts below use the raw field values; an empty field is counted as missing. Common literal null markers (NULL, N/A, NAN, NONE, <NA>) are counted separately if present. Lengths count Unicode characters in nonempty fields, including spaces and punctuation. Percentages use the file's row count unless stated otherwise.

## File inventory

| File | Bytes | MiB | Rows | Unique entity IDs | Duplicate ID rows |
| --- | ---: | ---: | ---: | ---: | ---: |
| `dataset/test/test_source1.tsv` | 175,022,086 | 166.91 | 1,732,544 | 1,732,544 | 0 |
| `dataset/test/test_source2.tsv` | 509,456,422 | 485.86 | 4,887,273 | 4,887,273 | 0 |
| `dataset/test/test_source3.tsv` | 506,002,772 | 482.56 | 5,082,316 | 5,082,316 | 0 |
| `dataset/train/train_ground_truth.tsv` | 127,015,583 | 121.13 | 2,206,821 | 2,206,821 | 0 |
| `dataset/train/train_source1.tsv` | 210,069,713 | 200.34 | 2,206,821 | 2,206,821 | 0 |
| `dataset/train/train_source2.tsv` | 489,301,488 | 466.63 | 5,034,616 | 5,034,616 | 0 |
| `dataset/train/train_source3.tsv` | 503,705,637 | 480.37 | 5,285,603 | 5,285,603 | 0 |

Duplicate ID rows count every occurrence after the first within that file. Unique IDs exclude empty IDs.

## Columns, dtypes and missing values

All populated fields are textual; pandas infers `object` for these columns when reading normally. Empty `matched_entity_ids` fields become missing values under pandas' default NA parsing.

| File | Column | Dtype | Empty / missing | Literal null markers |
| --- | --- | --- | ---: | ---: |
| `test_source1.tsv` | `entity_id` | object (text) | 0 (0.00%) | 0 |
| `test_source1.tsv` | `business_name` | object (text) | 0 (0.00%) | 0 |
| `test_source1.tsv` | `business_address` | object (text) | 0 (0.00%) | 0 |
| `test_source1.tsv` | `country` | object (text) | 0 (0.00%) | 0 |
| `test_source2.tsv` | `entity_id` | object (text) | 0 (0.00%) | 0 |
| `test_source2.tsv` | `business_name` | object (text) | 0 (0.00%) | 0 |
| `test_source2.tsv` | `business_address` | object (text) | 129,408 (2.65%) | 0 |
| `test_source2.tsv` | `country` | object (text) | 0 (0.00%) | 0 |
| `test_source3.tsv` | `entity_id` | object (text) | 0 (0.00%) | 0 |
| `test_source3.tsv` | `business_name` | object (text) | 0 (0.00%) | 1 |
| `test_source3.tsv` | `business_address` | object (text) | 136,098 (2.68%) | 0 |
| `test_source3.tsv` | `country` | object (text) | 0 (0.00%) | 0 |
| `train_ground_truth.tsv` | `source1_entity_id` | object (text) | 0 (0.00%) | 0 |
| `train_ground_truth.tsv` | `matched_entity_ids` | object (text) | 123,247 (5.58%) | 0 |
| `train_source1.tsv` | `entity_id` | object (text) | 0 (0.00%) | 0 |
| `train_source1.tsv` | `business_name` | object (text) | 0 (0.00%) | 0 |
| `train_source1.tsv` | `business_address` | object (text) | 0 (0.00%) | 0 |
| `train_source1.tsv` | `country` | object (text) | 0 (0.00%) | 0 |
| `train_source2.tsv` | `entity_id` | object (text) | 0 (0.00%) | 0 |
| `train_source2.tsv` | `business_name` | object (text) | 0 (0.00%) | 1 |
| `train_source2.tsv` | `business_address` | object (text) | 168,967 (3.36%) | 0 |
| `train_source2.tsv` | `country` | object (text) | 0 (0.00%) | 0 |
| `train_source3.tsv` | `entity_id` | object (text) | 0 (0.00%) | 0 |
| `train_source3.tsv` | `business_name` | object (text) | 0 (0.00%) | 0 |
| `train_source3.tsv` | `business_address` | object (text) | 175,916 (3.33%) | 0 |
| `train_source3.tsv` | `country` | object (text) | 0 (0.00%) | 0 |

## Five sample rows per file

### `dataset/test/test_source1.tsv`

| entity_id | business_name | business_address | country |
| --- | --- | --- | --- |
| S1-714132312 | Zephay Labs Inc | 2621 Cotten Road, Tyler, TX | US |
| S1-106407869 | Vision Partners Corp | IA, Iowa City, 1064 Newton Rd, Unit 11 | US |
| S1-156285671 | << Team Ecole | 175 Boulevard du Président Franklin Roosevelt, Bordeaux, Nouvelle-Aquitaine | France |
| S1-689823050 | Red Perfect Trading | Mirzapur, Ews 12, Uttar Pradesh, Mirzapursadar, Awas Vikas Colony | India |
| S1-921369899 | ZNB Club SARL | Nouvelle-Aquitaine, La Teste-de-Buch, 5 bis Rue Pierre Dignac | France |

### `dataset/test/test_source2.tsv`

| entity_id | business_name | business_address | country |
| --- | --- | --- | --- |
| S2-192345572 | Brahma Infosoft | COIMATORE COLONY, HUNSUR TQMYSORE DIST., Karnataka | India |
| S2-566025912 | Marina Ecole France Sarl | 63 R. DE DIEPPE, LILLE, Hauts-de-France | France |
| S2-158121477 | SCI Ptit Àmicale | 18 RUE JEN ZAY, Dunkerque, Nord | France |
| S2-89663826 | Apex Summit | 67 KENTUCKY ST, SALYERSVILLE, KY | US |
| S2-884102769 | Fresh Truist | 8264 FILLY COURT, ROANOKE COUNTY, VA | US |

### `dataset/test/test_source3.tsv`

| entity_id | business_name | business_address | country |
| --- | --- | --- | --- |
| S3-462677478 | मॉडर्न फाइनेंस | No 10 Enkay Square, 448A, Udyog Vihar Phase V, Gurugram, Gurgaon, HR | India |
| S3-374810425 | Shri Sai Infratech Co | 3/115, East Delhi, DL | India |
| S3-198586129 | Fractales Amis Groupe S.A.S | 23 Rue Icmre, La Teste-de-buch, Gironde | France |
| S3-10300249 | Shri Supreme Consulting Private  (Limited) | H.no 910 A 3503, Mumbai, महाराष्ट्र | India |
| S3-604980231 | Prime Realty Ventures Public Limited | G.t. Karnal Road, Industrial Area, New Delhi, null, A-68, दिल्ली | India |

### `dataset/train/train_ground_truth.tsv`

| source1_entity_id | matched_entity_ids |
| --- | --- |
| S1-965667 | S2-681193310,S2-743505751,S3-775321672,S3-11291185,S3-860443364 |
| S1-55344266 | S2-249013014,S2-197070651,S3-478195123,S3-384364074 |
| S1-343815751 | S2-790675320,S2-479876582,S3-878454467 |
| S1-656753428 | S2-153058913,S2-24659151,S3-679606215 |
| S1-102811957 | S2-478959098,S2-553508714,S2-625774905,S3-728090388,S3-928796641,S3-449308785 |

### `dataset/train/train_source1.tsv`

| entity_id | business_name | business_address | country |
| --- | --- | --- | --- |
| S1-925783039 | Orelee's Barbershop | 1795 Westchester Drive, High Point, NC | US |
| S1-773889195 | Prime Money | 17560 Ellis Road, Tahlequah, OK | US |
| S1-377745466 | B+ Retail Inc | 1712 Montebello Avenue, Phoenix, AZ | US |
| S1-133037285 | Christ Chapel | 2100 Cameron Drive, Unit APARTMENT G, Dundalk, MD | US |
| S1-755362802 | Prabhav Business Center | 797, Lake Town Block A, Kolkata, Howrah, West Bengal | India |

### `dataset/train/train_source2.tsv`

| entity_id | business_name | business_address | country |
| --- | --- | --- | --- |
| S2-166376419 | राम मार्केटिंग प्राइवेट लिमिटेड | KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi | India |
| S2-764573417 | -- Holloway Peak Inc Seafood | 105 ELM ST, MORGANTON, NC | US |
| S2-639257739 | आदित्य प्रॉपर्टीज एलएलपी | G-3/571, GULMOHAR COLONY, BHOPAL, Madhya Pradesh | India |
| S2-163963287 | Summit Inc | GREENSBORO, NC, 19 1/2 STARDUST TRAIL | US |
| S2-49942811 | Delta Tetlecommunication Inc | 914 PIERPONT AVE, CLEVELAND, OH | US |

### `dataset/train/train_source3.tsv`

| entity_id | business_name | business_address | country |
| --- | --- | --- | --- |
| S3-202863386 | wilfordhancock.com | Mack Rd, Haltom City, Texas | US |
| S3-859268022 | International South Consultants Private Ltd | *(empty)* | India |
| S3-22467283 | LLC Moncada Léarning Center | 5780 Fawn Ct, Fort Worth, Texas | US |
| S3-671162755 | Moyna's Coffee | 1 Ivanhoe Ave, PO Box 6009, Cincinnati, Ohio | US |
| S3-960981775 | Pvt. EFS Print Ventures Ltd. | Door No 183, 41St Cross, 22Nd Main 9Th Block Jayanagar, Bengaluru Urban, Bangalore, ಕರ್ನಾಟಕ | India |

## Country distribution by source

| Split / source | Country | Rows | Share |
| --- | --- | ---: | ---: |
| `test_source1` | India | 809,986 | 46.75% |
| `test_source1` | US | 663,106 | 38.27% |
| `test_source1` | France | 259,452 | 14.98% |
| `test_source2` | India | 2,312,565 | 47.32% |
| `test_source2` | US | 1,871,330 | 38.29% |
| `test_source2` | France | 703,378 | 14.39% |
| `test_source3` | India | 2,405,000 | 47.32% |
| `test_source3` | US | 1,945,701 | 38.28% |
| `test_source3` | France | 731,615 | 14.40% |
| `train_source1` | US | 1,323,633 | 59.98% |
| `train_source1` | India | 883,188 | 40.02% |
| `train_source2` | US | 3,016,817 | 59.92% |
| `train_source2` | India | 2,017,799 | 40.08% |
| `train_source3` | US | 3,170,056 | 59.98% |
| `train_source3` | India | 2,115,547 | 40.02% |

## Business name and address lengths

Statistics exclude empty values. The 50th and 90th percentiles use the nearest observed rank.

| File | Field | Nonempty | Mean | Min | Median | P90 | Max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `test_source1` | `business_name` | 1,732,544 | 23.84 | 3 | 24 | 34 | 92 |
| `test_source1` | `business_address` | 1,732,544 | 57.21 | 11 | 50 | 93 | 268 |
| `test_source2` | `business_name` | 4,887,273 | 25.70 | 2 | 25 | 38 | 102 |
| `test_source2` | `business_address` | 4,757,865 | 51.78 | 5 | 43 | 87 | 269 |
| `test_source3` | `business_name` | 5,082,316 | 25.66 | 2 | 25 | 38 | 103 |
| `test_source3` | `business_address` | 4,946,218 | 50.08 | 5 | 44 | 82 | 267 |
| `train_source1` | `business_name` | 2,206,821 | 24.03 | 3 | 24 | 34 | 105 |
| `train_source1` | `business_address` | 2,206,821 | 52.07 | 11 | 41 | 90 | 256 |
| `train_source2` | `business_name` | 5,034,616 | 25.10 | 2 | 25 | 37 | 104 |
| `train_source2` | `business_address` | 4,865,649 | 47.83 | 8 | 37 | 84 | 249 |
| `train_source3` | `business_name` | 5,285,603 | 25.20 | 2 | 25 | 37 | 123 |
| `train_source3` | `business_address` | 5,109,687 | 48.32 | 2 | 42 | 78 | 240 |

## Training ground truth

The ground truth has 2,206,821 rows and 2,206,821 unique S1 IDs. 123,247 S1 rows have zero matches (5.58%). The maximum is 11 matches for `S1-765235386`.

### Match count per S1

| Matches | S1 rows | Share |
| ---: | ---: | ---: |
| 0 | 123,247 | 5.58% |
| 1 | 119,157 | 5.40% |
| 2 | 375,212 | 17.00% |
| 3 | 530,841 | 24.05% |
| 4 | 484,115 | 21.94% |
| 5 | 321,957 | 14.59% |
| 6 | 164,868 | 7.47% |
| 7 | 63,968 | 2.90% |
| 8 | 18,680 | 0.85% |
| 9 | 4,205 | 0.19% |
| 10 | 534 | 0.02% |
| 11 | 37 | 0.00% |

### Sources represented in each S1 match list

| Category | S1 rows | Share of all S1 rows |
| --- | ---: | ---: |
| zero | 123,247 | 5.58% |
| S2 only | 143,029 | 6.48% |
| S3 only | 164,498 | 7.45% |
| both S2 and S3 | 1,776,047 | 80.48% |
| other prefix | 0 | 0.00% |

Across all lists there are 7,638,365 distinct S2/S3 target IDs. 0 target IDs map to more than one distinct S1 ID (S2: 0; S3: 0). Repeated IDs within a single list: 0. Unexpected target prefixes: none.

## Train versus test

| Source | Train rows | Test rows | Train countries | Test countries |
| --- | ---: | ---: | --- | --- |
| S1 | 2,206,821 | 1,732,544 | US: 59.98%, India: 40.02% | India: 46.75%, US: 38.27%, France: 14.98% |
| S2 | 5,034,616 | 4,887,273 | US: 59.92%, India: 40.08% | India: 47.32%, US: 38.29%, France: 14.39% |
| S3 | 5,285,603 | 5,082,316 | US: 59.98%, India: 40.02% | India: 47.32%, US: 38.28%, France: 14.40% |

Country share changes (test minus train, percentage points):

| Source | Country | Change |
| --- | --- | ---: |
| S1 | France | +14.98 pp |
| S1 | India | +6.73 pp |
| S1 | US | -21.71 pp |
| S2 | France | +14.39 pp |
| S2 | India | +7.24 pp |
| S2 | US | -21.63 pp |
| S3 | France | +14.40 pp |
| S3 | India | +7.30 pp |
| S3 | US | -21.69 pp |

The strongest shift is country coverage: France is absent from every training source but accounts for 14.98% of test S1, 14.39% of test S2, and 14.40% of test S3. The US share falls by about 21.6–21.7 percentage points in every source, while India's share rises by about 6.7–7.3 points. Test addresses are also longer on average by 5.14 characters in S1, 3.95 in S2, and 1.76 in S3. Missing addresses in S2/S3 fall from 3.36%/3.33% in train to 2.65%/2.68% in test. These are descriptive differences; the files alone do not establish their cause.

### Data integrity notes

- No rows had more fields than the header in the streaming parse.
