# Normalization audit

## Sampling method

Training source files only. An S1/S2/S3 record is selected when the first eight bytes of SHA-256(`2026:<entity_id>`) as an unsigned big-endian integer are below 922337203685477580 (approximately 5%). This rule is fixed, independent of row order, and reproducible with `reports/analyze_normalization.py`. Source files were read as UTF-8 tab-separated text and were not modified.

| File | Scanned rows | Selected rows | Selected share |
| --- | ---: | ---: | ---: |
| `train_source1.tsv` | 2,206,821 | 110,130 | 4.990% |
| `train_source2.tsv` | 5,034,616 | 251,700 | 4.999% |
| `train_source3.tsv` | 5,285,603 | 263,733 | 4.990% |

## Transformations

The baseline applies NFKC, Unicode casefolding, punctuation and whitespace separation, and whitespace collapse. It preserves letters from all scripts, accents, numeric content, symbols outside Unicode punctuation, legal suffixes, and address abbreviations. Tokens retain order and repetitions; numeric tokens are digit runs, including digits inside alphanumeric tokens. Every derived field is added alongside the original field.

The added fields are `name_norm`, `address_norm`, `name_tokens`, `address_tokens`, `name_numeric_tokens`, and `address_numeric_tokens`. Null, empty, and pandas-style missing values produce empty strings and empty token lists.

## Name collision audit

A collision group is a normalized name with at least two **different exact raw business names** in this sample. Repeated records with the same raw name alone do not count as a normalization collision. A collision means text was merged by the representation, not that the underlying businesses are the same.

| Measure | Value |
| --- | ---: |
| Sampled records | 625,563 |
| Distinct raw business names | 589,369 |
| Distinct normalized business names | 568,316 |
| Loss of distinct name strings | 21,053 (3.572% of distinct raw names) |
| Collision groups | 16,024 (2.820% of normalized names) |
| Distinct raw names in collision groups | 37,077 (6.291% of distinct raw names) |
| Sampled records in collision groups | 52,489 (8.391% of sampled records) |
| Groups requiring punctuation changes to merge | 7,951 |
| Sampled records in punctuation-induced groups | 30,818 (4.926% of sampled records) |

The last two measures compare against an intermediate form with only NFKC, casefolding, and whitespace collapse. They isolate groups where punctuation separation creates additional merging; the remaining collision groups arise from case, compatibility, or whitespace changes.

## Sampled raw → normalized records

### Indian numeric address — `S1-341506266` (India)

- Name: `Secunderabad Park Private Limited` → `secunderabad park private limited`
- Address: `Secunderabad, Hyderabad, Plot No.53, Flat No.402, Sr Estate-I Rainbow Colony, Ammuguda, Sainikpuri, Telangana, # 53/1` → `secunderabad hyderabad plot no 53 flat no 402 sr estate i rainbow colony ammuguda sainikpuri telangana 53 1`
- Name tokens: `['secunderabad', 'park', 'private', 'limited']`; numeric tokens: `[]`
- Address tokens: `['secunderabad', 'hyderabad', 'plot', 'no', '53', 'flat', 'no', '402', 'sr', 'estate', 'i', 'rainbow', 'colony', 'ammuguda', 'sainikpuri', 'telangana', '53', '1']`; numeric tokens: `['53', '402', '53', '1']`

### US punctuated name — `S1-304785953` (US)

- Name: `Lafferty Certified Lifesciences Inc.` → `lafferty certified lifesciences inc`
- Address: `13315 Schwenger Place, Fairfax County, VA` → `13315 schwenger place fairfax county va`
- Name tokens: `['lafferty', 'certified', 'lifesciences', 'inc']`; numeric tokens: `[]`
- Address tokens: `['13315', 'schwenger', 'place', 'fairfax', 'county', 'va']`; numeric tokens: `['13315']`

### US apostrophe — `S1-457615425` (US)

- Name: `Shanks's Interstate Physical Therapy` → `shanks s interstate physical therapy`
- Address: `6515 Belcrest Road, Unit 1416, Hyattsville, MD` → `6515 belcrest road unit 1416 hyattsville md`
- Name tokens: `['shanks', 's', 'interstate', 'physical', 'therapy']`; numeric tokens: `[]`
- Address tokens: `['6515', 'belcrest', 'road', 'unit', '1416', 'hyattsville', 'md']`; numeric tokens: `['6515', '1416']`

### Raw encoding artifact — `S1-403268983` (India)

- Name: `Royal Foundation Private Limited` → `royal foundation private limited`
- Address: `Hyderabad, VspâS Bhavana Surovar, Madhavapuri, Hill Flat No.101, First Floor, Pjr Enclave, Ch, Andanagar, Hyderabad, Plot No.277, Telangana` → `hyderabad vspâs bhavana surovar madhavapuri hill flat no 101 first floor pjr enclave ch andanagar hyderabad plot no 277 telangana`
- Name tokens: `['royal', 'foundation', 'private', 'limited']`; numeric tokens: `[]`
- Address tokens: `['hyderabad', 'vspâ\x80\x99s', 'bhavana', 'surovar', 'madhavapuri', 'hill', 'flat', 'no', '101', 'first', 'floor', 'pjr', 'enclave', 'ch', 'andanagar', 'hyderabad', 'plot', 'no', '277', 'telangana']`; numeric tokens: `['101', '277']`

### Repeated whitespace — `S1-694286467` (US)

- Name: `House Superior Meshflow LLC` → `house superior meshflow llc`
- Address: `7021 Memorial Drive, Unit SUITE  253, Tulsa, OK` → `7021 memorial drive unit suite 253 tulsa ok`
- Name tokens: `['house', 'superior', 'meshflow', 'llc']`; numeric tokens: `[]`
- Address tokens: `['7021', 'memorial', 'drive', 'unit', 'suite', '253', 'tulsa', 'ok']`; numeric tokens: `['7021', '253']`

### Indian script — `S2-228141367` (India)

- Name: `अल कंसल्टिंग प्रा. लि.` → `अल कंसल्टिंग प्रा लि`
- Address: `9A PATEL SHOPPING CENTRALCHANDAVARKAR BORIVALI (W), MUMBAI, MUMBAI CITY, Maharashtra` → `9a patel shopping centralchandavarkar borivali w mumbai mumbai city maharashtra`
- Name tokens: `['अल', 'कंसल्टिंग', 'प्रा', 'लि']`; numeric tokens: `[]`
- Address tokens: `['9a', 'patel', 'shopping', 'centralchandavarkar', 'borivali', 'w', 'mumbai', 'mumbai', 'city', 'maharashtra']`; numeric tokens: `['9']`

## Example collisions of different raw names

| Normalized name | Distinct raw names in group | Sample raw variants |
| --- | ---: | --- |
| `pediatric dental` | 10 | `*** PEDIATRIC DENTAL`; `... PEDIATRIC DENTAL`; `PEDIATRIC  DENTAL`; `PEDIATRIC (DENTAL)` |
| `family partners` | 10 | `*** FAMILY PARTNERS`; `-- FAMILY (PARTNERS,)`; `FAMILY  PARTNERS`; `Family  Partners` |
| `physical therapy` | 9 | `PHYSICAL  THERAPY`; `PHYSICAL THERAPY`; `PHYSICAL-THERAPY`; `Physical  Therapy` |
| `falcon inc` | 9 | `FALCON INC`; `FALCON INC.`; `Falcon  Inc`; `Falcon Inc` |
| `foot ankle medicine inc` | 9 | `FOOT &-ANKLE MEDICINE INC`; `Foot & Ankle Medicine  Inc`; `Foot & Ankle Medicine (Inc)`; `Foot & Ankle Medicine Inc` |
| `primary care associates llc` | 9 | `PRIMARY CARE ASSOCIATES-LLC`; `Primary  Care Associates LLC`; `Primary Care  Associates, LLC`; `Primary Care Associates  LLC` |
| `urgent care` | 8 | `URGENT  CARE`; `URGENT CARE`; `Urgent  Care`; `Urgent (Care)` |
| `pediatric dentistry` | 8 | `*** Pediatric Dentistry`; `... Pediatric Dentistry`; `PEDIATRIC  DENTISTRY`; `PEDIATRIC DENTISTRY` |

## Investigated transformations not applied

- **Transliteration / accent stripping:** French accents and Indian scripts are retained. Removing accents can collapse different names, and transliteration can create ambiguous Latin spellings. NFKC itself can still merge compatibility characters.
- **Legal suffix removal or standardization:** Terms such as `Inc`, `LLC`, `Pvt`, `Ltd`, and `SARL` remain in the text. Their interpretation varies by country and context.
- **Address abbreviation standardization:** `St`, `Rd`, `rue`, and other local terms remain as written; `St` can mean street or saint. No US/India-specific rule is used.

## Risks observed

Case, compatibility, whitespace, and punctuation normalization can make distinct raw names identical. In particular, punctuation separation can remove potentially meaningful apostrophes, hyphens, ampersands, or periods. The raw name and address remain available to disambiguate these cases. Training data has US and India only; accented French behavior is covered by unit tests, but this audit cannot measure collision rates in France.
One sampled address already contains apparent mojibake (`â\x80\x99`). The normalizer preserves this raw artifact and does not attempt to repair it.
