# Stage 4 blocking baseline

## Setup and rule definitions

Experiment `blocking_baseline_v1` uses the saved validation split: **440,522 S1 entities** and all **10,320,219** training S2/S3 target records as the search corpus. It evaluates **1,525,811** validation truth links. The target corpus includes records linked to development S1 entities as realistic distractors. All TSVs were read with tab separation; raw files were not changed. Candidate retrieval does not use ground truth, which is consulted only to score hits.

The indices are built over validation S1 records; S2/S3 are streamed once and looked up against those indices. This reverse index is equivalent to retrieving targets for each S1 and avoids storing the full target corpus or all candidate pairs. Each rule returns an S1 set per target; the union deduplicates overlaps.

| Rule | Key / eligibility |
| --- | --- |
| A | Exact nonempty `name_norm`; no country restriction. |
| B | Exact sorted multiset of at least two `name_tokens`; preserves repeated tokens and is insensitive to word order. No country restriction. |
| C1 | Exact nonempty `address_norm` plus same country; address must contain a numeric run of at least three digits. |
| C2 | Same country, shared address numeric run of at least three digits, and one of the S1 name's two rarest eligible tokens (length ≥3, validation S1 document frequency ≤5,000). Up to three longest distinct address numbers are used. |

C1/C2 skip an index key if it maps to more than 64 validation S1 records. In this run, **zero** target lookups triggered that guard. Country labels are handled as arbitrary strings; there are no country-specific normalization rules. No validation truth link crossed countries in these data.

## Rule results

Candidate recall = retrieved true links / all validation true links. All-true-S1 percentages include singletons, which are vacuously complete, and are also shown for non-singletons only.

| Rule | True links found | Candidate recall | All true links found, all S1 | All true links found, matched S1 | Candidate pairs | Avg/S1 | P95 | P99 | Max | Zero-candidate S1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| A: exact normalized name | 330,361 | 21.652% | 8.188% | 2.758% | 4,304,522 | 9.77 | 60 | 175 | 1,039 | 29.331% |
| B: sorted name token multiset | 402,582 | 26.385% | 9.101% | 3.726% | 4,648,885 | 10.55 | 63 | 176 | 1,040 | 23.956% |
| C1: exact numeric address | 93,367 | 6.119% | 6.017% | 0.459% | 115,019 | 0.26 | 1 | 3 | 20 | 82.000% |
| C2: address number + name token | 744,201 | 48.774% | 28.632% | 24.412% | 5,969,501 | 13.55 | 66 | 175 | 1,051 | 30.182% |
| Union A+B+C1+C2 | 924,782 | 60.609% | 35.542% | 31.730% | 10,707,669 | 24.31 | 118 | 241 | 1,075 | 8.095% |

The union's median is **6** candidates/S1 and P90 is **72**. It keeps 10,707,669 of 4,546,283,514,318 possible S1×target pairs (0.000236% of the Cartesian product).

### Incremental true links found in rule order

| Rule added | Newly retrieved true links | True links found by this rule alone among all four |
| --- | ---: | ---: |
| A: exact normalized name | 330,361 | 1,410 |
| B: sorted name token multiset | 76,570 | 31,614 |
| C1: exact numeric address | 73,949 | 13,901 |
| C2: address number + name token | 443,902 | 443,902 |

The numeric-plus-name rule contributes the most new links. Its country and number constraints keep its average candidate count modest, but the union still misses 39.39% of true links. This baseline does **not** meet the high-recall goal.

## Union recall breakdown

Country uses the S1 country. The source breakdown uses the true target's S2/S3 prefix.

| Target source | Retrieved | True links | Recall |
| --- | ---: | ---: | ---: |
| S2 | 440,542 | 738,024 | 59.692% |
| S3 | 484,240 | 787,787 | 61.468% |

| S1 country | Retrieved | True links | Recall |
| --- | ---: | ---: | ---: |
| India | 268,128 | 610,096 | 43.948% |
| US | 656,654 | 915,715 | 71.709% |

France does not occur in training/validation, so its candidate recall is unknown.

| True matches per S1 | Retrieved true links | True links | Recall |
| ---: | ---: | ---: | ---: |
| 1 | 14,251 | 23,704 | 60.121% |
| 2 | 89,872 | 149,432 | 60.142% |
| 3 | 192,850 | 318,288 | 60.590% |
| 4 | 234,272 | 386,088 | 60.678% |
| 5 | 194,723 | 321,105 | 60.642% |
| 6 | 120,864 | 199,068 | 60.715% |
| 7 | 54,712 | 89,880 | 60.872% |
| 8 | 18,015 | 29,584 | 60.894% |
| 9 | 4,610 | 7,578 | 60.834% |
| 10 | 584 | 1,040 | 56.154% |
| 11 | 29 | 44 | 65.909% |

## True singleton candidate behavior

Among **24,597** true singleton S1 entities, the union yields **552,385** candidate pairs: mean **22.46**, median **3**, P90 **71**, P95 **119**, P99 **237**, max **609**. **8,256** (33.565%) receive zero candidates. Candidates for a singleton are not predictions; a later matcher must reject them.

## Runtime, memory, and candidate tail

Wall runtime: **1214.5 seconds** (20.24 minutes), including 22.2 seconds to load validation S1 and 37.6 seconds to build indices and truth lookup. Approximate peak process RSS sampled during the run: **0.90 GiB**. Index key counts: A_exact_name 355,975, B_name_signature 354,423, C1_exact_numeric_address 319,872, C2_numeric_plus_name_token 615,494.

The maximum union set is 1,075 candidates for one S1, versus a median of 6 and P99 of 241. Exact-name and name-signature rules individually reach maxima around 1,040, so common names create a visible high tail. The 64-S1 posting guard on numeric/address rules never fired. The union's 10.71 million pairs are feasible to count and stream here, but later pair scoring would need to handle these outliers.

## Miss analysis: fixed hash sample

The **25** misses below are the smallest SHA-256 hash values of `2026:<S1 ID>:<target ID>` among all union-missed validation truth links. The sample is reproducible and was manually categorized. Primary category counts describe this small sample only; categories overlap in reality.

| Primary category | Sample count |
| --- | ---: |
| abbreviation / legal form | 9 |
| missing / repeated name tokens | 7 |
| transliteration | 5 |
| accent difference | 1 |
| address-only evidence | 1 |
| different name representation | 1 |
| spelling difference | 1 |

In the 25 misses, **18** share at least one exact name token, **15** share some address number, but only **3** share a number eligible for C2. Transliteration and wholly different names often leave address evidence as the only common signal. One target has an empty address.

### Miss 1: `S1-731436000` → `S3-605343849` (S3, India)

- Raw names: `Sadar Crafts Private Limited` → `Sadar Crafts Prsiem Limited`
- Normalized names: `sadar crafts private limited` → `sadar crafts prsiem limited`
- Raw addresses: `Hn 76/F65, Imliya, Sadar, Mau, Uttar Pradesh` → `Mau, Imliya, Sadar, Hn 76/F65, UP`
- Normalized addresses: `hn 76 f65 imliya sadar mau uttar pradesh` → `mau imliya sadar hn 76 f65 up`
- Primary category: **spelling difference**. `Private` becomes `Prsiem`; the shared address numbers are only two digits, and the address order differs.

### Miss 2: `S1-521020374` → `S3-961854123` (S3, India)

- Raw names: `Hari All Business` → `హరి ఆల్ బిజినెస్`
- Normalized names: `hari all business` → `హరి ఆల్ బిజినెస్`
- Raw addresses: `Plot No. 47-Iii, 8-2-293/82/Jiii/47, Rn 76, Shaikpet, Hyderabad, Telangana` → `H.no 47-Iii, 8-2-293/82/Jiii/47, Rn 76, Hyderabad, Shaikpet, Andhra Pradesh`
- Normalized addresses: `plot no 47 iii 8 2 293 82 jiii 47 rn 76 shaikpet hyderabad telangana` → `h no 47 iii 8 2 293 82 jiii 47 rn 76 hyderabad shaikpet andhra pradesh`
- Primary category: **transliteration**. Latin and Telugu names share no tokens. Numbers overlap, but C2 also requires a shared name token; addresses are not exact.

### Miss 3: `S1-670000585` → `S2-554833247` (S2, India)

- Raw names: `Star & Brothers Private Limited` → `Star +  Brothers Private Ltd`
- Normalized names: `star brothers private limited` → `star + brothers private ltd`
- Raw addresses: `12Th Floor, Dev Corpora, Near Cadbury Company, Near Khajana E.E. Highway, Khopat, Thane, Maharashtra` → `#12TH FLOOR, DEV CORPORA, NEAR CADBURY COMPANY, NEAR KHAJANA E.E. HIGHWAY, KHOPAT, Maharashtra`
- Normalized addresses: `12th floor dev corpora near cadbury company near khajana e e highway khopat thane maharashtra` → `12th floor dev corpora near cadbury company near khajana e e highway khopat maharashtra`
- Primary category: **abbreviation / legal form**. `Limited` becomes `Ltd` and `&` becomes `+`; only the two-digit floor number is shared.

### Miss 4: `S1-34119872` → `S2-868885400` (S2, India)

- Raw names: `Ever Services Ltd` → `Ever Services Limited`
- Normalized names: `ever services ltd` → `ever services limited`
- Raw addresses: `Vadodara, Gujarat, Vadodara, 292 Dharmsinh Desia Margnr Baroda Co Op Ind Estate Chhani Road` → `2-92 DHARMSINH DESIA MARGNR BARODA CO OP IND ESTATE CHHANI ROAD, VADODARA, ગુજરાત`
- Normalized addresses: `vadodara gujarat vadodara 292 dharmsinh desia margnr baroda co op ind estate chhani road` → `2 92 dharmsinh desia margnr baroda co op ind estate chhani road vadodara ગુજરાત`
- Primary category: **abbreviation / legal form**. `Ltd` becomes `Limited`; the address number changes from `292` to `2-92`, leaving no shared eligible number.

### Miss 5: `S1-160463661` → `S2-634974154` (S2, India)

- Raw names: `My Consulting Private Limited` → `மை கன்சல்டிங் பிரைவேட் லிமிடெட்`
- Normalized names: `my consulting private limited` → `மை கன்சல்டிங் பிரைவேட் லிமிடெட்`
- Raw addresses: `49/26, Ist Floor Ramasamy Kovil Agraharam, Palayamkottai, Tirunelveli, Tamil Nadu` → `49/26, PALAYAMKOTTAI, TIRUNELVELI, தமிழ்நாடு`
- Normalized addresses: `49 26 ist floor ramasamy kovil agraharam palayamkottai tirunelveli tamil nadu` → `49 26 palayamkottai tirunelveli தமிழ்நாடு`
- Primary category: **transliteration**. Latin and Tamil names share no tokens; the common address numbers `49` and `26` are below the three-digit rule.

### Miss 6: `S1-251684948` → `S3-454305563` (S3, US)

- Raw names: `Charley, Fredericka A., DO` → `frederickacharleya.com`
- Normalized names: `charley fredericka a do` → `frederickacharleya com`
- Raw addresses: `15105 Oak Street, Dolton, IL` → `Dolton Towship, Illinois, 15105 Oak St`
- Normalized addresses: `15105 oak street dolton il` → `dolton towship illinois 15105 oak st`
- Primary category: **different name representation**. A personal name is concatenated into a web domain. The five-digit address number matches, but no name token does.

### Miss 7: `S1-216816542` → `S2-909961405` (S2, India)

- Raw names: `Green Engineering` → `Green Green Engineering`
- Normalized names: `green engineering` → `green green engineering`
- Raw addresses: `525 Ward 6, At Asha Nagar, More Pops -Soh Sarai, Bihar, Nalanda, Bihar` → `25 WARD 6, AT ASHA NAGAR, MORE POPS -SOH SARAI, BIHAR, Bihar`
- Normalized addresses: `525 ward 6 at asha nagar more pops soh sarai bihar nalanda bihar` → `25 ward 6 at asha nagar more pops soh sarai bihar bihar`
- Primary category: **missing / repeated name tokens**. `Green` is repeated; address number `525` becomes `25`, leaving only the one-digit `6` shared.

### Miss 8: `S1-784938102` → `S2-725060927` (S2, US)

- Raw names: `Nelson's Diamond Apparel L.L.C.` → `Nelson's Diamond Apparel Llc`
- Normalized names: `nelson s diamond apparel l l c` → `nelson s diamond apparel llc`
- Raw addresses: `2216 Harrod Street, Ashland, KY` → `HARROD STREET, ASHLAND, NULL, KY`
- Normalized addresses: `2216 harrod street ashland ky` → `harrod street ashland null ky`
- Primary category: **abbreviation / legal form**. `L.L.C.` becomes `Llc`, changing the token multiset; the target address has no number.

### Miss 9: `S1-973056391` → `S3-471043429` (S3, US)

- Raw names: `Randene Waiters End LLC` → `Randene Waiters  End`
- Normalized names: `randene waiters end llc` → `randene waiters end`
- Raw addresses: `1505 Walters Avenue, Dierks, AR` → `150 Walters Ave, Dierks, Arkansas`
- Normalized addresses: `1505 walters avenue dierks ar` → `150 walters ave dierks arkansas`
- Primary category: **abbreviation / legal form**. `LLC` is omitted and the address number changes from `1505` to `150`.

### Miss 10: `S1-286219191` → `S3-779258965` (S3, US)

- Raw names: `Clean Freedom LLC` → `Clean Freedom`
- Normalized names: `clean freedom llc` → `clean freedom`
- Raw addresses: `7011 John Adams Way, Louisville, KY` → `John Adams Way, Louisville, Kentucky`
- Normalized addresses: `7011 john adams way louisville ky` → `john adams way louisville kentucky`
- Primary category: **abbreviation / legal form**. `LLC` is omitted; the target address also omits the street number.

### Miss 11: `S1-370403107` → `S3-413211188` (S3, US)

- Raw names: `Newcomb Cleaning PLLC` → `Newcomb Cleaning`
- Normalized names: `newcomb cleaning pllc` → `newcomb cleaning`
- Raw addresses: `56 Nancy Drive, Cropwell, AL` → `#56 Nancy Drive, Alabama, Cropwell`
- Normalized addresses: `56 nancy drive cropwell al` → `56 nancy drive alabama cropwell`
- Primary category: **abbreviation / legal form**. `PLLC` is omitted; the shared number `56` is below the three-digit threshold.

### Miss 12: `S1-773777308` → `S3-260189364` (S3, India)

- Raw names: `Engineering Point Foundation` → `Engineering Point`
- Normalized names: `engineering point foundation` → `engineering point`
- Raw addresses: `Patna, New Mainpura, Bihar, Pragati Nagar, Ward 14, Dinapur-Cum-Khagaul` → `Patna, Dinapur-cum-khagaul, Hn 151 New Mainpura, Pragati Nagar, Ward 14, BR`
- Normalized addresses: `patna new mainpura bihar pragati nagar ward 14 dinapur cum khagaul` → `patna dinapur cum khagaul hn 151 new mainpura pragati nagar ward 14 br`
- Primary category: **missing / repeated name tokens**. `Foundation` is omitted; the only shared address number is `14`, below the numeric threshold.

### Miss 13: `S1-521515409` → `S2-946804009` (S2, India)

- Raw names: `Shakti Agro Private Limited` → `Limited Shakti Private Partners`
- Normalized names: `shakti agro private limited` → `limited shakti private partners`
- Raw addresses: `G T Roadsherpur Ludhiana, Punjab, Ludhiana` → `ਪੰਜਾਬ, LUDHIANA, G T ROADSHERPUR LUDHIANA, LUDHIANA`
- Normalized addresses: `g t roadsherpur ludhiana punjab ludhiana` → `ਪੰਜਾਬ ludhiana g t roadsherpur ludhiana ludhiana`
- Primary category: **missing / repeated name tokens**. `Agro` changes to `Partners` and word order changes; neither address contains an eligible common number.

### Miss 14: `S1-718458221` → `S2-660333848` (S2, US)

- Raw names: `Leonore C. Johnson, Esq. Care` → `LEONORE C. JOHNSON,`
- Normalized names: `leonore c johnson esq care` → `leonore c johnson`
- Raw addresses: `303 Birch Street, Sauk Centre, MN` → `319 BIRCH STREET, SAUK CENTRE, MN`
- Normalized addresses: `303 birch street sauk centre mn` → `319 birch street sauk centre mn`
- Primary category: **missing / repeated name tokens**. The target truncates `Esq. Care`; street numbers `303` and `319` disagree.

### Miss 15: `S1-300438772` → `S3-839087912` (S3, US)

- Raw names: `Patton, Blazek and Luna Group` → `Patton, Blazek and Luna Inc`
- Normalized names: `patton blazek and luna group` → `patton blazek and luna inc`
- Raw addresses: `218 Scenic Hills, Parkersburg, WV` → `Scenic Hills, Parkersburg, West Virginia`
- Normalized addresses: `218 scenic hills parkersburg wv` → `scenic hills parkersburg west virginia`
- Primary category: **abbreviation / legal form**. The ending changes from `Group` to `Inc`; the target address omits its street number.

### Miss 16: `S1-148187034` → `S2-865553205` (S2, US)

- Raw names: `Gullion Elite Atlas Inc.` → `Gullion Élite Atlas Inc.`
- Normalized names: `gullion elite atlas inc` → `gullion élite atlas inc`
- Raw addresses: `2 Deon Way, Plattsburgh, NY` → `2-6 DEON WAY, PLATTSBURGH, NY`
- Normalized addresses: `2 deon way plattsburgh ny` → `2 6 deon way plattsburgh ny`
- Primary category: **accent difference**. `Elite` and `Élite` remain distinct under the accent-preserving normalizer; the shared number `2` is short.

### Miss 17: `S1-881367168` → `S2-134812572` (S2, India)

- Raw names: `Silk Services Private Limited` → `SILK SERVICES  PRIVATE`
- Normalized names: `silk services private limited` → `silk services private`
- Raw addresses: `Pune, Maharashtra, Pune, 5/4 Nagar Road Pune.` → `*(empty)*`
- Normalized addresses: `pune maharashtra pune 5 4 nagar road pune` → `*(empty)*`
- Primary category: **missing / repeated name tokens**. `Limited` is omitted, and the target address is empty.

### Miss 18: `S1-502888861` → `S2-604637307` (S2, US)

- Raw names: `Better Commercial Laboratories LLC` → `BETTER COMMERCIAL`
- Normalized names: `better commercial laboratories llc` → `better commercial`
- Raw addresses: `42 Hunns Lake Road, Stanford, NY` → `42B HUNNS LAKE ROAD, STANFORD, NY`
- Normalized addresses: `42 hunns lake road stanford ny` → `42b hunns lake road stanford ny`
- Primary category: **missing / repeated name tokens**. `Laboratories LLC` is omitted; common address number `42` is below the threshold.

### Miss 19: `S1-638569181` → `S3-8528430` (S3, India)

- Raw names: `Chennai 16 Exporters Private Limited` → `Chennai 16 Exporters Private Ltd`
- Normalized names: `chennai 16 exporters private limited` → `chennai 16 exporters private ltd`
- Raw addresses: `No.3, Muthaiyal Reddy Street, Alandur, Chennai 16, Tamil Nadu` → `Door No 3, Chennai 16, Alandur, தமிழ்நாடு`
- Normalized addresses: `no 3 muthaiyal reddy street alandur chennai 16 tamil nadu` → `door no 3 chennai 16 alandur தமிழ்நாடு`
- Primary category: **abbreviation / legal form**. `Limited` becomes `Ltd`; the common address numbers `3` and `16` are short.

### Miss 20: `S1-875801955` → `S2-479334485` (S2, US)

- Raw names: `Downtown Ice Cream` → `Downtown Ice`
- Normalized names: `downtown ice cream` → `downtown ice`
- Raw addresses: `16621 33rd Avenue, Lynnwood, WA` → `33ND AVE, LYNNWOOD, WA`
- Normalized addresses: `16621 33rd avenue lynnwood wa` → `33nd ave lynnwood wa`
- Primary category: **missing / repeated name tokens**. `Cream` is omitted; the target address lacks the S1 street number.

### Miss 21: `S1-632370376` → `S2-913851699` (S2, India)

- Raw names: `Jaipur Stocks Pvt Ltd` → `Jaipur Stocks Pvt Limited`
- Normalized names: `jaipur stocks pvt ltd` → `jaipur stocks pvt limited`
- Raw addresses: `Fl No-401, 3Rd Floor, Vaishali Elegance, Jaipur, Rajasthan` → `FL NO-40, 3RD FLOOR, VAISHALI ELEGANCE, JAIPUR, Rajasthan`
- Normalized addresses: `fl no 401 3rd floor vaishali elegance jaipur rajasthan` → `fl no 40 3rd floor vaishali elegance jaipur rajasthan`
- Primary category: **abbreviation / legal form**. `Ltd` becomes `Limited`, while `401` changes to `40`; only one-digit `3` remains shared.

### Miss 22: `S1-905781662` → `S3-86984213` (S3, India)

- Raw names: `Eastern Construction` → `ईस्टर्न कंस्ट्रक्शन`
- Normalized names: `eastern construction` → `ईस्टर्न कंस्ट्रक्शन`
- Raw addresses: `C/O Prakash Narayan Shukla Jahanpur (Near Osho School Bindki), Fatehpur, Uttar Pradesh` → `C/o Prakash Narayan Shukla Jahanpur (Near Osho School Bindki), Fatehpur, उत्तर प्रदेश`
- Normalized addresses: `c o prakash narayan shukla jahanpur near osho school bindki fatehpur uttar pradesh` → `c o prakash narayan shukla jahanpur near osho school bindki fatehpur उत्तर प्रदेश`
- Primary category: **transliteration**. Latin and Hindi names share no tokens. The addresses overlap as text but contain no numbers.

### Miss 23: `S1-370645971` → `S3-577357887` (S3, India)

- Raw names: `Bombay Foundation LLP` → `बॉम्बे फाउंडेशन एलएलपी`
- Normalized names: `bombay foundation llp` → `बॉम्बे फाउंडेशन एलएलपी`
- Raw addresses: `C/O Moradabad Road-Opp.Ekta Vihar Sambhal, Moradabad, Sambhal, Uttar Pradesh` → `Moradabad, Sambhal, उत्तर प्रदेश, C/o Moradabad Road-opp.ekta Vihar Sambhal`
- Normalized addresses: `c o moradabad road opp ekta vihar sambhal moradabad sambhal uttar pradesh` → `moradabad sambhal उत्तर प्रदेश c o moradabad road opp ekta vihar sambhal`
- Primary category: **transliteration**. Latin and Hindi names share no tokens. Address components are reordered, with no numeric anchor.

### Miss 24: `S1-106678782` → `S2-106319943` (S2, India)

- Raw names: `Vis College Pvt. Ltd.` → `Mr EVOEVOJAX`
- Normalized names: `vis college pvt ltd` → `mr evoevojax`
- Raw addresses: `Shop No.2, Jayant Krupa Sector-9, Dive Village, Navi Mumbai, Maharashtra` → `NAVI MUMBAI, महाराष्ट्र, NAVI MUMBAI, SHOP NO.2, JAYANT KRUPA SECTOR-9, DIVE VILLAGE`
- Normalized addresses: `shop no 2 jayant krupa sector 9 dive village navi mumbai maharashtra` → `navi mumbai महाराष्ट्र navi mumbai shop no 2 jayant krupa sector 9 dive village`
- Primary category: **address-only evidence**. The names are unrelated strings. The addresses overlap, but their common numbers `2` and `9` are short.

### Miss 25: `S1-48680569` → `S2-197425554` (S2, India)

- Raw names: `Star Classic Ventures Private Limited` → `స్టార్ క్లాసిక్ వెంచర్స్ ప్రైవేట్ లిమిటెడ్`
- Normalized names: `star classic ventures private limited` → `స్టార్ క్లాసిక్ వెంచర్స్ ప్రైవేట్ లిమిటెడ్`
- Raw addresses: `120-25/164 Taj Mahal Colony, Peerzadiguda, Hyderabade, Hyderabad, Telangana` → `G-120-25/164 TAJ MAHAL COLONY, PEERZADIGUDA, HYDERABADE, HYDERABAD, Andhra Pradesh`
- Normalized addresses: `120 25 164 taj mahal colony peerzadiguda hyderabade hyderabad telangana` → `g 120 25 164 taj mahal colony peerzadiguda hyderabade hyderabad andhra pradesh`
- Primary category: **transliteration**. Latin and Telugu names share no tokens; long address numbers match, but C2 requires a shared name token.

## Interpretation and next step

This is a measured first baseline, not a sufficient final candidate generator. The largest gaps are India (especially Latin ↔ Indian-script names), legal suffix or name truncation changes, and addresses whose numbers are short, altered, or absent. The next retrieval experiment should be chosen from these observed gaps and measured on the same validation split; no fuzzy, TF-IDF, embedding, feature, or ML retrieval was implemented in Stage 4.
