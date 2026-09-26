# Stage 5: character TF-IDF name retrieval

## Configuration and scalability

The query set is the saved validation split of **440,522 S1 entities**. TF-IDF is fit on their `name_norm` strings, then all training S2/S3 names are streamed as targets. It uses `char_wb` character **4–5 grams**, sublinear term frequency, L2 normalization, `min_df=2`, and `max_df=0.02`. A 4–5 gram range is compact enough to avoid the very common three-character fragments while retaining partial names, order-insensitive word evidence, and small spelling variations. The fitted vocabulary has **128,109** grams and **10,345,536** nonzeros in the S1 query matrix.

Country is a dynamic retrieval partition: same nonempty country is compared, missing-country S1 queries search every target partition, and missing-country targets search every S1 partition. No country list is hard-coded; France behavior is covered by a unit test. Sparse cosine products use `sparse-dot-topn`, batches of 100,000 targets, a score floor of 0.05, and at most **20 results per S1 per target source**. Global top lists are merged across batches. No dense S1×target matrix is formed.

Dependencies live in the project `.venv` (`requirements-stage5.txt`): scikit-learn 1.9.1 and sparse-dot-topn 1.2.0. The top-20 arrays are saved under `artifacts/` so K=5, 10, and 20 use the same retrieval run. Stage 4 pairs were reconstructed exactly and compared by packed S1/source/target IDs; the replay matched its published **10,707,669** pairs and **924,782** true hits.

## Recall and candidate-volume tradeoff

Candidate recall means the fraction of validation truth links present in the candidate set. Country recall uses S1 country. Each K is a maximum **per target source**, so TF-IDF alone can return up to 2K targets per S1. Union counts deduplicate exact pair overlap with Stage 4.

| Retrieval | True-link recall | S2 | S3 | India | US | Candidate pairs | Mean/S1 | P95 | P99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage 4 baseline | 60.609% | 59.692% | 61.468% | 43.948% | 71.709% | 10,707,669 | 24.31 | 118 | 241 |
| TF-IDF K=5 | 53.213% | 54.239% | 52.252% | 45.939% | 58.059% | 4,404,494 | 10.00 | 10 | 10 |
| Stage 4 + TF-IDF K=5 | 74.646% | 74.249% | 75.019% | 61.484% | 83.416% | 14,027,030 | 31.84 | 125 | 248 |
| TF-IDF K=10 | 59.687% | 60.708% | 58.730% | 51.384% | 65.218% | 8,808,773 | 20.00 | 20 | 20 |
| Stage 4 + TF-IDF K=10 | 77.085% | 76.728% | 77.419% | 64.224% | 85.653% | 18,043,178 | 40.96 | 132 | 255 |
| TF-IDF K=20 | 64.942% | 65.890% | 64.053% | 56.562% | 70.525% | 17,616,831 | 39.99 | 40 | 40 |
| Stage 4 + TF-IDF K=20 | 79.063% | 78.712% | 79.393% | 66.680% | 87.314% | 26,242,184 | 59.57 | 148 | 269 |

| K per source | New true links beyond Stage 4 | Recall gain over Stage 4 | TF-IDF pairs already in Stage 4 | Union median | Union P90 | Union max |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 214,178 | +14.037 pp | 1,085,133 | 13 | 79 | 1,076 |
| 10 | 251,386 | +16.476 pp | 1,473,264 | 22 | 86 | 1,080 |
| 20 | 281,573 | +18.454 pp | 2,082,316 | 42 | 100 | 1,090 |

**Selected configuration: K=10 per source.** The table above shows the gain and candidate cost of moving to K=20; this selection favors a smaller candidate set when the extra recall is limited. Stage 4 alone reaches every truth link for 31.730% of non-singleton S1 entities; the selected union reaches 53.105%.

France has no validation ground truth, so France recall cannot be measured here.

## Runtime and memory

TF-IDF fitting and retrieval took **2442.4 seconds** (40.71 minutes) with approximate sampled peak RSS **1.06 GiB**. Within this run, vectorization accounted for 550.5 seconds and sparse products for 1496.1 seconds. The exact Stage 4 pair replay took **214.3 seconds** and ran concurrently with TF-IDF's S3 pass. The union evaluation process peaked at **2.00 GiB**; its wall duration includes waiting for TF-IDF to finish, so it should not be added to the retrieval duration.

## Remaining misses: fixed hash sample

For K=10, the sample takes the 25 smallest SHA-256 hashes of `2026:<S1 ID>:<target ID>` among true links still missed by the union. These manually reviewed categories describe the sample, not population rates.

| Primary category | Sample count |
| --- | ---: |
| Different script / transliteration | 7 |
| Name truncation / addition | 4 |
| Spelling / text noise | 3 |
| Concatenated web / hashtag name | 3 |
| Legal suffix / punctuation variation | 2 |
| Major name difference | 2 |
| Weak name / strong address | 2 |
| Opaque name / strong address | 1 |
| Alias expansion | 1 |

### Miss 1: `S1-521020374` → `S3-961854123` (S3, India)

- Raw names: `Hari All Business` → `హరి ఆల్ బిజినెస్`
- Normalized names: `hari all business` → `హరి ఆల్ బిజినెస్`
- Raw addresses: `Plot No. 47-Iii, 8-2-293/82/Jiii/47, Rn 76, Shaikpet, Hyderabad, Telangana` → `H.no 47-Iii, 8-2-293/82/Jiii/47, Rn 76, Hyderabad, Shaikpet, Andhra Pradesh`
- Normalized addresses: `plot no 47 iii 8 2 293 82 jiii 47 rn 76 shaikpet hyderabad telangana` → `h no 47 iii 8 2 293 82 jiii 47 rn 76 hyderabad shaikpet andhra pradesh`
- **Different script / transliteration:** Latin business name is rendered in Telugu; the similar address cannot rescue name retrieval.

### Miss 2: `S1-670000585` → `S2-554833247` (S2, India)

- Raw names: `Star & Brothers Private Limited` → `Star +  Brothers Private Ltd`
- Normalized names: `star brothers private limited` → `star + brothers private ltd`
- Raw addresses: `12Th Floor, Dev Corpora, Near Cadbury Company, Near Khajana E.E. Highway, Khopat, Thane, Maharashtra` → `#12TH FLOOR, DEV CORPORA, NEAR CADBURY COMPANY, NEAR KHAJANA E.E. HIGHWAY, KHOPAT, Maharashtra`
- Normalized addresses: `12th floor dev corpora near cadbury company near khajana e e highway khopat thane maharashtra` → `12th floor dev corpora near cadbury company near khajana e e highway khopat maharashtra`
- **Legal suffix / punctuation variation:** The plus sign and Ltd versus Limited change the character grams; this link has zero-based rank 16 (17th candidate) and misses K=10.

### Miss 3: `S1-160463661` → `S2-634974154` (S2, India)

- Raw names: `My Consulting Private Limited` → `மை கன்சல்டிங் பிரைவேட் லிமிடெட்`
- Normalized names: `my consulting private limited` → `மை கன்சல்டிங் பிரைவேட் லிமிடெட்`
- Raw addresses: `49/26, Ist Floor Ramasamy Kovil Agraharam, Palayamkottai, Tirunelveli, Tamil Nadu` → `49/26, PALAYAMKOTTAI, TIRUNELVELI, தமிழ்நாடு`
- Normalized addresses: `49 26 ist floor ramasamy kovil agraharam palayamkottai tirunelveli tamil nadu` → `49 26 palayamkottai tirunelveli தமிழ்நாடு`
- **Different script / transliteration:** Latin business name is rendered in Tamil.

### Miss 4: `S1-216816542` → `S2-909961405` (S2, India)

- Raw names: `Green Engineering` → `Green Green Engineering`
- Normalized names: `green engineering` → `green green engineering`
- Raw addresses: `525 Ward 6, At Asha Nagar, More Pops -Soh Sarai, Bihar, Nalanda, Bihar` → `25 WARD 6, AT ASHA NAGAR, MORE POPS -SOH SARAI, BIHAR, Bihar`
- Normalized addresses: `525 ward 6 at asha nagar more pops soh sarai bihar nalanda bihar` → `25 ward 6 at asha nagar more pops soh sarai bihar bihar`
- **Spelling / text noise:** A name token is repeated and the address number also changes, so exact and fuzzy evidence weaken.

### Miss 5: `S1-521515409` → `S2-946804009` (S2, India)

- Raw names: `Shakti Agro Private Limited` → `Limited Shakti Private Partners`
- Normalized names: `shakti agro private limited` → `limited shakti private partners`
- Raw addresses: `G T Roadsherpur Ludhiana, Punjab, Ludhiana` → `ਪੰਜਾਬ, LUDHIANA, G T ROADSHERPUR LUDHIANA, LUDHIANA`
- Normalized addresses: `g t roadsherpur ludhiana punjab ludhiana` → `ਪੰਜਾਬ ludhiana g t roadsherpur ludhiana ludhiana`
- **Major name difference:** Only Shakti and generic legal words remain; the other distinctive name token changes.

### Miss 6: `S1-502888861` → `S2-604637307` (S2, US)

- Raw names: `Better Commercial Laboratories LLC` → `BETTER COMMERCIAL`
- Normalized names: `better commercial laboratories llc` → `better commercial`
- Raw addresses: `42 Hunns Lake Road, Stanford, NY` → `42B HUNNS LAKE ROAD, STANFORD, NY`
- Normalized addresses: `42 hunns lake road stanford ny` → `42b hunns lake road stanford ny`
- **Name truncation / addition:** The target omits Laboratories LLC; its shorter name has many plausible competitors.

### Miss 7: `S1-875801955` → `S2-479334485` (S2, US)

- Raw names: `Downtown Ice Cream` → `Downtown Ice`
- Normalized names: `downtown ice cream` → `downtown ice`
- Raw addresses: `16621 33rd Avenue, Lynnwood, WA` → `33ND AVE, LYNNWOOD, WA`
- Normalized addresses: `16621 33rd avenue lynnwood wa` → `33nd ave lynnwood wa`
- **Name truncation / addition:** The target omits Cream from the business name.

### Miss 8: `S1-905781662` → `S3-86984213` (S3, India)

- Raw names: `Eastern Construction` → `ईस्टर्न कंस्ट्रक्शन`
- Normalized names: `eastern construction` → `ईस्टर्न कंस्ट्रक्शन`
- Raw addresses: `C/O Prakash Narayan Shukla Jahanpur (Near Osho School Bindki), Fatehpur, Uttar Pradesh` → `C/o Prakash Narayan Shukla Jahanpur (Near Osho School Bindki), Fatehpur, उत्तर प्रदेश`
- Normalized addresses: `c o prakash narayan shukla jahanpur near osho school bindki fatehpur uttar pradesh` → `c o prakash narayan shukla jahanpur near osho school bindki fatehpur उत्तर प्रदेश`
- **Different script / transliteration:** Latin business name is rendered in Hindi.

### Miss 9: `S1-370645971` → `S3-577357887` (S3, India)

- Raw names: `Bombay Foundation LLP` → `बॉम्बे फाउंडेशन एलएलपी`
- Normalized names: `bombay foundation llp` → `बॉम्बे फाउंडेशन एलएलपी`
- Raw addresses: `C/O Moradabad Road-Opp.Ekta Vihar Sambhal, Moradabad, Sambhal, Uttar Pradesh` → `Moradabad, Sambhal, उत्तर प्रदेश, C/o Moradabad Road-opp.ekta Vihar Sambhal`
- Normalized addresses: `c o moradabad road opp ekta vihar sambhal moradabad sambhal uttar pradesh` → `moradabad sambhal उत्तर प्रदेश c o moradabad road opp ekta vihar sambhal`
- **Different script / transliteration:** Latin business name is rendered in Hindi.

### Miss 10: `S1-106678782` → `S2-106319943` (S2, India)

- Raw names: `Vis College Pvt. Ltd.` → `Mr EVOEVOJAX`
- Normalized names: `vis college pvt ltd` → `mr evoevojax`
- Raw addresses: `Shop No.2, Jayant Krupa Sector-9, Dive Village, Navi Mumbai, Maharashtra` → `NAVI MUMBAI, महाराष्ट्र, NAVI MUMBAI, SHOP NO.2, JAYANT KRUPA SECTOR-9, DIVE VILLAGE`
- Normalized addresses: `shop no 2 jayant krupa sector 9 dive village navi mumbai maharashtra` → `navi mumbai महाराष्ट्र navi mumbai shop no 2 jayant krupa sector 9 dive village`
- **Weak name / strong address:** The names are unrelated while the addresses provide the useful match evidence.

### Miss 11: `S1-48680569` → `S2-197425554` (S2, India)

- Raw names: `Star Classic Ventures Private Limited` → `స్టార్ క్లాసిక్ వెంచర్స్ ప్రైవేట్ లిమిటెడ్`
- Normalized names: `star classic ventures private limited` → `స్టార్ క్లాసిక్ వెంచర్స్ ప్రైవేట్ లిమిటెడ్`
- Raw addresses: `120-25/164 Taj Mahal Colony, Peerzadiguda, Hyderabade, Hyderabad, Telangana` → `G-120-25/164 TAJ MAHAL COLONY, PEERZADIGUDA, HYDERABADE, HYDERABAD, Andhra Pradesh`
- Normalized addresses: `120 25 164 taj mahal colony peerzadiguda hyderabade hyderabad telangana` → `g 120 25 164 taj mahal colony peerzadiguda hyderabade hyderabad andhra pradesh`
- **Different script / transliteration:** Latin business name is rendered in Telugu.

### Miss 12: `S1-833391908` → `S2-495340863` (S2, India)

- Raw names: `Modern Infotech Private Limited` → `ಮಾಡರ್ನ್ ಇನ್‌ಫೋಟೆಕ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್`
- Normalized names: `modern infotech private limited` → `ಮಾಡರ್ನ್ ಇನ್‌ಫೋಟೆಕ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್`
- Raw addresses: `No.174, 2Nd Floor, 2Nd Cross, 7Th Main, 1St Block, Koramangala, Bangalore, Karnataka` → `NO.174, BANGALORE, ಕರ್ನಾಟಕ`
- Normalized addresses: `no 174 2nd floor 2nd cross 7th main 1st block koramangala bangalore karnataka` → `no 174 bangalore ಕರ್ನಾಟಕ`
- **Different script / transliteration:** Latin business name is rendered in Kannada.

### Miss 13: `S1-108694605` → `S3-689689104` (S3, US)

- Raw names: `Primary Care Partners` → `The Primary Care Partners`
- Normalized names: `primary care partners` → `the primary care partners`
- Raw addresses: `ME, 42 Hillis Street, Portland` → `*(empty)*`
- Normalized addresses: `me 42 hillis street portland` → `*(empty)*`
- **Name truncation / addition:** The target adds The to a generic name and has an empty address.

### Miss 14: `S1-30008742` → `S2-892319502` (S2, India)

- Raw names: `East Constructions Pvt Ltd` → `ইস্ট কনস্ট্রাকশনস প্রাইভেট লিমিটেড`
- Normalized names: `east constructions pvt ltd` → `ইস্ট কনস্ট্রাকশনস প্রাইভেট লিমিটেড`
- Raw addresses: `C/O Akbar Ali Khan Vill Dewli, Ps Po Raghunathganj, Murshidabad, West Bengal` → `MURSHIDABAD, C/O AKBAR ALI KHAN VILL DEWLI, West Bengal, PS PO RAGHUNATHGANJ`
- Normalized addresses: `c o akbar ali khan vill dewli ps po raghunathganj murshidabad west bengal` → `murshidabad c o akbar ali khan vill dewli west bengal ps po raghunathganj`
- **Different script / transliteration:** Latin business name is rendered in Bengali.

### Miss 15: `S1-212604968` → `S3-798007045` (S3, US)

- Raw names: `Total Energy International LLC` → `Totalenergyintl.Com`
- Normalized names: `total energy international llc` → `totalenergyintl com`
- Raw addresses: `1632 Rockwell Court, Village Of Howard, WI` → `Rockwell Court, Village Of Howard, Wisconsin`
- Normalized addresses: `1632 rockwell court village of howard wi` → `rockwell court village of howard wisconsin`
- **Concatenated web / hashtag name:** The target uses a compressed domain-style name and omits words from the longer business name.

### Miss 16: `S1-36665438` → `S2-100070337` (S2, India)

- Raw names: `Shree Logistics Private Limited` → `Novimira`
- Normalized names: `shree logistics private limited` → `novimira`
- Raw addresses: `Nand Bhawan Main Market Near Patrick'S Bureau Office, Baran, Rajasthan` → `NAND BHAWAN MAIN MARKET NEAR PATRICK'S BUREAU OFFICE, BARAN, Rajasthan`
- Normalized addresses: `nand bhawan main market near patrick s bureau office baran rajasthan` → `nand bhawan main market near patrick s bureau office baran rajasthan`
- **Weak name / strong address:** The target name is unrelated, but the raw addresses are effectively identical.

### Miss 17: `S1-708579475` → `S3-949922323` (S3, India)

- Raw names: `High United Consulting` → `#highunited`
- Normalized names: `high united consulting` → `highunited`
- Raw addresses: `103 E Green Valley, Indore, Madhya Pradesh` → `#103 E Green Valley, Dewas, MP`
- Normalized addresses: `103 e green valley indore madhya pradesh` → `103 e green valley dewas mp`
- **Concatenated web / hashtag name:** The target reduces the business name to a joined hashtag form.

### Miss 18: `S1-573133231` → `S3-726467438` (S3, US)

- Raw names: `N 6 Z Kochav` → `N  6 Z`
- Normalized names: `n 6 z kochav` → `n 6 z`
- Raw addresses: `3001 Randolph Road, Unit GF16, Phoenix, AZ` → `Randolph Road, Phoenix, Arizona`
- Normalized addresses: `3001 randolph road unit gf16 phoenix az` → `randolph road phoenix arizona`
- **Name truncation / addition:** The target retains only the short initial-token part of the name.

### Miss 19: `S1-226093814` → `S2-961064678` (S2, US)

- Raw names: `Keystone Telecom Group` → `Keystone Group Partners`
- Normalized names: `keystone telecom group` → `keystone group partners`
- Raw addresses: `5305 Fox Hill Lane, Dallas, TX` → `Fox Hill Lane, DALLAS, TX`
- Normalized addresses: `5305 fox hill lane dallas tx` → `fox hill lane dallas tx`
- **Major name difference:** The target substitutes several distinctive words, leaving only Keystone and Group.

### Miss 20: `S1-950428644` → `S3-325608094` (S3, India)

- Raw names: `KM Friday Private Limited` → `*** KM Finrddoay Private Limited`
- Normalized names: `km friday private limited` → `km finrddoay private limited`
- Raw addresses: `No.16, Krishna Street, Sri Venkateswara Nagar, Anakaputhur, Chennai, Kancheepuram, Tamil Nadu` → `No 16, Chennai, Kancheepuram, தமிழ்நாடு`
- Normalized addresses: `no 16 krishna street sri venkateswara nagar anakaputhur chennai kancheepuram tamil nadu` → `no 16 chennai kancheepuram தமிழ்நாடு`
- **Spelling / text noise:** The target has a misspelled name token and added leading symbols.

### Miss 21: `S1-306998913` → `S3-457810613` (S3, India)

- Raw names: `High Developers` → `High Developers LLP`
- Normalized names: `high developers` → `high developers llp`
- Raw addresses: `No.40/44, Silanthi Kuttai, Kolathur Main Road, Ambattur, Tiruvallur, Tamil Nadu` → `Door No 40/44, Silanthi Kuttai, Kolathur Main Road, Tiruvallur, TN`
- Normalized addresses: `no 40 44 silanthi kuttai kolathur main road ambattur tiruvallur tamil nadu` → `door no 40 44 silanthi kuttai kolathur main road tiruvallur tn`
- **Legal suffix / punctuation variation:** The target adds LLP and the address is partial and reordered.

### Miss 22: `S1-175089979` → `S3-9795886` (S3, US)

- Raw names: `Royal Aerospace Company Partners` → `Royalaerospacepartners.Com`
- Normalized names: `royal aerospace company partners` → `royalaerospacepartners com`
- Raw addresses: `1620 Waterford Pointe Road, Lexington, NC` → `1620 Waterford Pointe Rd, Lexington, North Carolina`
- Normalized addresses: `1620 waterford pointe road lexington nc` → `1620 waterford pointe rd lexington north carolina`
- **Concatenated web / hashtag name:** The target joins words into a domain-style name and omits Company.

### Miss 23: `S1-512145492` → `S3-9144759` (S3, US)

- Raw names: `Cornerstone Diana` → `Cornerstone Dna`
- Normalized names: `cornerstone diana` → `cornerstone dna`
- Raw addresses: `4721 Carters Valley Road, Unit PAD, Church Hill, TN` → `Tennessee, 004721 Carters Valley Rd, Church Hill`
- Normalized addresses: `4721 carters valley road unit pad church hill tn` → `tennessee 004721 carters valley rd church hill`
- **Spelling / text noise:** Diana becomes Dna, and the address number has leading zeros.

### Miss 24: `S1-577958575` → `S3-924928088` (S3, India)

- Raw names: `Aeonian Consultants Private Limited` → `Ny1áirisyn`
- Normalized names: `aeonian consultants private limited` → `ny1áirisyn`
- Raw addresses: `No 401 E Block Bren Avalon Aprt Doddanakundi Extn, Bangalore, Bangalore Rural, Karnataka` → `No 401 E Block Bren Avalon Aprt Doddanakundi Extn, Bangalore, Bangalore Rural, KA`
- Normalized addresses: `no 401 e block bren avalon aprt doddanakundi extn bangalore bangalore rural karnataka` → `no 401 e block bren avalon aprt doddanakundi extn bangalore bangalore rural ka`
- **Opaque name / strong address:** The target name is opaque, possibly corrupted; the address retains the useful evidence.

### Miss 25: `S1-467956589` → `S3-924197768` (S3, India)

- Raw names: `Vpm (India) Pay Private Limited` → `Deltadovafayeio a/k/a Vpm (India) Pay Private Limited`
- Normalized names: `vpm india pay private limited` → `deltadovafayeio a k a vpm india pay private limited`
- Raw addresses: `Delhi, New Delhi, D 1/84 First Floor Janakpuri Delhi, South West Delhi` → `D 1/84 First Floor Janakpuri Delhi, Delhi, New Delhi, South West Delhi`
- Normalized addresses: `delhi new delhi d 1 84 first floor janakpuri delhi south west delhi` → `d 1 84 first floor janakpuri delhi delhi new delhi south west delhi`
- **Alias expansion:** The target adds an alias prefix before the original name; this link has zero-based rank 10 (11th candidate) and misses K=10.

## Interpretation

Character TF-IDF remains a name-based retrieval channel. Cross-script transliteration, substantially different names, and links supported mainly by addresses can remain unseen even at K=20. The next retrieval channel should be selected from the reviewed miss patterns; Stage 5 adds no address TF-IDF, transliteration retrieval, embeddings, pairwise features, or matching model.
