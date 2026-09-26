"""Render the measured blocking run and manually reviewed miss sample."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / "reports" / "blocking_results.json").read_text(encoding="utf-8"))
OUT = ROOT / "reports" / "blocking_baseline.md"

# Primary categories assigned by inspection of the fixed hash sample. Each
# reason also explains why the exact/signature rules and numeric rules miss.
MISS_LABELS = [
    ("spelling difference", "`Private` becomes `Prsiem`; the shared address numbers are only two digits, and the address order differs."),
    ("transliteration", "Latin and Telugu names share no tokens. Numbers overlap, but C2 also requires a shared name token; addresses are not exact."),
    ("abbreviation / legal form", "`Limited` becomes `Ltd` and `&` becomes `+`; only the two-digit floor number is shared."),
    ("abbreviation / legal form", "`Ltd` becomes `Limited`; the address number changes from `292` to `2-92`, leaving no shared eligible number."),
    ("transliteration", "Latin and Tamil names share no tokens; the common address numbers `49` and `26` are below the three-digit rule."),
    ("different name representation", "A personal name is concatenated into a web domain. The five-digit address number matches, but no name token does."),
    ("missing / repeated name tokens", "`Green` is repeated; address number `525` becomes `25`, leaving only the one-digit `6` shared."),
    ("abbreviation / legal form", "`L.L.C.` becomes `Llc`, changing the token multiset; the target address has no number."),
    ("abbreviation / legal form", "`LLC` is omitted and the address number changes from `1505` to `150`."),
    ("abbreviation / legal form", "`LLC` is omitted; the target address also omits the street number."),
    ("abbreviation / legal form", "`PLLC` is omitted; the shared number `56` is below the three-digit threshold."),
    ("missing / repeated name tokens", "`Foundation` is omitted; the only shared address number is `14`, below the numeric threshold."),
    ("missing / repeated name tokens", "`Agro` changes to `Partners` and word order changes; neither address contains an eligible common number."),
    ("missing / repeated name tokens", "The target truncates `Esq. Care`; street numbers `303` and `319` disagree."),
    ("abbreviation / legal form", "The ending changes from `Group` to `Inc`; the target address omits its street number."),
    ("accent difference", "`Elite` and `Élite` remain distinct under the accent-preserving normalizer; the shared number `2` is short."),
    ("missing / repeated name tokens", "`Limited` is omitted, and the target address is empty."),
    ("missing / repeated name tokens", "`Laboratories LLC` is omitted; common address number `42` is below the threshold."),
    ("abbreviation / legal form", "`Limited` becomes `Ltd`; the common address numbers `3` and `16` are short."),
    ("missing / repeated name tokens", "`Cream` is omitted; the target address lacks the S1 street number."),
    ("abbreviation / legal form", "`Ltd` becomes `Limited`, while `401` changes to `40`; only one-digit `3` remains shared."),
    ("transliteration", "Latin and Hindi names share no tokens. The addresses overlap as text but contain no numbers."),
    ("transliteration", "Latin and Hindi names share no tokens. Address components are reordered, with no numeric anchor."),
    ("address-only evidence", "The names are unrelated strings. The addresses overlap, but their common numbers `2` and `9` are short."),
    ("transliteration", "Latin and Telugu names share no tokens; long address numbers match, but C2 requires a shared name token."),
]


def pct(n: int, denominator: int) -> str:
    return f"{100 * n / denominator:.3f}%" if denominator else "—"


def quote(value: str) -> str:
    return value.replace("`", "\\`").replace("\n", " ").replace("\r", "") or "*(empty)*"


def number(value: int | float) -> str:
    return f"{value:,}" if isinstance(value, int) else f"{value:,.2f}"


def main() -> None:
    d = DATA
    total_true = d["true_matches"]
    n = d["validation_s1"]
    rule_order = ["A_exact_name", "B_name_signature", "C1_exact_numeric_address", "C2_numeric_plus_name_token", "union"]
    descriptions = {
        "A_exact_name": "A: exact normalized name",
        "B_name_signature": "B: sorted name token multiset",
        "C1_exact_numeric_address": "C1: exact numeric address",
        "C2_numeric_plus_name_token": "C2: address number + name token",
        "union": "Union A+B+C1+C2",
    }
    possible_pairs = n * sum(d["target_corpus"].values())
    lines = [
        "# Stage 4 blocking baseline",
        "",
        "## Setup and rule definitions",
        "",
        f"Experiment `blocking_baseline_v1` uses the saved validation split: **{n:,} S1 entities** and all **{sum(d['target_corpus'].values()):,}** training S2/S3 target records as the search corpus. It evaluates **{total_true:,}** validation truth links. The target corpus includes records linked to development S1 entities as realistic distractors. All TSVs were read with tab separation; raw files were not changed. Candidate retrieval does not use ground truth, which is consulted only to score hits.",
        "",
        "The indices are built over validation S1 records; S2/S3 are streamed once and looked up against those indices. This reverse index is equivalent to retrieving targets for each S1 and avoids storing the full target corpus or all candidate pairs. Each rule returns an S1 set per target; the union deduplicates overlaps.",
        "",
        "| Rule | Key / eligibility |",
        "| --- | --- |",
        "| A | Exact nonempty `name_norm`; no country restriction. |",
        "| B | Exact sorted multiset of at least two `name_tokens`; preserves repeated tokens and is insensitive to word order. No country restriction. |",
        "| C1 | Exact nonempty `address_norm` plus same country; address must contain a numeric run of at least three digits. |",
        "| C2 | Same country, shared address numeric run of at least three digits, and one of the S1 name's two rarest eligible tokens (length ≥3, validation S1 document frequency ≤5,000). Up to three longest distinct address numbers are used. |",
        "",
        "C1/C2 skip an index key if it maps to more than 64 validation S1 records. In this run, **zero** target lookups triggered that guard. Country labels are handled as arbitrary strings; there are no country-specific normalization rules. No validation truth link crossed countries in these data.",
        "",
        "## Rule results",
        "",
        "Candidate recall = retrieved true links / all validation true links. All-true-S1 percentages include singletons, which are vacuously complete, and are also shown for non-singletons only.",
        "",
        "| Rule | True links found | Candidate recall | All true links found, all S1 | All true links found, matched S1 | Candidate pairs | Avg/S1 | P95 | P99 | Max | Zero-candidate S1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for key in rule_order:
        metric = d["metrics"][key]
        counts = metric["candidate_counts"]
        lines.append(f"| {descriptions[key]} | {metric['true_hits']:,} | {pct(metric['true_hits'], total_true)} | {100*metric['all_true_s1_fraction']:.3f}% | {100*metric['all_true_matched_s1_fraction']:.3f}% | {counts['total_pairs']:,} | {counts['mean']:.2f} | {counts['p95']:,} | {counts['p99']:,} | {counts['max']:,} | {pct(counts['zero_entities'], n)} |")
    lines += [
        "",
        f"The union's median is **{d['metrics']['union']['candidate_counts']['median']}** candidates/S1 and P90 is **{d['metrics']['union']['candidate_counts']['p90']}**. It keeps {d['metrics']['union']['candidate_counts']['total_pairs']:,} of {possible_pairs:,} possible S1×target pairs ({100*d['metrics']['union']['candidate_counts']['total_pairs']/possible_pairs:.6f}% of the Cartesian product).",
        "",
        "### Incremental true links found in rule order",
        "",
        "| Rule added | Newly retrieved true links | True links found by this rule alone among all four |",
        "| --- | ---: | ---: |",
    ]
    for key in rule_order[:-1]:
        lines.append(f"| {descriptions[key]} | {d['incremental_true_hits'].get(key, 0):,} | {d['exclusive_true_hits'].get(key, 0):,} |")
    lines += ["", "The numeric-plus-name rule contributes the most new links. Its country and number constraints keep its average candidate count modest, but the union still misses 39.39% of true links. This baseline does **not** meet the high-recall goal.", "", "## Union recall breakdown", "", "Country uses the S1 country. The source breakdown uses the true target's S2/S3 prefix.", "", "| Target source | Retrieved | True links | Recall |", "| --- | ---: | ---: | ---: |"]
    for source, counts in d["source_recall"].items():
        lines.append(f"| {source} | {counts['hits']:,} | {counts['total']:,} | {pct(counts['hits'], counts['total'])} |")
    lines += ["", "| S1 country | Retrieved | True links | Recall |", "| --- | ---: | ---: | ---: |"]
    for country, counts in d["country_recall"].items():
        lines.append(f"| {country} | {counts['hits']:,} | {counts['total']:,} | {pct(counts['hits'], counts['total'])} |")
    lines += ["", "France does not occur in training/validation, so its candidate recall is unknown.", "", "| True matches per S1 | Retrieved true links | True links | Recall |", "| ---: | ---: | ---: | ---: |"]
    for count, counts in d["match_count_recall"].items():
        lines.append(f"| {count} | {counts['hits']:,} | {counts['total']:,} | {pct(counts['hits'], counts['total'])} |")
    single = d["singleton_candidates"]
    lines += ["", "## True singleton candidate behavior", "", f"Among **{d['validation_singletons']:,}** true singleton S1 entities, the union yields **{single['total_pairs']:,}** candidate pairs: mean **{single['mean']:.2f}**, median **{single['median']}**, P90 **{single['p90']}**, P95 **{single['p95']}**, P99 **{single['p99']}**, max **{single['max']}**. **{single['zero_entities']:,}** ({pct(single['zero_entities'], single['entities'])}) receive zero candidates. Candidates for a singleton are not predictions; a later matcher must reject them.", "", "## Runtime, memory, and candidate tail", "", f"Wall runtime: **{d['wall_seconds']:.1f} seconds** ({d['wall_seconds']/60:.2f} minutes), including {d['s1_load_seconds']:.1f} seconds to load validation S1 and {d['index_and_truth_seconds']:.1f} seconds to build indices and truth lookup. Approximate peak process RSS sampled during the run: **{d['approx_peak_rss_bytes']/(1024**3):.2f} GiB**. Index key counts: " + ", ".join(f"{key} {value:,}" for key, value in d["rule_index_keys"].items()) + ".", "", "The maximum union set is 1,075 candidates for one S1, versus a median of 6 and P99 of 241. Exact-name and name-signature rules individually reach maxima around 1,040, so common names create a visible high tail. The 64-S1 posting guard on numeric/address rules never fired. The union's 10.71 million pairs are feasible to count and stream here, but later pair scoring would need to handle these outliers.", "", "## Miss analysis: fixed hash sample", "", f"The **{len(d['miss_sample'])}** misses below are the smallest SHA-256 hash values of `{d['miss_sample_seed']}:<S1 ID>:<target ID>` among all union-missed validation truth links. The sample is reproducible and was manually categorized. Primary category counts describe this small sample only; categories overlap in reality.", ""]
    category_counts = {}
    for category, _ in MISS_LABELS:
        category_counts[category] = category_counts.get(category, 0) + 1
    lines += ["| Primary category | Sample count |", "| --- | ---: |"]
    for category, count in sorted(category_counts.items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| {category} | {count} |")
    sample = d["miss_sample"]
    lines += ["", f"In the 25 misses, **{sum(bool(item['shared_name_tokens']) for item in sample)}** share at least one exact name token, **{sum(bool(item['shared_address_numbers']) for item in sample)}** share some address number, but only **{sum(any(len(number) >= 3 for number in item['shared_address_numbers']) for item in sample)}** share a number eligible for C2. Transliteration and wholly different names often leave address evidence as the only common signal. One target has an empty address.", ""]
    if len(sample) != len(MISS_LABELS):
        raise ValueError("Miss sample size changed; review the manual labels")
    for number_in_sample, (item, label) in enumerate(zip(sample, MISS_LABELS), start=1):
        category, reason = label
        s1 = item["s1"]
        target = item["target"]
        lines += [
            f"### Miss {number_in_sample}: `{s1['entity_id']}` → `{target['entity_id']}` ({item['source']}, {item['country']})",
            "",
            f"- Raw names: `{quote(s1['business_name'])}` → `{quote(target['business_name'])}`",
            f"- Normalized names: `{quote(s1['name_norm'])}` → `{quote(target['name_norm'])}`",
            f"- Raw addresses: `{quote(s1['business_address'])}` → `{quote(target['business_address'])}`",
            f"- Normalized addresses: `{quote(s1['address_norm'])}` → `{quote(target['address_norm'])}`",
            f"- Primary category: **{category}**. {reason}",
            "",
        ]
    lines += ["## Interpretation and next step", "", "This is a measured first baseline, not a sufficient final candidate generator. The largest gaps are India (especially Latin ↔ Indian-script names), legal suffix or name truncation changes, and addresses whose numbers are short, altered, or absent. The next retrieval experiment should be chosen from these observed gaps and measured on the same validation split; no fuzzy, TF-IDF, embedding, feature, or ML retrieval was implemented in Stage 4.", ""]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
