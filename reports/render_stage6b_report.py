"""Render reviewed Stage 6B miss analysis and its labeled sample TSV."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "reports/stage6b_miss_summary.json"
LABELS = ROOT / "reports/stage6b_sample_labels.json"
SAMPLE = ROOT / "reports/stage6b_miss_sample.tsv"
LABELED_SAMPLE = ROOT / "reports/stage6b_miss_sample_labeled.tsv"
OUTPUT = ROOT / "reports/stage6b_miss_analysis.md"


def percent(count: int, denominator: int) -> str:
    return f"{100 * count / denominator:.3f}%" if denominator else "—"


def inline(value: str, max_length: int = 130) -> str:
    clean = value.replace("`", "\\`").replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    return clean[:max_length] + ("…" if len(clean) > max_length else "") if clean else "*(empty)*"


def main() -> None:
    data = json.loads(SUMMARY.read_text(encoding="utf-8"))
    labels = json.loads(LABELS.read_text(encoding="utf-8"))
    category_by_number = {}
    for category, numbers in labels["categories"].items():
        for number in numbers:
            if number in category_by_number:
                raise ValueError(f"Duplicate sample label for {number}")
            category_by_number[number] = category
    if set(category_by_number) != set(range(1, data["sample_size"] + 1)):
        raise ValueError("Manual labels do not cover exactly the fixed sample")
    with SAMPLE.open("r", encoding="utf-8-sig", newline="") as source, LABELED_SAMPLE.open("w", encoding="utf-8", newline="") as destination:
        reader = csv.DictReader(source, delimiter="\t")
        writer = csv.DictWriter(destination, fieldnames=["primary_category"] + reader.fieldnames,
                                delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for row in reader:
            number = int(row["sample_number"])
            writer.writerow({"primary_category": category_by_number[number], **row})

    n = data["missed_links"]
    distributions = data["distributions"]
    lines = [
        "# Stage 6B: misses after Stage 5 + address TF-IDF K=5",
        "",
        f"This analyzes the exact saved Stage 6A subset of **{data['subset_s1']:,} S1 entities** and **{data['true_links']:,} true links**. The Stage 4 packed pairs, Stage 5 name top-20 arrays, and Stage 6A address top-10 arrays are reused. No retrieval is run here. The reconstruction exactly matches both Stage 6A K=5 and K=10 truth-hit totals.",
        "",
        f"**Remaining missed links: {n:,} / {data['true_links']:,} ({percent(n, data['true_links'])}).** These are candidate-generation misses, not model false negatives. The analysis script is `reports/analyze_stage6b.py`; machine-readable counts and the fixed 150-link sample are in `reports/stage6b_miss_summary.json` and `reports/stage6b_miss_sample_labeled.tsv`.",
        "",
        "## Where misses occur",
        "",
        "| Country | True links | Missed links | Miss rate | Share of all misses |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for country, total in data["country_truth_links"].items():
        missed = distributions["country"].get(country, 0)
        lines.append(f"| {country} | {total:,} | {missed:,} | {percent(missed, total)} | {percent(missed, n)} |")
    lines += ["", "| Source | True links | Missed links | Miss rate | Share of all misses |", "| --- | ---: | ---: | ---: | ---: |"]
    for source, total in data["source_truth_links"].items():
        missed = distributions["source"].get(source, 0)
        lines.append(f"| {source} | {total:,} | {missed:,} | {percent(missed, total)} | {percent(missed, n)} |")

    lines += [
        "",
        "## Field and text evidence across all 11,596 misses",
        "",
        "A field is present when its conservative normalized value is nonempty. Script is the dominant Unicode script of alphabetic **name** characters, requiring at least 80% agreement within each name. Mixed or empty cases are uncertain. Token overlap is set Jaccard on normalized whitespace tokens; it includes common and legal words, so a high value alone need not identify a business.",
        "",
        "| Dimension | Category | Misses | Share of misses |",
        "| --- | --- | ---: | ---: |",
    ]
    display = [
        ("Name presence", "name_presence"),
        ("Address presence", "address_presence"),
        ("Name scripts", "name_script_relation"),
        ("Name token Jaccard", "name_token_overlap"),
        ("Address token Jaccard", "address_token_overlap"),
        ("Any shared address number", "shared_numeric_address"),
        ("Shared address number ≥3 digits", "shared_substantial_numeric_address"),
    ]
    for title, key in display:
        for category, count in sorted(distributions[key].items(), key=lambda pair: -pair[1]):
            lines.append(f"| {title} | {category} | {count:,} | {percent(count, n)} |")
    lines += [
        "",
        "Any shared number counts even one-digit fragments and can be weak evidence. The ≥3-digit row is a stricter diagnostic. Address Jaccard is not comparable when a target address is empty.",
        "",
        "## What cached deeper lists recover",
        "",
        f"The already-computed address K=10 list contains **{data['k10_recovered']:,}** of the {n:,} K=5 misses (**{percent(data['k10_recovered'], n)}**). It would reduce the miss count to **{n-data['k10_recovered']:,}** and subset miss rate to **{percent(n-data['k10_recovered'], data['true_links'])}**. Stage 6A measured its marginal cost at **490,332** further candidate pairs, or only **3.863** true links per 1,000 additional candidates.",
        "",
        f"The cached name top-20 list contains **{data['name_top20_present']:,}** K=5 misses (**{percent(data['name_top20_present'], n)}**), including **{data['name_top20_by_script'].get('same', 0):,}** same-script and **{data['name_top20_by_script'].get('different', 0):,}** detectably different-script pairs. This is a diagnostic membership check, not a new retrieval run; its candidate-volume tradeoff was not reevaluated for this subset.",
        "",
        "| Name-script relation | K=5 misses | Recovered by address K=10 | Recovery within group |",
        "| --- | ---: | ---: | ---: |",
    ]
    for category, count in distributions["name_script_relation"].items():
        recovered = data["k10_recovered_by_script"].get(category, 0)
        lines.append(f"| {category} | {count:,} | {recovered:,} | {percent(recovered, count)} |")

    category_counts = Counter(category_by_number.values())
    lines += [
        "",
        "## Reproducible 150-link review",
        "",
        f"The sample is the 150 smallest SHA-256 hashes of `{data['sample_seed']}:<S1 ID>:<target ID>` among all K=5 missed links. Each pair was assigned one manually reviewed primary category using its raw name, address, and diagnostics. Categories are exclusive for counting, although a link can have several failure modes. Percentages are approximate sample frequencies, not exact population rates.",
        "",
        "| Primary failure category | Sample links | Sample share |",
        "| --- | ---: | ---: |",
    ]
    for category, count in category_counts.most_common():
        lines.append(f"| {category} | {count:,} | {percent(count, data['sample_size'])} |")
    lines.append("| other | 0 | 0.000% |")
    lines += [
        "",
        "The opaque-name category includes possible encoding or generated-text corruption, but the raw text alone cannot distinguish corruption from a genuine alternate name. Address formatting and legal suffix variation also appear as secondary issues in many links assigned to larger categories.",
        "",
        "### Representative reviewed links",
        "",
        "The labeled TSV contains all 150 sampled links with complete raw names and addresses, source, country, overlap diagnostics, and K=10 recovery status. Examples below show why a single primary category was assigned.",
        "",
        "| Category | S1 → target | Raw names | Raw addresses |",
        "| --- | --- | --- | --- |",
    ]
    examples = [1, 8, 7, 24, 14, 35, 86, 80]
    for number in examples:
        item = data["sample"][number-1]
        category = category_by_number[number]
        lines.append(
            f"| {category} | `{item['s1_id']}` → `{item['target_id']}` | "
            f"`{inline(item['s1_name'], 75)}` → `{inline(item['target_name'], 75)}` | "
            f"`{inline(item['s1_address'], 85)}` → `{inline(item['target_address'], 85)}` |"
        )
    lines += [
        "",
        "## Next retrieval mechanism suggested by this evidence",
        "",
        f"**Investigate script-aware transliteration of business names followed by conservative name retrieval on the same-country partition.** Detectably different name scripts account for **{distributions['name_script_relation'].get('different', 0):,}** misses (**{percent(distributions['name_script_relation'].get('different', 0), n)}**); **{data['different_script_by_country'].get('India', 0):,}** are in India and **{data['different_script_by_country'].get('US', 0):,}** in the US. Cached name top-20 finds none of these, while address K=10 finds only **{data['k10_recovered_by_script'].get('different', 0):,}**. This gives a targeted pool that neither current name retrieval nor a modest address-K increase handles well. It should remain an added representation beside raw and conservative normalized text, and be measured for incremental recall and candidate volume before any full run.",
        "",
        "A second promising investigation is a precise address or locality anchor for pairs with same-script names but altered or missing address text. Shared numeric fragments are common, but only 22.4% of misses share a number of at least three digits, so an unrestricted numeric block would likely generate many false candidates. The current evidence favors targeted transliteration first; it does not establish downstream match precision or F0.5.",
        "",
        "No new retrieval channel, full-validation run, test run, production pairwise features, or matching model was implemented in Stage 6B.",
        "",
    ]
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT} and {LABELED_SAMPLE}")


if __name__ == "__main__":
    main()
