"""Render the Stage 6A subset experiment without starting any retrieval."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "reports/address_tfidf_subset_results.json"
OUTPUT = ROOT / "reports/address_tfidf_subset.md"


def pct(hits: int, total: int) -> str:
    return f"{100 * hits / total:.3f}%" if total else "—"


def main() -> None:
    data = json.loads(RESULT.read_text(encoding="utf-8"))
    metrics = data["metrics"]
    sample_size = data["sample_size"]
    scale = data["validation_size"] / sample_size
    target_count = sum(data["target_counts"].values())
    test_targets = 4_887_273 + 5_082_316
    test_queries = 1_732_544
    stage4_seconds = json.loads((ROOT / "reports/blocking_results.json").read_text(encoding="utf-8"))["wall_seconds"]
    stage5_seconds = json.loads((ROOT / "artifacts/tfidf_run_metadata.json").read_text(encoding="utf-8"))["total_seconds"]
    timing = data["tfidf"]
    scan_other = timing["scan_seconds"] - timing["vectorization_seconds"] - timing["sparse_product_seconds"]
    validation_projection = (
        timing["fit_seconds"] * scale + timing["vectorization_seconds"] +
        timing["sparse_product_seconds"] * scale + scan_other
    )
    test_scale = test_queries / sample_size
    target_scale = test_targets / target_count
    test_projection = (
        timing["fit_seconds"] * test_scale +
        (timing["vectorization_seconds"] + scan_other) * target_scale +
        timing["sparse_product_seconds"] * test_scale * target_scale
    )
    full_validation_sequential = stage4_seconds + stage5_seconds + validation_projection
    test_sequential = (stage4_seconds + stage5_seconds) * test_scale * target_scale / scale + test_projection
    lines = [
        "# Stage 6A: address TF-IDF on a validation subset",
        "",
        f"This experiment uses **{sample_size:,} S1 entities** and **{metrics['stage5']['true_links']:,} true links** drawn only from the saved {data['validation_size']:,}-entity validation split. The subset is saved in `{data['sample_file']}`. It uses seed **{data['seed']}** and proportionally allocates each S1 country × exact true-match-count stratum using largest remainders, then samples without replacement within each stratum. This is an experimental subset, not a new train/validation split.",
        "",
        "Stage 4 candidates come from the saved packed pair artifact. Stage 5 name candidates come from the saved top-20 results at its selected K=10 per source. Neither was recomputed. All rows below use exactly these same sampled S1 entities and their ground-truth links.",
        "",
        "## Sample balance",
        "",
        "| Distribution | Category | Full validation | 50k sample | Difference |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for distribution in ("country", "singleton", "match_count"):
        for category, item in data["balance"][distribution].items():
            difference = 100 * (item["sample_fraction"] - item["validation_fraction"])
            lines.append(f"| {distribution} | {category} | {pct(item['validation'], data['validation_size'])} ({item['validation']:,}) | {pct(item['sample'], sample_size)} ({item['sample']:,}) | {difference:+.3f} pp |")
    lines += [
        "",
        "## Candidate recall and volume",
        "",
        "| Retrieval | True-link recall | India | US | S2 | S3 | Candidate pairs | Mean/S1 | P95 | P99 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, label in (
        ("stage4", "Stage 4 baseline"),
        ("stage5", "Stage 5 current best"),
        ("stage5_address_k5", "Stage 5 + address K=5"),
        ("stage5_address_k10", "Stage 5 + address K=10"),
    ):
        item = metrics[name]
        countries = item["country_recall"]
        sources = item["source_recall"]
        counts = item["candidate_counts"]
        lines.append(f"| {label} | {pct(item['true_hits'], item['true_links'])} | {pct(**countries['India'])} | {pct(**countries['US'])} | {pct(**sources['S2'])} | {pct(**sources['S3'])} | {counts['total_pairs']:,} | {counts['mean']:.2f} | {counts['p95']:,} | {counts['p99']:,} |")
    lines += [
        "",
        "## Marginal value of address retrieval",
        "",
        "Incremental links and candidate pairs are measured against the **same Stage 5 subset candidate set**, with exact duplicate pairs removed. These figures are candidate recall and candidate purity, not final matching precision.",
        "",
        "| Address K per source | New true links | Added candidates | True links per 1,000 added candidates | Added candidates per new true link | Recall gain |",
        "| ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for k in (5, 10):
        item = data["incremental_over_stage5"][str(k)]
        recall_gain = 100 * item["additional_true_links"] / metrics["stage5"]["true_links"]
        lines.append(f"| {k} | {item['additional_true_links']:,} | {item['additional_candidates']:,} | {item['true_links_per_1000_additional_candidates']:.3f} | {item['additional_candidates_per_true_link']:.1f} | +{recall_gain:.3f} pp |")
    k5 = data["incremental_over_stage5"]["5"]
    k10 = data["incremental_over_stage5"]["10"]
    extra_hits = k10["additional_true_links"] - k5["additional_true_links"]
    extra_pairs = k10["additional_candidates"] - k5["additional_candidates"]
    lines += [
        "",
        f"Moving from address K=5 to K=10 adds **{extra_hits:,}** more true links and **{extra_pairs:,}** more candidates, or **{1000*extra_hits/extra_pairs:.3f} true links per 1,000 additional candidates**.",
        "",
        "**Subset choice: address K=5 per source.** It captures most of the observed address recall gain with much better marginal yield than K=10. This is an experimental choice only; no full run was started.",
        "",
        "## Configuration, runtime, and memory",
        "",
        f"The address channel uses normalized address text, `char_wb` 4–5 grams, `min_df=2`, `max_df=0.02`, sublinear term frequency, L2-normalized cosine similarity, a 0.05 score floor, 50,000-target sparse batches, and top 10 per source. Its vocabulary has **{timing['vocabulary_size']:,}** grams and its subset query matrix has **{timing['query_nonzeros']:,}** nonzeros. K=5 is a prefix of the same retrieval result; no second target scan was run. Country partitions and missing-country behavior follow Stage 5.",
        "",
        f"The subset address fit took **{timing['fit_seconds']:.1f} s**; the S2/S3 scan took **{timing['scan_seconds']:.1f} s**; total address fitting and retrieval took **{timing['fit_seconds']+timing['scan_seconds']:.1f} s** ({(timing['fit_seconds']+timing['scan_seconds'])/60:.2f} min). Of the scan, target vectorization used {timing['vectorization_seconds']:.1f} s and sparse products used {timing['sparse_product_seconds']:.1f} s. End-to-end sampling, retrieval, and evaluation took **{data['total_seconds']:.1f} s**. Approximate sampled peak process RSS was **{data['approx_peak_rss_bytes']/1024**3:.2f} GiB**.",
        "",
        f"For the selected K=5 configuration, a rough **incremental address retrieval** projection is **{validation_projection/3600:.2f} h** for all {data['validation_size']:,} validation queries and the same {target_count:,} training targets, or **{test_projection/3600:.2f} h** for {test_queries:,} test queries and {test_targets:,} test targets. These use the measured K=10 scan as a conservative proxy for K=5, hold target parsing/vectorization approximately fixed per target, and scale sparse products and query fitting with query count. If Stage 4 and Stage 5 also had to be rerun sequentially, a coarse total is **{full_validation_sequential/3600:.2f} h** for validation and **{test_sequential/3600:.2f} h** for test; the test total extrapolates their measured validation runtimes by query and target counts. Reusing the saved validation Stage 4/5 candidates requires only the incremental address work. Vocabulary size, country mix (including unseen France), memory pressure, and sparse-product density can change runtime materially. These are order-of-magnitude estimates, not measured full runs; candidate assembly time is excluded.",
        "",
        "The experiment stops at this subset. No full-validation or test address retrieval was run, and no matching model was trained.",
        "",
    ]
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
