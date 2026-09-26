"""Render Stage 5 retrieval metrics and reviewed miss examples."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STAGE4 = json.loads((ROOT / "reports/blocking_results.json").read_text(encoding="utf-8"))
TFIDF = json.loads((ROOT / "artifacts/tfidf_run_metadata.json").read_text(encoding="utf-8"))
RESULTS = json.loads((ROOT / "reports/blocking_tfidf_results.json").read_text(encoding="utf-8"))
LABEL_FILE = ROOT / "reports/tfidf_miss_labels.json"
LABELS = json.loads(LABEL_FILE.read_text(encoding="utf-8")) if LABEL_FILE.exists() else {}
OUTPUT = ROOT / "reports/blocking_tfidf.md"
SELECTED_K = 10


def percent(hits: int, total: int) -> str:
    return f"{100 * hits / total:.3f}%" if total else "—"


def inline(text: str) -> str:
    return text.replace("`", "\\`").replace("\n", " ").replace("\r", "") or "*(empty)*"


def main() -> None:
    n = RESULTS["validation_s1"]
    true_total = RESULTS["true_links"]
    baseline = STAGE4["metrics"]["union"]
    chosen = RESULTS["k_results"][str(SELECTED_K)]
    lines = [
        "# Stage 5: character TF-IDF name retrieval",
        "",
        "## Configuration and scalability",
        "",
        f"The query set is the saved validation split of **{n:,} S1 entities**. TF-IDF is fit on their `name_norm` strings, then all training S2/S3 names are streamed as targets. It uses `char_wb` character **4–5 grams**, sublinear term frequency, L2 normalization, `min_df=2`, and `max_df=0.02`. A 4–5 gram range is compact enough to avoid the very common three-character fragments while retaining partial names, order-insensitive word evidence, and small spelling variations. The fitted vocabulary has **{TFIDF['vocabulary_size']:,}** grams and **{TFIDF['query_nonzeros']:,}** nonzeros in the S1 query matrix.",
        "",
        f"Country is a dynamic retrieval partition: same nonempty country is compared, missing-country S1 queries search every target partition, and missing-country targets search every S1 partition. No country list is hard-coded; France behavior is covered by a unit test. Sparse cosine products use `sparse-dot-topn`, batches of {TFIDF['batch_size']:,} targets, a score floor of {TFIDF['score_floor']}, and at most **20 results per S1 per target source**. Global top lists are merged across batches. No dense S1×target matrix is formed.",
        "",
        f"Dependencies live in the project `.venv` (`requirements-stage5.txt`): scikit-learn {TFIDF['sklearn_version']} and sparse-dot-topn {TFIDF['sparse_dot_topn_version']}. The top-20 arrays are saved under `artifacts/` so K=5, 10, and 20 use the same retrieval run. Stage 4 pairs were reconstructed exactly and compared by packed S1/source/target IDs; the replay matched its published **{RESULTS['baseline_pairs']:,}** pairs and **{RESULTS['baseline_true_hits']:,}** true hits.",
        "",
        "## Recall and candidate-volume tradeoff",
        "",
        "Candidate recall means the fraction of validation truth links present in the candidate set. Country recall uses S1 country. Each K is a maximum **per target source**, so TF-IDF alone can return up to 2K targets per S1. Union counts deduplicate exact pair overlap with Stage 4.",
        "",
        "| Retrieval | True-link recall | S2 | S3 | India | US | Candidate pairs | Mean/S1 | P95 | P99 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    base_country = STAGE4["country_recall"]
    base_source = STAGE4["source_recall"]
    base_counts = baseline["candidate_counts"]
    lines.append(f"| Stage 4 baseline | {percent(baseline['true_hits'], true_total)} | {percent(base_source['S2']['hits'], base_source['S2']['total'])} | {percent(base_source['S3']['hits'], base_source['S3']['total'])} | {percent(base_country['India']['hits'], base_country['India']['total'])} | {percent(base_country['US']['hits'], base_country['US']['total'])} | {base_counts['total_pairs']:,} | {base_counts['mean']:.2f} | {base_counts['p95']:,} | {base_counts['p99']:,} |")
    for k, k_result in RESULTS["k_results"].items():
        for kind in ("tfidf", "union"):
            metric = k_result[kind]
            source = metric["source_recall"]
            country = metric["country_recall"]
            counts = metric["candidate_counts"]
            label = f"TF-IDF K={k}" if kind == "tfidf" else f"Stage 4 + TF-IDF K={k}"
            lines.append(f"| {label} | {percent(metric['true_hits'], true_total)} | {percent(source['S2']['hits'], source['S2']['total'])} | {percent(source['S3']['hits'], source['S3']['total'])} | {percent(country['India']['hits'], country['India']['total'])} | {percent(country['US']['hits'], country['US']['total'])} | {counts['total_pairs']:,} | {counts['mean']:.2f} | {counts['p95']:,} | {counts['p99']:,} |")
    lines += ["", "| K per source | New true links beyond Stage 4 | Recall gain over Stage 4 | TF-IDF pairs already in Stage 4 | Union median | Union P90 | Union max |", "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for k, k_result in RESULTS["k_results"].items():
        metric = k_result["union"]["candidate_counts"]
        incremental = k_result["incremental_true_hits_over_stage4"]
        lines.append(f"| {k} | {incremental:,} | +{100*incremental/true_total:.3f} pp | {k_result['candidate_overlap_with_stage4']:,} | {metric['median']} | {metric['p90']} | {metric['max']:,} |")
    lines += ["", f"**Selected configuration: K={SELECTED_K} per source.** The table above shows the gain and candidate cost of moving to K=20; this selection favors a smaller candidate set when the extra recall is limited. Stage 4 alone reaches every truth link for {100*baseline['all_true_matched_s1_fraction']:.3f}% of non-singleton S1 entities; the selected union reaches {100*chosen['union']['all_true_matched_s1_fraction']:.3f}%.", "", "France has no validation ground truth, so France recall cannot be measured here.", "", "## Runtime and memory", "", f"TF-IDF fitting and retrieval took **{TFIDF['total_seconds']:.1f} seconds** ({TFIDF['total_seconds']/60:.2f} minutes) with approximate sampled peak RSS **{TFIDF['approx_peak_rss_bytes']/(1024**3):.2f} GiB**. Within this run, vectorization accounted for {TFIDF['vectorization_seconds']:.1f} seconds and sparse products for {TFIDF['sparse_product_seconds']:.1f} seconds. The exact Stage 4 pair replay took **{RESULTS['stage4_recompute_seconds']:.1f} seconds** and ran concurrently with TF-IDF's S3 pass. The union evaluation process peaked at **{RESULTS['approx_peak_rss_bytes']/(1024**3):.2f} GiB**; its wall duration includes waiting for TF-IDF to finish, so it should not be added to the retrieval duration.", "", "## Remaining misses: fixed hash sample", "", f"For K={SELECTED_K}, the sample takes the 25 smallest SHA-256 hashes of `2026:<S1 ID>:<target ID>` among true links still missed by the union. These manually reviewed categories describe the sample, not population rates.", ""]
    sample = RESULTS["miss_samples"][str(SELECTED_K)]
    categories = Counter(LABELS.get(item["target_id"], {}).get("category", "unreviewed") for item in sample)
    lines += ["| Primary category | Sample count |", "| --- | ---: |"]
    for category, count in categories.most_common():
        lines.append(f"| {category} | {count} |")
    lines.append("")
    for number, item in enumerate(sample, 1):
        s1 = item["s1"]
        target = item["target"]
        label = LABELS.get(item["target_id"], {})
        category = label.get("category", "unreviewed")
        reason = label.get("reason", "")
        lines += [f"### Miss {number}: `{s1['entity_id']}` → `{target['entity_id']}` ({item['source']}, {item['country']})", "", f"- Raw names: `{inline(s1['business_name'])}` → `{inline(target['business_name'])}`", f"- Normalized names: `{inline(s1['name_norm'])}` → `{inline(target['name_norm'])}`", f"- Raw addresses: `{inline(s1['business_address'])}` → `{inline(target['business_address'])}`", f"- Normalized addresses: `{inline(s1['address_norm'])}` → `{inline(target['address_norm'])}`", f"- **{category}:** {reason}", ""]
    lines += ["## Interpretation", "", "Character TF-IDF remains a name-based retrieval channel. Cross-script transliteration, substantially different names, and links supported mainly by addresses can remain unseen even at K=20. The next retrieval channel should be selected from the reviewed miss patterns; Stage 5 adds no address TF-IDF, transliteration retrieval, embeddings, pairwise features, or matching model.", ""]
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
