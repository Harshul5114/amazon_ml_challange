"""Evaluate Stage 4 blocking on the fixed validation S1 entities."""

from __future__ import annotations

import csv
import hashlib
import heapq
import json
import os
import time
from array import array
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from src.blocking import Business, build_rules, retrieve_by_rule
from src.evaluate import parse_match_ids, read_split_ids


ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "splits" / "s1_train_validation.tsv"
SOURCE1 = ROOT / "dataset" / "train" / "train_source1.tsv"
TRUTH = ROOT / "dataset" / "train" / "train_ground_truth.tsv"
TARGETS = [ROOT / "dataset" / "train" / f"train_source{i}.tsv" for i in (2, 3)]
OUTPUT = ROOT / "reports" / "blocking_results.json"
MISS_SAMPLE_SIZE = 25
MISS_SEED = 2026


def pick_percentile(sorted_values: list[int], quantile: float) -> int:
    position = round((len(sorted_values) - 1) * quantile)
    return sorted_values[position]


def candidate_summary(values: array, chosen_indices: list[int] | None = None) -> dict:
    selected = sorted(values if chosen_indices is None else (values[idx] for idx in chosen_indices))
    n = len(selected)
    return {
        "entities": n,
        "total_pairs": sum(selected),
        "mean": sum(selected) / n,
        "median": pick_percentile(selected, 0.5),
        "p90": pick_percentile(selected, 0.9),
        "p95": pick_percentile(selected, 0.95),
        "p99": pick_percentile(selected, 0.99),
        "max": selected[-1],
        "zero_entities": selected.count(0),
        "zero_fraction": selected.count(0) / n,
    }


def main() -> None:
    start = time.perf_counter()
    try:
        import psutil
        process = psutil.Process(os.getpid())
        rss = lambda: process.memory_info().rss
    except ImportError:
        rss = lambda: 0
    peak_rss = rss()

    validation_ids = read_split_ids(SPLIT, "validation")
    s1_records = []
    s1_index = {}
    with SOURCE1.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["entity_id"] in validation_ids:
                index = len(s1_records)
                s1_index[row["entity_id"]] = index
                s1_records.append(Business.from_raw(row))
    if len(s1_records) != len(validation_ids):
        raise ValueError("Validation split has IDs missing from source1")
    del validation_ids
    s1_load_seconds = time.perf_counter() - start
    peak_rss = max(peak_rss, rss())
    print(f"Loaded {len(s1_records):,} validation S1 records in {s1_load_seconds:.1f}s", flush=True)

    rules = build_rules(s1_records)
    names = [rule.name for rule in rules]
    truth_counts = array("I", [0]) * len(s1_records)
    target_owner = {}
    truth_by_source = Counter()
    truth_by_country = Counter()
    with TRUTH.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            index = s1_index.get(row["source1_entity_id"])
            if index is None:
                continue
            matches = parse_match_ids(row["matched_entity_ids"])
            truth_counts[index] = len(matches)
            for target_id in matches:
                if target_id in target_owner:
                    raise ValueError(f"Target ID assigned twice: {target_id}")
                target_owner[target_id] = index
                truth_by_source[target_id.split("-", 1)[0]] += 1
                truth_by_country[s1_records[index].country] += 1
    if len(target_owner) != sum(truth_counts):
        raise ValueError("Ground truth target count mismatch")
    index_seconds = time.perf_counter() - start - s1_load_seconds
    peak_rss = max(peak_rss, rss())
    print(f"Built {len(rules)} indices and loaded {len(target_owner):,} true target IDs in {index_seconds:.1f}s", flush=True)

    pair_counts = {name: array("I", [0]) * len(s1_records) for name in names + ["union"]}
    hit_counts = {name: array("I", [0]) * len(s1_records) for name in names + ["union"]}
    hit_by_source = Counter()
    hit_by_country = Counter()
    hit_by_match_count = Counter()
    exclusive_hits = Counter()
    incremental_hits = Counter()
    true_targets_seen = 0
    cross_country_true_matches = 0
    scanned_targets = Counter()
    sample_heap = []
    for path in TARGETS:
        print(f"Streaming {path.name}", flush=True)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                target = Business.from_raw(row)
                scanned_targets[path.name] += 1
                retrieved = retrieve_by_rule(rules, target)
                union = set()
                for name, indices in retrieved.items():
                    for index in indices:
                        pair_counts[name][index] += 1
                    union.update(indices)
                for index in union:
                    pair_counts["union"][index] += 1
                owner = target_owner.get(target.entity_id)
                if owner is not None:
                    true_targets_seen += 1
                    if target.country != s1_records[owner].country:
                        cross_country_true_matches += 1
                    hits = [name for name in names if owner in retrieved[name]]
                    for name in hits:
                        hit_counts[name][owner] += 1
                    if hits:
                        hit_counts["union"][owner] += 1
                        hit_by_source[target.entity_id.split("-", 1)[0]] += 1
                        hit_by_country[s1_records[owner].country] += 1
                        hit_by_match_count[truth_counts[owner]] += 1
                        incremental_hits[hits[0]] += 1
                        if len(hits) == 1:
                            exclusive_hits[hits[0]] += 1
                    else:
                        sample_key = int.from_bytes(hashlib.sha256(f"{MISS_SEED}:{s1_records[owner].entity_id}:{target.entity_id}".encode("utf-8")).digest()[:8], "big")
                        item = (-sample_key, s1_records[owner], target)
                        if len(sample_heap) < MISS_SAMPLE_SIZE:
                            heapq.heappush(sample_heap, item)
                        elif item[0] > sample_heap[0][0]:
                            heapq.heapreplace(sample_heap, item)
                if scanned_targets[path.name] % 100_000 == 0:
                    peak_rss = max(peak_rss, rss())
        print(f"  scanned {scanned_targets[path.name]:,}", flush=True)
    peak_rss = max(peak_rss, rss())
    if true_targets_seen != len(target_owner):
        raise ValueError(f"Only {true_targets_seen:,} of {len(target_owner):,} validation true targets were found")

    total_truth = len(target_owner)
    singleton_indices = [idx for idx, count in enumerate(truth_counts) if count == 0]
    truth_by_match_count = Counter()
    for count in truth_counts:
        truth_by_match_count[count] += count
    metrics = {}
    for name in names + ["union"]:
        hit_total = sum(hit_counts[name])
        all_true_all = sum(hit_counts[name][idx] == truth_counts[idx] for idx in range(len(s1_records)))
        matched_indices = len(s1_records) - len(singleton_indices)
        all_true_matched = all_true_all - len(singleton_indices)
        metrics[name] = {
            "true_hits": hit_total,
            "true_total": total_truth,
            "true_match_recall": hit_total / total_truth,
            "all_true_s1_count": all_true_all,
            "all_true_s1_fraction": all_true_all / len(s1_records),
            "all_true_matched_s1_fraction": all_true_matched / matched_indices,
            "candidate_counts": candidate_summary(pair_counts[name]),
        }
    missed = []
    for minus_key, s1, target in sorted(sample_heap, reverse=True):
        missed.append({
            "sample_hash": -minus_key,
            "s1": asdict(s1),
            "target": asdict(target),
            "source": target.entity_id.split("-", 1)[0],
            "country": s1.country,
            "shared_name_tokens": sorted(set(s1.name_tokens) & set(target.name_tokens)),
            "shared_address_numbers": sorted(set(s1.address_numeric_tokens) & set(target.address_numeric_tokens)),
        })
    result = {
        "experiment_id": "blocking_baseline_v1",
        "validation_s1": len(s1_records),
        "validation_singletons": len(singleton_indices),
        "target_corpus": dict(scanned_targets),
        "true_matches": total_truth,
        "true_targets_seen": true_targets_seen,
        "cross_country_true_matches": cross_country_true_matches,
        "rule_index_keys": {rule.name: len(rule.index) for rule in rules},
        "rule_skipped_postings": {rule.name: rule.skipped_postings for rule in rules},
        "metrics": metrics,
        "exclusive_true_hits": dict(exclusive_hits),
        "incremental_true_hits": dict(incremental_hits),
        "source_recall": {source: {"hits": hit_by_source[source], "total": truth_by_source[source]} for source in sorted(truth_by_source)},
        "country_recall": {country: {"hits": hit_by_country[country], "total": truth_by_country[country]} for country in sorted(truth_by_country)},
        "match_count_recall": {str(count): {"hits": hit_by_match_count[count], "total": truth_by_match_count[count]} for count in sorted(truth_by_match_count) if count},
        "singleton_candidates": candidate_summary(pair_counts["union"], singleton_indices),
        "wall_seconds": time.perf_counter() - start,
        "s1_load_seconds": s1_load_seconds,
        "index_and_truth_seconds": index_seconds,
        "approx_peak_rss_bytes": peak_rss,
        "miss_sample_seed": MISS_SEED,
        "miss_sample": missed,
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"union recall={metrics['union']['true_match_recall']:.4%} pairs={metrics['union']['candidate_counts']['total_pairs']:,} runtime={result['wall_seconds']:.1f}s peak_rss={peak_rss / (1024**3):.2f} GiB", flush=True)
    print(f"Wrote {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
