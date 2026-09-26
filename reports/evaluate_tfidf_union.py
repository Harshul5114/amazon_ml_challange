"""Evaluate TF-IDF top-K and exact unions with the Stage 4 candidate pairs."""

from __future__ import annotations

import csv
import hashlib
import json
import threading
import time
from array import array
from collections import Counter
from pathlib import Path

import numpy as np
import psutil

from src.blocking import Business, build_rules, retrieve_by_rule
from src.evaluate import parse_match_ids, read_split_ids


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
SOURCE1 = ROOT / "dataset/train/train_source1.tsv"
GROUND_TRUTH = ROOT / "dataset/train/train_ground_truth.tsv"
SPLIT = ROOT / "splits/s1_train_validation.tsv"
TARGETS = [ROOT / "dataset/train/train_source2.tsv", ROOT / "dataset/train/train_source3.tsv"]
STAGE4 = json.loads((ROOT / "reports/blocking_results.json").read_text(encoding="utf-8"))
K_VALUES = (5, 10, 20)
TOP_K = 20
MISS_SAMPLE_SIZE = 25
MISS_SEED = 2026


def pack_pair(s1_index: int, source_code: int, target_number: int) -> int:
    return (s1_index << 34) | (source_code << 32) | target_number


def target_id_from_key(key: int) -> str:
    source = "S2" if (key >> 32) & 3 == 0 else "S3"
    return f"{source}-{key & 0xFFFFFFFF}"


def summary(counts: np.ndarray) -> dict:
    sorted_counts = np.sort(counts)
    n = len(counts)
    at = lambda p: int(sorted_counts[round((n - 1) * p)])
    return {
        "total_pairs": int(np.sum(counts, dtype=np.int64)),
        "mean": float(np.mean(counts)),
        "median": at(0.5),
        "p90": at(0.9),
        "p95": at(0.95),
        "p99": at(0.99),
        "max": int(sorted_counts[-1]),
        "zero_entities": int(np.count_nonzero(counts == 0)),
    }


def exact_membership(sorted_keys: np.ndarray, query_keys: np.ndarray) -> np.ndarray:
    positions = np.searchsorted(sorted_keys, query_keys)
    in_bounds = positions < len(sorted_keys)
    output = np.zeros(len(query_keys), dtype=bool)
    output[in_bounds] = sorted_keys[positions[in_bounds]] == query_keys[in_bounds]
    return output


def main() -> None:
    start = time.perf_counter()
    process = psutil.Process()
    stopping = threading.Event()
    peak = [process.memory_info().rss]

    def monitor() -> None:
        while not stopping.wait(0.5):
            peak[0] = max(peak[0], process.memory_info().rss)

    monitor_thread = threading.Thread(target=monitor, daemon=True)
    monitor_thread.start()
    try:
        valid_ids = read_split_ids(SPLIT)
        s1_records = []
        s1_index = {}
        with SOURCE1.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row["entity_id"] in valid_ids:
                    s1_index[row["entity_id"]] = len(s1_records)
                    s1_records.append(Business.from_raw(row))
        if len(s1_records) != len(valid_ids):
            raise ValueError("Validation split and source1 mismatch")
        del valid_ids
        n = len(s1_records)
        print(f"Loaded {n:,} validation S1 records", flush=True)

        rules = build_rules(s1_records)
        baseline_pairs = array("Q")
        baseline_start = time.perf_counter()
        for source_code, path in enumerate(TARGETS):
            print(f"Regenerating Stage 4 pairs from {path.name}", flush=True)
            processed = 0
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    target = Business.from_raw(row)
                    number = int(target.entity_id.split("-", 1)[1])
                    found_by_rule = retrieve_by_rule(rules, target)
                    union = set().union(*found_by_rule.values())
                    baseline_pairs.extend(pack_pair(index, source_code, number) for index in union)
                    processed += 1
                    if processed % 1_000_000 == 0:
                        print(f"  {processed:,} targets; {len(baseline_pairs):,} pairs", flush=True)
            print(f"  finished {processed:,} targets", flush=True)
        baseline_seconds = time.perf_counter() - baseline_start
        baseline_keys = np.frombuffer(baseline_pairs, dtype=np.uint64)
        expected_pairs = STAGE4["metrics"]["union"]["candidate_counts"]["total_pairs"]
        if len(baseline_keys) != expected_pairs:
            raise ValueError(f"Stage 4 pair count mismatch: {len(baseline_keys)} != {expected_pairs}")
        np.save(ARTIFACTS / "stage4_pairs_packed.npy", baseline_keys)
        print(f"Regenerated {len(baseline_keys):,} Stage 4 pairs in {baseline_seconds:.1f}s", flush=True)

        metadata_path = ARTIFACTS / "tfidf_run_metadata.json"
        while not metadata_path.exists():
            print("Waiting for TF-IDF top-K retrieval to finish", flush=True)
            time.sleep(20)
        top = np.load(ARTIFACTS / "tfidf_top20.npz")
        if not np.array_equal(top["s1_ids"], np.asarray([record.entity_id for record in s1_records])):
            raise ValueError("TF-IDF top-K S1 order does not match the saved split")
        print("Loaded completed TF-IDF top-K arrays", flush=True)

        truth_keys_list = array("Q")
        with GROUND_TRUTH.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                index = s1_index.get(row["source1_entity_id"])
                if index is None:
                    continue
                for target_id in parse_match_ids(row["matched_entity_ids"]):
                    source_text, numeric_text = target_id.split("-", 1)
                    source_code = 0 if source_text == "S2" else 1 if source_text == "S3" else None
                    if source_code is None:
                        raise ValueError(f"Unexpected truth target: {target_id}")
                    truth_keys_list.append(pack_pair(index, source_code, int(numeric_text)))
        truth_keys = np.frombuffer(truth_keys_list, dtype=np.uint64)
        if len(truth_keys) != STAGE4["true_matches"]:
            raise ValueError("Truth link count differs from Stage 4")
        truth_s1 = (truth_keys >> 34).astype(np.int32)
        truth_source = ((truth_keys >> 32) & 3).astype(np.int8)
        country_by_s1 = np.asarray([record.country for record in s1_records])

        base_sorted = np.sort(baseline_keys)
        base_truth_hit = exact_membership(base_sorted, truth_keys)
        if np.count_nonzero(base_truth_hit) != STAGE4["metrics"]["union"]["true_hits"]:
            raise ValueError("Baseline truth hits differ from Stage 4")
        baseline_counts = np.bincount((baseline_keys >> 34).astype(np.int32), minlength=n)

        tfidf_key_parts = []
        tfidf_rank_parts = []
        tfidf_count_by_k = {k: np.zeros(n, dtype=np.int32) for k in K_VALUES}
        for source_code, source in enumerate(("S2", "S3")):
            scores = top[f"{source}_scores"]
            ids = top[f"{source}_ids"]
            if scores.shape != (n, TOP_K) or ids.shape != (n, TOP_K):
                raise ValueError(f"Unexpected top-K matrix shape for {source}")
            for k in K_VALUES:
                tfidf_count_by_k[k] += np.count_nonzero(scores[:, :k] >= 0, axis=1).astype(np.int32)
            rows, ranks = np.nonzero(scores >= 0)
            keys = (rows.astype(np.uint64) << np.uint64(34)) | (np.uint64(source_code) << np.uint64(32)) | ids[rows, ranks].astype(np.uint64)
            tfidf_key_parts.append(keys)
            tfidf_rank_parts.append(ranks.astype(np.uint8))
        tfidf_keys = np.concatenate(tfidf_key_parts)
        tfidf_ranks = np.concatenate(tfidf_rank_parts)
        order = np.argsort(tfidf_keys)
        tfidf_sorted = tfidf_keys[order]
        rank_sorted = tfidf_ranks[order]
        positions = np.searchsorted(tfidf_sorted, truth_keys)
        truth_rank = np.full(len(truth_keys), TOP_K, dtype=np.uint8)
        valid_positions = positions < len(tfidf_sorted)
        matched_positions = valid_positions & (tfidf_sorted[np.minimum(positions, len(tfidf_sorted)-1)] == truth_keys)
        truth_rank[matched_positions] = rank_sorted[positions[matched_positions]]

        overlaps = exact_membership(base_sorted, tfidf_keys)
        overlap_rows = (tfidf_keys[overlaps] >> 34).astype(np.int32)
        overlap_ranks = tfidf_ranks[overlaps].astype(np.int32)
        overlap_by_rank = np.bincount(overlap_rows * TOP_K + overlap_ranks, minlength=n * TOP_K).reshape(n, TOP_K)
        overlap_cumulative = np.cumsum(overlap_by_rank, axis=1)
        print(f"TF-IDF pairs at K=20: {len(tfidf_keys):,}; Stage 4 overlap: {int(np.count_nonzero(overlaps)):,}", flush=True)

        results = {
            "validation_s1": n,
            "true_links": len(truth_keys),
            "baseline_pairs": len(baseline_keys),
            "baseline_true_hits": int(np.count_nonzero(base_truth_hit)),
            "tfidf_pairs_at_20": len(tfidf_keys),
            "tfidf_stage4_overlap_at_20": int(np.count_nonzero(overlaps)),
            "k_results": {},
        }
        miss_ids_to_find = set()
        missed_by_k = {}
        for k in K_VALUES:
            tfidf_hit = truth_rank < k
            union_hit = base_truth_hit | tfidf_hit
            tfidf_counts = tfidf_count_by_k[k]
            union_counts = baseline_counts + tfidf_counts - overlap_cumulative[:, k-1]
            if np.any(union_counts < baseline_counts) or np.any(union_counts < tfidf_counts):
                raise ValueError("Candidate union counts are inconsistent")
            one = {}
            for label, hit, counts in (("tfidf", tfidf_hit, tfidf_counts), ("union", union_hit, union_counts)):
                source_recall = {}
                for source_code, source in enumerate(("S2", "S3")):
                    source_mask = truth_source == source_code
                    source_recall[source] = {"hits": int(np.count_nonzero(hit & source_mask)), "total": int(np.count_nonzero(source_mask))}
                country_recall = {}
                for country in sorted(set(country_by_s1)):
                    country_mask = country_by_s1[truth_s1] == country
                    country_recall[country] = {"hits": int(np.count_nonzero(hit & country_mask)), "total": int(np.count_nonzero(country_mask))}
                hits_per_s1 = np.bincount(truth_s1[hit], minlength=n)
                true_per_s1 = np.bincount(truth_s1, minlength=n)
                one[label] = {
                    "true_hits": int(np.count_nonzero(hit)),
                    "true_recall": float(np.mean(hit)),
                    "candidate_counts": summary(counts),
                    "source_recall": source_recall,
                    "country_recall": country_recall,
                    "all_true_s1_fraction": float(np.mean(hits_per_s1 == true_per_s1)),
                    "all_true_matched_s1_fraction": float(np.mean(hits_per_s1[true_per_s1 > 0] == true_per_s1[true_per_s1 > 0])),
                }
            one["incremental_true_hits_over_stage4"] = int(np.count_nonzero(tfidf_hit & ~base_truth_hit))
            one["candidate_overlap_with_stage4"] = int(np.sum(overlap_cumulative[:, k-1]))
            results["k_results"][str(k)] = one
            missed_indices = np.flatnonzero(~union_hit)
            ranked = sorted(missed_indices, key=lambda idx: hashlib.sha256(f"{MISS_SEED}:{s1_records[truth_s1[idx]].entity_id}:{target_id_from_key(int(truth_keys[idx]))}".encode()).digest()[:8])[:MISS_SAMPLE_SIZE]
            sampled = []
            for idx in ranked:
                target_id = target_id_from_key(int(truth_keys[idx]))
                miss_ids_to_find.add(target_id)
                sampled.append({"s1_index": int(truth_s1[idx]), "target_id": target_id,
                                "source": "S2" if truth_source[idx] == 0 else "S3"})
            missed_by_k[str(k)] = sampled
            print(f"K={k}: tfidf={one['tfidf']['true_recall']:.3%} union={one['union']['true_recall']:.3%} union_pairs={one['union']['candidate_counts']['total_pairs']:,}", flush=True)

        target_rows = {}
        for path in TARGETS:
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    if row["entity_id"] in miss_ids_to_find:
                        target_rows[row["entity_id"]] = Business.from_raw(row)
        if len(target_rows) != len(miss_ids_to_find):
            raise ValueError("Could not find every sampled missed target")
        for k, sample in missed_by_k.items():
            for item in sample:
                s1 = s1_records[item.pop("s1_index")]
                target = target_rows[item["target_id"]]
                item["country"] = s1.country
                item["s1"] = {
                    "entity_id": s1.entity_id, "business_name": s1.business_name,
                    "business_address": s1.business_address, "name_norm": s1.name_norm,
                    "address_norm": s1.address_norm,
                }
                item["target"] = {
                    "entity_id": target.entity_id, "business_name": target.business_name,
                    "business_address": target.business_address, "name_norm": target.name_norm,
                    "address_norm": target.address_norm,
                }
            results.setdefault("miss_samples", {})[k] = sample
        results["stage4_recompute_seconds"] = baseline_seconds
        results["union_evaluation_seconds"] = time.perf_counter() - start
        results["approx_peak_rss_bytes"] = peak[0]
        (ROOT / "reports/blocking_tfidf_results.json").write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote results; evaluation runtime={results['union_evaluation_seconds']:.1f}s peak_RSS={peak[0]/(1024**3):.2f} GiB", flush=True)
    finally:
        stopping.set()
        monitor_thread.join(timeout=2)


if __name__ == "__main__":
    main()
