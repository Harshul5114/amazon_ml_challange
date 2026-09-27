"""Stage 7A targeted offline transliteration retrieval on the fixed 50k subset."""

from __future__ import annotations

import csv
import json
import threading
import time
from collections import Counter
from pathlib import Path

import numpy as np
import psutil

from reports.analyze_stage6b import script_of
from reports.evaluate_tfidf_union import exact_membership, pack_pair
from reports.run_address_tfidf_subset import metrics, ranked_keys
from src.blocking import Business
from src.evaluate import parse_match_ids
from src.transliterate import contains_non_latin_letter
from src.transliteration_retrieval import TransliteratedNameRetriever


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
SAMPLE = ROOT / "splits/stage6a_validation_subset.tsv"
S1_FILE = ROOT / "dataset/train/train_source1.tsv"
TRUTH_FILE = ROOT / "dataset/train/train_ground_truth.tsv"
TARGETS = [ROOT / f"dataset/train/train_source{i}.tsv" for i in (2, 3)]
STAGE6A = ROOT / "reports/address_tfidf_subset_results.json"
STAGE6B = ROOT / "reports/stage6b_miss_summary.json"
LABELS = ROOT / "reports/stage6b_sample_labels.json"
TOP_FILE = ARTIFACTS / "translit_top10_subset.npz"
RESULT_FILE = ROOT / "reports/transliteration_subset_results.json"


def main() -> None:
    start = time.perf_counter()
    process = psutil.Process()
    stop = threading.Event()
    peak = [process.memory_info().rss]

    def monitor() -> None:
        while not stop.wait(0.5):
            peak[0] = max(peak[0], process.memory_info().rss)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        prior = json.loads(STAGE6A.read_text(encoding="utf-8"))
        miss_analysis = json.loads(STAGE6B.read_text(encoding="utf-8"))
        name_top = np.load(ARTIFACTS / "tfidf_top20.npz")
        address_top = np.load(ARTIFACTS / "address_tfidf_subset_top10.npz")
        all_ids = name_top["s1_ids"]
        selected = address_top["global_indices"]
        sample_ids = []
        country_by_id = {}
        with SAMPLE.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                sample_ids.append(row["source1_entity_id"])
                country_by_id[row["source1_entity_id"]] = row["country"]
        if sample_ids != all_ids[selected].tolist() or len(sample_ids) != 50_000:
            raise ValueError("Fixed subset does not match cached Stage 6A arrays")
        needed_ids = set(sample_ids)
        s1_by_id = {}
        with S1_FILE.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row["entity_id"] in needed_ids:
                    s1_by_id[row["entity_id"]] = Business.from_raw(row)
        records = [s1_by_id[s1] for s1 in sample_ids]
        non_latin_s1 = sum(contains_non_latin_letter(record.name_norm) for record in records)
        if non_latin_s1:
            raise ValueError("This targeted experiment expects only Latin-script S1 names")
        global_by_s1 = {s1: int(index) for s1, index in zip(sample_ids, selected)}
        truth_keys = []
        target_truth_ids = set()
        with TRUTH_FILE.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                global_index = global_by_s1.get(row["source1_entity_id"])
                if global_index is None:
                    continue
                for target_id in parse_match_ids(row["matched_entity_ids"]):
                    source, number = target_id.split("-", 1)
                    truth_keys.append(pack_pair(global_index, {"S2": 0, "S3": 1}[source], int(number)))
                    target_truth_ids.add(target_id)
        truth_keys = np.asarray(truth_keys, dtype=np.uint64)
        if len(truth_keys) != prior["metrics"]["stage5_address_k5"]["true_links"]:
            raise ValueError("Truth link total differs from Stage 6A")
        print(f"Loaded {len(records):,} Latin S1 names and {len(truth_keys):,} truth links", flush=True)

        fit_start = time.perf_counter()
        retriever = TransliteratedNameRetriever(
            records, truth_target_ids=target_truth_ids, top_k=10, batch_size=50_000,
            n_threads=4, min_df=2, max_df=0.02, score_floor=0.05,
        )
        fit_seconds = time.perf_counter() - fit_start
        scan_start = time.perf_counter()
        for target_path in TARGETS:
            retriever.scan_transliterated_source(target_path)
        scan_seconds = time.perf_counter() - scan_start
        for source in ("S2", "S3"):
            retriever._source_arrays(source)
        np.savez_compressed(
            TOP_FILE, global_indices=selected,
            S2_scores=retriever.source_scores["S2"], S2_ids=retriever.source_ids["S2"],
            S3_scores=retriever.source_scores["S3"], S3_ids=retriever.source_ids["S3"],
        )
        print(f"Saved transliteration top-10 in {fit_seconds+scan_seconds:.1f}s; peak RSS {peak[0]/1024**3:.2f} GiB", flush=True)

        selected_mask = np.zeros(len(all_ids), dtype=bool)
        selected_mask[selected] = True
        baseline_all = np.load(ARTIFACTS / "stage4_pairs_packed.npy", mmap_mode="r")
        stage4_keys = baseline_all[selected_mask[(baseline_all >> np.uint64(34)).astype(np.int32)]]
        stage5_keys = np.union1d(stage4_keys, ranked_keys(name_top, selected, 10, "stage5"))
        baseline = np.union1d(stage5_keys, ranked_keys(address_top, selected, 5, "address"))
        if len(baseline) != prior["metrics"]["stage5_address_k5"]["candidate_counts"]["total_pairs"]:
            raise ValueError("Baseline pair count differs from Stage 6A")
        baseline_hit = exact_membership(baseline, truth_keys)
        if int(np.count_nonzero(baseline_hit)) != prior["metrics"]["stage5_address_k5"]["true_hits"]:
            raise ValueError("Baseline truth hits differ from Stage 6A")

        truth_sources = ((truth_keys >> np.uint64(32)) & np.uint64(3)).astype(np.int8)
        id_by_global = {int(index): s1 for s1, index in zip(sample_ids, selected)}
        truth_country = np.asarray([country_by_id[id_by_global[int(key >> np.uint64(34))]] for key in truth_keys])
        target_ids = [f"S{int(source)+2}-{int(key & np.uint64(0xFFFFFFFF))}" for key, source in zip(truth_keys, truth_sources)]
        eligible = np.asarray([target_id in retriever.truth_target_names for target_id in target_ids], dtype=bool)
        cross_script = np.asarray([
            script_of(retriever.truth_target_names[target_id]) not in {"LATIN", "mixed", "unknown"}
            if target_id in retriever.truth_target_names else False
            for target_id in target_ids
        ], dtype=bool)
        cross_script_missed_before = int(np.count_nonzero(cross_script & ~baseline_hit))
        if cross_script_missed_before != miss_analysis["distributions"]["name_script_relation"]["different"]:
            raise ValueError(f"Cross-script miss count differs from Stage 6B: {cross_script_missed_before}")

        top = np.load(TOP_FILE)
        measured = {"current_best": metrics(baseline, truth_keys, truth_sources, truth_country, selected)}
        incremental = {}
        for k in (5, 10):
            retrieved = ranked_keys(top, selected, k, "address")
            union = np.union1d(baseline, retrieved)
            alone_hit = exact_membership(np.sort(retrieved), truth_keys)
            union_hit = baseline_hit | alone_hit
            if int(np.count_nonzero(union_hit)) != int(np.count_nonzero(exact_membership(union, truth_keys))):
                raise ValueError("Candidate hit union mismatch")
            label = f"translit_k{k}"
            measured[label] = metrics(retrieved, truth_keys, truth_sources, truth_country, selected)
            measured[f"union_k{k}"] = metrics(union, truth_keys, truth_sources, truth_country, selected)
            new_hits = int(np.count_nonzero(alone_hit & ~baseline_hit))
            added_pairs = len(union) - len(baseline)
            if new_hits != measured[f"union_k{k}"]["true_hits"] - measured["current_best"]["true_hits"]:
                raise ValueError("Incremental hit mismatch")
            incremental[str(k)] = {
                "new_true_links": new_hits,
                "additional_candidate_pairs": added_pairs,
                "true_links_per_1000_added_candidates": 1000 * new_hits / added_pairs if added_pairs else 0.0,
                "cross_script_misses_recovered": int(np.count_nonzero(alone_hit & ~baseline_hit & cross_script)),
                "same_or_uncertain_script_misses_recovered": int(np.count_nonzero(alone_hit & ~baseline_hit & ~cross_script)),
                "eligible_truth_retrieved_alone": int(np.count_nonzero(alone_hit & eligible)),
                "eligible_truth_total": int(np.count_nonzero(eligible)),
                "cross_script_truth_retrieved_alone": int(np.count_nonzero(alone_hit & cross_script)),
                "cross_script_truth_total": int(np.count_nonzero(cross_script)),
            }
            print(f"K={k}: union recall {measured[f'union_k{k}']['true_recall']:.3%}; +{new_hits:,} links, +{added_pairs:,} pairs", flush=True)

        label_data = json.loads(LABELS.read_text(encoding="utf-8"))["categories"]
        category_by_number = {number: category for category, numbers in label_data.items() for number in numbers}
        if len(category_by_number) != miss_analysis["sample_size"]:
            raise ValueError("Stage 6B sample labels incomplete")
        recovered_sample = {}
        remaining_sample = {}
        for k in (5, 10):
            retrieved = np.sort(ranked_keys(top, selected, k, "address"))
            recovered = Counter()
            remaining = Counter()
            for number, item in enumerate(miss_analysis["sample"], 1):
                global_index = global_by_s1[item["s1_id"]]
                source, number_text = item["target_id"].split("-", 1)
                key = np.asarray([pack_pair(global_index, {"S2": 0, "S3": 1}[source], int(number_text))], dtype=np.uint64)
                category = category_by_number[number]
                if exact_membership(retrieved, key)[0]:
                    recovered[category] += 1
                else:
                    remaining[category] += 1
            recovered_sample[str(k)] = dict(recovered)
            remaining_sample[str(k)] = dict(remaining)
        result = {
            "subset_s1": len(records), "true_links": len(truth_keys),
            "s1_non_latin_names": non_latin_s1,
            "target_scanned": dict(retriever.source_scanned),
            "target_transliterated": dict(retriever.source_selected),
            "tfidf": {"analyzer": "char_wb", "ngram_range": [4, 5], "min_df": 2, "max_df": 0.02,
                      "score_floor": 0.05, "batch_size": 50_000, "top_k": 10,
                      "vocabulary_size": len(retriever.vectorizer.vocabulary_),
                      "query_nonzeros": int(retriever.query_matrix.nnz)},
            "cross_script_missed_before": cross_script_missed_before,
            "eligible_truth_links": int(np.count_nonzero(eligible)),
            "cross_script_truth_links": int(np.count_nonzero(cross_script)),
            "metrics": measured, "incremental": incremental,
            "collision_audit": retriever.collision_summary(),
            "stage6b_sample_recovered": recovered_sample,
            "stage6b_sample_remaining": remaining_sample,
            "timing": {"fit_seconds": fit_seconds, "scan_seconds": scan_seconds,
                       "vectorization_seconds": retriever.vectorization_seconds,
                       "sparse_product_seconds": retriever.product_seconds,
                       "total_seconds": time.perf_counter() - start,
                       "approx_peak_rss_bytes": peak[0]},
            "dependency": {"package": "anyascii", "version": "0.3.3", "license": "ISC",
                           "license_url": "https://github.com/anyascii/anyascii/blob/master/LICENSE"},
        }
        RESULT_FILE.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {RESULT_FILE}; total {result['timing']['total_seconds']:.1f}s, peak {peak[0]/1024**3:.2f} GiB", flush=True)
    finally:
        stop.set()
        thread.join(timeout=2)


if __name__ == "__main__":
    main()
