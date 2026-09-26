"""Stage 6A: stratified validation subset and address TF-IDF retrieval only."""

from __future__ import annotations

import csv
import json
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import psutil

from reports.evaluate_tfidf_union import exact_membership, pack_pair, summary
from src.blocking import Business
from src.evaluate import parse_match_ids
from src.tfidf_retrieval import TfidfNameRetriever


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
S1_PATH = ROOT / "dataset/train/train_source1.tsv"
TRUTH_PATH = ROOT / "dataset/train/train_ground_truth.tsv"
TARGETS = [ROOT / f"dataset/train/train_source{i}.tsv" for i in (2, 3)]
SAMPLE_PATH = ROOT / "splits/stage6a_validation_subset.tsv"
RESULT_PATH = ROOT / "reports/address_tfidf_subset_results.json"
TOP_PATH = ARTIFACTS / "address_tfidf_subset_top10.npz"
SAMPLE_SIZE = 50_000
SEED = 2061
K_VALUES = (5, 10)


def choose_stratified(labels: list[tuple[str, int]], sample_size: int, seed: int) -> np.ndarray:
    """Proportionally allocate exact country × match-count strata, then sample IDs."""
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index, label in enumerate(labels):
        groups[label].append(index)
    total = len(labels)
    if not 0 < sample_size <= total:
        raise ValueError("Invalid sample size")
    quotas = {label: sample_size * len(indices) // total for label, indices in groups.items()}
    remaining = sample_size - sum(quotas.values())
    fractional_order = sorted(
        groups,
        key=lambda label: (-(sample_size * len(groups[label]) % total), label),
    )
    for label in fractional_order[:remaining]:
        quotas[label] += 1
    rng = np.random.default_rng(seed)
    selected = []
    for label in sorted(groups):
        selected.extend(rng.choice(groups[label], size=quotas[label], replace=False).tolist())
    return np.asarray(sorted(selected), dtype=np.int32)


def balance_table(countries: list[str], match_counts: np.ndarray, selected: np.ndarray) -> dict:
    result = {}
    for label, values in (
        ("country", countries),
        ("match_count", match_counts.tolist()),
        ("singleton", (match_counts == 0).tolist()),
    ):
        full = Counter(values)
        subset = Counter(values[int(index)] for index in selected)
        result[label] = {
            str(key): {
                "validation": int(full[key]),
                "sample": int(subset[key]),
                "validation_fraction": full[key] / len(values),
                "sample_fraction": subset[key] / len(selected),
            }
            for key in sorted(full)
        }
    return result


def metrics(keys: np.ndarray, truth_keys: np.ndarray, truth_source: np.ndarray,
            truth_country: np.ndarray, sample_global: np.ndarray) -> dict:
    if len(keys) != len(np.unique(keys)):
        raise ValueError("Duplicate candidate keys")
    hit = exact_membership(np.sort(keys), truth_keys)
    global_to_local = np.full(int(sample_global[-1]) + 1, -1, dtype=np.int32)
    global_to_local[sample_global] = np.arange(len(sample_global), dtype=np.int32)
    candidate_local = global_to_local[(keys >> np.uint64(34)).astype(np.int32)]
    if np.any(candidate_local < 0):
        raise ValueError("Candidate points outside subset")
    counts = np.bincount(candidate_local, minlength=len(sample_global))
    source = {}
    country = {}
    for code, name in enumerate(("S2", "S3")):
        mask = truth_source == code
        source[name] = {"hits": int(np.count_nonzero(hit & mask)), "total": int(np.count_nonzero(mask))}
    for name in sorted(set(truth_country)):
        mask = truth_country == name
        country[name] = {"hits": int(np.count_nonzero(hit & mask)), "total": int(np.count_nonzero(mask))}
    return {
        "true_hits": int(np.count_nonzero(hit)),
        "true_links": len(truth_keys),
        "true_recall": float(np.mean(hit)),
        "source_recall": source,
        "country_recall": country,
        "candidate_counts": summary(counts),
    }


def ranked_keys(top: np.lib.npyio.NpzFile, global_indices: np.ndarray,
                k: int, kind: str) -> np.ndarray:
    pieces = []
    for source_code, source in enumerate(("S2", "S3")):
        scores = top[f"{source}_scores"]
        ids = top[f"{source}_ids"]
        if kind == "stage5":
            scores = scores[global_indices, :k]
            ids = ids[global_indices, :k]
        else:
            scores = scores[:, :k]
            ids = ids[:, :k]
        local, ranks = np.nonzero(scores >= 0)
        global_rows = global_indices[local].astype(np.uint64)
        pieces.append((global_rows << np.uint64(34)) |
                      (np.uint64(source_code) << np.uint64(32)) |
                      ids[local, ranks].astype(np.uint64))
    return np.concatenate(pieces)


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
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
        stage5 = np.load(ARTIFACTS / "tfidf_top20.npz")
        all_ids = stage5["s1_ids"]
        id_to_global = {value: index for index, value in enumerate(all_ids)}
        countries = [""] * len(all_ids)
        with S1_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                index = id_to_global.get(row["entity_id"])
                if index is not None:
                    countries[index] = row["country"] or ""
        match_counts = np.full(len(all_ids), -1, dtype=np.int16)
        with TRUTH_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                index = id_to_global.get(row["source1_entity_id"])
                if index is not None:
                    match_counts[index] = len(parse_match_ids(row["matched_entity_ids"]))
        if np.any(match_counts < 0) or not all(countries):
            raise ValueError("Incomplete validation metadata")
        labels = list(zip(countries, match_counts.tolist()))
        selected = choose_stratified(labels, SAMPLE_SIZE, SEED)
        sample_ids = set(all_ids[selected])
        with SAMPLE_PATH.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
            writer.writerow(["source1_entity_id", "country", "true_match_count"])
            for index in selected:
                writer.writerow([all_ids[index], countries[index], int(match_counts[index])])
        balance = balance_table(countries, match_counts, selected)
        print(f"Sampled {len(selected):,}/{len(all_ids):,} validation S1; singleton={balance['singleton']['True']['sample_fraction']:.3%}", flush=True)

        sample_records = {}
        with S1_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row["entity_id"] in sample_ids:
                    sample_records[row["entity_id"]] = Business.from_raw(row)
        records = [sample_records[value] for value in all_ids[selected]]
        del sample_records, sample_ids, id_to_global
        print(f"Loaded subset in {time.perf_counter()-start:.1f}s", flush=True)

        fit_start = time.perf_counter()
        retriever = TfidfNameRetriever(
            records, text_field="address", top_k=10, batch_size=50_000,
            n_threads=4, min_df=2, max_df=0.02, score_floor=0.05,
        )
        fit_seconds = time.perf_counter() - fit_start
        scan_start = time.perf_counter()
        for target_path in TARGETS:
            retriever.scan_source(target_path)
        scan_seconds = time.perf_counter() - scan_start
        np.savez_compressed(
            TOP_PATH, global_indices=selected,
            S2_scores=retriever.source_scores["S2"], S2_ids=retriever.source_ids["S2"],
            S3_scores=retriever.source_scores["S3"], S3_ids=retriever.source_ids["S3"],
        )
        print(f"Address retrieval completed in {fit_seconds+scan_seconds:.1f}s; peak RSS={peak[0]/1024**3:.2f} GiB", flush=True)

        baseline_all = np.load(ARTIFACTS / "stage4_pairs_packed.npy", mmap_mode="r")
        selected_mask = np.zeros(len(all_ids), dtype=bool)
        selected_mask[selected] = True
        baseline_keys = np.asarray(baseline_all[selected_mask[(baseline_all >> np.uint64(34)).astype(np.int32)]], dtype=np.uint64)
        stage5_keys = ranked_keys(stage5, selected, 10, "stage5")
        stage5_union = np.union1d(baseline_keys, stage5_keys)
        address_top = np.load(TOP_PATH)
        address_keys = {k: ranked_keys(address_top, selected, k, "address") for k in K_VALUES}

        truth_list = []
        selected_lookup = {str(all_ids[index]): int(index) for index in selected}
        with TRUTH_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                global_index = selected_lookup.get(row["source1_entity_id"])
                if global_index is None:
                    continue
                for match in parse_match_ids(row["matched_entity_ids"]):
                    source_text, numeric = match.split("-", 1)
                    source_code = {"S2": 0, "S3": 1}.get(source_text)
                    if source_code is None:
                        raise ValueError(f"Unexpected truth target: {match}")
                    truth_list.append(pack_pair(global_index, source_code, int(numeric)))
        truth_keys = np.asarray(truth_list, dtype=np.uint64)
        if len(truth_keys) != int(match_counts[selected].sum()):
            raise ValueError("Truth link count mismatch")
        truth_source = ((truth_keys >> np.uint64(32)) & np.uint64(3)).astype(np.int8)
        truth_country = np.asarray(countries)[(truth_keys >> np.uint64(34)).astype(np.int32)]
        methods = {"stage4": baseline_keys, "stage5": stage5_union}
        for k in K_VALUES:
            methods[f"stage5_address_k{k}"] = np.union1d(stage5_union, address_keys[k])
        measured = {name: metrics(keys, truth_keys, truth_source, truth_country, selected)
                    for name, keys in methods.items()}
        stage5_hit = exact_membership(stage5_union, truth_keys)
        incremental = {}
        for k in K_VALUES:
            name = f"stage5_address_k{k}"
            added_candidates = len(methods[name]) - len(stage5_union)
            added_hits = measured[name]["true_hits"] - measured["stage5"]["true_hits"]
            direct_added_hits = int(np.count_nonzero(exact_membership(np.sort(address_keys[k]), truth_keys) & ~stage5_hit))
            if added_hits != direct_added_hits:
                raise ValueError("Incremental truth hit mismatch")
            incremental[str(k)] = {
                "additional_candidates": added_candidates,
                "additional_true_links": added_hits,
                "true_links_per_1000_additional_candidates": 1000 * added_hits / added_candidates,
                "additional_candidates_per_true_link": added_candidates / added_hits if added_hits else None,
            }
        result = {
            "seed": SEED, "sample_size": len(selected), "validation_size": len(all_ids),
            "sample_file": str(SAMPLE_PATH.relative_to(ROOT)),
            "strata": "S1 country x exact number of true matches; largest-remainder proportional allocation and seeded random sampling",
            "balance": balance, "target_counts": dict(retriever.source_processed),
            "tfidf": {"field": "address_norm", "analyzer": "char_wb", "ngram_range": [4, 5],
                      "min_df": 2, "max_df": 0.02, "score_floor": 0.05,
                      "batch_size": 50_000, "vocabulary_size": len(retriever.vectorizer.vocabulary_),
                      "query_nonzeros": int(retriever.query_matrix.nnz), "fit_seconds": fit_seconds,
                      "scan_seconds": scan_seconds, "vectorization_seconds": retriever.vectorization_seconds,
                      "sparse_product_seconds": retriever.product_seconds},
            "metrics": measured, "incremental_over_stage5": incremental,
            "total_seconds": time.perf_counter() - start,
            "approx_peak_rss_bytes": peak[0],
        }
        RESULT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {RESULT_PATH}; total={result['total_seconds']:.1f}s peak_RSS={peak[0]/1024**3:.2f} GiB", flush=True)
        for name, value in measured.items():
            print(f"{name}: recall={value['true_recall']:.3%} pairs={value['candidate_counts']['total_pairs']:,}", flush=True)
    finally:
        stop.set()
        thread.join(timeout=2)


if __name__ == "__main__":
    main()
