"""Stage 8: Full validation candidate generation (Address TF-IDF + Name TF-IDF + Baseline)."""

from __future__ import annotations

import argparse
import csv
import json
import threading
import time
from array import array
from collections import defaultdict
from pathlib import Path

import numpy as np
import psutil

from reports.evaluate_tfidf_union import exact_membership, pack_pair, summary
from src.blocking import Business, build_rules, retrieve_by_rule
from src.evaluate import parse_match_ids, read_split_ids
from src.tfidf_retrieval import TfidfNameRetriever


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
SPLIT = ROOT / "splits" / "s1_train_validation.tsv"
SOURCE1 = ROOT / "dataset" / "train" / "train_source1.tsv"
TARGETS = [ROOT / "dataset" / "train" / f"train_source{i}.tsv" for i in (2, 3)]
TRUTH_PATH = ROOT / "dataset" / "train" / "train_ground_truth.tsv"

RESULT_PATH = ROOT / "reports" / "full_validation_retrieval_results.json"
BASELINE_PATH = ARTIFACTS / "stage4_pairs_packed.npy"
NAME_TOP_PATH = ARTIFACTS / "tfidf_top20.npz"
ADDRESS_TOP_PATH = ARTIFACTS / "address_tfidf_top10.npz"
UNION_PATH = ARTIFACTS / "full_validation_candidates_packed.npy"
CANDIDATE_TSV_PATH = ARTIFACTS / "validation_candidate_pairs.tsv"


def ranked_keys(top: np.lib.npyio.NpzFile, k: int) -> np.ndarray:
    pieces = []
    for source_code, source in enumerate(("S2", "S3")):
        scores = top[f"{source}_scores"][:, :k]
        ids = top[f"{source}_ids"][:, :k]
        local, ranks = np.nonzero(scores >= 0)
        global_rows = local.astype(np.uint64)
        pieces.append((global_rows << np.uint64(34)) |
                      (np.uint64(source_code) << np.uint64(32)) |
                      ids[local, ranks].astype(np.uint64))
    return np.concatenate(pieces)


def generate_stage4_baseline(s1_records: list[Business]) -> np.ndarray:
    print("Generating Stage 4 baseline pairs...", flush=True)
    start = time.perf_counter()
    rules = build_rules(s1_records)
    baseline_pairs = array("Q")
    for source_code, path in enumerate(TARGETS):
        print(f"Scanning {path.name} for baseline rules...", flush=True)
        processed = 0
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = handle.readline().rstrip("\r\n").split("\t")
            id_idx = header.index("entity_id")
            name_idx = header.index("business_name")
            addr_idx = header.index("business_address")
            country_idx = header.index("country")
            min_cols = max(id_idx, name_idx, addr_idx, country_idx) + 1
            for line in handle:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < min_cols:
                    continue
                target_id = parts[id_idx]
                target = Business.from_raw({
                    "entity_id": target_id,
                    "business_name": parts[name_idx],
                    "business_address": parts[addr_idx],
                    "country": parts[country_idx],
                })
                numeric = int(target_id.split("-", 1)[1])
                found_by_rule = retrieve_by_rule(rules, target)
                union = set().union(*found_by_rule.values())
                baseline_pairs.extend(pack_pair(index, source_code, numeric) for index in union)
                processed += 1
                if processed % 1_000_000 == 0:
                    print(f"  {path.name}: {processed:,} targets processed; pairs={len(baseline_pairs):,}", flush=True)
        print(f"Finished {path.name}: {processed:,} targets in {time.perf_counter()-start:.1f}s", flush=True)
    baseline_keys = np.frombuffer(baseline_pairs, dtype=np.uint64)
    np.save(BASELINE_PATH, baseline_keys)
    print(f"Saved {len(baseline_keys):,} baseline pairs to {BASELINE_PATH.name} in {time.perf_counter()-start:.1f}s", flush=True)
    return baseline_keys


def generate_name_tfidf(s1_records: list[Business], all_ids: np.ndarray) -> None:
    print("Generating Name TF-IDF top 10...", flush=True)
    start = time.perf_counter()
    retriever = TfidfNameRetriever(
        s1_records, text_field="name", top_k=10, batch_size=100_000,
        n_threads=12, min_df=2, max_df=0.02, score_floor=0.05
    )
    print(f"Fitted {len(retriever.vectorizer.vocabulary_):,} name char grams in {time.perf_counter()-start:.1f}s", flush=True)
    for path in TARGETS:
        t0 = time.perf_counter()
        retriever.scan_source(path)
        print(f"Finished Name TF-IDF for {path.name} in {time.perf_counter()-t0:.1f}s", flush=True)
    np.savez_compressed(
        NAME_TOP_PATH,
        s1_ids=all_ids,
        S2_scores=retriever.source_scores["S2"],
        S2_ids=retriever.source_ids["S2"],
        S3_scores=retriever.source_scores["S3"],
        S3_ids=retriever.source_ids["S3"],
    )
    print(f"Saved Name TF-IDF to {NAME_TOP_PATH.name} in {time.perf_counter()-start:.1f}s", flush=True)


def generate_address_tfidf(s1_records: list[Business], all_ids: np.ndarray) -> None:
    print("Generating Address TF-IDF top 10...", flush=True)
    start = time.perf_counter()
    retriever = TfidfNameRetriever(
        s1_records, text_field="address", top_k=10, batch_size=100_000,
        n_threads=12, min_df=2, max_df=0.02, score_floor=0.05
    )
    print(f"Fitted {len(retriever.vectorizer.vocabulary_):,} address char grams in {time.perf_counter()-start:.1f}s", flush=True)
    for path in TARGETS:
        t0 = time.perf_counter()
        retriever.scan_source(path)
        print(f"Finished Address TF-IDF for {path.name} in {time.perf_counter()-t0:.1f}s", flush=True)
    np.savez_compressed(
        ADDRESS_TOP_PATH,
        s1_ids=all_ids,
        S2_scores=retriever.source_scores["S2"],
        S2_ids=retriever.source_ids["S2"],
        S3_scores=retriever.source_scores["S3"],
        S3_ids=retriever.source_ids["S3"],
    )
    print(f"Saved Address TF-IDF to {ADDRESS_TOP_PATH.name} in {time.perf_counter()-start:.1f}s", flush=True)


def export_candidate_tsv(s1_records: list[Business], union_keys: np.ndarray, out_path: Path) -> None:
    print("Exporting candidate pairs TSV...", flush=True)
    t0 = time.perf_counter()
    s1_indices = (union_keys >> np.uint64(34)).astype(np.int32)
    sources = ((union_keys >> np.uint64(32)) & np.uint64(3)).astype(np.int8)
    numerics = (union_keys & np.uint64(0xFFFFFFFF)).astype(np.uint32)

    total_keys = len(union_keys)
    k_idx = 0
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_idx, biz in enumerate(s1_records):
            cands = []
            while k_idx < total_keys and s1_indices[k_idx] == s1_idx:
                src = "S2" if sources[k_idx] == 0 else "S3"
                cands.append(f"{src}-{numerics[k_idx]}")
                k_idx += 1
            handle.write(f"{biz.entity_id}\t{','.join(cands)}\n")
    print(f"Exported candidate TSV to {out_path.name} in {time.perf_counter()-t0:.1f}s", flush=True)


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    start = time.perf_counter()
    process = psutil.Process()
    stopping = threading.Event()
    peak = [process.memory_info().rss]

    def monitor() -> None:
        while not stopping.wait(0.5):
            peak[0] = max(peak[0], process.memory_info().rss)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        validation = read_split_ids(SPLIT)
        s1 = []
        all_ids = []
        id_to_global = {}
        with SOURCE1.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row["entity_id"] in validation:
                    all_ids.append(row["entity_id"])
                    id_to_global[row["entity_id"]] = len(s1)
                    s1.append(Business.from_raw(row))
        if len(s1) != len(validation):
            raise ValueError("Validation S1 coverage mismatch")
        all_ids = np.array(all_ids)
        print(f"Loaded {len(s1):,} S1 records in {time.perf_counter()-start:.1f}s", flush=True)

        # 1. Baseline Stage 4
        if not BASELINE_PATH.exists():
            generate_stage4_baseline(s1)
        baseline_keys = np.load(BASELINE_PATH, mmap_mode="r")

        # 2. Name TF-IDF
        if not NAME_TOP_PATH.exists():
            generate_name_tfidf(s1, all_ids)
        stage5_top = np.load(NAME_TOP_PATH)
        stage5_keys = ranked_keys(stage5_top, k=10)

        # 3. Address TF-IDF
        if not ADDRESS_TOP_PATH.exists():
            generate_address_tfidf(s1, all_ids)
        address_top = np.load(ADDRESS_TOP_PATH)
        address_keys = ranked_keys(address_top, k=5)

        # 4. Union
        print("Computing candidate union...", flush=True)
        union_keys = np.union1d(np.union1d(baseline_keys, stage5_keys), address_keys)
        print(f"Union keys count: {len(union_keys):,}", flush=True)
        np.save(UNION_PATH, union_keys)
        print("Saved full validation candidate union.", flush=True)

        # 5. Export TSV
        if not CANDIDATE_TSV_PATH.exists():
            export_candidate_tsv(s1, union_keys, CANDIDATE_TSV_PATH)

        # 6. Evaluate recall against ground truth
        truth_list = []
        with TRUTH_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                global_index = id_to_global.get(row["source1_entity_id"])
                if global_index is None:
                    continue
                for match in parse_match_ids(row["matched_entity_ids"]):
                    source_text, numeric = match.split("-", 1)
                    source_code = {"S2": 0, "S3": 1}.get(source_text)
                    if source_code is None:
                        raise ValueError(f"Unexpected truth target: {match}")
                    truth_list.append(pack_pair(global_index, source_code, int(numeric)))
        truth_keys = np.asarray(truth_list, dtype=np.uint64)
        
        hit = exact_membership(np.sort(union_keys), truth_keys)
        true_hits = int(np.count_nonzero(hit))
        recall = float(np.mean(hit))

        # Candidate stats per S1
        s1_indices = (union_keys >> np.uint64(34)).astype(np.int32)
        counts = np.bincount(s1_indices, minlength=len(s1))
        cand_stats = summary(counts)

        result = {
            "validation_size": len(s1),
            "union_candidate_count": len(union_keys),
            "true_hits": true_hits,
            "true_links": len(truth_keys),
            "recall": recall,
            "candidate_stats": cand_stats,
            "total_seconds": time.perf_counter() - start,
            "approx_peak_rss_bytes": peak[0],
        }
        RESULT_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"Results saved to {RESULT_PATH.name}")
        print(f"Total candidates: {len(union_keys):,} | Recall: {recall:.3%}")
        print(f"Candidate counts: mean={cand_stats['mean']:.2f}, median={cand_stats['median']}, P95={cand_stats['p95']}, max={cand_stats['max']}")
        
    finally:
        stopping.set()
        thread.join(timeout=2)


if __name__ == "__main__":
    main()
