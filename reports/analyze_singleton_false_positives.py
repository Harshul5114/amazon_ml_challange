"""Rebuild fixed-subset candidate scores and audit singleton false positives."""

from __future__ import annotations

import csv
import json
import time
from collections import Counter
from pathlib import Path

import joblib
import numpy as np

from reports.run_address_tfidf_subset import ranked_keys
from src.blocking import Business
from src.evaluate import parse_match_ids
from src.similarity_features import FEATURE_NAMES, compute_pair_features

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
OUT = ROOT / "reports" / "singleton_false_positive_audit.json"


def candidate_keys() -> tuple[np.ndarray, np.ndarray]:
    name = np.load(ARTIFACTS / "tfidf_top20.npz")
    address = np.load(ARTIFACTS / "address_tfidf_subset_top10.npz")
    selected = address["global_indices"]
    assert np.array_equal(selected, np.sort(selected))
    baseline = np.load(ARTIFACTS / "stage4_pairs_packed.npy", mmap_mode="r")
    baseline_s1 = (baseline >> np.uint64(34)).astype(np.int32)
    mask = np.isin(baseline_s1, selected)
    keys = np.unique(np.concatenate((
        baseline[mask],
        ranked_keys(name, selected, 10, "stage5"),
        ranked_keys(address, selected, 5, "address"),
    )))
    return selected, keys


def load_subset(selected: np.ndarray) -> tuple[list[Business], list[set[str]]]:
    all_ids = np.load(ARTIFACTS / "tfidf_top20.npz")["s1_ids"]
    wanted = set(all_ids[selected].tolist())
    records_by_id = {}
    with (ROOT / "dataset/train/train_source1.tsv").open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row["entity_id"] in wanted:
                records_by_id[row["entity_id"]] = Business.from_raw(row)
    records = [records_by_id[sid] for sid in all_ids[selected]]
    truth_by_id = {record.entity_id: set() for record in records}
    with (ROOT / "dataset/train/train_ground_truth.tsv").open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            sid = row["source1_entity_id"]
            if sid in truth_by_id:
                truth_by_id[sid] = parse_match_ids(row["matched_entity_ids"])
    return records, [truth_by_id[record.entity_id] for record in records]


def score_candidates(selected: np.ndarray, keys: np.ndarray, records: list[Business]) -> tuple[np.ndarray, np.ndarray]:
    global_s1 = (keys >> np.uint64(34)).astype(np.int32)
    local_s1 = np.searchsorted(selected, global_s1).astype(np.int32)
    if not np.array_equal(selected[local_s1], global_s1):
        raise ValueError("Candidate S1 index outside fixed subset")
    features = np.empty((len(keys), len(FEATURE_NAMES)), dtype=np.float32)
    filled = np.zeros(len(keys), dtype=bool)
    source_codes = ((keys >> np.uint64(32)) & np.uint64(3)).astype(np.uint8)
    target_numbers = (keys & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    for source_code, source in enumerate(("S2", "S3")):
        indices = np.flatnonzero(source_codes == source_code)
        order = indices[np.argsort(target_numbers[indices], kind="stable")]
        numbers, starts = np.unique(target_numbers[order], return_index=True)
        ends = np.append(starts[1:], len(order))
        lookup = {int(number): (int(start), int(end)) for number, start, end in zip(numbers, starts, ends)}
        path = ROOT / f"dataset/train/train_source{source_code + 2}.tsv"
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            header = f.readline().rstrip("\r\n").split("\t")
            id_idx = header.index("entity_id")
            name_idx = header.index("business_name")
            address_idx = header.index("business_address")
            country_idx = header.index("country")
            for line in f:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) <= max(id_idx, name_idx, address_idx, country_idx):
                    continue
                number = int(parts[id_idx].split("-", 1)[1])
                span = lookup.get(number)
                if span is None:
                    continue
                target = Business.from_raw({
                    "entity_id": parts[id_idx], "business_name": parts[name_idx],
                    "business_address": parts[address_idx], "country": parts[country_idx],
                })
                for row_idx in order[span[0]:span[1]]:
                    features[row_idx] = compute_pair_features(records[local_s1[row_idx]], target, source)
                    filled[row_idx] = True
        del lookup
    if not np.all(filled):
        raise ValueError(f"Missing {np.count_nonzero(~filled)} target records")
    model = joblib.load(ARTIFACTS / "tree_match_model.joblib")
    probabilities = model.predict_proba(features)[:, 1].astype(np.float32)
    return features, probabilities


def main() -> None:
    start = time.perf_counter()
    selected, keys = candidate_keys()
    records, truth = load_subset(selected)
    print(f"Subset: {len(records):,} S1; candidates: {len(keys):,}", flush=True)
    features, probabilities = score_candidates(selected, keys, records)
    local_s1 = np.searchsorted(selected, (keys >> np.uint64(34)).astype(np.int32))
    sources = ((keys >> np.uint64(32)) & np.uint64(3)).astype(np.uint8)
    numbers = (keys & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    # Stable ordering makes equal-score choices deterministic.
    order = np.lexsort((keys, -probabilities, local_s1))
    top = {}
    for idx in order:
        identity = (int(local_s1[idx]), int(sources[idx]))
        if identity not in top:
            top[identity] = int(idx)
    singleton_indices = [i for i, matches in enumerate(truth) if not matches]
    singleton_fp = []
    for s1_idx in singleton_indices:
        for source_code in (0, 1):
            row_idx = top.get((s1_idx, source_code))
            if row_idx is not None and probabilities[row_idx] >= 0.97:
                singleton_fp.append(row_idx)
    fp = np.asarray(singleton_fp, dtype=np.int32)
    thresholds = (0.97, 0.99, 0.995, 0.998, 0.999)
    threshold_stats = {}
    for threshold in thresholds:
        affected = {int(local_s1[row]) for row in fp if probabilities[row] >= threshold}
        threshold_stats[str(threshold)] = {"singleton_false_matches": int(np.count_nonzero(probabilities[fp] >= threshold)),
                                          "singleton_entities_with_false_match": len(affected)}
    feature_groups = {}
    for feature in ("exact_name_match", "exact_address_match", "numeric_conflict", "same_country", "target_is_s2"):
        values = features[fp, FEATURE_NAMES.index(feature)]
        feature_groups[feature] = int(np.count_nonzero(values >= 0.5))
    sample_rows = sorted(fp.tolist(), key=lambda i: (-float(probabilities[i]), int(keys[i])))[:20]
    examples = []
    for row_idx in sample_rows:
        s1_idx = int(local_s1[row_idx])
        target_id = f"S{int(sources[row_idx]) + 2}-{int(numbers[row_idx])}"
        examples.append({"source1_entity_id": records[s1_idx].entity_id, "target_id": target_id,
                         "score": float(probabilities[row_idx]),
                         "s1_name": records[s1_idx].business_name,
                         "s1_address": records[s1_idx].business_address,
                         "features": dict(zip(FEATURE_NAMES, features[row_idx].astype(float).tolist()))})
    example_by_target = {row["target_id"]: row for row in examples}
    for source_code in (2, 3):
        path = ROOT / f"dataset/train/train_source{source_code}.tsv"
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f, delimiter="\t"):
                example = example_by_target.get(row["entity_id"])
                if example is not None:
                    example["target_name"] = row["business_name"]
                    example["target_address"] = row["business_address"]
    result = {"subset_s1": len(records), "candidate_pairs": len(keys),
              "singleton_count": len(singleton_indices), "singleton_false_matches_at_097": len(fp),
              "singleton_entities_affected_at_097": len({int(local_s1[i]) for i in fp}),
              "threshold_stats": threshold_stats, "false_match_features_at_097": feature_groups,
              "false_match_probability_quantiles": np.quantile(probabilities[fp], [0, .1, .25, .5, .75, .9, 1]).tolist(),
              "examples": examples, "runtime_seconds": time.perf_counter() - start}
    OUT.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    np.savez_compressed(ARTIFACTS / "singleton_subset_scored.npz", keys=keys, probabilities=probabilities,
                        features=features, selected=selected)
    print(json.dumps({k: v for k, v in result.items() if k not in ("examples",)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
