"""Stage 9: Feature extraction for training candidate pairs."""

from __future__ import annotations

import csv
import json
import time
from array import array
from collections import defaultdict
from pathlib import Path

import numpy as np
import psutil

from src.blocking import Business, build_rules, retrieve_by_rule
from src.evaluate import parse_match_ids, read_split_ids
from src.similarity_features import (
    FEATURE_NAMES,
    compute_pair_features,
    compute_pair_features_array,
)


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
SPLIT = ROOT / "splits" / "s1_train_validation.tsv"
SOURCE1 = ROOT / "dataset" / "train" / "train_source1.tsv"
TARGETS = [ROOT / "dataset" / "train" / f"train_source{i}.tsv" for i in (2, 3)]
TRUTH_PATH = ROOT / "dataset" / "train" / "train_ground_truth.tsv"

TRAIN_FEATURES_PATH = ARTIFACTS / "train_features.npz"
DEFAULT_TRAIN_SAMPLE = 20_000
SEED = 2026


def build_training_pairs(
    sample_size: int = DEFAULT_TRAIN_SAMPLE, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Build training pairs (X, y) with positive true links and hard negative candidate pairs."""
    start = time.perf_counter()
    ARTIFACTS.mkdir(exist_ok=True)

    print(f"Selecting {sample_size:,} training S1 entities...", flush=True)
    train_ids_all = sorted(read_split_ids(SPLIT, "train"))
    rng = np.random.default_rng(seed)
    chosen_ids = set(rng.choice(train_ids_all, size=sample_size, replace=False))

    s1_records: list[Business] = []
    s1_index: dict[str, int] = {}
    with SOURCE1.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["entity_id"] in chosen_ids:
                s1_index[row["entity_id"]] = len(s1_records)
                s1_records.append(Business.from_raw(row))
    print(f"Loaded {len(s1_records):,} training S1 businesses in {time.perf_counter() - start:.1f}s", flush=True)

    # Load ground truth for these S1
    truth_pairs: set[tuple[int, str, int]] = set()  # (s1_idx, source_prefix, target_number)
    all_truth_target_ids: set[str] = set()
    with TRUTH_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            s1_idx = s1_index.get(row["source1_entity_id"])
            if s1_idx is None:
                continue
            for match in parse_match_ids(row["matched_entity_ids"]):
                prefix, num = match.split("-", 1)
                truth_pairs.add((s1_idx, prefix, int(num)))
                all_truth_target_ids.add(match)

    print(f"Identified {len(truth_pairs):,} true match pairs for sampled S1 entities", flush=True)

    # Build blocking rules and address number index to retrieve realistic candidate negatives
    rules = build_rules(s1_records)
    addr_num_to_s1 = defaultdict(list)
    for idx, s1 in enumerate(s1_records):
        for num in s1.address_numeric_tokens:
            if len(num) >= 3 and len(addr_num_to_s1[(s1.country, num)]) < 5:
                addr_num_to_s1[(s1.country, num)].append(idx)

    candidate_pairs: set[tuple[int, str, int]] = set()
    target_cache: dict[str, Business] = {}
    random_targets: dict[str, list[tuple[int, Business]]] = {"S2": [], "S3": []}

    for source_code, path in enumerate(TARGETS):
        prefix = "S2" if source_code == 0 else "S3"
        print(f"Scanning {path.name} for candidates and truth targets...", flush=True)
        scan_start = time.perf_counter()
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
                tid = parts[id_idx]
                is_truth = tid in all_truth_target_ids

                # Parse business
                target = Business.from_raw({
                    "entity_id": tid,
                    "business_name": parts[name_idx],
                    "business_address": parts[addr_idx],
                    "country": parts[country_idx],
                })
                numeric = int(tid.split("-", 1)[1])

                retrieved = retrieve_by_rule(rules, target)
                union = set().union(*retrieved.values())

                # Address-only candidates
                for num in target.address_numeric_tokens:
                    if len(num) >= 3:
                        for s1_idx in addr_num_to_s1.get((target.country, num), ()):
                            union.add(s1_idx)

                if union or is_truth:
                    target_cache[tid] = target

                for idx in union:
                    candidate_pairs.add((idx, prefix, numeric))

                if len(random_targets[prefix]) < 40_000:
                    random_targets[prefix].append((numeric, target))

        print(f"Finished {path.name} in {time.perf_counter() - scan_start:.1f}s", flush=True)

    # Add random background negatives
    print("Adding background negative pairs...", flush=True)
    for s1_idx in range(len(s1_records)):
        for prefix in ("S2", "S3"):
            sample_pool = random_targets[prefix]
            if sample_pool:
                for _ in range(2):
                    r_num, r_target = sample_pool[rng.integers(len(sample_pool))]
                    candidate_pairs.add((s1_idx, prefix, r_num))
                    target_cache[f"{prefix}-{r_num}"] = r_target

    print(f"Generated {len(candidate_pairs):,} total candidate pairs (name, address, background)", flush=True)

    # Union candidate pairs and truth pairs so all true positives are included
    all_pairs = candidate_pairs | truth_pairs
    print(f"Total training pairs to featurize: {len(all_pairs):,}", flush=True)

    # Compute features and labels
    feature_rows = []
    labels = []
    feat_start = time.perf_counter()

    for s1_idx, prefix, numeric in all_pairs:
        tid = f"{prefix}-{numeric}"
        target = target_cache.get(tid)
        if target is None:
            continue
        s1 = s1_records[s1_idx]
        feat = compute_pair_features(s1, target, prefix)
        is_match = 1.0 if (s1_idx, prefix, numeric) in truth_pairs else 0.0

        feature_rows.append(feat)
        labels.append(is_match)

    X = np.asarray(feature_rows, dtype=np.float32)
    y = np.asarray(labels, dtype=np.float32)
    print(f"Computed feature matrix {X.shape} (positives: {int(np.sum(y)):,}, negatives: {int(len(y) - np.sum(y)):,}) in {time.perf_counter() - feat_start:.1f}s", flush=True)

    np.savez_compressed(
        TRAIN_FEATURES_PATH,
        X=X,
        y=y,
        feature_names=np.asarray(FEATURE_NAMES),
    )
    print(f"Saved training features to {TRAIN_FEATURES_PATH.name} ({TRAIN_FEATURES_PATH.stat().st_size / (1024**2):.1f} MB)", flush=True)
    return X, y, FEATURE_NAMES


def main() -> None:
    build_training_pairs()


if __name__ == "__main__":
    main()
