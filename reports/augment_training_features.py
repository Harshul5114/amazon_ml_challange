"""Augment training feature matrix with background and address-sharing negatives."""

from __future__ import annotations

import csv
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.blocking import Business
from src.similarity_features import FEATURE_NAMES, compute_pair_features


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
SOURCE1 = ROOT / "dataset" / "train" / "train_source1.tsv"
TARGETS = [ROOT / "dataset" / "train" / f"train_source{i}.tsv" for i in (2, 3)]
TRAIN_FEATURES_PATH = ARTIFACTS / "train_features.npz"


def augment_features(n_negatives: int = 150_000, seed: int = 2026) -> None:
    t0 = time.perf_counter()
    data = np.load(TRAIN_FEATURES_PATH)
    X_orig, y_orig = data["X"], data["y"]
    print(f"Loaded existing training features: X={X_orig.shape}, y_pos={int(np.sum(y_orig)):,}, y_neg={int(len(y_orig)-np.sum(y_orig)):,}", flush=True)

    # Sample S1 records
    rng = np.random.default_rng(seed)
    s1_sample: list[Business] = []
    with SOURCE1.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for i, row in enumerate(reader):
            if i >= 100_000:
                break
            if rng.random() < 0.2:
                s1_sample.append(Business.from_raw(row))
    print(f"Sampled {len(s1_sample):,} S1 businesses for negative pairing in {time.perf_counter()-t0:.1f}s", flush=True)

    # Sample S2 and S3 targets
    target_pool: dict[str, list[Business]] = {"S2": [], "S3": []}
    for source_code, path in enumerate(TARGETS):
        prefix = "S2" if source_code == 0 else "S3"
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for i, row in enumerate(reader):
                if i >= 100_000:
                    break
                if rng.random() < 0.2:
                    target_pool[prefix].append(Business.from_raw(row))

    print(f"Sampled target pool: S2={len(target_pool['S2']):,}, S3={len(target_pool['S3']):,}", flush=True)

    # Generate random background negatives and address-sharing negatives
    neg_rows = []
    # 1. Random background negatives (different business names)
    for _ in range(n_negatives // 2):
        s1 = s1_sample[rng.integers(len(s1_sample))]
        prefix = "S2" if rng.random() < 0.5 else "S3"
        pool = target_pool[prefix]
        target = pool[rng.integers(len(pool))]
        neg_rows.append(compute_pair_features(s1, target, prefix))

    # 2. Hard address-sharing negatives (create pairs with synthetic/modified names but shared address)
    for _ in range(n_negatives // 2):
        s1 = s1_sample[rng.integers(len(s1_sample))]
        # Pair with another business in the same country
        prefix = "S2" if rng.random() < 0.5 else "S3"
        pool = target_pool[prefix]
        target = pool[rng.integers(len(pool))]
        # Swap address so they share address but have completely different names
        fake_target = Business(
            entity_id=target.entity_id,
            country=s1.country,
            business_name=target.business_name,
            business_address=s1.business_address,
            name_norm=target.name_norm,
            address_norm=s1.address_norm,
            name_tokens=target.name_tokens,
            address_numeric_tokens=s1.address_numeric_tokens,
        )
        neg_rows.append(compute_pair_features(s1, fake_target, prefix))

    X_neg = np.asarray(neg_rows, dtype=np.float32)
    y_neg = np.zeros(len(neg_rows), dtype=np.float32)

    X_combined = np.vstack([X_orig, X_neg])
    y_combined = np.concatenate([y_orig, y_neg])

    print(f"Combined features shape: {X_combined.shape}, positives={int(np.sum(y_combined)):,}, negatives={int(len(y_combined)-np.sum(y_combined)):,}", flush=True)

    np.savez_compressed(
        TRAIN_FEATURES_PATH,
        X=X_combined,
        y=y_combined,
        feature_names=np.asarray(FEATURE_NAMES),
    )
    print(f"Saved augmented features to {TRAIN_FEATURES_PATH.name} ({TRAIN_FEATURES_PATH.stat().st_size / (1024**2):.1f} MB) in {time.perf_counter()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    augment_features()
