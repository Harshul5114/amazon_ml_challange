"""Stage 10: Train tree-based matching model and optimize threshold for entity-level macro F0.5."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from src.blocking import Business, build_rules, retrieve_by_rule
from src.evaluate import entity_f05, macro_f05, parse_match_ids, read_split_ids
from src.similarity_features import (
    FEATURE_NAMES,
    compute_pair_features,
    compute_pair_features_array,
)
from reports.generate_features import build_training_pairs


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
SPLIT = ROOT / "splits" / "s1_train_validation.tsv"
SUBSET_SPLIT = ROOT / "splits" / "stage6a_validation_subset.tsv"
SOURCE1 = ROOT / "dataset" / "train" / "train_source1.tsv"
TARGETS = [ROOT / "dataset" / "train" / f"train_source{i}.tsv" for i in (2, 3)]
TRUTH_PATH = ROOT / "dataset" / "train" / "train_ground_truth.tsv"

MODEL_PATH = ARTIFACTS / "tree_match_model.joblib"
TRAIN_FEATURES_PATH = ARTIFACTS / "train_features.npz"
RESULTS_PATH = ROOT / "reports" / "model_evaluation_results.json"
ADDRESS_TOP_PATH = ARTIFACTS / "address_tfidf_top10.npz"


def train_tree_model(
    X_train: np.ndarray, y_train: np.ndarray, model_type: str = "hist_gb",
    model_path: Path | None = None,
) -> HistGradientBoostingClassifier | RandomForestClassifier:
    """Train a gradient-boosted decision tree matching model."""
    print(f"Training {model_type} model on {X_train.shape[0]:,} pairs with {X_train.shape[1]} features...", flush=True)
    t0 = time.perf_counter()

    if model_type == "hist_gb":
        model = HistGradientBoostingClassifier(
            max_iter=150,
            learning_rate=0.08,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            random_state=2026,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
        )
        model.fit(X_train, y_train)
    elif model_type == "rf":
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            min_samples_leaf=10,
            random_state=2026,
            n_jobs=-1,
            class_weight="balanced_subsample",
        )
        model.fit(X_train, y_train)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    train_time = time.perf_counter() - t0
    print(f"Model trained in {train_time:.1f}s", flush=True)

    # In-sample evaluation
    train_probs = model.predict_proba(X_train)[:, 1]
    auc = roc_auc_score(y_train, train_probs)
    print(f"Train ROC-AUC: {auc:.4f}", flush=True)

    output_path = MODEL_PATH if model_path is None else model_path
    joblib.dump(model, output_path)
    print(f"Saved model to {output_path.name}", flush=True)
    return model


def evaluate_thresholds_on_validation(
    model,
    val_s1_records: list[Business],
    val_candidate_pairs: list[tuple[int, str, int]],
    truth_by_s1: dict[str, set[str]],
    target_cache: dict[str, Business],
    threshold_range: np.ndarray | None = None,
) -> dict:
    """Score validation candidate pairs, grid-search decision thresholds, and compute entity-level macro F0.5."""
    if threshold_range is None:
        threshold_range = np.concatenate([
            np.arange(0.85, 0.98, 0.02),
            np.arange(0.98, 0.999, 0.002),
            [0.999, 0.9995]
        ])
    print(f"Scoring {len(val_candidate_pairs):,} validation candidate pairs...", flush=True)
    t0 = time.perf_counter()

    # Featurize validation pairs
    rows = []
    pair_metadata = []  # (s1_idx, target_id)
    for s1_idx, prefix, num in val_candidate_pairs:
        tid = f"{prefix}-{num}"
        target = target_cache.get(tid)
        if target is None:
            continue
        s1 = val_s1_records[s1_idx]
        rows.append(compute_pair_features(s1, target, prefix))
        pair_metadata.append((s1_idx, tid))

    X_val = np.asarray(rows, dtype=np.float32)
    probs = model.predict_proba(X_val)[:, 1]
    score_time = time.perf_counter() - t0
    print(f"Scored {len(probs):,} pairs in {score_time:.1f}s", flush=True)

    # Organize predictions by S1 index and probability
    s1_candidates = defaultdict(list)
    for i, (s1_idx, tid) in enumerate(pair_metadata):
        s1_candidates[s1_idx].append((probs[i], tid))

    # Pre-sort each S1's candidates by probability descending
    for idx in s1_candidates:
        s1_candidates[idx].sort(key=lambda item: item[0], reverse=True)

    best_threshold = 0.5
    best_f05 = -1.0
    threshold_metrics = []

    s1_id_list = [b.entity_id for b in val_s1_records]
    total_s1 = len(val_s1_records)
    singleton_ids = {s1 for s1, matches in truth_by_s1.items() if not matches}

    for thresh in threshold_range:
        thresh = round(float(thresh), 4)
        predictions_by_s1: dict[str, set[str]] = {}
        total_pred_matches = 0
        singleton_correct = 0

        for idx, s1_id in enumerate(s1_id_list):
            cand_list = s1_candidates.get(idx, [])
            # Top-1 per target source above threshold
            s2_cands = [tid for p, tid in cand_list if p >= thresh and tid.startswith("S2")][:1]
            s3_cands = [tid for p, tid in cand_list if p >= thresh and tid.startswith("S3")][:1]
            preds = set(s2_cands + s3_cands)
            predictions_by_s1[s1_id] = preds
            total_pred_matches += len(preds)
            if s1_id in singleton_ids and not preds:
                singleton_correct += 1

        score = macro_f05(truth_by_s1, predictions_by_s1)
        singleton_acc = singleton_correct / max(len(singleton_ids), 1)
        avg_matches = total_pred_matches / total_s1

        threshold_metrics.append({
            "threshold": thresh,
            "macro_f05": score,
            "singleton_accuracy": singleton_acc,
            "avg_matches_per_s1": avg_matches,
        })

        if score > best_f05:
            best_f05 = score
            best_threshold = thresh

    print(f"\nBest threshold: {best_threshold:.2f} with Macro F0.5: {best_f05:.6f}")
    
    # Detailed evaluation at best threshold
    best_predictions: dict[str, set[str]] = {}
    for idx, s1_id in enumerate(s1_id_list):
        cand_list = s1_candidates.get(idx, [])
        s2_cands = [tid for p, tid in cand_list if p >= best_threshold and tid.startswith("S2")][:1]
        s3_cands = [tid for p, tid in cand_list if p >= best_threshold and tid.startswith("S3")][:1]
        best_predictions[s1_id] = set(s2_cands + s3_cands)

    # Entity level precision and recall breakdown
    tp_total = 0
    fp_total = 0
    fn_total = 0
    for s1_id, matches in truth_by_s1.items():
        preds = best_predictions[s1_id]
        tp = len(matches & preds)
        fp = len(preds - matches)
        fn = len(matches - preds)
        tp_total += tp
        fp_total += fp
        fn_total += fn

    pair_precision = tp_total / max(tp_total + fp_total, 1)
    pair_recall = tp_total / max(tp_total + fn_total, 1)

    result = {
        "best_threshold": best_threshold,
        "best_macro_f05": best_f05,
        "pair_precision": pair_precision,
        "pair_recall": pair_recall,
        "true_positives": tp_total,
        "false_positives": fp_total,
        "false_negatives": fn_total,
        "validation_s1_count": total_s1,
        "validation_candidate_pairs": len(val_candidate_pairs),
        "threshold_curve": threshold_metrics,
        "model_file": MODEL_PATH.name,
    }
    return result, best_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Train tree-based matching model and evaluate macro F0.5")
    parser.add_argument("--model-type", choices=("hist_gb", "rf"), default="hist_gb", help="Tree model type")
    parser.add_argument("--sample-size", type=int, default=20_000, help="Training S1 sample size")
    parser.add_argument("--eval-subset", action="store_true", default=True, help="Evaluate on validation subset")
    args = parser.parse_args()

    ARTIFACTS.mkdir(exist_ok=True)
    start_total = time.perf_counter()

    # 1. Load or build training features
    if TRAIN_FEATURES_PATH.exists():
        print(f"Loading cached training features from {TRAIN_FEATURES_PATH.name}...", flush=True)
        data = np.load(TRAIN_FEATURES_PATH)
        X_train, y_train = data["X"], data["y"]
    else:
        print("Generating training features from scratch...", flush=True)
        X_train, y_train, _ = build_training_pairs(sample_size=args.sample_size)

    # 2. Train model
    model = train_tree_model(X_train, y_train, model_type=args.model_type)

    # 3. Load validation set for evaluation
    val_ids_to_use = set()
    if args.eval_subset and SUBSET_SPLIT.exists():
        print(f"Evaluating on stratified 50,000 validation subset ({SUBSET_SPLIT.name})...", flush=True)
        with SUBSET_SPLIT.open("r", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                val_ids_to_use.add(row["source1_entity_id"])
    else:
        print("Evaluating on full validation split...", flush=True)
        val_ids_to_use = read_split_ids(SPLIT, "validation")

    val_s1: list[Business] = []
    val_s1_index: dict[str, int] = {}
    with SOURCE1.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["entity_id"] in val_ids_to_use:
                val_s1_index[row["entity_id"]] = len(val_s1)
                val_s1.append(Business.from_raw(row))
    print(f"Loaded {len(val_s1):,} validation S1 businesses", flush=True)

    # Truth for validation
    truth_by_s1 = {b.entity_id: set() for b in val_s1}
    all_val_truth_targets = set()
    with TRUTH_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            s1_id = row["source1_entity_id"]
            if s1_id in truth_by_s1:
                matches = parse_match_ids(row["matched_entity_ids"])
                truth_by_s1[s1_id] = matches
                all_val_truth_targets.update(matches)

    # Generate or load candidate pairs for validation
    # If full_validation_candidates_packed.npy exists, we can use it!
    candidates_packed_path = ARTIFACTS / "full_validation_candidates_packed.npy"
    val_candidate_pairs: list[tuple[int, str, int]] = []
    target_cache: dict[str, Business] = {}

    if candidates_packed_path.exists():
        print(f"Loading candidate pairs from {candidates_packed_path.name}...", flush=True)
        if ADDRESS_TOP_PATH.exists():
            all_val_ids = np.load(ADDRESS_TOP_PATH)["s1_ids"]
        else:
            all_val_ids = []
            all_val_set = read_split_ids(SPLIT, "validation")
            with SOURCE1.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle, delimiter="\t"):
                    if row["entity_id"] in all_val_set:
                        all_val_ids.append(row["entity_id"])
            all_val_ids = np.array(all_val_ids)

        global_to_local = {}
        for local_idx, b in enumerate(val_s1):
            global_to_local[b.entity_id] = local_idx

        packed = np.load(candidates_packed_path, mmap_mode="r")
        s1_indices = (packed >> np.uint64(34)).astype(np.int32)
        sources = ((packed >> np.uint64(32)) & np.uint64(3)).astype(np.int8)
        numerics = (packed & np.uint64(0xFFFFFFFF)).astype(np.uint32)

        # Load S1 IDs list in validation order
        val_s1_order = []
        with SOURCE1.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row["entity_id"] in val_ids_to_use:
                    val_s1_order.append(row["entity_id"])

        val_needed_targets = set()
        for i in range(len(packed)):
            g_idx = int(s1_indices[i])
            if g_idx < len(all_val_ids):
                s1_id = all_val_ids[g_idx]
                loc_idx = global_to_local.get(s1_id)
                if loc_idx is not None:
                    src = "S2" if sources[i] == 0 else "S3"
                    num = int(numerics[i])
                    val_candidate_pairs.append((loc_idx, src, num))
                    val_needed_targets.add(f"{src}-{num}")
    else:
        # Build Stage 4 blocking candidates on the validation subset
        print("Candidate union artifact not present yet; generating Stage 4 candidates for validation subset...", flush=True)
        val_rules = build_rules(val_s1)
        val_needed_targets = set(all_val_truth_targets)

        for source_code, path in enumerate(TARGETS):
            prefix = "S2" if source_code == 0 else "S3"
            print(f"Scanning {path.name} for validation candidates...", flush=True)
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
                    numeric = int(tid.split("-", 1)[1])
                    target = Business.from_raw({
                        "entity_id": tid,
                        "business_name": parts[name_idx],
                        "business_address": parts[addr_idx],
                        "country": parts[country_idx],
                    })
                    retrieved = retrieve_by_rule(val_rules, target)
                    union = set().union(*retrieved.values())
                    if union or tid in all_val_truth_targets:
                        target_cache[tid] = target
                    for s1_idx in union:
                        val_candidate_pairs.append((s1_idx, prefix, numeric))

    # Populate target cache for needed targets if not already loaded
    missing_targets = val_needed_targets - set(target_cache.keys())
    if missing_targets:
        print(f"Caching {len(missing_targets):,} target records for feature evaluation...", flush=True)
        for path in TARGETS:
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
                    if tid in missing_targets:
                        target_cache[tid] = Business.from_raw({
                            "entity_id": tid,
                            "business_name": parts[name_idx],
                            "business_address": parts[addr_idx],
                            "country": parts[country_idx],
                        })
                        missing_targets.remove(tid)
                        if not missing_targets:
                            break

    # 4. Evaluate thresholds
    results, best_preds = evaluate_thresholds_on_validation(
        model, val_s1, val_candidate_pairs, truth_by_s1, target_cache
    )
    results["total_pipeline_seconds"] = time.perf_counter() - start_total

    RESULTS_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\nFinal evaluation saved to {RESULTS_PATH.name}")
    print(f"Macro F0.5: {results['best_macro_f05']:.6f} at Threshold {results['best_threshold']:.2f}")
    print(f"Pair Precision: {results['pair_precision']:.4f} | Pair Recall: {results['pair_recall']:.4f}")


if __name__ == "__main__":
    main()
