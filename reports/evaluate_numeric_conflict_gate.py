"""Evaluate one numeric-conflict rejection rule on the fixed validation subset."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from reports.analyze_singleton_false_positives import load_subset
from reports.evaluate_tfidf_union import pack_pair
from src.evaluate import entity_f05
from src.similarity_features import FEATURE_NAMES

ROOT = Path(__file__).resolve().parents[1]
SCORED = ROOT / "artifacts/singleton_subset_scored.npz"
OUTPUT = ROOT / "reports/numeric_conflict_gate_results.json"
THRESHOLD = 0.97


def select_predictions(
    keys: np.ndarray, scores: np.ndarray, eligible: np.ndarray,
    selected: np.ndarray, threshold: float,
) -> list[set[str]]:
    local_s1 = np.searchsorted(selected, (keys >> np.uint64(34)).astype(np.int32))
    sources = ((keys >> np.uint64(32)) & np.uint64(3)).astype(np.uint8)
    numbers = (keys & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    order = np.lexsort((keys, -scores, local_s1))
    predictions = [set() for _ in selected]
    chosen = np.zeros((len(selected), 2), dtype=bool)
    for row_idx in order:
        if scores[row_idx] < threshold or not eligible[row_idx]:
            continue
        s1_idx = int(local_s1[row_idx])
        source_code = int(sources[row_idx])
        if not chosen[s1_idx, source_code]:
            predictions[s1_idx].add(f"S{source_code + 2}-{int(numbers[row_idx])}")
            chosen[s1_idx, source_code] = True
    return predictions


def summarize(truth: list[set[str]], predictions: list[set[str]]) -> dict:
    tp = fp = fn = 0
    scores = []
    singleton_scores = []
    non_singleton_scores = []
    non_singleton_precision = []
    non_singleton_recall = []
    false_singleton_entities = false_singleton_links = predicted_links = 0
    for true_ids, pred_ids in zip(truth, predictions):
        n_tp = len(true_ids & pred_ids)
        tp += n_tp
        fp += len(pred_ids - true_ids)
        fn += len(true_ids - pred_ids)
        predicted_links += len(pred_ids)
        score = entity_f05(true_ids, pred_ids)
        scores.append(score)
        if true_ids:
            non_singleton_scores.append(score)
            non_singleton_precision.append(n_tp / len(pred_ids) if pred_ids else 0.0)
            non_singleton_recall.append(n_tp / len(true_ids))
        else:
            singleton_scores.append(score)
            if pred_ids:
                false_singleton_entities += 1
                false_singleton_links += len(pred_ids)
    return {
        "macro_f05": float(np.mean(scores)),
        "pair_precision": tp / (tp + fp) if tp + fp else 0.0,
        "pair_recall": tp / (tp + fn) if tp + fn else 0.0,
        "true_positives": tp, "false_positives": fp, "false_negatives": fn,
        "singleton_accuracy": float(np.mean(singleton_scores)),
        "singleton_false_match_entities": false_singleton_entities,
        "singleton_false_match_links": false_singleton_links,
        "non_singleton_macro_f05": float(np.mean(non_singleton_scores)),
        "non_singleton_macro_precision": float(np.mean(non_singleton_precision)),
        "non_singleton_macro_recall": float(np.mean(non_singleton_recall)),
        "average_predicted_matches_per_s1": predicted_links / len(truth),
    }


def main() -> None:
    start = time.perf_counter()
    data = np.load(SCORED)
    keys, scores = data["keys"], data["probabilities"]
    selected, features = data["selected"], data["features"]
    records, truth = load_subset(selected)
    truth_keys = np.asarray([
        pack_pair(int(global_idx), 0 if target_id.startswith("S2-") else 1,
                  int(target_id.split("-", 1)[1]))
        for global_idx, matches in zip(selected, truth)
        for target_id in matches
    ], dtype=np.uint64)
    positions = np.searchsorted(keys, truth_keys)
    hits = np.zeros(len(truth_keys), dtype=bool)
    in_bounds = positions < len(keys)
    hits[in_bounds] = keys[positions[in_bounds]] == truth_keys[in_bounds]
    true_hits = int(np.count_nonzero(hits))
    all_eligible = np.ones(len(keys), dtype=bool)
    no_conflict = features[:, FEATURE_NAMES.index("numeric_conflict")] < 0.5
    baseline_predictions = select_predictions(keys, scores, all_eligible, selected, THRESHOLD)
    gated_predictions = select_predictions(keys, scores, no_conflict, selected, THRESHOLD)
    baseline = summarize(truth, baseline_predictions)
    gated = summarize(truth, gated_predictions)
    country_results = {}
    for country in sorted({record.country for record in records}):
        indices = [i for i, record in enumerate(records) if record.country == country]
        country_results[country] = {
            "s1_count": len(indices),
            "baseline": summarize([truth[i] for i in indices], [baseline_predictions[i] for i in indices]),
            "gated": summarize([truth[i] for i in indices], [gated_predictions[i] for i in indices]),
        }
    result = {
        "experiment_id": "numeric_conflict_gate_v1",
        "fixed_validation_subset_s1": len(selected),
        "candidate_pairs": len(keys),
        "candidate_true_hits": true_hits,
        "candidate_true_links": len(truth_keys),
        "candidate_recall": true_hits / len(truth_keys),
        "model": "hist_gb; 5,000 sampled training S1; 50 candidates per S1/source; 150,000 augmented negatives",
        "threshold": THRESHOLD,
        "gate": "reject candidate if numeric_conflict feature is 1",
        "baseline": baseline, "gated": gated,
        "country_results": country_results,
        "runtime_seconds": time.perf_counter() - start,
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
