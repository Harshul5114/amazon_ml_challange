"""Evaluate the submitted exact-key rule on the fixed S1 validation split."""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path

from src.evaluate import entity_f05, read_match_tsv, read_split_ids
from src.generate_quick_submission import key, rows


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "dataset" / "train"
SPLIT = ROOT / "splits" / "s1_train_validation.tsv"
RESULT = ROOT / "reports" / "quick_submission_validation_results.json"
PREDICTIONS = ROOT / "artifacts" / "quick_validation_predictions.tsv"


def main() -> None:
    start = time.perf_counter()
    allowed = read_split_ids(SPLIT, "validation")
    truth = read_match_tsv(TRAIN / "train_ground_truth.tsv", allowed)
    if set(truth) != allowed:
        raise ValueError("Ground truth does not cover the saved validation split")

    s1_ids: list[str] = []
    countries: list[str] = []
    unique_keys: dict[tuple[str, str, str], int] = {}
    ambiguous: set[tuple[str, str, str]] = set()
    for entity_id, name, address, country in rows(TRAIN / "train_source1.tsv"):
        if entity_id not in allowed:
            continue
        index = len(s1_ids)
        s1_ids.append(entity_id)
        countries.append(country)
        record_key = key(name, address, country)
        if record_key is None or record_key in ambiguous:
            continue
        if record_key in unique_keys:
            del unique_keys[record_key]
            ambiguous.add(record_key)
        else:
            unique_keys[record_key] = index
    if len(s1_ids) != len(allowed):
        raise ValueError("S1 source does not cover the saved validation split")
    print(f"Indexed {len(s1_ids):,} validation S1 records", flush=True)

    matches: dict[int, list[str]] = defaultdict(list)
    for source in (2, 3):
        count = 0
        for target_id, name, address, country in rows(TRAIN / f"train_source{source}.tsv"):
            record_key = key(name, address, country)
            if record_key is not None:
                index = unique_keys.get(record_key)
                if index is not None:
                    matches[index].append(target_id)
            count += 1
            if count % 2_000_000 == 0:
                print(f"S{source}: scanned {count:,} targets", flush=True)
        print(f"S{source}: finished {count:,} targets", flush=True)

    PREDICTIONS.parent.mkdir(exist_ok=True)
    with PREDICTIONS.open("w", encoding="utf-8", newline="") as handle:
        handle.write("source1_entity_id\tmatched_entity_ids\n")
        for index, entity_id in enumerate(s1_ids):
            handle.write(f"{entity_id}\t{','.join(matches.get(index, ()))}\n")

    totals = defaultdict(float)
    country_totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for index, entity_id in enumerate(s1_ids):
        actual = truth[entity_id]
        predicted = set(matches.get(index, ()))
        tp = len(actual & predicted)
        fp = len(predicted - actual)
        fn = len(actual - predicted)
        score = entity_f05(actual, predicted)
        for bucket in (totals, country_totals[countries[index]]):
            bucket["entities"] += 1
            bucket["f05_sum"] += score
            bucket["true_positives"] += tp
            bucket["false_positives"] += fp
            bucket["false_negatives"] += fn
            bucket["predicted_links"] += len(predicted)
            bucket["nonempty_predictions"] += bool(predicted)
            if not actual:
                bucket["singletons"] += 1
                bucket["singleton_correct"] += not predicted
                bucket["singleton_false_match_links"] += len(predicted)
            else:
                bucket["non_singletons"] += 1
                bucket["non_singleton_f05_sum"] += score

    def summarize(bucket: dict[str, float]) -> dict:
        entities = int(bucket["entities"])
        tp, fp, fn = (int(bucket[name]) for name in
                      ("true_positives", "false_positives", "false_negatives"))
        singletons = int(bucket["singletons"])
        non_singletons = int(bucket["non_singletons"])
        return {
            "s1_entities": entities,
            "macro_f05": bucket["f05_sum"] / entities,
            "pair_precision": tp / (tp + fp) if tp + fp else 0.0,
            "pair_recall": tp / (tp + fn) if tp + fn else 0.0,
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "predicted_links": int(bucket["predicted_links"]),
            "nonempty_predictions": int(bucket["nonempty_predictions"]),
            "average_predicted_matches_per_s1": bucket["predicted_links"] / entities,
            "singletons": singletons,
            "singleton_accuracy": bucket["singleton_correct"] / singletons if singletons else None,
            "singleton_false_match_entities": singletons - int(bucket["singleton_correct"]),
            "singleton_false_match_links": int(bucket["singleton_false_match_links"]),
            "non_singletons": non_singletons,
            "non_singleton_macro_f05": bucket["non_singleton_f05_sum"] / non_singletons if non_singletons else None,
        }

    result = {
        "method": "exact country + normalized business name + normalized business address; unique S1 keys only",
        "split": str(SPLIT.relative_to(ROOT)),
        "target_pool": "all train S2 and S3 records; no truth filtering",
        "overall": summarize(totals),
        "by_country": {country: summarize(bucket) for country, bucket in sorted(country_totals.items())},
        "runtime_seconds": time.perf_counter() - start,
        "predictions": str(PREDICTIONS.relative_to(ROOT)),
    }
    RESULT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
