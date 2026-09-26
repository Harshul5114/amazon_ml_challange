"""Entity-level macro F0.5 for Business Entity Resolution."""

from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from pathlib import Path


def parse_match_ids(value: str | None) -> set[str]:
    """Parse the challenge's comma-separated match ID field as a set."""
    if not value:
        return set()
    return {item.strip() for item in value.split(",") if item.strip()}


def entity_f05(truth: Iterable[str], prediction: Iterable[str]) -> float:
    """Score one S1 entity, including the challenge's singleton convention."""
    true_ids = set(truth)
    predicted_ids = set(prediction)
    if not true_ids:
        return 1.0 if not predicted_ids else 0.0
    true_positives = len(true_ids & predicted_ids)
    if not true_positives:
        return 0.0
    precision = true_positives / len(predicted_ids)
    recall = true_positives / len(true_ids)
    return 1.25 * precision * recall / (0.25 * precision + recall)


def macro_f05(
    truth_by_s1: Mapping[str, Iterable[str]],
    prediction_by_s1: Mapping[str, Iterable[str]],
) -> float:
    """Average entity scores; require exactly one prediction per truth S1 ID."""
    if not truth_by_s1:
        raise ValueError("Cannot score an empty evaluation set")
    truth_keys = truth_by_s1.keys()
    prediction_keys = prediction_by_s1.keys()
    if truth_keys != prediction_keys:
        missing = len(truth_keys - prediction_keys)
        extra = len(prediction_keys - truth_keys)
        raise ValueError(f"S1 ID mismatch: {missing} missing predictions, {extra} extra predictions")
    return sum(entity_f05(matches, prediction_by_s1[s1]) for s1, matches in truth_by_s1.items()) / len(truth_by_s1)


def read_match_tsv(path: str | Path, allowed_ids: set[str] | None = None) -> dict[str, set[str]]:
    """Read a ground truth or prediction TSV using tab separation."""
    result = {}
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["source1_entity_id", "matched_entity_ids"]:
            raise ValueError(f"Unexpected columns in {path}: {reader.fieldnames}")
        for row in reader:
            s1 = row["source1_entity_id"]
            if allowed_ids is not None and s1 not in allowed_ids:
                continue
            if not s1 or s1 in result:
                raise ValueError(f"Empty or duplicate S1 ID in {path}: {s1!r}")
            result[s1] = parse_match_ids(row["matched_entity_ids"])
    return result


def read_split_ids(path: str | Path, partition: str = "validation") -> set[str]:
    """Load S1 IDs for one partition of the saved split definition."""
    if partition not in {"train", "validation"}:
        raise ValueError(f"Unknown split partition: {partition}")
    ids = set()
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["source1_entity_id", "split"]:
            raise ValueError(f"Unexpected split columns in {path}: {reader.fieldnames}")
        for row in reader:
            if row["split"] == partition:
                s1 = row["source1_entity_id"]
                if not s1 or s1 in ids:
                    raise ValueError(f"Empty or duplicate S1 ID in {path}: {s1!r}")
                ids.add(s1)
            elif row["split"] not in {"train", "validation"}:
                raise ValueError(f"Unknown split value in {path}: {row['split']!r}")
    return ids


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Compute challenge macro F0.5")
    parser.add_argument("--truth", required=True, type=Path)
    parser.add_argument("--predictions", required=True, type=Path)
    parser.add_argument("--split-file", type=Path, help="Optional saved S1 split membership TSV")
    parser.add_argument("--partition", choices=("train", "validation"), default="validation")
    args = parser.parse_args()
    allowed_ids = read_split_ids(args.split_file, args.partition) if args.split_file else None
    truth = read_match_tsv(args.truth, allowed_ids)
    predictions = read_match_tsv(args.predictions, allowed_ids)
    print(f"macro F0.5: {macro_f05(truth, predictions):.12f}")


if __name__ == "__main__":
    main()
