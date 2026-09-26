"""Create a fixed S1-level split and summarize its balance."""

from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from src.evaluate import entity_f05, parse_match_ids


ROOT = Path(__file__).resolve().parents[1]
SOURCE1 = ROOT / "dataset" / "train" / "train_source1.tsv"
TRUTH = ROOT / "dataset" / "train" / "train_ground_truth.tsv"
SPLIT_PATH = ROOT / "splits" / "s1_train_validation.tsv"
MANIFEST_PATH = ROOT / "splits" / "s1_split_manifest.json"
REPORT_PATH = ROOT / "reports" / "validation_split.md"
SEED = 2026
VALIDATION_THRESHOLD = (1 << 64) // 5
FALSE_ID = "__sanity_false_match__"


def partition(s1_id: str) -> str:
    digest = hashlib.sha256(f"{SEED}:{s1_id}".encode("utf-8")).digest()
    return "validation" if int.from_bytes(digest[:8], "big") < VALIDATION_THRESHOLD else "train"


def percent(n: int, total: int) -> str:
    return f"{100 * n / total:.3f}%" if total else "—"


def split_table(name: str, counts: dict[str, Counter], categories: list) -> list[str]:
    lines = [f"### {name}", "", "| Category | Train count | Train share | Validation count | Validation share | Difference (validation − train) |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    train_total = sum(counts["train"].values())
    val_total = sum(counts["validation"].values())
    for cat in categories:
        train = counts["train"][cat]
        val = counts["validation"][cat]
        difference = 100 * (val / val_total - train / train_total)
        lines.append(f"| {cat} | {train:,} | {percent(train, train_total)} | {val:,} | {percent(val, val_total)} | {difference:+.3f} pp |")
    lines.append("")
    return lines


def main() -> None:
    SPLIT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.parent.mkdir(exist_ok=True)
    split_by_id = {}
    countries = defaultdict(Counter)
    with SOURCE1.open("r", encoding="utf-8-sig", newline="") as source, SPLIT_PATH.open("w", encoding="utf-8", newline="") as destination:
        reader = csv.DictReader(source, delimiter="\t")
        if reader.fieldnames != ["entity_id", "business_name", "business_address", "country"]:
            raise ValueError(f"Unexpected source1 columns: {reader.fieldnames}")
        writer = csv.writer(destination, delimiter="\t", lineterminator="\n")
        writer.writerow(["source1_entity_id", "split"])
        for row in reader:
            s1 = row["entity_id"]
            if not s1 or s1 in split_by_id:
                raise ValueError(f"Empty or duplicate S1 ID: {s1!r}")
            split = partition(s1)
            split_by_id[s1] = split
            countries[split][row["country"]] += 1
            writer.writerow([s1, split])
    print(f"Saved split membership for {len(split_by_id):,} S1 IDs", flush=True)

    match_counts = defaultdict(Counter)
    source_categories = defaultdict(Counter)
    sanity = defaultdict(float)
    with TRUTH.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames != ["source1_entity_id", "matched_entity_ids"]:
            raise ValueError(f"Unexpected ground truth columns: {reader.fieldnames}")
        for row in reader:
            s1 = row["source1_entity_id"]
            split = split_by_id.pop(s1, None)
            if split is None:
                raise ValueError(f"Ground truth has unknown or duplicate S1 ID: {s1!r}")
            matches = parse_match_ids(row["matched_entity_ids"])
            match_counts[split][len(matches)] += 1
            prefixes = {match.split("-", 1)[0] for match in matches}
            category = "zero" if not prefixes else "S2 only" if prefixes == {"S2"} else "S3 only" if prefixes == {"S3"} else "both" if prefixes == {"S2", "S3"} else "other prefix"
            source_categories[split][category] += 1
            if split == "validation":
                if FALSE_ID in matches:
                    raise ValueError("Synthetic false match ID is present in ground truth")
                sanity["perfect"] += entity_f05(matches, matches)
                sanity["all_empty"] += entity_f05(matches, set())
                sanity["one_extra"] += entity_f05(matches, matches | {FALSE_ID})
    if split_by_id:
        raise ValueError(f"Ground truth is missing {len(split_by_id):,} S1 IDs from source1")

    train_n = sum(match_counts["train"].values())
    val_n = sum(match_counts["validation"].values())
    manifest = {
        "seed": SEED,
        "algorithm": "SHA-256 of UTF-8 text '<seed>:<source1_entity_id>'; first 8 digest bytes as unsigned big-endian integer",
        "validation_rule": f"hash_integer < {VALIDATION_THRESHOLD}",
        "membership_file": "s1_train_validation.tsv",
        "train_count": train_n,
        "validation_count": val_n,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    perfect = sanity["perfect"] / val_n
    all_empty = sanity["all_empty"] / val_n
    one_extra = sanity["one_extra"] / val_n
    lines = ["# Fixed S1 train/validation split", "", f"Explicit membership: `splits/s1_train_validation.tsv`. Reproduction rule and seed: `splits/s1_split_manifest.json`. Each S1 ID is assigned once by SHA-256 with seed {SEED} and a 20% validation threshold. All matches for an S1 remain with that S1.", "", f"Train: **{train_n:,}** S1 entities ({percent(train_n, train_n + val_n)}). Validation: **{val_n:,}** S1 entities ({percent(val_n, train_n + val_n)}). The ground truth S1 IDs exactly cover the source1 IDs.", "", "## Balance checks", ""]
    lines += split_table("Singleton status", {split: Counter({"singleton": match_counts[split][0], "non-singleton": sum(v for k, v in match_counts[split].items() if k)}) for split in ("train", "validation")}, ["singleton", "non-singleton"])
    lines += split_table("Matches per S1", match_counts, sorted(set(match_counts["train"]) | set(match_counts["validation"])))
    lines += split_table("Matched source category", source_categories, ["zero", "S2 only", "S3 only", "both", "other prefix"])
    lines += split_table("S1 country", countries, sorted(set(countries["train"]) | set(countries["validation"])))
    lines += ["## Evaluation sanity checks on validation", "", "| Prediction | Macro F0.5 |", "| --- | ---: |", f"| Perfect ground truth copy | {perfect:.12f} |", f"| Empty for every S1 | {all_empty:.12f} |", f"| One synthetic false ID added to every S1 | {one_extra:.12f} |", "", "The all-empty score equals the validation singleton fraction. Adding a false ID to every S1 lowers the score. The synthetic ID is used only for this metric check and is not a challenge prediction.", ""]
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"train={train_n:,} validation={val_n:,} singleton_train={percent(match_counts['train'][0], train_n)} singleton_validation={percent(match_counts['validation'][0], val_n)}", flush=True)
    print(f"perfect={perfect:.12f} all_empty={all_empty:.12f} one_extra={one_extra:.12f}", flush=True)
    print(f"Wrote {REPORT_PATH}", flush=True)


if __name__ == "__main__":
    main()
