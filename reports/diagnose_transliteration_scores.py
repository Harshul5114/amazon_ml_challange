"""Inspect direct cosine scores on the fixed Stage 6B cross-script sample."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from src.normalize import normalize_text
from src.transliterate import name_translit


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_FILE = ROOT / "splits/stage6a_validation_subset.tsv"
S1_FILE = ROOT / "dataset/train/train_source1.tsv"
MISSES = ROOT / "reports/stage6b_miss_summary.json"
TOP = ROOT / "artifacts/translit_top10_subset.npz"
OUTPUT = ROOT / "reports/transliteration_score_diagnostic.json"


def main() -> None:
    ids = [row["source1_entity_id"] for row in csv.DictReader(SAMPLE_FILE.open("r", encoding="utf-8"), delimiter="\t")]
    needed = set(ids)
    names = {}
    with S1_FILE.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["entity_id"] in needed:
                names[row["entity_id"]] = normalize_text(row["business_name"])
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 5), min_df=2,
                                 max_df=0.02, sublinear_tf=True, dtype=np.float32)
    queries = vectorizer.fit_transform(names[s1] for s1 in ids)
    position = {s1: index for index, s1 in enumerate(ids)}
    misses = json.loads(MISSES.read_text(encoding="utf-8"))["sample"]
    cross = [item for item in misses if item["name_script_relation"] == "different"]
    converted = vectorizer.transform(name_translit(item["target_name_norm"]) for item in cross)
    top = np.load(TOP)
    details = []
    for index, item in enumerate(cross):
        local = position[item["s1_id"]]
        direct = float(queries[local].multiply(converted[index]).sum())
        scores = top[f"{item['source']}_scores"][local]
        cutoff = float(scores[9]) if scores[9] >= 0 else 0.0
        target_number = int(item["target_id"].split("-", 1)[1])
        target_ranks = np.flatnonzero(top[f"{item['source']}_ids"][local] == target_number)
        details.append({"s1_id": item["s1_id"], "target_id": item["target_id"],
                        "direct_cosine": direct, "rank_0_based": int(target_ranks[0]) if len(target_ranks) else None,
                        "k10_cutoff": cutoff,
                        "s1_name": item["s1_name_norm"],
                        "target_name_translit": name_translit(item["target_name_norm"])})
    direct = np.asarray([row["direct_cosine"] for row in details])
    cutoffs = np.asarray([row["k10_cutoff"] for row in details])
    result = {
        "cross_script_sample_size": len(details),
        "median_true_cosine": float(np.median(direct)),
        "median_k10_cutoff": float(np.median(cutoffs)),
        "true_cosine_below_k10_cutoff": int(np.count_nonzero(direct < cutoffs)),
        "true_cosine_zero": int(np.count_nonzero(direct == 0)),
        "true_pair_in_top10": int(sum(row["rank_0_based"] is not None for row in details)),
        "examples": details[:12],
    }
    OUTPUT.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}: median true={result['median_true_cosine']:.3f}, median cutoff={result['median_k10_cutoff']:.3f}")


if __name__ == "__main__":
    main()
