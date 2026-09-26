"""Analyze Stage 6A K=5 misses using cached candidates, without retrieval."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import time
import unicodedata
from collections import Counter
from pathlib import Path

import numpy as np

from reports.evaluate_tfidf_union import exact_membership, pack_pair
from reports.run_address_tfidf_subset import ranked_keys
from src.blocking import Business
from src.evaluate import parse_match_ids
from src.normalize import numeric_tokens


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_FILE = ROOT / "splits/stage6a_validation_subset.tsv"
S1_FILE = ROOT / "dataset/train/train_source1.tsv"
TRUTH_FILE = ROOT / "dataset/train/train_ground_truth.tsv"
TARGET_FILES = [ROOT / f"dataset/train/train_source{i}.tsv" for i in (2, 3)]
TOP_NAME = ROOT / "artifacts/tfidf_top20.npz"
TOP_ADDRESS = ROOT / "artifacts/address_tfidf_subset_top10.npz"
STAGE4 = ROOT / "artifacts/stage4_pairs_packed.npy"
STAGE6A = ROOT / "reports/address_tfidf_subset_results.json"
SUMMARY = ROOT / "reports/stage6b_miss_summary.json"
SAMPLE_TSV = ROOT / "reports/stage6b_miss_sample.tsv"
SAMPLE_SIZE = 150
SAMPLE_SEED = 2062
NUMBER_PATTERN = re.compile(r"\d+")


def script_of(text: str) -> str:
    scripts = Counter()
    for char in text:
        if not char.isalpha():
            continue
        unicode_name = unicodedata.name(char, "")
        prefix = unicode_name.split(" ", 1)[0]
        if prefix in {"LATIN", "DEVANAGARI", "BENGALI", "TAMIL", "TELUGU", "KANNADA",
                      "MALAYALAM", "GUJARATI", "GURMUKHI", "ARABIC", "CYRILLIC", "GREEK",
                      "HEBREW", "THAI", "HANGUL", "HIRAGANA", "KATAKANA"}:
            scripts[prefix] += 1
        elif "CJK UNIFIED IDEOGRAPH" in unicode_name:
            scripts["HAN"] += 1
        else:
            scripts["OTHER"] += 1
    if not scripts:
        return "unknown"
    primary, count = scripts.most_common(1)[0]
    return primary if count / sum(scripts.values()) >= 0.8 else "mixed"


def token_overlap(first: str, second: str) -> tuple[float | None, int]:
    a, b = set(first.split()), set(second.split())
    if not a or not b:
        return None, 0
    common = len(a & b)
    return common / len(a | b), common


def overlap_bin(value: float | None) -> str:
    if value is None:
        return "not comparable: empty field"
    if value == 0:
        return "zero"
    if value < 0.25:
        return "low (<0.25)"
    if value < 0.5:
        return "medium (0.25–<0.5)"
    return "high (>=0.5)"


def presence(first: str, second: str) -> str:
    if first and second:
        return "both present"
    if first:
        return "S1 only"
    if second:
        return "target only"
    return "neither present"


def pair_id(s1: str, target: str) -> str:
    return f"{s1}|{target}"


def main() -> None:
    start = time.perf_counter()
    stage6a = json.loads(STAGE6A.read_text(encoding="utf-8"))
    name_top = np.load(TOP_NAME)
    address_top = np.load(TOP_ADDRESS)
    all_ids = name_top["s1_ids"]
    selected = address_top["global_indices"]
    sample_ids = []
    sample_country = {}
    with SAMPLE_FILE.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            sample_ids.append(row["source1_entity_id"])
            sample_country[row["source1_entity_id"]] = row["country"]
    if sample_ids != all_ids[selected].tolist() or len(sample_ids) != 50_000:
        raise ValueError("Stage 6A fixed subset and cached address top lists differ")
    id_to_global = {s1: int(index) for s1, index in zip(sample_ids, selected)}

    selected_mask = np.zeros(len(all_ids), dtype=bool)
    selected_mask[selected] = True
    stage4_all = np.load(STAGE4, mmap_mode="r")
    stage4_keys = stage4_all[selected_mask[(stage4_all >> np.uint64(34)).astype(np.int32)]]
    stage5_keys = np.union1d(stage4_keys, ranked_keys(name_top, selected, 10, "stage5"))
    address5 = ranked_keys(address_top, selected, 5, "address")
    address10 = ranked_keys(address_top, selected, 10, "address")
    best5 = np.union1d(stage5_keys, address5)
    best10 = np.union1d(stage5_keys, address10)

    truth_keys = []
    with TRUTH_FILE.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            global_index = id_to_global.get(row["source1_entity_id"])
            if global_index is None:
                continue
            for target_id in parse_match_ids(row["matched_entity_ids"]):
                source, number = target_id.split("-", 1)
                truth_keys.append(pack_pair(global_index, {"S2": 0, "S3": 1}[source], int(number)))
    truth_keys = np.asarray(truth_keys, dtype=np.uint64)
    hit5 = exact_membership(best5, truth_keys)
    hit10 = exact_membership(best10, truth_keys)
    if len(truth_keys) != stage6a["metrics"]["stage5_address_k5"]["true_links"]:
        raise ValueError("Truth link total differs from Stage 6A")
    if int(np.count_nonzero(hit5)) != stage6a["metrics"]["stage5_address_k5"]["true_hits"]:
        raise ValueError("K=5 hit count differs from Stage 6A")
    if int(np.count_nonzero(hit10)) != stage6a["metrics"]["stage5_address_k10"]["true_hits"]:
        raise ValueError("K=10 hit count differs from Stage 6A")
    missing_keys = truth_keys[~hit5]
    recovered_k10 = hit10[~hit5]
    name20_keys = np.sort(ranked_keys(name_top, selected, 20, "stage5"))
    recovered_name20 = exact_membership(name20_keys, missing_keys)
    global_to_id = {int(index): s1 for s1, index in zip(sample_ids, selected)}
    missed = []
    needed_targets = set()
    for key, recovers, name20_recovers in zip(missing_keys, recovered_k10, recovered_name20):
        global_index = int(key >> np.uint64(34))
        source_code = int((key >> np.uint64(32)) & np.uint64(3))
        target_id = f"S{source_code + 2}-{int(key & np.uint64(0xFFFFFFFF))}"
        s1 = global_to_id[global_index]
        missed.append({"s1_id": s1, "target_id": target_id, "country": sample_country[s1],
                       "source": f"S{source_code + 2}", "recovered_at_address_k10": bool(recovers),
                       "present_in_cached_name_top20": bool(name20_recovers)})
        needed_targets.add(target_id)
    print(f"Reconstructed {len(missed):,} misses; K=10 recovers {sum(item['recovered_at_address_k10'] for item in missed):,}", flush=True)

    s1_needed = {item["s1_id"] for item in missed}
    s1_records = {}
    with S1_FILE.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["entity_id"] in s1_needed:
                s1_records[row["entity_id"]] = Business.from_raw(row)
    target_records = {}
    for path in TARGET_FILES:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row["entity_id"] in needed_targets:
                    target_records[row["entity_id"]] = Business.from_raw(row)
    if len(s1_records) != len(s1_needed) or len(target_records) != len(needed_targets):
        raise ValueError("Could not load all missed raw records")

    distributions = {key: Counter() for key in (
        "country", "source", "name_presence", "address_presence", "name_script_relation",
        "address_script_relation", "name_token_overlap", "address_token_overlap",
        "shared_numeric_address", "shared_substantial_numeric_address", "k10_recovery", "name_top20_presence",
    )}
    country_truth = {country: item["total"] for country, item in stage6a["metrics"]["stage5_address_k5"]["country_recall"].items()}
    source_truth = {source: item["total"] for source, item in stage6a["metrics"]["stage5_address_k5"]["source_recall"].items()}
    for item in missed:
        s1 = s1_records[item["s1_id"]]
        target = target_records[item["target_id"]]
        name_overlap, name_shared = token_overlap(s1.name_norm, target.name_norm)
        address_overlap, address_shared = token_overlap(s1.address_norm, target.address_norm)
        s1_name_script, target_name_script = script_of(s1.name_norm), script_of(target.name_norm)
        s1_address_script, target_address_script = script_of(s1.address_norm), script_of(target.address_norm)

        def relation(first: str, second: str) -> str:
            if "unknown" in {first, second} or "mixed" in {first, second}:
                return "uncertain"
            return "same" if first == second else "different"

        name_relation = relation(s1_name_script, target_name_script)
        address_relation = relation(s1_address_script, target_address_script)
        numeric1 = set(numeric_tokens(s1.address_norm))
        numeric2 = set(numeric_tokens(target.address_norm))
        shared_numeric = numeric1 & numeric2
        shared_substantial = {number for number in shared_numeric if len(number) >= 3}
        item.update({
            "s1_name": s1.business_name, "target_name": target.business_name,
            "s1_address": s1.business_address, "target_address": target.business_address,
            "s1_name_norm": s1.name_norm, "target_name_norm": target.name_norm,
            "s1_address_norm": s1.address_norm, "target_address_norm": target.address_norm,
            "name_presence": presence(s1.name_norm, target.name_norm),
            "address_presence": presence(s1.address_norm, target.address_norm),
            "s1_name_script": s1_name_script, "target_name_script": target_name_script,
            "name_script_relation": name_relation,
            "address_script_relation": address_relation,
            "name_token_jaccard": name_overlap, "name_shared_tokens": name_shared,
            "address_token_jaccard": address_overlap, "address_shared_tokens": address_shared,
            "shared_numeric_address_tokens": sorted(shared_numeric),
            "shared_substantial_numeric_address_tokens": sorted(shared_substantial),
        })
        distributions["country"][item["country"]] += 1
        distributions["source"][item["source"]] += 1
        distributions["name_presence"][item["name_presence"]] += 1
        distributions["address_presence"][item["address_presence"]] += 1
        distributions["name_script_relation"][name_relation] += 1
        distributions["address_script_relation"][address_relation] += 1
        distributions["name_token_overlap"][overlap_bin(name_overlap)] += 1
        distributions["address_token_overlap"][overlap_bin(address_overlap)] += 1
        distributions["shared_numeric_address"]["yes" if shared_numeric else "no"] += 1
        distributions["shared_substantial_numeric_address"]["yes" if shared_substantial else "no"] += 1
        distributions["k10_recovery"]["yes" if item["recovered_at_address_k10"] else "no"] += 1
        distributions["name_top20_presence"]["yes" if item["present_in_cached_name_top20"] else "no"] += 1

    sampled = sorted(missed, key=lambda item: hashlib.sha256(
        f"{SAMPLE_SEED}:{item['s1_id']}:{item['target_id']}".encode("utf-8")
    ).digest())[:SAMPLE_SIZE]
    with SAMPLE_TSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["sample_number", "s1_id", "target_id", "country", "source", "recovered_at_address_k10", "present_in_cached_name_top20",
                         "name_script_relation", "name_token_jaccard", "address_token_jaccard",
                         "shared_numeric_address_tokens", "s1_name", "target_name", "s1_address", "target_address"])
        for number, item in enumerate(sampled, 1):
            writer.writerow([number, item["s1_id"], item["target_id"], item["country"], item["source"],
                             item["recovered_at_address_k10"], item["present_in_cached_name_top20"], item["name_script_relation"],
                             item["name_token_jaccard"], item["address_token_jaccard"],
                             ",".join(item["shared_numeric_address_tokens"]),
                             item["s1_name"], item["target_name"], item["s1_address"], item["target_address"]])
    result = {
        "subset_s1": len(sample_ids), "true_links": len(truth_keys), "missed_links": len(missed),
        "miss_rate": len(missed) / len(truth_keys), "k10_recovered": int(np.count_nonzero(recovered_k10)),
        "name_top20_present": int(np.count_nonzero(recovered_name20)),
        "k10_recovered_by_country": dict(Counter(item["country"] for item in missed if item["recovered_at_address_k10"])),
        "k10_recovered_by_script": dict(Counter(item["name_script_relation"] for item in missed if item["recovered_at_address_k10"])),
        "name_top20_by_script": dict(Counter(item["name_script_relation"] for item in missed if item["present_in_cached_name_top20"])),
        "different_script_by_country": dict(Counter(item["country"] for item in missed if item["name_script_relation"] == "different")),
        "target_address_missing_by_country": dict(Counter(item["country"] for item in missed if item["address_presence"] == "S1 only")),
        "country_truth_links": country_truth, "source_truth_links": source_truth,
        "distributions": {name: dict(counter) for name, counter in distributions.items()},
        "sample_seed": SAMPLE_SEED, "sample_size": len(sampled),
        "sample": sampled, "analysis_seconds": time.perf_counter() - start,
    }
    SUMMARY.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {SUMMARY} and {SAMPLE_TSV} in {result['analysis_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
