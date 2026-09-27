"""Generate test candidates, score them, and assemble the challenge submission.

Run each country as a separate process to bound memory. All retrieval channels use
the existing normalization and parameters from the validation experiments.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import time
from array import array
from pathlib import Path

import joblib
import numpy as np

from src.blocking import Business, build_rules, retrieve_by_rule
from src.similarity_features import FEATURE_NAMES, compute_pair_features
from src.tfidf_retrieval import TfidfNameRetriever

ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "dataset" / "test"
ARTIFACTS = ROOT / "artifacts"
OUTPUT = ROOT / "output"
COUNTRIES = ("France", "US", "India")
THRESHOLD = 0.97
FLUSH_PAIRS = 100_000


def pack_pair(s1_index: int, source_code: int, target_number: int) -> int:
    return (s1_index << 34) | (source_code << 32) | target_number


def ranked_keys(top: dict[str, np.ndarray], k: int) -> np.ndarray:
    pieces = []
    for source_code, source in enumerate(("S2", "S3")):
        scores = top[f"{source}_scores"][:, :k]
        ids = top[f"{source}_ids"][:, :k]
        local, ranks = np.nonzero(scores >= 0)
        pieces.append((local.astype(np.uint64) << np.uint64(34)) |
                      (np.uint64(source_code) << np.uint64(32)) |
                      ids[local, ranks].astype(np.uint64))
    return np.concatenate(pieces)


def slug(country: str) -> str:
    return country.lower()


def paths(country: str) -> tuple[Path, Path, Path, Path]:
    stem = slug(country)
    return (
        ARTIFACTS / f"test_{stem}_s1_ids.npy",
        ARTIFACTS / f"test_{stem}_candidates_packed.npy",
        OUTPUT / f"candidate_pairs_{stem}.tsv",
        OUTPUT / f"matching_results_{stem}.tsv",
    )


def load_s1(country: str) -> list[Business]:
    records = []
    with (TEST / "test_source1.tsv").open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if (row["country"] or "") == country:
                records.append(Business.from_raw(row))
    if not records:
        raise ValueError(f"No S1 records for {country}")
    return records


def export_candidates(records: list[Business], keys: np.ndarray, path: Path) -> None:
    position = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_idx, record in enumerate(records):
            ids = []
            while position < len(keys) and int(keys[position] >> np.uint64(34)) == s1_idx:
                key = int(keys[position])
                ids.append(f"S{2 + ((key >> 32) & 3)}-{key & 0xFFFFFFFF}")
                position += 1
            handle.write(f"{record.entity_id}\t{','.join(ids)}\n")
    if position != len(keys):
        raise ValueError("Candidate keys include an S1 outside the country partition")


def generate_candidates(country: str, threads: int) -> None:
    start = time.perf_counter()
    ARTIFACTS.mkdir(exist_ok=True)
    OUTPUT.mkdir(exist_ok=True)
    ids_path, keys_path, candidate_path, _ = paths(country)
    records = load_s1(country)
    ids = np.asarray([record.entity_id for record in records])
    np.save(ids_path, ids)
    print(f"{country}: loaded {len(records):,} S1 records", flush=True)

    rules = build_rules(records)
    name = TfidfNameRetriever(records, text_field="name", top_k=10,
                              batch_size=50_000, n_threads=threads,
                              min_df=2, max_df=0.02, score_floor=0.05)
    address = TfidfNameRetriever(records, text_field="address", top_k=5,
                                 batch_size=50_000, n_threads=threads,
                                 min_df=2, max_df=0.02, score_floor=0.05)
    print(f"{country}: fitted name and address retrieval", flush=True)
    baseline = array("Q")
    for source_code in (0, 1):
        source = f"S{source_code + 2}"
        source_path = TEST / f"test_source{source_code + 2}.tsv"
        buffers: dict[str, tuple[list[str], list[str], list[int]]] = {}
        processed = 0
        with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = handle.readline().rstrip("\r\n").split("\t")
            id_idx = header.index("entity_id")
            name_idx = header.index("business_name")
            addr_idx = header.index("business_address")
            country_idx = header.index("country")
            required_cols = max(id_idx, name_idx, addr_idx, country_idx) + 1
            for line in handle:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < required_cols:
                    continue
                target_country = parts[country_idx] or ""
                if target_country not in (country, ""):
                    continue
                target = Business.from_raw({
                    "entity_id": parts[id_idx], "business_name": parts[name_idx],
                    "business_address": parts[addr_idx], "country": target_country,
                })
                prefix, numeric = target.entity_id.split("-", 1)
                if prefix != source:
                    raise ValueError(f"Unexpected target ID {target.entity_id}")
                number = int(numeric)
                found = retrieve_by_rule(rules, target)
                union = set().union(*found.values())
                baseline.extend(pack_pair(s1_idx, source_code, number) for s1_idx in union)
                names, addresses, numbers = buffers.setdefault(target_country, ([], [], []))
                names.append(target.name_norm)
                addresses.append(target.address_norm)
                numbers.append(number)
                if len(numbers) >= 50_000:
                    name._process_batch(source, target_country, names, numbers)
                    address._process_batch(source, target_country, addresses, numbers)
                    names.clear()
                    addresses.clear()
                    numbers.clear()
                processed += 1
                if processed % 1_000_000 == 0:
                    print(f"{country}: {source} {processed:,} targets; {len(baseline):,} baseline pairs", flush=True)
        for target_country, (names, addresses, numbers) in buffers.items():
            if numbers:
                name._process_batch(source, target_country, names, numbers)
                address._process_batch(source, target_country, addresses, numbers)
        print(f"{country}: finished {source} ({processed:,} targets)", flush=True)

    baseline_keys = np.frombuffer(baseline, dtype=np.uint64)
    name_keys = ranked_keys({
        "S2_scores": name.source_scores["S2"], "S2_ids": name.source_ids["S2"],
        "S3_scores": name.source_scores["S3"], "S3_ids": name.source_ids["S3"],
    }, k=10)
    address_keys = ranked_keys({
        "S2_scores": address.source_scores["S2"], "S2_ids": address.source_ids["S2"],
        "S3_scores": address.source_scores["S3"], "S3_ids": address.source_ids["S3"],
    }, k=5)
    keys = np.union1d(np.union1d(baseline_keys, name_keys), address_keys)
    np.save(keys_path, keys)
    print(f"{country}: {len(keys):,} union candidate pairs; exporting", flush=True)
    export_candidates(records, keys, candidate_path)
    print(f"{country}: candidates complete in {time.perf_counter() - start:.1f}s", flush=True)


def write_matching(records: list[Business], best_score: np.ndarray,
                   best_number: np.ndarray, matching_path: Path) -> None:
    with matching_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_idx, record in enumerate(records):
            matches = [
                f"S{source_code + 2}-{best_number[s1_idx, source_code]}"
                for source_code in (0, 1)
                if best_score[s1_idx, source_code] >= THRESHOLD
            ]
            handle.write(f"{record.entity_id}\t{','.join(matches)}\n")


def score_country(country: str, source_only: str | None = None) -> None:
    start = time.perf_counter()
    ids_path, keys_path, _, matching_path = paths(country)
    records = load_s1(country)
    ids = np.load(ids_path)
    if not np.array_equal(ids, np.asarray([record.entity_id for record in records])):
        raise ValueError("S1 order differs from candidate generation")
    keys = np.load(keys_path, mmap_mode="r")
    model = joblib.load(ARTIFACTS / "tree_match_model.joblib")
    best_score = np.full((len(records), 2), -1.0, dtype=np.float32)
    best_number = np.zeros((len(records), 2), dtype=np.uint32)
    conflict_idx = FEATURE_NAMES.index("numeric_conflict")
    significant_numbers = [
        frozenset(number for number in record.address_numeric_tokens if len(number) >= 3)
        for record in records
    ]

    source_codes = (0, 1) if source_only is None else (int(source_only[-1]) - 2,)
    for source_code in source_codes:
        source = f"S{source_code + 2}"
        all_sources = ((keys >> np.uint64(32)) & np.uint64(3)).astype(np.uint8)
        row_indices = np.flatnonzero(all_sources == source_code)
        del all_sources
        target_numbers = (keys[row_indices] & np.uint64(0xFFFFFFFF)).astype(np.uint32)
        order = np.argsort(target_numbers, kind="stable")
        row_indices = row_indices[order]
        target_numbers = target_numbers[order]
        unique_numbers, starts = np.unique(target_numbers, return_index=True)
        ends = np.append(starts[1:], len(row_indices))
        del order, target_numbers
        gc.collect()
        feature_rows: list[list[float]] = []
        s1_rows: list[int] = []
        candidate_numbers: list[int] = []
        scored = 0

        def flush() -> None:
            nonlocal scored
            if not feature_rows:
                return
            matrix = np.asarray(feature_rows, dtype=np.float32)
            probabilities = model.predict_proba(matrix)[:, 1]
            for s1_idx, number, probability in zip(s1_rows, candidate_numbers, probabilities):
                if probability > best_score[s1_idx, source_code]:
                    best_score[s1_idx, source_code] = probability
                    best_number[s1_idx, source_code] = number
            scored += len(feature_rows)
            feature_rows.clear()
            s1_rows.clear()
            candidate_numbers.clear()
            if scored % 1_000_000 < FLUSH_PAIRS:
                print(f"{country}: scored {source} {scored:,} eligible pairs", flush=True)

        path = TEST / f"test_source{source_code + 2}.tsv"
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            header = handle.readline().rstrip("\r\n").split("\t")
            id_idx = header.index("entity_id")
            name_idx = header.index("business_name")
            addr_idx = header.index("business_address")
            country_idx = header.index("country")
            required_cols = max(id_idx, name_idx, addr_idx, country_idx) + 1
            for line in handle:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) < required_cols:
                    continue
                target_country = parts[country_idx] or ""
                if target_country not in (country, ""):
                    continue
                number = int(parts[id_idx].split("-", 1)[1])
                position = np.searchsorted(unique_numbers, number)
                if position == len(unique_numbers) or unique_numbers[position] != number:
                    continue
                target = Business.from_raw({
                    "entity_id": parts[id_idx], "business_name": parts[name_idx],
                    "business_address": parts[addr_idx], "country": target_country,
                })
                target_significant = frozenset(
                    number for number in target.address_numeric_tokens if len(number) >= 3
                )
                for row_idx in row_indices[starts[position]:ends[position]]:
                    s1_idx = int(keys[row_idx] >> np.uint64(34))
                    s1_significant = significant_numbers[s1_idx]
                    if s1_significant and target_significant and s1_significant.isdisjoint(target_significant):
                        continue
                    features = compute_pair_features(records[s1_idx], target, source)
                    if features[conflict_idx] >= 0.5:
                        continue
                    feature_rows.append(features)
                    s1_rows.append(s1_idx)
                    candidate_numbers.append(number)
                    if len(feature_rows) >= FLUSH_PAIRS:
                        flush()
        flush()
        print(f"{country}: {source} scoring complete, {scored:,} eligible pairs", flush=True)
        del row_indices, unique_numbers, starts, ends
        gc.collect()

    if source_only is None:
        write_matching(records, best_score, best_number, matching_path)
        print(f"{country}: matches complete in {time.perf_counter() - start:.1f}s", flush=True)
    else:
        score_path = ARTIFACTS / f"test_{slug(country)}_{source_only.lower()}_scores.npz"
        source_code = source_codes[0]
        np.savez(score_path, scores=best_score[:, source_code], numbers=best_number[:, source_code])
        print(f"{country}: {source_only} scores complete in {time.perf_counter() - start:.1f}s", flush=True)


def combine_scores(country: str) -> None:
    records = load_s1(country)
    _, _, _, matching_path = paths(country)
    best_score = np.full((len(records), 2), -1.0, dtype=np.float32)
    best_number = np.zeros((len(records), 2), dtype=np.uint32)
    for source_code, source in enumerate(("S2", "S3")):
        score_path = ARTIFACTS / f"test_{slug(country)}_{source.lower()}_scores.npz"
        with np.load(score_path) as data:
            if len(data["scores"]) != len(records):
                raise ValueError(f"Wrong score length in {score_path}")
            best_score[:, source_code] = data["scores"]
            best_number[:, source_code] = data["numbers"]
    write_matching(records, best_score, best_number, matching_path)
    print(f"{country}: combined S2/S3 matches", flush=True)


def assemble() -> None:
    OUTPUT.mkdir(exist_ok=True)
    for kind, final_name in (("candidate", "candidate_pairs.tsv"), ("matching", "matching_results.tsv")):
        final_path = OUTPUT / final_name
        expected_header = "source1_entity_id\tcandidate_entity_ids\n" if kind == "candidate" else "source1_entity_id\tmatched_entity_ids\n"
        total = 0
        with final_path.open("w", encoding="utf-8", newline="") as output:
            output.write(expected_header)
            for country in COUNTRIES:
                _, _, candidate_path, matching_path = paths(country)
                part_path = candidate_path if kind == "candidate" else matching_path
                with part_path.open("r", encoding="utf-8", newline="") as part:
                    if part.readline() != expected_header:
                        raise ValueError(f"Wrong header in {part_path}")
                    for line in part:
                        output.write(line)
                        total += 1
        print(f"Assembled {final_path}: {total:,} S1 rows", flush=True)
    (OUTPUT / "submission_manifest.json").write_text(json.dumps({
        "countries": list(COUNTRIES), "threshold": THRESHOLD,
        "model": "artifacts/tree_match_model.joblib",
        "candidate_channels": ["stage4", "name_tfidf_k10", "address_tfidf_k5"],
        "numeric_conflict_gate": True,
    }, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("candidates", "score", "combine", "assemble"))
    parser.add_argument("--country", choices=COUNTRIES)
    parser.add_argument("--source", choices=("S2", "S3"), help="Score one target source for parallel execution")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if args.stage != "assemble" and args.country is None:
        parser.error("--country is required for country stages")
    if args.stage == "candidates":
        generate_candidates(args.country, args.threads)
    elif args.stage == "score":
        score_country(args.country, args.source)
    elif args.stage == "combine":
        combine_scores(args.country)
    else:
        assemble()


if __name__ == "__main__":
    main()
