"""Generate a conservative deterministic submission from exact normalized records.

This fallback uses only competition test records. A target is linked only when its
country, normalized name, and normalized address identify exactly one S1 record.
"""

from __future__ import annotations

import time
from collections import defaultdict
from pathlib import Path

from src.normalize import normalize_text


ROOT = Path(__file__).resolve().parents[1]
TEST = ROOT / "dataset" / "test"
OUTPUT = ROOT / "output"


def rows(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = handle.readline().rstrip("\r\n").split("\t")
        indices = [header.index(name) for name in
                   ("entity_id", "business_name", "business_address", "country")]
        limit = max(indices)
        for line in handle:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) <= limit:
                raise ValueError(f"Malformed TSV row in {path}")
            yield tuple(parts[index] for index in indices)


def key(name: str, address: str, country: str) -> tuple[str, str, str] | None:
    normalized_name = normalize_text(name)
    normalized_address = normalize_text(address)
    if not normalized_name or not normalized_address:
        return None
    return country, normalized_name, normalized_address


def main() -> None:
    start = time.perf_counter()
    OUTPUT.mkdir(exist_ok=True)
    s1_ids: list[str] = []
    unique_keys: dict[tuple[str, str, str], int] = {}
    ambiguous = set()
    for entity_id, name, address, country in rows(TEST / "test_source1.tsv"):
        index = len(s1_ids)
        s1_ids.append(entity_id)
        record_key = key(name, address, country)
        if record_key is None or record_key in ambiguous:
            continue
        if record_key in unique_keys:
            del unique_keys[record_key]
            ambiguous.add(record_key)
        else:
            unique_keys[record_key] = index
    print(f"Indexed {len(s1_ids):,} S1 rows; {len(unique_keys):,} unique exact keys", flush=True)

    matches: dict[int, list[str]] = defaultdict(list)
    for source in (2, 3):
        count = 0
        matched = 0
        for target_id, name, address, country in rows(TEST / f"test_source{source}.tsv"):
            record_key = key(name, address, country)
            if record_key is not None:
                index = unique_keys.get(record_key)
                if index is not None:
                    matches[index].append(target_id)
                    matched += 1
            count += 1
            if count % 1_000_000 == 0:
                print(f"S{source}: scanned {count:,}; exact links {matched:,}", flush=True)
        print(f"S{source}: finished {count:,}; exact links {matched:,}", flush=True)

    matching_tmp = OUTPUT / "matching_results.tmp.tsv"
    candidate_tmp = OUTPUT / "candidate_pairs.tmp.tsv"
    with matching_tmp.open("w", encoding="utf-8", newline="") as matching, \
         candidate_tmp.open("w", encoding="utf-8", newline="") as candidate:
        matching.write("source1_entity_id\tmatched_entity_ids\n")
        candidate.write("source1_entity_id\tcandidate_entity_ids\n")
        for index, entity_id in enumerate(s1_ids):
            ids = matches.get(index, ())
            value = ",".join(ids)
            matching.write(f"{entity_id}\t{value}\n")
            candidate.write(f"{entity_id}\t{value}\n")
    matching_tmp.replace(OUTPUT / "matching_results.tsv")
    candidate_tmp.replace(OUTPUT / "candidate_pairs.tsv")
    print(f"Wrote {len(s1_ids):,} S1 rows and {sum(map(len, matches.values())):,} links "
          f"in {time.perf_counter() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
