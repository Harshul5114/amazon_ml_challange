"""Memory-bounded cross-check of the two complete submission TSV files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from itertools import zip_longest


def validate(matching: Path, candidate: Path, test_s1: Path) -> dict:
    required = set()
    with test_s1.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            sid = row["entity_id"]
            if not sid or sid in required:
                raise ValueError(f"Invalid or repeated test S1 ID: {sid}")
            required.add(sid)

    seen = set()
    candidate_links = predicted_links = empty_predictions = france_rows = 0
    with matching.open("r", encoding="utf-8", newline="") as m, candidate.open("r", encoding="utf-8", newline="") as c:
        if m.readline() != "source1_entity_id\tmatched_entity_ids\n":
            raise ValueError("Invalid matching header")
        if c.readline() != "source1_entity_id\tcandidate_entity_ids\n":
            raise ValueError("Invalid candidate header")
        for line_number, (match_line, candidate_line) in enumerate(zip_longest(m, c), start=2):
            if match_line is None or candidate_line is None:
                raise ValueError("Matching and candidate files have different row counts")
            mid, tab_m, match_text = match_line.rstrip("\r\n").partition("\t")
            cid, tab_c, candidate_text = candidate_line.rstrip("\r\n").partition("\t")
            if not tab_m or not tab_c or mid != cid:
                raise ValueError(f"Bad or misaligned TSV row {line_number}")
            if mid not in required or mid in seen:
                raise ValueError(f"Unknown or repeated S1 ID on row {line_number}: {mid}")
            seen.add(mid)
            candidates = candidate_text.split(",") if candidate_text else []
            matches = match_text.split(",") if match_text else []
            if len(candidates) != len(set(candidates)) or len(matches) != len(set(matches)):
                raise ValueError(f"Repeated target ID on row {line_number}")
            if any(not target.startswith(("S2-", "S3-")) for target in candidates + matches):
                raise ValueError(f"Wrong target prefix on row {line_number}")
            if not set(matches).issubset(candidates):
                raise ValueError(f"Prediction missing from candidates on row {line_number}")
            candidate_links += len(candidates)
            predicted_links += len(matches)
            empty_predictions += not matches
    if seen != required:
        raise ValueError(f"Missing {len(required - seen)} test S1 rows")
    return {"source1_rows": len(seen), "candidate_links": candidate_links,
            "predicted_links": predicted_links, "empty_predictions": empty_predictions}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matching", type=Path, default=Path("output/matching_results.tsv"))
    parser.add_argument("--candidate", type=Path, default=Path("output/candidate_pairs.tsv"))
    parser.add_argument("--test-s1", type=Path, default=Path("dataset/test/test_source1.tsv"))
    args = parser.parse_args()
    print(json.dumps(validate(args.matching, args.candidate, args.test_s1), indent=2))


if __name__ == "__main__":
    main()
