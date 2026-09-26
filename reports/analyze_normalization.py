"""Reproducible sample audit for the conservative normalizer."""

from __future__ import annotations

import csv
import hashlib
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from src.normalize import normalize_record, normalize_text


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "normalization_audit.md"
SEED = 2026
THRESHOLD = (1 << 64) // 20  # 5% of IDs
FILES = [ROOT / "dataset" / "train" / f"train_source{i}.tsv" for i in (1, 2, 3)]


def selected(entity_id: str) -> bool:
    digest = hashlib.sha256(f"{SEED}:{entity_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") < THRESHOLD


def before_punctuation(value: str) -> str:
    """Intermediate NFKC + casefold + whitespace form for collision diagnosis."""
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def md(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", "<br>").replace("\r", "")


def percent(n: int, total: int) -> str:
    return f"{100 * n / total:.3f}%" if total else "—"


def main() -> None:
    scanned = Counter()
    sample = Counter()
    raw_counts = Counter()
    norm_to_raws = defaultdict(set)
    norm_to_intermediate = defaultdict(set)
    example_conditions = {
        "US apostrophe": lambda row: row["country"] == "US" and "'" in row["business_name"],
        "US punctuated name": lambda row: row["country"] == "US" and any(c in row["business_name"] for c in ".-/&"),
        "Indian script": lambda row: row["country"] == "India" and re.search(r"[\u0900-\u097f]", row["business_name"]),
        "Indian numeric address": lambda row: row["country"] == "India" and "/" in row["business_address"] and any(c.isdigit() for c in row["business_address"]),
        "Repeated whitespace": lambda row: bool(re.search(r"\s{2,}", row["business_name"] + " " + row["business_address"])),
        "Raw encoding artifact": lambda row: "â\x80" in row["business_name"] + row["business_address"],
    }
    examples = {}
    for path in FILES:
        print(f"Scanning {path.name}", flush=True)
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                scanned[path.name] += 1
                if not selected(row["entity_id"]):
                    continue
                sample[path.name] += 1
                raw = row["business_name"] or ""
                norm = normalize_text(raw)
                raw_counts[raw] += 1
                norm_to_raws[norm].add(raw)
                norm_to_intermediate[norm].add(before_punctuation(raw))
                for label, condition in example_conditions.items():
                    if label not in examples and condition(row):
                        examples[label] = normalize_record(row)
        print(f"  selected {sample[path.name]:,} / {scanned[path.name]:,}", flush=True)

    collision_groups = {norm: raws for norm, raws in norm_to_raws.items() if len(raws) > 1}
    punctuation_groups = {norm: raws for norm, raws in collision_groups.items() if len(norm_to_intermediate[norm]) > 1}
    colliding_raw_count = sum(len(raws) for raws in collision_groups.values())
    colliding_rows = sum(sum(raw_counts[raw] for raw in raws) for raws in collision_groups.values())
    punctuation_rows = sum(sum(raw_counts[raw] for raw in raws) for raws in punctuation_groups.values())
    unique_raw = len(raw_counts)
    unique_norm = len(norm_to_raws)
    sample_total = sum(sample.values())

    lines = ["# Normalization audit", "", "## Sampling method", "", f"Training source files only. An S1/S2/S3 record is selected when the first eight bytes of SHA-256(`{SEED}:<entity_id>`) as an unsigned big-endian integer are below {THRESHOLD} (approximately 5%). This rule is fixed, independent of row order, and reproducible with `reports/analyze_normalization.py`. Source files were read as UTF-8 tab-separated text and were not modified.", "", "| File | Scanned rows | Selected rows | Selected share |", "| --- | ---: | ---: | ---: |"]
    for path in FILES:
        name = path.name
        lines.append(f"| `{name}` | {scanned[name]:,} | {sample[name]:,} | {percent(sample[name], scanned[name])} |")
    lines += ["", "## Transformations", "", "The baseline applies NFKC, Unicode casefolding, punctuation and whitespace separation, and whitespace collapse. It preserves letters from all scripts, accents, numeric content, symbols outside Unicode punctuation, legal suffixes, and address abbreviations. Tokens retain order and repetitions; numeric tokens are digit runs, including digits inside alphanumeric tokens. Every derived field is added alongside the original field.", "", "The added fields are `name_norm`, `address_norm`, `name_tokens`, `address_tokens`, `name_numeric_tokens`, and `address_numeric_tokens`. Null, empty, and pandas-style missing values produce empty strings and empty token lists.", "", "## Name collision audit", "", "A collision group is a normalized name with at least two **different exact raw business names** in this sample. Repeated records with the same raw name alone do not count as a normalization collision. A collision means text was merged by the representation, not that the underlying businesses are the same.", "", "| Measure | Value |", "| --- | ---: |", f"| Sampled records | {sample_total:,} |", f"| Distinct raw business names | {unique_raw:,} |", f"| Distinct normalized business names | {unique_norm:,} |", f"| Loss of distinct name strings | {unique_raw - unique_norm:,} ({percent(unique_raw - unique_norm, unique_raw)} of distinct raw names) |", f"| Collision groups | {len(collision_groups):,} ({percent(len(collision_groups), unique_norm)} of normalized names) |", f"| Distinct raw names in collision groups | {colliding_raw_count:,} ({percent(colliding_raw_count, unique_raw)} of distinct raw names) |", f"| Sampled records in collision groups | {colliding_rows:,} ({percent(colliding_rows, sample_total)} of sampled records) |", f"| Groups requiring punctuation changes to merge | {len(punctuation_groups):,} |", f"| Sampled records in punctuation-induced groups | {punctuation_rows:,} ({percent(punctuation_rows, sample_total)} of sampled records) |", "", "The last two measures compare against an intermediate form with only NFKC, casefolding, and whitespace collapse. They isolate groups where punctuation separation creates additional merging; the remaining collision groups arise from case, compatibility, or whitespace changes.", "", "## Sampled raw → normalized records", ""]
    for label, record in examples.items():
        lines += [f"### {label} — `{record['entity_id']}` ({record['country']})", "", f"- Name: `{md(record['business_name'])}` → `{md(record['name_norm'])}`", f"- Address: `{md(record['business_address'])}` → `{md(record['address_norm'])}`", f"- Name tokens: `{md(record['name_tokens'])}`; numeric tokens: `{md(record['name_numeric_tokens'])}`", f"- Address tokens: `{md(record['address_tokens'])}`; numeric tokens: `{md(record['address_numeric_tokens'])}`", ""]
    lines += ["## Example collisions of different raw names", "", "| Normalized name | Distinct raw names in group | Sample raw variants |", "| --- | ---: | --- |"]
    chosen = sorted(punctuation_groups.items(), key=lambda item: (-len(item[1]), -sum(raw_counts[name] for name in item[1]), item[0]))[:8]
    for norm, raws in chosen:
        names = sorted(raws)[:4]
        lines.append(f"| `{md(norm)}` | {len(raws)} | " + "; ".join(f"`{md(name)}`" for name in names) + " |")
    lines += ["", "## Investigated transformations not applied", "", "- **Transliteration / accent stripping:** French accents and Indian scripts are retained. Removing accents can collapse different names, and transliteration can create ambiguous Latin spellings. NFKC itself can still merge compatibility characters.", "- **Legal suffix removal or standardization:** Terms such as `Inc`, `LLC`, `Pvt`, `Ltd`, and `SARL` remain in the text. Their interpretation varies by country and context.", "- **Address abbreviation standardization:** `St`, `Rd`, `rue`, and other local terms remain as written; `St` can mean street or saint. No US/India-specific rule is used.", "", "## Risks observed", "", "Case, compatibility, whitespace, and punctuation normalization can make distinct raw names identical. In particular, punctuation separation can remove potentially meaningful apostrophes, hyphens, ampersands, or periods. The raw name and address remain available to disambiguate these cases. Training data has US and India only; accented French behavior is covered by unit tests, but this audit cannot measure collision rates in France.", ""]
    lines.insert(-1, "One sampled address already contains apparent mojibake (`â\\x80\\x99`). The normalizer preserves this raw artifact and does not attempt to repair it.")
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"sample={sample_total:,} raw_unique={unique_raw:,} normalized_unique={unique_norm:,} collision_groups={len(collision_groups):,} collision_rows={colliding_rows:,} punctuation_groups={len(punctuation_groups):,}", flush=True)
    print(f"Wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
