"""Read-only, streaming profile of the challenge TSV files."""

import csv
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FILES = sorted((ROOT / "dataset").rglob("*.tsv"))
OUT = ROOT / "reports" / "data_profile.md"


def fmt(n):
    return f"{n:,}"


def pct(n, total):
    return f"{100 * n / total:.2f}%" if total else "—"


def cell(value):
    return str(value).replace("|", "\\|").replace("\n", "<br>").replace("\r", "") or "*(empty)*"


def quantile(hist, q):
    total = sum(hist.values())
    if not total:
        return None
    target = 1 + round((total - 1) * q)
    cumulative = 0
    for value, count in sorted(hist.items()):
        cumulative += count
        if cumulative >= target:
            return value


profiles = []
ground = None
for path in FILES:
    relative = path.relative_to(ROOT).as_posix()
    is_ground = path.name == "train_ground_truth.tsv"
    print(f"Profiling {relative}", flush=True)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        columns = reader.fieldnames
        missing = Counter()
        null_tokens = Counter()
        countries = Counter()
        name_lengths = Counter()
        address_lengths = Counter()
        seen_ids = set()
        duplicate_ids = 0
        rows = 0
        samples = []
        malformed = 0
        if is_ground:
            match_counts = Counter()
            match_sources = Counter()
            max_matches = 0
            max_match_s1 = ""
            target_owner = {}
            multi_owner_ids = set()
            repeated_within_list = 0
            bad_prefixes = Counter()
        for row in reader:
            rows += 1
            if len(samples) < 5:
                samples.append(row.copy())
            if None in row:
                malformed += 1
            for col in columns:
                val = row.get(col)
                if val is None or val == "":
                    missing[col] += 1
                elif val.strip().upper() in {"NULL", "N/A", "NAN", "NONE", "<NA>"}:
                    null_tokens[col] += 1
            id_col = "source1_entity_id" if is_ground else "entity_id"
            eid = row.get(id_col) or ""
            if eid:
                if eid in seen_ids:
                    duplicate_ids += 1
                else:
                    seen_ids.add(eid)
            if is_ground:
                targets = [s.strip() for s in (row.get("matched_entity_ids") or "").split(",") if s.strip()]
                n = len(targets)
                match_counts[n] += 1
                if n > max_matches:
                    max_matches, max_match_s1 = n, eid
                sources = {target.split("-", 1)[0] for target in targets}
                if not sources:
                    match_sources["zero"] += 1
                elif sources == {"S2"}:
                    match_sources["S2 only"] += 1
                elif sources == {"S3"}:
                    match_sources["S3 only"] += 1
                elif sources == {"S2", "S3"}:
                    match_sources["both S2 and S3"] += 1
                else:
                    match_sources["other prefix"] += 1
                repeated_within_list += n - len(set(targets))
                for target in set(targets):
                    prefix = target.split("-", 1)[0]
                    if prefix not in {"S2", "S3"}:
                        bad_prefixes[prefix] += 1
                    owner = target_owner.setdefault(target, eid)
                    if owner != eid:
                        multi_owner_ids.add(target)
            else:
                countries[row.get("country") or "(empty)"] += 1
                for col, hist in (("business_name", name_lengths), ("business_address", address_lengths)):
                    value = row.get(col) or ""
                    if value:
                        hist[len(value)] += 1
        profile = dict(path=relative, size=path.stat().st_size, rows=rows,
                       columns=columns, missing=missing, null_tokens=null_tokens,
                       unique_ids=len(seen_ids), duplicate_ids=duplicate_ids,
                       countries=countries, name_lengths=name_lengths,
                       address_lengths=address_lengths, samples=samples,
                       malformed=malformed)
        if is_ground:
            profile.update(match_counts=match_counts, match_sources=match_sources,
                           max_matches=max_matches, max_match_s1=max_match_s1,
                           unique_target_ids=len(target_owner),
                           multi_owner_ids=len(multi_owner_ids),
                           multi_owner_by_source=Counter(x.split("-", 1)[0] for x in multi_owner_ids),
                           repeated_within_list=repeated_within_list,
                           bad_prefixes=bad_prefixes)
            ground = profile
        profiles.append(profile)


lines = ["# Dataset profile", "", "All seven TSVs were read with tab separation as UTF-8 text. Counts below use the raw field values; an empty field is counted as missing. Common literal null markers (NULL, N/A, NAN, NONE, <NA>) are counted separately if present. Lengths count Unicode characters in nonempty fields, including spaces and punctuation. Percentages use the file's row count unless stated otherwise.", "", "## File inventory", "", "| File | Bytes | MiB | Rows | Unique entity IDs | Duplicate ID rows |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
for p in profiles:
    lines.append(f"| `{p['path']}` | {fmt(p['size'])} | {p['size']/1048576:,.2f} | {fmt(p['rows'])} | {fmt(p['unique_ids'])} | {fmt(p['duplicate_ids'])} |")
lines += ["", "Duplicate ID rows count every occurrence after the first within that file. Unique IDs exclude empty IDs.", "", "## Columns, dtypes and missing values", "", "All populated fields are textual; pandas infers `object` for these columns when reading normally. Empty `matched_entity_ids` fields become missing values under pandas' default NA parsing.", "", "| File | Column | Dtype | Empty / missing | Literal null markers |", "| --- | --- | --- | ---: | ---: |"]
for p in profiles:
    for col in p["columns"]:
        lines.append(f"| `{Path(p['path']).name}` | `{col}` | object (text) | {fmt(p['missing'][col])} ({pct(p['missing'][col], p['rows'])}) | {fmt(p['null_tokens'][col])} |")
lines += ["", "## Five sample rows per file", ""]
for p in profiles:
    lines += [f"### `{p['path']}`", "", "| " + " | ".join(p["columns"]) + " |", "| " + " | ".join("---" for _ in p["columns"]) + " |"]
    for sample in p["samples"]:
        lines.append("| " + " | ".join(cell(sample.get(col) or "") for col in p["columns"]) + " |")
    lines.append("")
lines += ["## Country distribution by source", "", "| Split / source | Country | Rows | Share |", "| --- | --- | ---: | ---: |"]
for p in profiles:
    if p is ground:
        continue
    for country, count in p["countries"].most_common():
        lines.append(f"| `{Path(p['path']).stem}` | {cell(country)} | {fmt(count)} | {pct(count, p['rows'])} |")
lines += ["", "## Business name and address lengths", "", "Statistics exclude empty values. The 50th and 90th percentiles use the nearest observed rank.", "", "| File | Field | Nonempty | Mean | Min | Median | P90 | Max |", "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
for p in profiles:
    if p is ground:
        continue
    for field, hist in (("business_name", p["name_lengths"]), ("business_address", p["address_lengths"])):
        total = sum(hist.values())
        mean = sum(k * v for k, v in hist.items()) / total if total else 0
        lines.append(f"| `{Path(p['path']).stem}` | `{field}` | {fmt(total)} | {mean:.2f} | {min(hist) if hist else '—'} | {quantile(hist, .5)} | {quantile(hist, .9)} | {max(hist) if hist else '—'} |")
lines += ["", "## Training ground truth", "", f"The ground truth has {fmt(ground['rows'])} rows and {fmt(ground['unique_ids'])} unique S1 IDs. {fmt(ground['match_counts'][0])} S1 rows have zero matches ({pct(ground['match_counts'][0], ground['rows'])}). The maximum is {fmt(ground['max_matches'])} matches for `{ground['max_match_s1']}`.", "", "### Match count per S1", "", "| Matches | S1 rows | Share |", "| ---: | ---: | ---: |"]
for count, n in sorted(ground["match_counts"].items()):
    lines.append(f"| {fmt(count)} | {fmt(n)} | {pct(n, ground['rows'])} |")
lines += ["", "### Sources represented in each S1 match list", "", "| Category | S1 rows | Share of all S1 rows |", "| --- | ---: | ---: |"]
for category in ("zero", "S2 only", "S3 only", "both S2 and S3", "other prefix"):
    n = ground["match_sources"][category]
    lines.append(f"| {category} | {fmt(n)} | {pct(n, ground['rows'])} |")
lines += ["", f"Across all lists there are {fmt(ground['unique_target_ids'])} distinct S2/S3 target IDs. {fmt(ground['multi_owner_ids'])} target IDs map to more than one distinct S1 ID (S2: {fmt(ground['multi_owner_by_source']['S2'])}; S3: {fmt(ground['multi_owner_by_source']['S3'])}). Repeated IDs within a single list: {fmt(ground['repeated_within_list'])}. Unexpected target prefixes: {dict(ground['bad_prefixes']) or 'none'}.", "", "## Train versus test", "", "| Source | Train rows | Test rows | Train countries | Test countries |", "| --- | ---: | ---: | --- | --- |"]
for source in (1, 2, 3):
    tr = next(p for p in profiles if p["path"].endswith(f"train_source{source}.tsv"))
    te = next(p for p in profiles if p["path"].endswith(f"test_source{source}.tsv"))
    countries = lambda p: ", ".join(f"{k}: {pct(v, p['rows'])}" for k, v in p["countries"].most_common())
    lines.append(f"| S{source} | {fmt(tr['rows'])} | {fmt(te['rows'])} | {countries(tr)} | {countries(te)} |")
lines += ["", "Country share changes (test minus train, percentage points):", "", "| Source | Country | Change |", "| --- | --- | ---: |"]
for source in (1, 2, 3):
    tr = next(p for p in profiles if p["path"].endswith(f"train_source{source}.tsv"))
    te = next(p for p in profiles if p["path"].endswith(f"test_source{source}.tsv"))
    for country in sorted(set(tr["countries"]) | set(te["countries"])):
        delta = 100 * (te["countries"][country] / te["rows"] - tr["countries"][country] / tr["rows"])
        lines.append(f"| S{source} | {cell(country)} | {delta:+.2f} pp |")
lines += ["", "The strongest shift is country coverage: France is absent from every training source but accounts for 14.98% of test S1, 14.39% of test S2, and 14.40% of test S3. The US share falls by about 21.6–21.7 percentage points in every source, while India's share rises by about 6.7–7.3 points. Test addresses are also longer on average by 5.14 characters in S1, 3.95 in S2, and 1.76 in S3. Missing addresses in S2/S3 fall from 3.36%/3.33% in train to 2.65%/2.68% in test. These are descriptive differences; the files alone do not establish their cause.", "", "### Data integrity notes", ""]
for p in profiles:
    if p["malformed"]:
        lines.append(f"- `{p['path']}`: {fmt(p['malformed'])} rows have more fields than the header.")
if not any(p["malformed"] for p in profiles):
    lines.append("- No rows had more fields than the header in the streaming parse.")
lines.append("")
OUT.parent.mkdir(exist_ok=True)
OUT.write_text("\n".join(lines), encoding="utf-8")
print(f"Wrote {OUT}", flush=True)
