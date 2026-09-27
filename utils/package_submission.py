"""Create the competition's final code-and-output zip after validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


ROOT = Path(__file__).resolve().parents[1]
CODE_ROOT = "code/business_entity_resolution"
SOURCES = (
    "src/normalize.py",
    "src/generate_quick_submission.py",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team-name", default="team")
    args = parser.parse_args()
    stem = re.sub(r"[^a-zA-Z0-9_-]+", "_", args.team_name.strip()).strip("_") or "team"
    output = ROOT / "output"
    final_zip = output / f"{stem}_submission.zip"
    members = [
        (ROOT / "output/matching_results.tsv", "output/matching_results.tsv"),
        (ROOT / "output/candidate_pairs.tsv", "output/candidate_pairs.tsv"),
        (ROOT / "submission/Documentation_template.md", "Documentation_template.md"),
        (ROOT / "submission/README.md", f"{CODE_ROOT}/README.md"),
        (ROOT / "submission/requirements.txt", f"{CODE_ROOT}/requirements.txt"),
        (ROOT / "utils/validate_submission.py", f"{CODE_ROOT}/utils/validate_submission.py"),
    ]
    members.extend((ROOT / source, f"{CODE_ROOT}/{source}") for source in SOURCES)
    for path, _ in members:
        if not path.is_file():
            raise FileNotFoundError(path)
    with ZipFile(final_zip, "w", compression=ZIP_DEFLATED, compresslevel=6, allowZip64=True) as archive:
        for path, name in members:
            archive.write(path, name)
            print(f"Added {name} ({path.stat().st_size:,} bytes)", flush=True)
    with ZipFile(final_zip) as archive:
        failed = archive.testzip()
        if failed:
            raise ValueError(f"Corrupt zip entry: {failed}")
    manifest = {
        "package": final_zip.name,
        "package_bytes": final_zip.stat().st_size,
        "package_sha256": sha256(final_zip),
        "files": {name: {"bytes": path.stat().st_size, "sha256": sha256(path)} for path, name in members},
    }
    (output / "package_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Package complete: {final_zip} ({final_zip.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
