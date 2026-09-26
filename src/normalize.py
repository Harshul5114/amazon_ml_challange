"""Conservative, language-neutral text representations for business records.

Raw fields are never mutated. Accents, scripts, legal suffixes, symbols, and
address abbreviations remain available in the normalized text.
"""

from __future__ import annotations

import re
import unicodedata
from math import isnan
from collections.abc import Mapping


_NUMBER_RE = re.compile(r"\d+")


def normalize_text(value: str | None) -> str:
    """Apply NFKC, casefolding, and punctuation/whitespace separation."""
    if value is None:
        return ""
    if isinstance(value, str):
        if value == "":
            return ""
    else:
        try:
            if isnan(value):
                return ""
        except (TypeError, ValueError):
            pass
        if type(value).__name__ == "NAType" and type(value).__module__.startswith("pandas"):
            return ""
        raise TypeError(f"Expected str or None, got {type(value).__name__}")
    text = unicodedata.normalize("NFKC", value).casefold()
    text = unicodedata.normalize("NFKC", text)
    text = "".join(
        " " if char.isspace() or unicodedata.category(char).startswith(("P", "Z")) else char
        for char in text
    )
    return " ".join(text.split())


def text_tokens(normalized: str) -> list[str]:
    """Keep token order and repetitions."""
    return normalized.split()


def numeric_tokens(normalized: str) -> list[str]:
    """Extract digit runs, including those inside alphanumeric tokens."""
    return _NUMBER_RE.findall(normalized)


def normalize_record(record: Mapping[str, str | None]) -> dict:
    """Return a copy with six additional fields; retain all original values."""
    result = dict(record)
    for raw_field, stem in (("business_name", "name"), ("business_address", "address")):
        normalized = normalize_text(record.get(raw_field))
        result[f"{stem}_norm"] = normalized
        result[f"{stem}_tokens"] = text_tokens(normalized)
        result[f"{stem}_numeric_tokens"] = numeric_tokens(normalized)
    return result
