"""Optional offline name representation; never replaces name_norm or raw text."""

from __future__ import annotations

import unicodedata

from anyascii import anyascii

from src.normalize import normalize_text


def contains_non_latin_letter(text: str) -> bool:
    """Select names containing letters from scripts other than Latin."""
    return any(
        ord(char) > 127 and char.isalpha() and not unicodedata.name(char, "").startswith("LATIN ")
        for char in text
    )


def name_translit(name_norm: str) -> str:
    """Create a separate ASCII transliteration from a normalized name."""
    return normalize_text(anyascii(name_norm)) if name_norm else ""


def predominantly_non_latin(text: str, threshold: float = 0.8) -> bool:
    """Require most alphabetic characters to come from non-Latin scripts."""
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return False
    non_latin = sum(
        not unicodedata.name(char, "").startswith("LATIN ") for char in letters
    )
    return non_latin / len(letters) >= threshold
