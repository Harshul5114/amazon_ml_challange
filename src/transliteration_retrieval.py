"""Targeted transliterated-name channel over non-Latin target names."""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

from src.normalize import normalize_text
from src.tfidf_retrieval import TfidfNameRetriever
from src.transliterate import contains_non_latin_letter, name_translit, predominantly_non_latin


class TransliteratedNameRetriever(TfidfNameRetriever):
    """Compare Latin S1 names with an added ASCII form of non-Latin targets."""

    def __init__(self, *args, truth_target_ids: set[str] | None = None,
                 require_predominant_non_latin: bool = True, **kwargs) -> None:
        super().__init__(*args, text_field="name", **kwargs)
        self.truth_target_ids = truth_target_ids or set()
        self.require_predominant_non_latin = require_predominant_non_latin
        self.truth_target_names: dict[str, str] = {}
        self.source_scanned = defaultdict(int)
        self.source_selected = defaultdict(int)
        self.empty_transliterations = 0
        self.translit_record_counts = Counter()
        self.first_original: dict[str, str] = {}
        self.colliding_originals: dict[str, set[str]] = {}

    def _audit_collision(self, original: str, converted: str) -> None:
        self.translit_record_counts[converted] += 1
        first = self.first_original.setdefault(converted, original)
        if first != original:
            self.colliding_originals.setdefault(converted, {first}).add(original)

    def scan_transliterated_source(self, path: str | Path) -> None:
        path = Path(path)
        source = path.stem.split("_")[-1].upper().replace("SOURCE", "S")
        buffers = defaultdict(lambda: ([], []))
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                self.source_scanned[source] += 1
                raw_name = row["business_name"] or ""
                if not contains_non_latin_letter(raw_name):
                    continue
                if self.require_predominant_non_latin and not predominantly_non_latin(raw_name):
                    continue
                original = normalize_text(raw_name)
                converted = name_translit(original)
                if row["entity_id"] in self.truth_target_ids:
                    self.truth_target_names[row["entity_id"]] = original
                self.source_selected[source] += 1
                self._audit_collision(original, converted)
                if not converted:
                    self.empty_transliterations += 1
                    continue
                prefix, number = row["entity_id"].split("-", 1)
                if prefix != source:
                    raise ValueError(f"Unexpected target ID prefix: {row['entity_id']}")
                numeric = int(number)
                if not 0 <= numeric < 2**32:
                    raise ValueError(f"Target ID exceeds uint32 range: {row['entity_id']}")
                country = row["country"] or ""
                texts, ids = buffers[country]
                texts.append(converted)
                ids.append(numeric)
                if len(texts) >= self.batch_size:
                    self._process_batch(source, country, texts, ids)
                    texts.clear()
                    ids.clear()
                    print(f"{path.name}: {self.source_scanned[source]:,} scanned, {self.source_selected[source]:,} transliterated", flush=True)
        for country, (texts, ids) in buffers.items():
            if texts:
                self._process_batch(source, country, texts, ids)
        print(f"{path.name}: finished {self.source_scanned[source]:,} scanned, {self.source_selected[source]:,} transliterated", flush=True)

    def collision_summary(self) -> dict:
        colliding = self.colliding_originals
        samples = sorted(colliding, key=lambda text: (-len(colliding[text]), -self.translit_record_counts[text], text))[:12]
        return {
            "distinct_transliterations": len(self.first_original),
            "collision_groups": len(colliding),
            "records_in_collision_groups": sum(self.translit_record_counts[text] for text in colliding),
            "distinct_original_names_in_collision_groups": sum(len(names) for names in colliding.values()),
            "empty_transliterations": self.empty_transliterations,
            "examples": [
                {"name_translit": text, "distinct_name_norm": len(colliding[text]),
                 "record_count": self.translit_record_counts[text],
                 "name_norm_examples": sorted(colliding[text])[:6]}
                for text in samples
            ],
        }
