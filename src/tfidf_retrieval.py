"""Chunked, country-aware character TF-IDF name retrieval.

Sparse cosine products are capped at top K per query and target chunk; those
local winners are merged to obtain global top K per S1 and target source.
"""

from __future__ import annotations

import csv
import time
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from src.blocking import Business
from src.normalize import normalize_text


class TfidfNameRetriever:
    def __init__(
        self,
        s1_records: Sequence[Business],
        *,
        top_k: int = 20,
        batch_size: int = 100_000,
        n_threads: int = 4,
        min_df: int = 2,
        max_df: float = 0.02,
        score_floor: float = 0.05,
        text_field: str = "name",
    ) -> None:
        if text_field not in {"name", "address"}:
            raise ValueError(f"Unsupported text field: {text_field}")
        self.text_field = text_field
        self.top_k = top_k
        self.batch_size = batch_size
        self.n_threads = n_threads
        self.score_floor = score_floor
        self.s1_records = s1_records
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            ngram_range=(4, 5),
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.query_matrix = self.vectorizer.fit_transform(
            getattr(record, f"{text_field}_norm") for record in s1_records
        )
        self.query_country_indices: dict[str, np.ndarray] = {}
        raw_indices = defaultdict(list)
        for index, record in enumerate(s1_records):
            raw_indices[record.country].append(index)
        self.missing_query_indices = np.asarray(raw_indices.get("", []), dtype=np.int32)
        for country, indices in raw_indices.items():
            if country:
                combined = indices + raw_indices.get("", [])
                self.query_country_indices[country] = np.asarray(combined, dtype=np.int32)
        self.all_indices = np.arange(len(s1_records), dtype=np.int32)
        self.source_scores: dict[str, np.ndarray] = {}
        self.source_ids: dict[str, np.ndarray] = {}
        self.source_processed = defaultdict(int)
        self.product_seconds = 0.0
        self.vectorization_seconds = 0.0

    def _query_indices(self, target_country: str) -> np.ndarray:
        if not target_country:
            return self.all_indices
        return self.query_country_indices.get(target_country, self.missing_query_indices)

    def _source_arrays(self, source: str) -> tuple[np.ndarray, np.ndarray]:
        if source not in self.source_scores:
            shape = (len(self.s1_records), self.top_k)
            self.source_scores[source] = np.full(shape, -1.0, dtype=np.float32)
            self.source_ids[source] = np.zeros(shape, dtype=np.uint32)
        return self.source_scores[source], self.source_ids[source]

    def _process_batch(self, source: str, country: str, names: list[str], ids: list[int]) -> None:
        query_indices = self._query_indices(country)
        self.source_processed[source] += len(names)
        if not len(query_indices):
            return
        vector_start = time.perf_counter()
        target_matrix = self.vectorizer.transform(names)
        self.vectorization_seconds += time.perf_counter() - vector_start
        product_start = time.perf_counter()
        similarities = sp_matmul_topn(
            self.query_matrix[query_indices],
            target_matrix.T,
            top_n=self.top_k,
            threshold=self.score_floor,
            sort=True,
            n_threads=self.n_threads,
        )
        self.product_seconds += time.perf_counter() - product_start
        if similarities.nnz == 0:
            return
        current_scores, current_ids = self._source_arrays(source)
        rows_count = len(query_indices)
        batch_scores = np.full((rows_count, self.top_k), -1.0, dtype=np.float32)
        batch_ids = np.zeros((rows_count, self.top_k), dtype=np.uint32)
        nonzeros_per_row = np.diff(similarities.indptr)
        row_numbers = np.repeat(np.arange(rows_count, dtype=np.int32), nonzeros_per_row)
        within_row = np.arange(similarities.nnz, dtype=np.int32) - np.repeat(
            similarities.indptr[:-1], nonzeros_per_row
        )
        batch_scores[row_numbers, within_row] = similarities.data
        batch_ids[row_numbers, within_row] = np.asarray(ids, dtype=np.uint32)[similarities.indices]
        merged_scores = np.concatenate((current_scores[query_indices], batch_scores), axis=1)
        merged_ids = np.concatenate((current_ids[query_indices], batch_ids), axis=1)
        order = np.argsort(-merged_scores, axis=1, kind="stable")[:, :self.top_k]
        current_scores[query_indices] = np.take_along_axis(merged_scores, order, axis=1)
        current_ids[query_indices] = np.take_along_axis(merged_ids, order, axis=1)

    def scan_source(self, path: str | Path) -> None:
        path = Path(path)
        source = path.stem.split("_")[-1].upper().replace("SOURCE", "S")
        buffers = defaultdict(lambda: ([], []))
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                country = row["country"] or ""
                target_id = row["entity_id"]
                prefix, numeric = target_id.split("-", 1)
                if prefix != source:
                    raise ValueError(f"Unexpected target ID prefix in {path}: {target_id}")
                numeric_id = int(numeric)
                if not 0 <= numeric_id < 2**32:
                    raise ValueError(f"Target ID exceeds uint32 range: {target_id}")
                names, ids = buffers[country]
                names.append(normalize_text(row[f"business_{self.text_field}"]))
                ids.append(numeric_id)
                if len(names) >= self.batch_size:
                    self._process_batch(source, country, names, ids)
                    names.clear()
                    ids.clear()
                    print(f"{path.name}: {self.source_processed[source]:,} processed", flush=True)
        for country, (names, ids) in buffers.items():
            if names:
                self._process_batch(source, country, names, ids)
                print(f"{path.name}: {self.source_processed[source]:,} processed", flush=True)
