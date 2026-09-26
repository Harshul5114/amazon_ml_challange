"""Run sparse character TF-IDF retrieval on the fixed validation S1 set."""

from __future__ import annotations

import csv
import json
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import sklearn
import sparse_dot_topn

from src.blocking import Business
from src.evaluate import read_split_ids
from src.tfidf_retrieval import TfidfNameRetriever


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
SPLIT = ROOT / "splits" / "s1_train_validation.tsv"
SOURCE1 = ROOT / "dataset" / "train" / "train_source1.tsv"
TARGETS = [ROOT / "dataset" / "train" / f"train_source{i}.tsv" for i in (2, 3)]


def main() -> None:
    ARTIFACTS.mkdir(exist_ok=True)
    start = time.perf_counter()
    process = psutil.Process()
    stopping = threading.Event()
    peak = [process.memory_info().rss]

    def monitor() -> None:
        while not stopping.wait(0.5):
            peak[0] = max(peak[0], process.memory_info().rss)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        validation = read_split_ids(SPLIT)
        s1 = []
        with SOURCE1.open("r", encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                if row["entity_id"] in validation:
                    s1.append(Business.from_raw(row))
        if len(s1) != len(validation):
            raise ValueError("Validation S1 coverage mismatch")
        del validation
        print(f"Loaded {len(s1):,} S1 records in {time.perf_counter()-start:.1f}s", flush=True)
        fit_start = time.perf_counter()
        retriever = TfidfNameRetriever(s1, top_k=20, batch_size=100_000, n_threads=4,
                                       min_df=2, max_df=0.02, score_floor=0.05)
        fit_seconds = time.perf_counter() - fit_start
        print(f"Fitted {len(retriever.vectorizer.vocabulary_):,} character grams in {fit_seconds:.1f}s; query nnz={retriever.query_matrix.nnz:,}", flush=True)
        for path in TARGETS:
            retriever.scan_source(path)
            print(f"Finished {path.name} in {time.perf_counter()-start:.1f}s", flush=True)
        np.savez_compressed(
            ARTIFACTS / "tfidf_top20.npz",
            s1_ids=np.asarray([record.entity_id for record in s1]),
            S2_scores=retriever.source_scores["S2"],
            S2_ids=retriever.source_ids["S2"],
            S3_scores=retriever.source_scores["S3"],
            S3_ids=retriever.source_ids["S3"],
        )
        metadata = {
            "s1_count": len(s1),
            "target_counts": dict(retriever.source_processed),
            "top_k_per_source": retriever.top_k,
            "ngram_analyzer": "char_wb",
            "ngram_range": [4, 5],
            "min_df": 2,
            "max_df": 0.02,
            "sublinear_tf": True,
            "score_floor": 0.05,
            "country_partition": "same country; missing country searches all / is searched by all",
            "batch_size": retriever.batch_size,
            "n_threads": retriever.n_threads,
            "vocabulary_size": len(retriever.vectorizer.vocabulary_),
            "query_nonzeros": int(retriever.query_matrix.nnz),
            "fit_seconds": fit_seconds,
            "vectorization_seconds": retriever.vectorization_seconds,
            "sparse_product_seconds": retriever.product_seconds,
            "total_seconds": time.perf_counter() - start,
            "approx_peak_rss_bytes": peak[0],
            "sklearn_version": sklearn.__version__,
            "sparse_dot_topn_version": sparse_dot_topn.__version__,
        }
        (ARTIFACTS / "tfidf_run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"Saved top-20 lists; runtime={metadata['total_seconds']:.1f}s peak_RSS={peak[0]/(1024**3):.2f}GiB", flush=True)
    finally:
        stopping.set()
        thread.join(timeout=2)


if __name__ == "__main__":
    main()
