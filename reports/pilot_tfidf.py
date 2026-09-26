"""Small runtime pilot for sparse character TF-IDF retrieval."""

import csv
import time
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse_dot_topn import sp_matmul_topn

from src.evaluate import read_split_ids
from src.normalize import normalize_text


ROOT = Path(__file__).resolve().parents[1]
SPLIT = ROOT / "splits" / "s1_train_validation.tsv"


def main() -> None:
    start = time.perf_counter()
    val_ids = read_split_ids(SPLIT)
    queries = []
    with (ROOT / "dataset/train/train_source1.tsv").open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["entity_id"] in val_ids and row["country"] == "US":
                queries.append(normalize_text(row["business_name"]))
                if len(queries) == 100_000:
                    break
    targets = []
    with (ROOT / "dataset/train/train_source2.tsv").open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["country"] == "US":
                targets.append(normalize_text(row["business_name"]))
                if len(targets) == 100_000:
                    break
    print(f"loaded q={len(queries)} t={len(targets)} seconds={time.perf_counter()-start:.1f}", flush=True)
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(4, 5), min_df=2, max_df=0.02,
                                  sublinear_tf=True, dtype=np.float32)
    q = vectorizer.fit_transform(queries)
    t = vectorizer.transform(targets)
    print(f"vectorized vocab={len(vectorizer.vocabulary_)} q_nnz={q.nnz} t_nnz={t.nnz} seconds={time.perf_counter()-start:.1f}", flush=True)
    sims = sp_matmul_topn(q, t.T, top_n=20, threshold=0.15, sort=True, n_threads=4)
    print(f"scored nnz={sims.nnz} seconds={time.perf_counter()-start:.1f}", flush=True)


if __name__ == "__main__":
    main()
