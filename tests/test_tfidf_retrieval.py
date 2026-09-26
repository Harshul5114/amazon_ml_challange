import unittest

from src.blocking import Business
from src.tfidf_retrieval import TfidfNameRetriever


def business(entity_id, name, country):
    return Business.from_raw({
        "entity_id": entity_id,
        "business_name": name,
        "business_address": "",
        "country": country,
    })


class TfidfRetrieverTests(unittest.TestCase):
    def test_unseen_country_and_missing_country(self):
        queries = [business("S1-1", "Café Paris", "France"),
                   business("S1-2", "Acme Boston", "US"),
                   business("S1-3", "Global Cafe", "")]
        retriever = TfidfNameRetriever(queries, top_k=2, min_df=1, max_df=1.0, score_floor=0.01)
        retriever._process_batch("S2", "France", ["café paris"], [1])
        self.assertGreater(retriever.source_scores["S2"][0, 0], 0)
        self.assertEqual(retriever.source_scores["S2"][1, 0], -1)
        retriever._process_batch("S2", "", ["acme boston"], [2])
        self.assertEqual(retriever.source_ids["S2"][1, 0], 2)

    def test_global_top_k_merges_batches(self):
        queries = [business("S1-1", "Alpha Consulting", "US")]
        retriever = TfidfNameRetriever(queries, top_k=2, min_df=1, max_df=1.0, score_floor=0.01)
        retriever._process_batch("S3", "US", ["alpha consult"], [10])
        retriever._process_batch("S3", "US", ["alpha consulting"], [11])
        self.assertEqual(retriever.source_ids["S3"][0, 0], 11)
        self.assertEqual(retriever.source_ids["S3"][0, 1], 10)


if __name__ == "__main__":
    unittest.main()
