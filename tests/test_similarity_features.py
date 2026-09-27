import unittest
import numpy as np

from src.blocking import Business
from src.similarity_features import (
    FEATURE_NAMES,
    compute_pair_features,
    compute_pair_features_array,
    char_ngrams,
    jaccard,
    containment,
)


def make_biz(entity_id: str, name: str, addr: str, country: str = "US") -> Business:
    return Business.from_raw({
        "entity_id": entity_id,
        "business_name": name,
        "business_address": addr,
        "country": country,
    })


class SimilarityFeaturesTest(unittest.TestCase):
    def test_feature_names_length(self):
        self.assertEqual(len(FEATURE_NAMES), 15)

    def test_char_ngrams(self):
        ng = char_ngrams("abc", 3)
        self.assertEqual(ng, {"abc"})
        ng4 = char_ngrams("abcd", 3)
        self.assertEqual(ng4, {"abc", "bcd"})
        self.assertEqual(char_ngrams("", 3), set())
        self.assertEqual(char_ngrams("ab", 3), {"ab"})

    def test_jaccard_and_containment(self):
        self.assertEqual(jaccard(set(), {"a"}), 0.0)
        self.assertEqual(containment(set(), {"a"}), 0.0)
        s1 = {"a", "b"}
        s2 = {"a", "b", "c"}
        self.assertAlmostEqual(jaccard(s1, s2), 2 / 3)
        self.assertAlmostEqual(containment(s1, s2), 1.0)

    def test_exact_match(self):
        b1 = make_biz("S1-1", "Starbucks Coffee", "123 Main St Seattle WA 98101", "US")
        b2 = make_biz("S2-1", "Starbucks Coffee", "123 Main St Seattle WA 98101", "US")
        feats = compute_pair_features(b1, b2, "S2")
        self.assertEqual(len(feats), len(FEATURE_NAMES))
        d = dict(zip(FEATURE_NAMES, feats))
        self.assertEqual(d["exact_name_match"], 1.0)
        self.assertAlmostEqual(d["name_jaccard"], 1.0)
        self.assertEqual(d["exact_address_match"], 1.0)
        self.assertAlmostEqual(d["address_jaccard"], 1.0)
        self.assertEqual(d["same_country"], 1.0)
        self.assertEqual(d["missing_s1_address"], 0.0)
        self.assertEqual(d["missing_target_address"], 0.0)
        self.assertEqual(d["target_is_s2"], 1.0)
        self.assertEqual(d["numeric_conflict"], 0.0)
        self.assertGreater(d["numeric_overlap_count"], 0)

    def test_numeric_conflict(self):
        # Different house numbers length >= 3
        b1 = make_biz("S1-1", "Acme Corp", "500 Broadway New York", "US")
        b2 = make_biz("S3-2", "Acme Corp", "700 Broadway New York", "US")
        feats = compute_pair_features(b1, b2, "S3")
        d = dict(zip(FEATURE_NAMES, feats))
        self.assertEqual(d["numeric_conflict"], 1.0)
        self.assertEqual(d["numeric_overlap_count"], 0.0)
        self.assertEqual(d["target_is_s2"], 0.0)

    def test_missing_address_edge_case(self):
        b1 = make_biz("S1-1", "Acme Corp", "", "US")
        b2 = make_biz("S2-2", "Acme Corp", None, "India")
        feats = compute_pair_features(b1, b2, "S2")
        d = dict(zip(FEATURE_NAMES, feats))
        self.assertEqual(d["missing_s1_address"], 1.0)
        self.assertEqual(d["missing_target_address"], 1.0)
        self.assertEqual(d["exact_address_match"], 0.0)
        self.assertEqual(d["address_jaccard"], 0.0)
        self.assertEqual(d["same_country"], 0.0)
        self.assertEqual(d["numeric_conflict"], 0.0)

    def test_compute_pair_features_array(self):
        b1 = make_biz("S1-1", "Alpha", "100 Street", "US")
        b2 = make_biz("S2-1", "Alpha", "100 Street", "US")
        b3 = make_biz("S3-2", "Beta", "200 Avenue", "US")
        arr = compute_pair_features_array([(b1, b2, "S2"), (b1, b3, "S3")])
        self.assertEqual(arr.shape, (2, len(FEATURE_NAMES)))
        self.assertEqual(arr.dtype, np.float32)

    def test_empty_array(self):
        arr = compute_pair_features_array([])
        self.assertEqual(arr.shape, (0, len(FEATURE_NAMES)))


if __name__ == "__main__":
    unittest.main()
