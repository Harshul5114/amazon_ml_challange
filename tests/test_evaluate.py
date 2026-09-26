import unittest
from pathlib import Path

from src.evaluate import entity_f05, macro_f05, parse_match_ids, read_match_tsv, read_split_ids


class EvaluateTests(unittest.TestCase):
    def test_perfect_prediction(self):
        self.assertEqual(macro_f05({"S1-1": {"S2-1"}}, {"S1-1": {"S2-1"}}), 1.0)

    def test_macro_average_includes_singletons(self):
        truth = {"S1-1": set(), "S1-2": {"S2-1"}}
        predictions = {"S1-1": set(), "S1-2": set()}
        self.assertEqual(macro_f05(truth, predictions), 0.5)

    def test_missed_true_match(self):
        self.assertAlmostEqual(entity_f05({"S2-1", "S3-1"}, {"S2-1"}), 5 / 6)

    def test_one_extra_false_match(self):
        self.assertAlmostEqual(entity_f05({"S2-1"}, {"S2-1", "S3-2"}), 5 / 9)

    def test_correct_singleton(self):
        self.assertEqual(entity_f05(set(), set()), 1.0)

    def test_false_positive_on_singleton(self):
        self.assertEqual(entity_f05(set(), {"S2-1"}), 0.0)

    def test_multiple_true_matches(self):
        self.assertEqual(entity_f05({"S2-1", "S2-2", "S3-1"}, {"S3-1", "S2-2", "S2-1"}), 1.0)

    def test_empty_prediction_for_non_singleton(self):
        self.assertEqual(entity_f05({"S2-1"}, set()), 0.0)

    def test_sets_ignore_repeated_ids(self):
        self.assertEqual(entity_f05(["S2-1", "S2-1"], ["S2-1", "S2-1"]), 1.0)

    def test_missing_s1_prediction_is_rejected(self):
        with self.assertRaises(ValueError):
            macro_f05({"S1-1": set()}, {})

    def test_parse_empty_field(self):
        self.assertEqual(parse_match_ids(""), set())

    def test_saved_split_filters_match_tsv(self):
        fixtures = Path(__file__).resolve().parent / "fixtures"
        allowed = read_split_ids(fixtures / "split.tsv")
        self.assertEqual(allowed, {"S1-2"})
        self.assertEqual(read_match_tsv(fixtures / "matches.tsv", allowed), {"S1-2": {"S3-2"}})


if __name__ == "__main__":
    unittest.main()
