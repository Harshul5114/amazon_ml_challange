import unittest
import numpy as np

from src.blocking import Business
from src.similarity_features import FEATURE_NAMES, compute_pair_features
from src.train_match_model import train_tree_model, evaluate_thresholds_on_validation


def make_biz(entity_id: str, name: str, addr: str, country: str = "US") -> Business:
    return Business.from_raw({
        "entity_id": entity_id,
        "business_name": name,
        "business_address": addr,
        "country": country,
    })


class TrainMatchModelTest(unittest.TestCase):
    def test_train_tree_model(self):
        # Create synthetic training set with 15 features
        rng = np.random.default_rng(2026)
        n_samples = 200
        n_features = len(FEATURE_NAMES)
        X = rng.random((n_samples, n_features), dtype=np.float32)
        # target is 1 if feature 0 (exact_name) or feature 1 (jaccard) > 0.5
        y = ((X[:, 0] > 0.5) | (X[:, 1] > 0.6)).astype(np.float32)

        model = train_tree_model(X, y, model_type="hist_gb")
        self.assertIsNotNone(model)
        probs = model.predict_proba(X[:5])[:, 1]
        self.assertEqual(len(probs), 5)
        self.assertTrue(np.all((probs >= 0.0) & (probs <= 1.0)))

    def test_evaluate_thresholds_on_validation(self):
        b_s1_1 = make_biz("S1-1", "Alpha Tech", "100 Innovation Way", "US")
        b_s1_2 = make_biz("S1-2", "Beta Cafe", "200 Market Street", "US")
        b_s1_3 = make_biz("S1-3", "Gamma Singleton", "300 Quiet Road", "US")

        t_s2_1 = make_biz("S2-10", "Alpha Tech", "100 Innovation Way", "US")
        t_s3_1 = make_biz("S3-20", "Alpha Tech Inc", "100 Innovation Way", "US")
        t_s2_2 = make_biz("S2-30", "Beta Cafe", "200 Market Street", "US")
        t_s3_wrong = make_biz("S3-40", "Completely Unrelated", "999 Far Away", "US")

        val_s1 = [b_s1_1, b_s1_2, b_s1_3]
        target_cache = {
            "S2-10": t_s2_1,
            "S3-20": t_s3_1,
            "S2-30": t_s2_2,
            "S3-40": t_s3_wrong,
        }
        val_candidate_pairs = [
            (0, "S2", 10),
            (0, "S3", 20),
            (0, "S3", 40), # negative candidate for s1_1
            (1, "S2", 30),
            (2, "S3", 40), # false candidate for singleton
        ]
        truth_by_s1 = {
            "S1-1": {"S2-10", "S3-20"},
            "S1-2": {"S2-30"},
            "S1-3": set(), # singleton
        }

        # Train model with real features
        X_rows = []
        y_rows = []
        for s1, t, src, lbl in [
            (b_s1_1, t_s2_1, "S2", 1.0),
            (b_s1_1, t_s3_1, "S3", 1.0),
            (b_s1_2, t_s2_2, "S2", 1.0),
            (b_s1_1, t_s3_wrong, "S3", 0.0),
            (b_s1_3, t_s3_wrong, "S3", 0.0),
        ]:
            X_rows.append(compute_pair_features(s1, t, src))
            y_rows.append(lbl)

        # Duplicate to have enough samples for hist_gb
        X_train = np.repeat(np.asarray(X_rows, dtype=np.float32), 40, axis=0)
        y_train = np.repeat(np.asarray(y_rows, dtype=np.float32), 40, axis=0)
        model = train_tree_model(X_train, y_train, model_type="hist_gb")

        results, preds = evaluate_thresholds_on_validation(
            model, val_s1, val_candidate_pairs, truth_by_s1, target_cache
        )

        self.assertGreater(results["best_macro_f05"], 0.8)
        self.assertIn("threshold_curve", results)
        self.assertEqual(len(preds), 3)
        self.assertEqual(preds["S1-3"], set()) # singleton should be empty


if __name__ == "__main__":
    unittest.main()
