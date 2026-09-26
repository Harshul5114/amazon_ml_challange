import unittest

import numpy as np

from reports.evaluate_tfidf_union import exact_membership, pack_pair, target_id_from_key


class TfidfUnionTests(unittest.TestCase):
    def test_packed_pair_identity_and_membership(self):
        a = pack_pair(7, 0, 123)
        b = pack_pair(7, 1, 123)
        c = pack_pair(8, 0, 123)
        self.assertEqual(len({a, b, c}), 3)
        self.assertEqual(target_id_from_key(a), "S2-123")
        self.assertEqual(target_id_from_key(b), "S3-123")
        found = exact_membership(np.array(sorted([a, c]), dtype=np.uint64),
                                 np.array([a, b, c], dtype=np.uint64))
        self.assertEqual(found.tolist(), [True, False, True])


if __name__ == "__main__":
    unittest.main()
