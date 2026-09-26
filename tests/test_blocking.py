import unittest

from src.blocking import Business, build_rules, retrieve_by_rule


def business(entity_id, name, address, country="US"):
    return Business.from_raw({
        "entity_id": entity_id,
        "business_name": name,
        "business_address": address,
        "country": country,
    })


class BlockingTests(unittest.TestCase):
    def test_exact_normalized_name(self):
        rules = build_rules([business("S1-1", "Acme, Inc.", "15 Oak Road")])
        found = retrieve_by_rule(rules, business("S2-1", "ACME INC", "different address"))
        self.assertEqual(found["A_exact_name"], {0})

    def test_name_signature_reorders_tokens(self):
        rules = build_rules([business("S1-1", "Blue River Labs", "15 Oak Road")])
        found = retrieve_by_rule(rules, business("S2-1", "Labs River Blue", "different address"))
        self.assertEqual(found["A_exact_name"], set())
        self.assertEqual(found["B_name_signature"], {0})

    def test_exact_numeric_address_with_different_name(self):
        rules = build_rules([business("S1-1", "North Clinic", "15 Oak Road, 12345")])
        found = retrieve_by_rule(rules, business("S3-1", "South Clinic", "15 Oak Road, 12345"))
        self.assertEqual(found["C1_exact_numeric_address"], {0})

    def test_numeric_address_and_shared_name_token(self):
        rules = build_rules([business("S1-1", "Acme Lighting Company", "Unit 215, Oak Road, 12345")])
        found = retrieve_by_rule(rules, business("S2-1", "Lighting & Design", "215 Pine Avenue, 12345"))
        self.assertEqual(found["C2_numeric_plus_name_token"], {0})
        other_country = retrieve_by_rule(rules, business("S2-2", "Lighting & Design", "215 Pine Avenue, 12345", "France"))
        self.assertEqual(other_country["C2_numeric_plus_name_token"], set())


if __name__ == "__main__":
    unittest.main()
