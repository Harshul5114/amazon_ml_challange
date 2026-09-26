import unittest

from src.normalize import normalize_record, normalize_text


class NormalizeTests(unittest.TestCase):
    def test_capitalization(self):
        self.assertEqual(normalize_text("AcMe CORP"), "acme corp")

    def test_punctuation(self):
        self.assertEqual(normalize_text("ACME, Inc. / North-East"), "acme inc north east")

    def test_repeated_whitespace(self):
        self.assertEqual(normalize_text("  New\t\tYork\u00a0  Office  "), "new york office")

    def test_nfkc_and_accent_preservation(self):
        self.assertEqual(normalize_text("Ｃａｆé Ｎº １２"), "café no 12")

    def test_indian_address_and_script(self):
        record = normalize_record({"business_name": "राम मार्केटिंग", "business_address": "KH NO. -570/13, नई दिल्ली, Delhi"})
        self.assertEqual(record["name_norm"], "राम मार्केटिंग")
        self.assertEqual(record["address_norm"], "kh no 570 13 नई दिल्ली delhi")
        self.assertEqual(record["address_numeric_tokens"], ["570", "13"])

    def test_french_accented_text(self):
        record = normalize_record({"business_name": "École de l'Église", "business_address": "12 Rue de l'Église, 75001 Paris"})
        self.assertEqual(record["name_norm"], "école de l église")
        self.assertEqual(record["address_norm"], "12 rue de l église 75001 paris")
        self.assertEqual(record["address_numeric_tokens"], ["12", "75001"])

    def test_empty_and_null_fields(self):
        record = {"entity_id": "S1-1", "business_name": None, "business_address": ""}
        derived = normalize_record(record)
        self.assertEqual(record, {"entity_id": "S1-1", "business_name": None, "business_address": ""})
        self.assertIsNone(derived["business_name"])
        for field in ("name_norm", "address_norm"):
            self.assertEqual(derived[field], "")
        for field in ("name_tokens", "address_tokens", "name_numeric_tokens", "address_numeric_tokens"):
            self.assertEqual(derived[field], [])

    def test_nan_from_default_tsv_parsing(self):
        derived = normalize_record({"business_name": float("nan"), "business_address": float("nan")})
        self.assertEqual(derived["name_norm"], "")
        self.assertEqual(derived["address_tokens"], [])

    def test_numeric_components_inside_tokens(self):
        record = normalize_record({"business_name": "Studio 54B", "business_address": "Flat 12A, Block-7/3"})
        self.assertEqual(record["name_tokens"], ["studio", "54b"])
        self.assertEqual(record["name_numeric_tokens"], ["54"])
        self.assertEqual(record["address_numeric_tokens"], ["12", "7", "3"])

    def test_raw_fields_are_preserved(self):
        original = {"entity_id": "S1-2", "business_name": "A&B", "business_address": "1 Main St.", "country": "France"}
        normalized = normalize_record(original)
        self.assertEqual({key: normalized[key] for key in original}, original)
        self.assertEqual(original["business_name"], "A&B")


if __name__ == "__main__":
    unittest.main()
