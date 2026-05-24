import unittest

from cc_insights.data.load import CLEANING_VERSION, clean_text


class CleanTextTests(unittest.TestCase):
    def test_removes_gibberish_tail_after_real_message(self) -> None:
        raw = (
            "The Partner Offer hasn\u2019t arrived yet. Can you check? "
            "ubfln qfx bac gqprjxvog scyn qmxuupyc mkjpflymzi"
        )
        self.assertEqual(
            clean_text(raw),
            "The Partner Offer hasn\u2019t arrived yet. Can you check?",
        )

    def test_keeps_real_hinglish_sentence(self) -> None:
        raw = "Thoda jaldi please, flight in 2 hours."
        self.assertEqual(clean_text(raw), raw)

    def test_removes_sentence_level_gibberish_tail(self) -> None:
        raw = (
            "I have applied the coupon manually and adjusted your invoice. "
            "mkgbiie pznff gfbverpnge abwypvmq srvvrklbf zxdvais ngkuy"
        )
        self.assertEqual(
            clean_text(raw),
            "I have applied the coupon manually and adjusted your invoice.",
        )

    def test_cleaning_version_is_declared(self) -> None:
        self.assertTrue(CLEANING_VERSION)


if __name__ == "__main__":
    unittest.main()
