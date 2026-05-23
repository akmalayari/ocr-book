"""
Unit tests for the Stats dataclass in src/progress.py.
No filesystem, llama-server, or model stack required.
"""

import sys
import tempfile
import unittest
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from config import Config
from progress import Stats


class TestStatsAccumulators(unittest.TestCase):
    def setUp(self):
        self.stats = Stats(total=10)

    def test_record_success_increments_done(self):
        self.stats.record_success(elapsed=5.0, chars=100)
        self.assertEqual(self.stats.done, 1)

    def test_record_success_accumulates_chars(self):
        self.stats.record_success(elapsed=5.0, chars=200)
        self.stats.record_success(elapsed=3.0, chars=300)
        self.assertEqual(self.stats.total_chars, 500)

    def test_record_success_appends_time(self):
        self.stats.record_success(elapsed=4.0, chars=0)
        self.stats.record_success(elapsed=6.0, chars=0)
        self.assertEqual(self.stats.times, [4.0, 6.0])

    def test_record_error_increments_errors(self):
        self.stats.record_error()
        self.assertEqual(self.stats.errors, 1)

    def test_record_error_adds_to_pages(self):
        self.stats.record_error(page_name="page_007")
        self.assertEqual(self.stats._pages[-1]["name"], "page_007")
        self.assertTrue(self.stats._pages[-1]["error"])

    def test_record_skip_increments_skipped(self):
        self.stats.record_skip()
        self.stats.record_skip()
        self.assertEqual(self.stats.skipped, 2)

    def test_looped_flag_increments_loop_stops(self):
        self.stats.record_success(elapsed=5.0, chars=0, looped=True)
        self.assertEqual(self.stats.loop_stops, 1)

    def test_no_loop_does_not_increment_loop_stops(self):
        self.stats.record_success(elapsed=5.0, chars=0, looped=False)
        self.assertEqual(self.stats.loop_stops, 0)

    def test_no_layout_adds_to_fallback_pages(self):
        self.stats.record_success(elapsed=5.0, chars=0, no_layout=True, page_name="page_003")
        self.assertIn("page_003", self.stats.fallback_pages)

    def test_layout_does_not_add_to_fallback_pages(self):
        self.stats.record_success(elapsed=5.0, chars=0, no_layout=False, page_name="page_003")
        self.assertNotIn("page_003", self.stats.fallback_pages)


class TestStatsProperties(unittest.TestCase):
    def test_avg_time_computed_correctly(self):
        stats = Stats(total=3)
        stats.record_success(elapsed=4.0, chars=0)
        stats.record_success(elapsed=6.0, chars=0)
        self.assertAlmostEqual(stats.avg_time, 5.0)

    def test_avg_time_zero_when_no_pages(self):
        stats = Stats(total=3)
        self.assertEqual(stats.avg_time, 0.0)

    def test_eta_s_computed_correctly(self):
        stats = Stats(total=5)
        stats.record_success(elapsed=10.0, chars=0)  # avg = 10s, remaining = 4
        self.assertAlmostEqual(stats.eta_s, 40.0)

    def test_eta_s_none_when_no_pages_done(self):
        stats = Stats(total=5)
        self.assertIsNone(stats.eta_s)

    def test_eta_s_none_when_all_done(self):
        stats = Stats(total=1)
        stats.record_success(elapsed=5.0, chars=0)
        self.assertIsNone(stats.eta_s)

    def test_eta_accounts_for_skipped_and_errors(self):
        # total=5, done=1, skipped=1, errors=1 → remaining=2
        stats = Stats(total=5)
        stats.record_success(elapsed=10.0, chars=0)
        stats.record_skip()
        stats.record_error()
        self.assertAlmostEqual(stats.eta_s, 20.0)


class TestStatsWriteReport(unittest.TestCase):
    def test_write_report_creates_file(self):
        stats = Stats(total=2)
        stats.record_success(elapsed=5.0, chars=300, t_ocr=4.5, t_post=0.1, page_name="page_001")
        stats.record_error(page_name="page_002")

        cfg = Config()
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "ocr_report.md"
            stats.write_report(report_path, cfg)
            self.assertTrue(report_path.exists())
            content = report_path.read_text(encoding="utf-8")
            self.assertIn("# OCR Run Report", content)
            self.assertIn("page_001", content)
            self.assertIn("ERROR", content)

    def test_write_report_summary_counts(self):
        stats = Stats(total=3)
        stats.record_success(elapsed=5.0, chars=100, page_name="p1")
        stats.record_skip()
        stats.record_error(page_name="p3")

        cfg = Config()
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "report.md"
            stats.write_report(report_path, cfg)
            content = report_path.read_text(encoding="utf-8")
            self.assertIn("| Processed successfully | 1 |", content)
            self.assertIn("| Skipped (resume) | 1 |", content)
            self.assertIn("| Errors | 1 |", content)


if __name__ == "__main__":
    unittest.main()
