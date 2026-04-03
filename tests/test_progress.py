"""
test_progress.py — Tests unitaires pour progress.py

Couvre :
  - Stats : valeurs initiales, record_success/skip/error, compteurs cumulatifs
  - Stats : propriétés avg_time, elapsed_total, eta_s (cas normaux + edge cases)
  - Stats : log_page et log_summary (vérification qu'ils n'explosent pas)
  - setup_logging : niveaux, handlers, isolation des bibliothèques tierces
"""

import logging
import time
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from progress import Stats, setup_logging


# ── Stats : état initial ──────────────────────────────────────────────────────

class TestStatsInitialState:

    def test_defaults_all_zero(self):
        s = Stats()
        assert s.done == 0
        assert s.skipped == 0
        assert s.errors == 0
        assert s.total_chars == 0

    def test_total_set_at_creation(self):
        s = Stats(total=150)
        assert s.total == 150

    def test_times_list_empty(self):
        s = Stats()
        assert s.times == []

    def test_start_time_close_to_now(self):
        before = time.time()
        s = Stats()
        after = time.time()
        assert before <= s.start_time <= after


# ── Stats : méthodes de mise à jour ──────────────────────────────────────────

class TestStatsRecordMethods:

    def test_record_success_increments_done(self):
        s = Stats()
        s.record_success(10.0, 500)
        assert s.done == 1

    def test_record_success_appends_time(self):
        s = Stats()
        s.record_success(15.5, 300)
        assert 15.5 in s.times

    def test_record_success_adds_chars(self):
        s = Stats()
        s.record_success(10.0, 400)
        s.record_success(8.0, 600)
        assert s.total_chars == 1000

    def test_record_success_stores_latency(self):
        s = Stats()
        s.record_success(10.0, 500, latency=8.5)
        assert 8.5 in s.latencies

    def test_record_success_latency_defaults_to_zero(self):
        s = Stats()
        s.record_success(10.0, 500)
        assert s.latencies == [0.0]

    def test_record_skip_increments_skipped(self):
        s = Stats()
        s.record_skip()
        s.record_skip()
        assert s.skipped == 2

    def test_record_skip_does_not_affect_done(self):
        s = Stats()
        s.record_skip()
        assert s.done == 0

    def test_record_error_increments_errors(self):
        s = Stats()
        s.record_error()
        assert s.errors == 1

    def test_record_error_does_not_affect_done_or_skipped(self):
        s = Stats()
        s.record_error()
        assert s.done == 0
        assert s.skipped == 0

    def test_mixed_operations(self):
        s = Stats(total=10)
        s.record_success(5.0, 100)
        s.record_success(7.0, 200)
        s.record_skip()
        s.record_error()
        assert s.done == 2
        assert s.skipped == 1
        assert s.errors == 1
        assert s.total_chars == 300


# ── Stats : propriétés calculées ─────────────────────────────────────────────

class TestStatsProperties:

    def test_avg_time_single_record(self):
        s = Stats()
        s.record_success(10.0, 100)
        assert s.avg_time == pytest.approx(10.0)

    def test_avg_time_multiple_records(self):
        s = Stats()
        s.record_success(10.0, 100)
        s.record_success(20.0, 100)
        assert s.avg_time == pytest.approx(15.0)

    def test_avg_time_zero_when_no_records(self):
        s = Stats()
        assert s.avg_time == 0.0

    def test_avg_time_no_division_by_zero(self):
        """Doit retourner 0.0 sans lever ZeroDivisionError."""
        s = Stats()
        result = s.avg_time
        assert result == 0.0

    def test_elapsed_total_increases_over_time(self):
        s = Stats()
        t1 = s.elapsed_total
        time.sleep(0.05)
        t2 = s.elapsed_total
        assert t2 > t1

    def test_elapsed_total_is_positive(self):
        s = Stats()
        assert s.elapsed_total >= 0

    def test_eta_none_when_no_records(self):
        s = Stats(total=10)
        assert s.eta_s is None

    def test_eta_none_when_total_zero(self):
        s = Stats(total=0)
        assert s.eta_s is None

    def test_eta_computed_correctly(self):
        """ETA = avg_time × pages_restantes."""
        s = Stats(total=10)
        s.record_success(5.0, 100)   # done=1, avg=5s
        # remaining = 10 - 1 - 0 - 0 = 9
        eta = s.eta_s
        assert eta is not None
        assert eta == pytest.approx(9 * 5.0, rel=0.01)

    def test_eta_none_when_all_done(self):
        s = Stats(total=2)
        s.record_success(5.0, 100)
        s.record_success(5.0, 100)
        assert s.eta_s is None

    def test_eta_accounts_for_skipped_and_errors(self):
        """Les pages skippées et en erreur comptent dans le « traité »."""
        s = Stats(total=10)
        s.record_success(10.0, 100)  # done=1
        s.record_skip()               # skipped=1
        s.record_error()              # errors=1
        # remaining = 10 - 1 - 1 - 1 = 7
        eta = s.eta_s
        assert eta == pytest.approx(7 * 10.0, rel=0.01)

    def test_avg_latency_single_record(self):
        s = Stats()
        s.record_success(10.0, 100, latency=8.0)
        assert s.avg_latency == pytest.approx(8.0)

    def test_avg_latency_multiple_records(self):
        s = Stats()
        s.record_success(10.0, 100, latency=6.0)
        s.record_success(12.0, 100, latency=10.0)
        assert s.avg_latency == pytest.approx(8.0)

    def test_avg_latency_zero_when_no_records(self):
        assert Stats().avg_latency == 0.0


# ── Stats : méthodes d'affichage ──────────────────────────────────────────────

class TestStatsLogging:

    def test_log_page_does_not_raise(self, caplog):
        s = Stats(total=10)
        s.record_success(16.0, 1800)
        with caplog.at_level(logging.INFO):
            s.log_page(1, "page_001.jpg", 16.0, 1800)  # ne doit pas exploser

    def test_log_page_contains_index(self, caplog):
        s = Stats(total=50)
        with caplog.at_level(logging.INFO):
            s.log_page(7, "page_007.jpg", 10.0, 500)
        assert any("7" in r.message for r in caplog.records)

    def test_log_page_contains_filename(self, caplog):
        s = Stats(total=50)
        with caplog.at_level(logging.INFO):
            s.log_page(1, "page_001.jpg", 10.0, 500)
        assert any("page_001.jpg" in r.message for r in caplog.records)

    def test_log_summary_does_not_raise(self, caplog):
        s = Stats(total=5)
        s.record_success(10.0, 500)
        s.record_skip()
        s.record_error()
        with caplog.at_level(logging.INFO):
            s.log_summary()

    def test_log_summary_with_zero_done(self, caplog):
        """Résumé sans pages traitées ne doit pas lever d'exception."""
        s = Stats(total=0)
        with caplog.at_level(logging.INFO):
            s.log_summary()

    def test_log_page_with_eta(self, caplog):
        """Quand ETA est disponible, elle doit apparaître dans le log."""
        s = Stats(total=20)
        s.record_success(10.0, 500)
        with caplog.at_level(logging.INFO):
            s.log_page(1, "page_001.jpg", 10.0, 500)
        # ETA = (20-1) * 10 = 190s → ~3min dans le message
        combined = " ".join(r.message for r in caplog.records)
        assert "ETA" in combined or "min" in combined


# ── setup_logging ─────────────────────────────────────────────────────────────

class TestSetupLogging:

    def test_does_not_raise(self, tmp_path):
        cfg = Config(log_file=str(tmp_path / "test.log"), verbose=False)
        # Réinitialiser le root logger pour isoler le test
        root = logging.getLogger()
        original_handlers = root.handlers[:]
        root.handlers.clear()
        try:
            setup_logging(cfg)
        finally:
            root.handlers = original_handlers

    def test_verbose_sets_debug_level(self, tmp_path):
        cfg = Config(log_file="", verbose=True)
        root = logging.getLogger()
        original_handlers = root.handlers[:]
        original_level = root.level
        root.handlers.clear()
        try:
            setup_logging(cfg)
            assert root.level <= logging.DEBUG
        finally:
            root.handlers = original_handlers
            root.setLevel(original_level)

    def test_non_verbose_sets_info_level(self, tmp_path):
        cfg = Config(log_file="", verbose=False)
        root = logging.getLogger()
        original_handlers = root.handlers[:]
        original_level = root.level
        root.handlers.clear()
        try:
            setup_logging(cfg)
            assert root.level <= logging.INFO
        finally:
            root.handlers = original_handlers
            root.setLevel(original_level)

    def test_creates_log_file(self, tmp_path):
        log_path = tmp_path / "test_run.log"
        cfg = Config(log_file=str(log_path), verbose=False)
        root = logging.getLogger()
        original_handlers = root.handlers[:]
        root.handlers.clear()
        try:
            setup_logging(cfg)
            logging.getLogger("test").info("Message test")
            # Fermer les handlers pour libérer le fichier
            for h in root.handlers:
                h.close()
        finally:
            root.handlers = original_handlers
        assert log_path.exists()

    def test_urllib3_logger_set_to_warning(self, tmp_path):
        cfg = Config(log_file="", verbose=True)
        root = logging.getLogger()
        original_handlers = root.handlers[:]
        root.handlers.clear()
        try:
            setup_logging(cfg)
            assert logging.getLogger("urllib3").level == logging.WARNING
        finally:
            root.handlers = original_handlers

    def test_requests_logger_set_to_warning(self, tmp_path):
        cfg = Config(log_file="", verbose=True)
        root = logging.getLogger()
        original_handlers = root.handlers[:]
        root.handlers.clear()
        try:
            setup_logging(cfg)
            assert logging.getLogger("requests").level == logging.WARNING
        finally:
            root.handlers = original_handlers
