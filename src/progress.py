"""
progress.py — Suivi de progression, statistiques et logging
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from config import Config


def setup_logging(cfg: Config) -> None:
    """
    Configure le logging :
      - console  : INFO (ou DEBUG si cfg.verbose)
      - fichier  : DEBUG toujours (cfg.log_file)
    """
    level = logging.DEBUG if cfg.verbose else logging.INFO

    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    datefmt = "%H:%M:%S"

    handlers: list[logging.Handler] = [
        logging.StreamHandler(),
    ]
    if cfg.log_file:
        handlers.append(logging.FileHandler(cfg.log_file, encoding="utf-8"))

    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)

    # Réduire le bruit des bibliothèques tierces
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)


@dataclass
class Stats:
    total: int = 0
    done: int = 0
    skipped: int = 0
    errors: int = 0
    total_chars: int = 0
    times: list[float] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)

    # ── Mise à jour ──────────────────────────────────────────────────────────

    def record_success(self, elapsed: float, chars: int) -> None:
        self.done += 1
        self.times.append(elapsed)
        self.total_chars += chars

    def record_skip(self) -> None:
        self.skipped += 1

    def record_error(self) -> None:
        self.errors += 1

    # ── Calculs ──────────────────────────────────────────────────────────────

    @property
    def avg_time(self) -> float:
        return sum(self.times) / len(self.times) if self.times else 0.0

    @property
    def elapsed_total(self) -> float:
        return time.time() - self.start_time

    @property
    def eta_s(self) -> float | None:
        """Estimation du temps restant (secondes)."""
        remaining = self.total - self.done - self.skipped - self.errors
        if self.avg_time and remaining > 0:
            return self.avg_time * remaining
        return None

    # ── Affichage ────────────────────────────────────────────────────────────

    def log_page(self, index: int, name: str, elapsed: float, chars: int) -> None:
        eta = self.eta_s
        eta_str = f"  ETA ~{eta/60:.0f}min" if eta else ""
        logging.getLogger(__name__).info(
            "[%d/%d] %-30s  %5.1fs  %d car.%s",
            index, self.total, name, elapsed, chars, eta_str,
        )

    def log_summary(self) -> None:
        logger = logging.getLogger(__name__)
        minutes = self.elapsed_total / 60
        logger.info("─" * 60)
        logger.info(
            "Terminé en %.1f min  |  %d pages OK  |  %d skippées  |  %d erreurs",
            minutes, self.done, self.skipped, self.errors,
        )
        if self.done:
            logger.info(
                "Vitesse moyenne : %.1fs/page  |  %d caractères total",
                self.avg_time, self.total_chars,
            )
