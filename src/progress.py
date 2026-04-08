"""
progress.py — Suivi de progression, statistiques, logging et rapport final
"""

import datetime
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
        Path(cfg.log_file).parent.mkdir(parents=True, exist_ok=True)
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
    loop_stops: int = 0
    total_chars: int = 0

    # Temps par page (total + par étape)
    times: list[float] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)
    preprocess_times: list[float] = field(default_factory=list)
    ocr_times: list[float] = field(default_factory=list)
    postprocess_times: list[float] = field(default_factory=list)

    # Détail par page (pour le rapport)
    _pages: list[dict] = field(default_factory=list)

    model_load_time: float = 0.0
    start_time: float = field(default_factory=time.time)

    # ── Mise à jour ──────────────────────────────────────────────────────────

    def record_success(
        self,
        elapsed: float,
        chars: int,
        latency: float = 0.0,
        t_pre: float = 0.0,
        t_ocr: float = 0.0,
        t_post: float = 0.0,
        looped: bool = False,
        page_name: str = "",
    ) -> None:
        self.done += 1
        self.times.append(elapsed)
        self.latencies.append(latency)
        self.total_chars += chars
        self.preprocess_times.append(t_pre)
        self.ocr_times.append(t_ocr)
        self.postprocess_times.append(t_post)
        if looped:
            self.loop_stops += 1
        self._pages.append({
            "name": page_name,
            "t_pre": t_pre,
            "t_ocr": t_ocr,
            "t_post": t_post,
            "total": elapsed,
            "chars": chars,
            "looped": looped,
            "error": False,
        })

    def record_skip(self) -> None:
        self.skipped += 1

    def record_error(self, page_name: str = "") -> None:
        self.errors += 1
        self._pages.append({"name": page_name, "error": True})

    # ── Calculs ──────────────────────────────────────────────────────────────

    @property
    def avg_time(self) -> float:
        return sum(self.times) / len(self.times) if self.times else 0.0

    @property
    def avg_latency(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0

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
        latency = self.latencies[-1] if self.latencies else elapsed
        logging.getLogger(__name__).info(
            "[%d/%d] %-30s  %5.1fs (latence %.1fs)  %d car.%s",
            index, self.total, name, elapsed, latency, chars, eta_str,
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
                "Vitesse moyenne : %.1fs/page  |  Latence moy. : %.1fs  |  %d caractères total",
                self.avg_time, self.avg_latency, self.total_chars,
            )

    # ── Rapport ──────────────────────────────────────────────────────────────

    def write_report(self, report_path: Path, cfg: Config) -> None:
        """Écrit un rapport Markdown détaillé du run."""
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        def _row(*cells) -> str:
            return "| " + " | ".join(str(c) for c in cells) + " |"

        def _lst_stats(lst: list[float]):
            """Retourne (avg, min, max, total) ou (0,0,0,0) si vide."""
            if not lst:
                return 0.0, 0.0, 0.0, 0.0
            return sum(lst) / len(lst), min(lst), max(lst), sum(lst)

        lines: list[str] = []

        # ── En-tête ─────────────────────────────────────────────────────────
        lines += [
            "# Rapport de run OCR\n",
            f"**Date** : {now}  ",
            f"**Modèle** : PaddleOCR-VL-1.5  ",
            f"**Layout detection** : {cfg.use_layout_detection}  ",
            f"**Sortie** : `{cfg.output_file}`  ",
            "",
        ]

        # ── Résumé ──────────────────────────────────────────────────────────
        elapsed_min = self.elapsed_total / 60
        avg_chars = self.total_chars // self.done if self.done else 0
        lines += [
            "## Résumé\n",
            "| Indicateur | Valeur |",
            "|---|---|",
            _row("Pages total", self.total),
            _row("Traitées avec succès", self.done),
            _row("Ignorées (reprise)", self.skipped),
            _row("Erreurs", self.errors),
            _row("Chargement modèle", f"{self.model_load_time:.1f}s"),
            _row("Durée totale (pipeline)", f"{elapsed_min:.1f} min"),
            _row("Caractères total", f"{self.total_chars:,}"),
            _row("Caractères moy./page", f"{avg_chars:,}"),
            "",
        ]

        # ── Décomposition du temps ──────────────────────────────────────────
        if self.done:
            avg_ocr, min_ocr, max_ocr, tot_ocr = _lst_stats(self.ocr_times)
            avg_post, min_post, max_post, tot_post = _lst_stats(self.postprocess_times)
            avg_tot, min_tot, max_tot, tot_tot = _lst_stats(self.times)

            pct_ocr  = 100 * tot_ocr  / tot_tot if tot_tot else 0
            pct_post = 100 * tot_post / tot_tot if tot_tot else 0

            lines += [
                "## Décomposition du temps d'exécution\n",
                "| Étape | Moy. | Min | Max | Total | % |\n|---|---|---|---|---|---|",
                _row("OCR (PaddleOCR + save)",
                     f"{avg_ocr:.2f}s", f"{min_ocr:.2f}s", f"{max_ocr:.2f}s",
                     f"{tot_ocr:.1f}s", f"{pct_ocr:.1f}%"),
                _row("Post-traitement",
                     f"{avg_post:.3f}s", f"{min_post:.3f}s", f"{max_post:.3f}s",
                     f"{tot_post:.2f}s", f"{pct_post:.1f}%"),
                _row("**Total/page**",
                     f"**{avg_tot:.2f}s**", f"**{min_tot:.2f}s**", f"**{max_tot:.2f}s**",
                     f"**{tot_tot:.1f}s**", "—"),
                "",
            ]

        # ── Détail par page ─────────────────────────────────────────────────
        if self._pages:
            lines += [
                "## Détail par page\n",
                "| Page | OCR | Post-traitement | Total | Caractères |",
                "|---|---|---|---|---|",
            ]
            for p in self._pages:
                if p.get("error"):
                    lines.append(_row(p["name"], "—", "—", "—", "—", "ERREUR"))
                else:
                    lines.append(_row(
                        p["name"],
                        f"{p['t_ocr']:.2f}s",
                        f"{p['t_post']:.3f}s",
                        f"{p['total']:.2f}s",
                        f"{p['chars']:,}",
                    ))
            lines.append("")

        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(lines), encoding="utf-8")
        logging.getLogger(__name__).info("Rapport : %s", report_path.resolve())
