"""
progress.py — Progress tracking, statistics, logging and final report
"""

import datetime
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from config import Config


def setup_logging(cfg: Config) -> None:
    """
    Configures logging:
      - console : INFO (or DEBUG if cfg.verbose)
      - file    : DEBUG always (cfg.log_file)
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

    # Reduce noise from third-party libraries
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

    # Time per page (total + per step)
    times: list[float] = field(default_factory=list)
    preprocess_times: list[float] = field(default_factory=list)
    ocr_times: list[float] = field(default_factory=list)
    postprocess_times: list[float] = field(default_factory=list)

    # Per-page detail (for the report)
    _pages: list[dict] = field(default_factory=list)
    fallback_pages: list[str] = field(default_factory=list)  # pages processed without layout

    model_load_time: float = 0.0
    start_time: float = field(default_factory=time.time)

    # ── Updates ──────────────────────────────────────────────────────────────

    def record_success(
        self,
        elapsed: float,
        chars: int,
        t_pre: float = 0.0,
        t_ocr: float = 0.0,
        t_post: float = 0.0,
        looped: bool = False,
        page_name: str = "",
        no_layout: bool = False,
    ) -> None:
        self.done += 1
        self.times.append(elapsed)
        self.total_chars += chars
        self.preprocess_times.append(t_pre)
        self.ocr_times.append(t_ocr)
        self.postprocess_times.append(t_post)
        if looped:
            self.loop_stops += 1
        if no_layout and page_name:
            self.fallback_pages.append(page_name)
        self._pages.append({
            "name": page_name,
            "t_pre": t_pre,
            "t_ocr": t_ocr,
            "t_post": t_post,
            "total": elapsed,
            "chars": chars,
            "looped": looped,
            "no_layout": no_layout,
            "error": False,
        })

    def record_skip(self) -> None:
        self.skipped += 1

    def record_error(self, page_name: str = "") -> None:
        self.errors += 1
        self._pages.append({"name": page_name, "error": True})

    # ── Calculations ─────────────────────────────────────────────────────────

    @property
    def avg_time(self) -> float:
        return sum(self.times) / len(self.times) if self.times else 0.0

    @property
    def elapsed_total(self) -> float:
        return time.time() - self.start_time

    @property
    def eta_s(self) -> float | None:
        """Estimated time remaining (seconds)."""
        remaining = self.total - self.done - self.skipped - self.errors
        if self.avg_time and remaining > 0:
            return self.avg_time * remaining
        return None

    # ── Display ──────────────────────────────────────────────────────────────

    def log_page(self, index: int, name: str, elapsed: float, chars: int) -> None:
        eta = self.eta_s
        eta_str = f"  ETA ~{eta/60:.0f}min" if eta else ""
        logging.getLogger(__name__).info(
            "[%d/%d] %-30s  %5.1fs  %d chars%s",
            index, self.total, name, elapsed, chars, eta_str,
        )

    def log_summary(self) -> None:
        logger = logging.getLogger(__name__)
        elapsed_s = self.elapsed_total
        elapsed_str = f"{elapsed_s:.0f}s" if elapsed_s < 60 else f"{elapsed_s / 60:.1f} min"
        logger.info("─" * 60)
        logger.info(
            "Finished in %s  |  %d pages OK  |  %d skipped  |  %d errors",
            elapsed_str, self.done, self.skipped, self.errors,
        )
        if self.done:
            logger.info(
                "Average speed: %.1fs/page  |  %d characters total",
                self.avg_time, self.total_chars,
            )

    # ── Report ───────────────────────────────────────────────────────────────

    def write_report(self, report_path: Path, cfg: Config) -> None:
        """Writes a detailed Markdown report of the run."""
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        def _row(*cells) -> str:
            return "| " + " | ".join(str(c) for c in cells) + " |"

        def _lst_stats(lst: list[float]):
            """Returns (avg, min, max, total) or (0,0,0,0) if empty."""
            if not lst:
                return 0.0, 0.0, 0.0, 0.0
            return sum(lst) / len(lst), min(lst), max(lst), sum(lst)

        lines: list[str] = []

        # ── Header ─────────────────────────────────────────────────────────
        lines += [
            "# OCR Run Report\n",
            f"**Date** : {now}  ",
            f"**Model** : PaddleOCR-VL-1.5  ",
            f"**Layout detection** : {cfg.use_layout_detection}  ",
            f"**Output** : `{cfg.output_file}`  ",
            "",
        ]

        # ── Summary ─────────────────────────────────────────────────────────
        elapsed_s = self.elapsed_total
        elapsed_str = f"{elapsed_s:.0f}s" if elapsed_s < 60 else f"{elapsed_s / 60:.1f} min"
        avg_chars = self.total_chars // self.done if self.done else 0
        lines += [
            "## Summary\n",
            "| Metric | Value |",
            "|---|---|",
            _row("Total pages", self.total),
            _row("Processed successfully", self.done),
            _row("Skipped (resume)", self.skipped),
            _row("Errors", self.errors),
            _row("Fallback no-layout", len(self.fallback_pages)),
            _row("Model load", f"{self.model_load_time:.1f}s"),
            _row("Total duration (pipeline)", elapsed_str),
            _row("Total characters", f"{self.total_chars:,}"),
            _row("Average characters/page", f"{avg_chars:,}"),
            "",
        ]

        # ── Time breakdown ──────────────────────────────────────────────────
        if self.done:
            avg_ocr, min_ocr, max_ocr, tot_ocr = _lst_stats(self.ocr_times)
            avg_post, min_post, max_post, tot_post = _lst_stats(self.postprocess_times)
            avg_tot, min_tot, max_tot, tot_tot = _lst_stats(self.times)

            pct_ocr  = 100 * tot_ocr  / tot_tot if tot_tot else 0
            pct_post = 100 * tot_post / tot_tot if tot_tot else 0

            lines += [
                "## Execution Time Breakdown\n",
                "| Step | Avg | Min | Max | Total | % |\n|---|---|---|---|---|---|",
                _row("Inference (pipeline.predict)",
                     f"{avg_ocr:.2f}s", f"{min_ocr:.2f}s", f"{max_ocr:.2f}s",
                     f"{tot_ocr:.1f}s", f"{pct_ocr:.1f}%"),
                _row("Post-processing",
                     f"{avg_post:.3f}s", f"{min_post:.3f}s", f"{max_post:.3f}s",
                     f"{tot_post:.2f}s", f"{pct_post:.1f}%"),
                _row("**Total/page**",
                     f"**{avg_tot:.2f}s**", f"**{min_tot:.2f}s**", f"**{max_tot:.2f}s**",
                     f"**{tot_tot:.1f}s**", "—"),
                "",
            ]

        # ── Fallback no-layout pages ────────────────────────────────────────
        if self.fallback_pages:
            lines += [
                "## Pages processed in fallback (without layout) — to review\n",
                "These pages triggered a timeout and were reprocessed without layout detection.",
                "Quality may be degraded (missing headers, broken words).\n",
            ]
            for name in self.fallback_pages:
                lines.append(f"- {name}")
            lines.append("")

        # ── Per-page detail ─────────────────────────────────────────────────
        if self._pages:
            lines += [
                "## Per-page Detail\n",
                "| Page | OCR | Post-processing | Total | Characters | Notes |",
                "|---|---|---|---|---|---|",
            ]
            for p in self._pages:
                if p.get("error"):
                    lines.append(_row(p["name"], "—", "—", "—", "ERROR", ""))
                else:
                    note = "fallback" if p.get("no_layout") else ""
                    lines.append(_row(
                        p["name"],
                        f"{p['t_ocr']:.2f}s",
                        f"{p['t_post']:.3f}s",
                        f"{p['total']:.2f}s",
                        f"{p['chars']:,}",
                        note,
                    ))
            lines.append("")

        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
        logging.getLogger(__name__).info("Report: %s", report_path.resolve())
