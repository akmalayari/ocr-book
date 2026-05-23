"""
config.py — Central configuration for the OCR pipeline
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    # ── llama-server ─────────────────────────────────────────────────────────
    # Set via environment variables, CLI arguments, or edit this file directly.
    llama_server_path: str | None = os.environ.get("LLAMA_SERVER_PATH")
    model_path: str | None        = os.environ.get("MODEL_PATH")
    mmproj_path: str | None       = os.environ.get("MMPROJ_PATH")
    server_base_port: int  = 8080   # ports 8080, 8081, … (one per server)
    server_timeout: int    = 60     # seconds to wait before declaring the server dead

    # ── llama-server parameters (tuning) ─────────────────────────────────────
    # n_ctx: None = auto (n_parallel * 2048). Override for books with large/dense
    # tables where the default per-slot budget is too small (e.g. n_parallel * 4096).
    n_ctx: int | None     = None
    n_gpu_layers: int     = 99
    n_batch: int          = 512
    n_ubatch: int         = 512
    n_threads: int        = 4      # P-cores
    prio: int             = 2
    kv_offload: bool      = True
    max_tokens: int       = 4096
    temperature: float    = 0.0
    # n_parallel > 1 requires apply_paddlex_patch_parallel.py; leave at 1 otherwise.
    n_parallel: int       = 1
    n_servers: int        = 1      # number of parallel llama-server instances
    page_timeout: int     = 120    # max seconds per page before giving up (0 = disabled)

    # ── PaddleOCR ─────────────────────────────────────────────────────────────
    use_layout_detection: bool = True   # False = fallback without layout

    # ── Images ───────────────────────────────────────────────────────────────
    rename_prefix: str          = "page"
    images_dir: str             = "./photos"
    extensions: tuple           = (".jpg", ".jpeg", ".png", ".webp")
    image_files: list | None    = None   # if provided, bypasses images_dir

    # ── PDF ────────────────────────────────────────────────────────────────────
    pdf_dpi: int = 200
    pdf_force_ocr: bool = False
    extraction_method: str = "paddleocrvl"   # "text" | "docling" | "paddleocrvl"
    temp_dir: Path = field(default_factory=lambda: Path("output/temp"))

    # ── EPUB ───────────────────────────────────────────────────────────────────
    epub_file: str | None = None

    # ── Output ───────────────────────────────────────────────────────────────
    output_file: str = "./output/book.md"
    figures_dir: str = "./output/figures"
    resume: bool     = True

    # ── Post-processing ──────────────────────────────────────────────────────
    postprocess: bool                   = True
    keep_html: bool                     = False  # keep HTML tables/figures instead of converting to Markdown
    mode: str                           = "base"   # "base" | "obsidian"
    vault_root: str | None               = os.environ.get("OBSIDIAN_VAULT_ROOT")
    vault_path: str                     = os.environ.get("OBSIDIAN_VAULT_PATH", "Documents/OCR")
    vault_figures_dir: str              = os.environ.get("OBSIDIAN_VAULT_FIGURES_DIR", "Files/OCR")
    remove_isolated_page_numbers: bool  = True
    rejoin_hyphenated_words: bool       = True
    collapse_blank_lines: bool          = True
    # List of (line_start_regex, level) for header detection on the final file.
    # [] = disabled (default). Set via --header-pattern CLI flag to enable.
    # Ex: [("^[IVX]+\\.", 2), ("^[A-Z]\\.", 3)]
    header_patterns: list[tuple[str, int]] = field(default_factory=list)

    # ── Logging ──────────────────────────────────────────────────────────────
    log_file: str    = "output/ocr_run.log"
    report_file: str = "output/ocr_report.md"
    verbose: bool    = False

    def __post_init__(self):
        if self.n_ctx is None:
            self.n_ctx = self.n_parallel * 2048

    @property
    def images_path(self) -> Path:
        return Path(self.images_dir)

    @property
    def output_path(self) -> Path:
        return Path(self.output_file)

    @property
    def figures_path(self) -> Path:
        return Path(self.figures_dir)
