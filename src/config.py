"""
config.py — Central configuration for the OCR pipeline
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _env_int(name: str, default: int) -> int:
    """Reads an int from the environment, falling back to `default` if unset."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"Invalid integer for {name}: {raw!r}") from None


def _env_opt_int(name: str) -> int | None:
    """Same as `_env_int` but returns None when the variable is unset."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"Invalid integer for {name}: {raw!r}") from None


def _env_bool(name: str, default: bool) -> bool:
    """Reads a bool from the environment (1/true/yes/on vs 0/false/no/off)."""
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"Invalid boolean for {name}: {raw!r}")


@dataclass
class Config:
    # ── llama-server ─────────────────────────────────────────────────────────
    # Set via environment variables, CLI arguments, or edit this file directly.
    llama_server_path: str | None = os.environ.get("LLAMA_SERVER_PATH")
    model_path: str | None        = os.environ.get("MODEL_PATH")
    mmproj_path: str | None       = os.environ.get("MMPROJ_PATH")
    server_base_port: int  = _env_int("OCR_SERVER_BASE_PORT", 8080)  # 8080, 8081, … (one per server)
    server_timeout: int    = _env_int("OCR_SERVER_TIMEOUT", 60)      # seconds before declaring the server dead

    # ── llama-server parameters (tuning) ─────────────────────────────────────
    # Every value below is a conservative default that runs anywhere. Machine
    # specific tuning belongs in .env (OCR_* variables, see .env.example), which
    # is gitignored; CLI flags still take precedence over both.
    #
    # n_ctx: None = auto (n_parallel * 2048), resolved in __post_init__. Override
    # for books with large/dense tables (e.g. n_parallel * 4096).
    n_ctx: int | None     = None
    n_gpu_layers: int     = _env_int("OCR_N_GPU_LAYERS", 99)
    n_batch: int          = _env_int("OCR_N_BATCH", 512)
    n_ubatch: int         = _env_int("OCR_N_UBATCH", 512)
    n_threads: int        = _env_int("OCR_N_THREADS", 4)      # P-cores
    prio: int             = _env_int("OCR_PRIO", 2)
    kv_offload: bool      = _env_bool("OCR_KV_OFFLOAD", True)
    max_tokens: int       = _env_int("OCR_MAX_TOKENS", 4096)
    temperature: float    = 0.0
    # n_parallel > 1 requires apply_paddlex_patch_parallel.py; leave at 1 otherwise.
    n_parallel: int       = _env_int("OCR_N_PARALLEL", 1)
    n_servers: int        = _env_int("OCR_N_SERVERS", 1)      # parallel llama-server instances
    page_timeout: int     = _env_int("OCR_PAGE_TIMEOUT", 120) # max seconds per page (0 = disabled)

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
        # Resolved here rather than in the field default so that the CLI passing
        # n_ctx=None (its default) still falls back to OCR_N_CTX, then to auto.
        if self.n_ctx is None:
            self.n_ctx = _env_opt_int("OCR_N_CTX") or self.n_parallel * 2048

    def validate_ocr_paths(self) -> None:
        """Raise ValueError listing any missing required OCR server paths."""
        missing = [
            name for name, val in [
                ("llama_server_path", self.llama_server_path),
                ("model_path", self.model_path),
                ("mmproj_path", self.mmproj_path),
            ]
            if not val
        ]
        if missing:
            raise ValueError(
                f"Missing required configuration: {', '.join(missing)}.\n"
                "Set them via:\n"
                "  CLI: --llama-server PATH --model PATH --mmproj PATH\n"
                "  Env: LLAMA_SERVER_PATH, MODEL_PATH, MMPROJ_PATH\n"
                "  Or edit src/config.py directly."
            )

    @property
    def images_path(self) -> Path:
        return Path(self.images_dir)

    @property
    def output_path(self) -> Path:
        return Path(self.output_file)

    @property
    def figures_path(self) -> Path:
        return Path(self.figures_dir)
