"""
config.py — Central configuration for the OCR pipeline
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ── llama-server ─────────────────────────────────────────────────────────
    llama_server_path: str = r"C:\path\to\llama.cpp\llama-b8683-bin-win-vulkan-x64\llama-server.exe"
    model_path: str        = r"C:\path\to\models\PaddlePaddle-PaddleOCR-VL-1.5-GGUF\PaddleOCR-VL-1.5.gguf"
    mmproj_path: str       = r"C:\path\to\models\PaddlePaddle-PaddleOCR-VL-1.5-GGUF\PaddleOCR-VL-1.5-mmproj.gguf"
    server_base_port: int  = 8080   # ports 8080, 8081, … (one per server)
    server_timeout: int    = 60     # seconds to wait before declaring the server dead
    n_servers: int         = 1      # number of parallel llama-server instances

    # ── llama-server parameters (tuning) ─────────────────────────────────────
    n_ctx: int            = 6144   # 2048 tokens/slot × np=3
    n_gpu_layers: int     = 99
    n_batch: int          = 512
    n_ubatch: int         = 512
    n_threads: int        = 4      # P-cores
    prio: int             = 2
    kv_offload: bool      = True
    max_tokens: int       = 4096
    temperature: float    = 0.0
    n_parallel: int       = 3      # intra-page parallelism (apply_paddlex_patch_parallel.py required)
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
    temp_dir: Path = field(default_factory=lambda: Path("output/temp"))

    # ── Output ───────────────────────────────────────────────────────────────
    output_file: str = "./output/book.md"
    figures_dir: str = "./output/figures"
    resume: bool     = True

    # ── Post-processing ──────────────────────────────────────────────────────
    postprocess: bool                   = True
    mode: str                           = "base"   # "base" | "obsidian"
    vault_root: str | None               = "C:/path/to/Documents/Classique Obsidian"  # vault root
    vault_path: str | None              = "Documents/OCR"          # output subfolder, relative to vault_root
    vault_figures_dir: str | None       = "Files/OCR"    # figures path relative to vault_root
    remove_isolated_page_numbers: bool  = True
    rejoin_hyphenated_words: bool       = True
    collapse_blank_lines: bool          = True
    # List of (line_start_regex, level) for header detection on the final file.
    # None = disabled (prompt at launch). [] = disabled without prompt.
    # Ex: [("^[IVX]+\\.", 2), ("^[A-Z]\\.", 3)]
    header_patterns: list[tuple[str, int]] | None = field(default_factory=list)

    # ── Logging ──────────────────────────────────────────────────────────────
    log_file: str    = "output/ocr_run.log"
    report_file: str = "output/ocr_report.md"
    verbose: bool    = False

    @property
    def images_path(self) -> Path:
        return Path(self.images_dir)

    @property
    def output_path(self) -> Path:
        return Path(self.output_file)

    @property
    def figures_path(self) -> Path:
        return Path(self.figures_dir)
