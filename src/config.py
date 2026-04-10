"""
config.py — Configuration centrale du pipeline OCR
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # ── llama-server ─────────────────────────────────────────────────────────
    llama_server_path: str = r"C:\path\to\llama.cpp\llama-b8683-bin-win-vulkan-x64\llama-server.exe"
    model_path: str        = r"C:\path\to\models\PaddlePaddle-PaddleOCR-VL-1.5-GGUF\PaddleOCR-VL-1.5.gguf"
    mmproj_path: str       = r"C:\path\to\models\PaddlePaddle-PaddleOCR-VL-1.5-GGUF\PaddleOCR-VL-1.5-mmproj.gguf"
    server_base_port: int  = 8080   # ports 8080, 8081, … (un par serveur)
    server_timeout: int    = 60     # secondes à attendre avant de déclarer le serveur mort
    n_servers: int         = 1      # nombre de llama-server parallèles

    # ── Paramètres llama-server (tuning) ─────────────────────────────────────
    n_ctx: int            = 12288   # 2048 tokens/slot × np=3
    n_gpu_layers: int     = 99
    n_batch: int          = 512
    n_ubatch: int         = 512
    n_threads: int        = 4      # P-cores
    prio: int             = 2
    kv_offload: bool      = True
    max_tokens: int       = 4096
    temperature: float    = 0.0
    n_parallel: int       = 3      # intra-page parallelism (apply_paddlex_patch_parallel.py requis)
    page_timeout: int     = 120    # secondes max par page avant abandon (0 = désactivé)

    # ── PaddleOCR ─────────────────────────────────────────────────────────────
    use_layout_detection: bool = True   # False = fallback sans layout

    # ── Images ───────────────────────────────────────────────────────────────
    rename_prefix: str          = "page"
    images_dir: str             = "./photos"
    extensions: tuple           = (".jpg", ".jpeg", ".png", ".webp")
    image_files: list | None    = None   # si fourni, court-circuite images_dir

    # ── Sortie ───────────────────────────────────────────────────────────────
    output_file: str = "./output/livre.md"
    figures_dir: str = "./output/figures"
    resume: bool     = True

    # ── Post-traitement ──────────────────────────────────────────────────────
    postprocess: bool                   = True
    mode: str                           = "base"   # "base" | "obsidian"
    vault_path: str | None              = "C:/path/to/Documents/Classique Obsidian/Documents/OCR"
    vault_figures_dir: str | None       = "Files"      # chemin figures relatif à la racine du vault
    remove_isolated_page_numbers: bool  = True
    rejoin_hyphenated_words: bool       = True
    collapse_blank_lines: bool          = True
    # Liste de (regex_début_de_ligne, niveau) pour la détection de headers sur le fichier final.
    # None = désactivé (prompt au lancement). [] = désactivé sans prompt.
    # Ex : [("^[IVX]+\\.", 2), ("^[A-Z]\\.", 3)]
    header_patterns: list[tuple[str, int]] | None = None

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
