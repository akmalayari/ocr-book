"""
config.py — Configuration centrale du pipeline OCR
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ── Modèle VLM ───────────────────────────────────────────────────────────
    model: str = "NexaAI/DeepSeek-OCR-GGUF"

    # ── Paramètres ModelConfig ────────────────────────────────────────────────
    n_ctx: int        = 4096
    n_threads: int    = 4
    n_gpu_layers: int = 999
    n_batch: int      = 2048

    # ── Inférence ────────────────────────────────────────────────────────────
    max_tokens: int   = 2048
    temperature: float = 0.0

    # ── Pré-traitement ───────────────────────────────────────────────────────
    #   "none"     → image originale
    #   "binarize" → binarisation adaptative (GAUSSIAN_C, blockSize=31, C=10)
    preprocess_mode: str = "binarize"

    # ── Prompts disponibles ──────────────────────────────────────────────────
    #   "markdown"  → structure complète (titres, tableaux, listes)
    #   "plain"     → texte brut sans mise en forme
    #   "figure"    → analyse d'une figure ou d'un graphique
    prompt_mode: str = "markdown"
    PROMPTS: dict = field(default_factory=lambda: {
        "markdown": "Convert the document to markdown.",
        "plain":    "Free OCR.",
        "figure":   "Parse the figure.",
    })

    # ── Images ───────────────────────────────────────────────────────────────
    images_dir: str = "./photos"
    extensions: tuple = (".jpg", ".jpeg", ".png", ".webp")

    # ── Sortie ───────────────────────────────────────────────────────────────
    output_file: str = "./output/livre.md"
    resume: bool = True

    # ── Post-traitement ──────────────────────────────────────────────────────
    remove_isolated_page_numbers: bool = True
    rejoin_hyphenated_words: bool = True
    collapse_blank_lines: bool = True

    # ── Logging ──────────────────────────────────────────────────────────────
    log_file: str = "output/ocr_run.log"
    verbose: bool = False

    @property
    def prompt(self) -> str:
        return self.PROMPTS.get(self.prompt_mode, self.PROMPTS["markdown"])

    @property
    def images_path(self) -> Path:
        return Path(self.images_dir)

    @property
    def output_path(self) -> Path:
        return Path(self.output_file)

    def to_model_config(self):
        from nexaai.nexa_sdk.types import ModelConfig
        return ModelConfig(
            n_ctx=self.n_ctx,
            n_threads=self.n_threads,
            n_gpu_layers=self.n_gpu_layers,
            n_batch=self.n_batch,
        )
