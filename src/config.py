"""
config.py — Configuration centrale du pipeline OCR
"""

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Config:
    # ── Serveur Nexa ──────────────────────────────────────────────────────────
    model: str = "NexaAI/DeepSeek-OCR-GGUF"
    port: int = 18181
    server_timeout_s: int = 60       # attente max démarrage serveur

    # ── Inférence ────────────────────────────────────────────────────────────
    max_tokens: int = 2048
    temperature: float = 0.0
    request_timeout_s: int = 180     # timeout par image

    # ── Prompts disponibles ──────────────────────────────────────────────────
    #   "markdown"  → structure complète (titres, tableaux, listes)
    #   "plain"     → texte brut sans mise en forme
    #   "figure"    → analyse d'une figure ou d'un graphique
    prompt_mode: str = "markdown"
    PROMPTS: dict = field(default_factory=lambda: {
        "markdown": "<image>\n<|grounding|>Convert the document to markdown.",
        "plain":    "<image>\nFree OCR.",
        "figure":   "<image>\nParse the figure.",
    })

    # ── Images ───────────────────────────────────────────────────────────────
    images_dir: str = "./photos"
    extensions: tuple = (".jpg", ".jpeg", ".png", ".webp")

    # ── Sortie ───────────────────────────────────────────────────────────────
    output_file: str = "./output/livre.md"
    resume: bool = True              # reprendre si interruption

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
