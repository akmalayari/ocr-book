"""
config.py — Configuration centrale du pipeline OCR
"""

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar


@dataclass
class Config:
    # ── Modèle VLM ───────────────────────────────────────────────────────────
    model: str = "NexaAI/DeepSeek-OCR-GGUF"
    quant: str = "bf16"
    QUANTS: ClassVar[tuple] = ("q8_0", "bf16")

    # ── Paramètres ModelConfig ────────────────────────────────────────────────
    n_ctx: int        = 8192
    n_threads: int    = 4
    n_gpu_layers: int = 999
    n_batch: int      = 1024

    # ── Inférence ────────────────────────────────────────────────────────────
    max_tokens: int        = 2048
    temperature: float     = 0.0
    repetition_penalty: float = 1.5

    # ── Pré-traitement ───────────────────────────────────────────────────────
    #   "none"     → image originale
    #   "binarize" → Gaussian blur et binarisation adaptative GAUSSIAN_C
    preprocess_mode: str = "binarize"
    binarize_block_size: int = 31
    binarize_c: int = 10

    blur_ksize: int = 5
    blur_sigma: float = 0.0

    # ── Prompts disponibles ──────────────────────────────────────────────────
    #   "plain"     → texte brut sans mise en forme
    #   "layout"    → texte brut avec balises spatiales
    #   "describe"  → description générale de l'image
    #   "parse"     → analyse détaillée des éléments de l'image
    #   "rec"       → localisation d'un élément (requiert locate_target)
    prompt_mode: str = "plain"
    PROMPTS: ClassVar[dict] = {
        "plain":    "Free OCR.",
        "layout":   "<|grounding|>Convert the document to markdown.",
        "describe": "Describe this image in detail.",
        "parse":    "Parse the figure.",
        "rec":      "Locate <|ref|>{target}<|/ref|> in the image.",
    }
    locate_target: str = "everything"

    # ── Images ───────────────────────────────────────────────────────────────
    rename_prefix: str = "page"
    images_dir: str    = "./photos"
    extensions: tuple  = (".jpg", ".jpeg", ".png", ".webp", ".pdf")

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
        template = self.PROMPTS.get(self.prompt_mode, self.PROMPTS[Config.prompt_mode])
        if self.prompt_mode == "rec":
            return template.format(target=self.locate_target)
        return template

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

    def to_sampler_config(self):
        from nexaai.nexa_sdk.types import SamplerConfig
        return SamplerConfig(
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
        )
