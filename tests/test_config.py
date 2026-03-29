"""
test_config.py — Tests unitaires pour config.py

Couvre :
  - Valeurs par défaut
  - Propriété prompt (tous les modes + mode inconnu)
  - Propriétés images_path et output_path (conversion str → Path)
  - Surcharge partielle via le constructeur
  - Immutabilité logique du dictionnaire PROMPTS
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import Config


class TestConfigDefaults:
    """Vérifie que les valeurs par défaut correspondent aux spécifications."""

    def test_model_default(self):
        cfg = Config()
        assert cfg.model == "NexaAI/DeepSeek-OCR-GGUF"

    def test_port_default(self):
        cfg = Config()
        assert cfg.port == 18181

    def test_server_timeout_default(self):
        cfg = Config()
        assert cfg.server_timeout_s == 60

    def test_max_tokens_default(self):
        cfg = Config()
        assert cfg.max_tokens == 2048

    def test_temperature_default(self):
        cfg = Config()
        assert cfg.temperature == 0.0

    def test_request_timeout_default(self):
        cfg = Config()
        assert cfg.request_timeout_s == 180

    def test_prompt_mode_default(self):
        cfg = Config()
        assert cfg.prompt_mode == "markdown"

    def test_images_dir_default(self):
        cfg = Config()
        assert cfg.images_dir == "./photos"

    def test_extensions_default(self):
        cfg = Config()
        assert ".jpg" in cfg.extensions
        assert ".jpeg" in cfg.extensions
        assert ".png" in cfg.extensions
        assert ".webp" in cfg.extensions

    def test_output_file_default(self):
        cfg = Config()
        assert cfg.output_file == "./output/livre.md"

    def test_resume_default(self):
        cfg = Config()
        assert cfg.resume is True

    def test_cleanups_all_enabled_by_default(self):
        cfg = Config()
        assert cfg.remove_isolated_page_numbers is True
        assert cfg.rejoin_hyphenated_words is True
        assert cfg.collapse_blank_lines is True

    def test_verbose_default_false(self):
        cfg = Config()
        assert cfg.verbose is False

    def test_prompts_dict_has_three_modes(self):
        cfg = Config()
        assert "markdown" in cfg.PROMPTS
        assert "plain" in cfg.PROMPTS
        assert "figure" in cfg.PROMPTS


class TestConfigPromptProperty:
    """Vérifie la propriété calculée prompt."""

    def test_prompt_markdown_mode(self):
        cfg = Config(prompt_mode="markdown")
        assert "markdown" in cfg.prompt.lower() or "grounding" in cfg.prompt

    def test_prompt_plain_mode(self):
        cfg = Config(prompt_mode="plain")
        assert "Free OCR" in cfg.prompt

    def test_prompt_figure_mode(self):
        cfg = Config(prompt_mode="figure")
        assert "figure" in cfg.prompt.lower() or "Parse" in cfg.prompt

    def test_prompt_unknown_mode_falls_back_to_markdown(self):
        cfg = Config(prompt_mode="inexistant")
        # Doit retourner le prompt markdown par défaut sans lever d'exception
        assert cfg.prompt == cfg.PROMPTS["markdown"]

    def test_prompt_contains_image_tag(self):
        """Tous les prompts doivent commencer par <image> pour DeepSeek-OCR."""
        cfg = Config()
        for mode in ["markdown", "plain", "figure"]:
            cfg.prompt_mode = mode
            assert cfg.prompt.startswith("<image>"), (
                f"Le prompt '{mode}' ne commence pas par <image>"
            )

    def test_prompt_is_string(self):
        cfg = Config()
        assert isinstance(cfg.prompt, str)


class TestConfigPathProperties:
    """Vérifie les propriétés images_path et output_path."""

    def test_images_path_returns_path_object(self):
        cfg = Config(images_dir="./photos")
        assert isinstance(cfg.images_path, Path)

    def test_images_path_value(self):
        cfg = Config(images_dir="/tmp/mes_photos")
        assert cfg.images_path == Path("/tmp/mes_photos")

    def test_output_path_returns_path_object(self):
        cfg = Config(output_file="livre.md")
        assert isinstance(cfg.output_path, Path)

    def test_output_path_value(self):
        cfg = Config(output_file="/tmp/sortie.md")
        assert cfg.output_path == Path("/tmp/sortie.md")

    def test_images_path_computed_from_images_dir(self):
        cfg = Config(images_dir="custom_folder")
        assert cfg.images_path == Path("custom_folder")

    def test_output_path_computed_from_output_file(self):
        cfg = Config(output_file="custom_output.md")
        assert cfg.output_path == Path("custom_output.md")


class TestConfigPartialOverride:
    """Vérifie que le constructeur permet de ne surcharger que certains champs."""

    def test_override_single_field(self):
        cfg = Config(port=9999)
        assert cfg.port == 9999
        assert cfg.model == "NexaAI/DeepSeek-OCR-GGUF"  # valeur par défaut conservée

    def test_override_multiple_fields(self):
        cfg = Config(port=8888, max_tokens=512, verbose=True)
        assert cfg.port == 8888
        assert cfg.max_tokens == 512
        assert cfg.verbose is True
        assert cfg.temperature == 0.0  # non modifié

    def test_override_resume_false(self):
        cfg = Config(resume=False)
        assert cfg.resume is False

    def test_override_cleanups(self):
        cfg = Config(
            remove_isolated_page_numbers=False,
            rejoin_hyphenated_words=False,
            collapse_blank_lines=False,
        )
        assert cfg.remove_isolated_page_numbers is False
        assert cfg.rejoin_hyphenated_words is False
        assert cfg.collapse_blank_lines is False

    def test_two_configs_are_independent(self):
        """Deux Config créés séparément ne partagent pas leur état."""
        cfg1 = Config(port=1111)
        cfg2 = Config(port=2222)
        assert cfg1.port != cfg2.port

    def test_prompts_dict_is_independent_per_instance(self):
        """Chaque instance a son propre dict PROMPTS (pas de partage via default_factory)."""
        cfg1 = Config()
        cfg2 = Config()
        cfg1.PROMPTS["custom"] = "custom_prompt"
        assert "custom" not in cfg2.PROMPTS
