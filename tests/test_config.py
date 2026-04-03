"""
test_config.py — Tests unitaires pour config.py

Couvre :
  - Valeurs par défaut (modèle, quantization, tokens, détection de boucle, prompts)
  - Propriété prompt (tous les modes, mode rec, mode inconnu)
  - Propriétés images_path et output_path
  - Surcharge partielle via le constructeur
  - to_model_config() et to_sampler_config()
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config import Config


class TestConfigDefaults:

    def test_model_default(self):
        assert Config().model == "NexaAI/DeepSeek-OCR-GGUF"

    def test_quant_default(self):
        assert Config().quant == "bf16"

    def test_quants_available(self):
        assert "q8_0" in Config.QUANTS
        assert "bf16" in Config.QUANTS

    def test_max_tokens_default(self):
        assert Config().max_tokens == 4096

    def test_temperature_default(self):
        assert Config().temperature == 0.0

    def test_prompt_mode_default(self):
        assert Config().prompt_mode == "plain"

    def test_loop_check_every_default(self):
        assert Config().loop_check_every == 200

    def test_loop_window_words_default(self):
        assert Config().loop_window_words == 50

    def test_loop_divisor_threshold_default(self):
        assert Config().loop_divisor_threshold == 0.7

    def test_locate_target_default(self):
        assert Config().locate_target == "everything"

    def test_images_dir_default(self):
        assert Config().images_dir == "./photos"

    def test_extensions_default(self):
        cfg = Config()
        assert ".jpg" in cfg.extensions
        assert ".jpeg" in cfg.extensions
        assert ".png" in cfg.extensions
        assert ".webp" in cfg.extensions

    def test_output_file_default(self):
        assert Config().output_file == "./output/livre.md"

    def test_resume_default(self):
        assert Config().resume is True

    def test_cleanups_all_enabled_by_default(self):
        cfg = Config()
        assert cfg.remove_isolated_page_numbers is True
        assert cfg.rejoin_hyphenated_words is True
        assert cfg.collapse_blank_lines is True

    def test_verbose_default_false(self):
        assert Config().verbose is False

    def test_prompts_dict_has_five_modes(self):
        prompts = Config.PROMPTS
        assert "plain" in prompts
        assert "layout" in prompts
        assert "describe" in prompts
        assert "parse" in prompts
        assert "rec" in prompts


class TestConfigPromptProperty:

    def test_prompt_plain_mode(self):
        cfg = Config(prompt_mode="plain")
        assert "Free OCR" in cfg.prompt

    def test_prompt_layout_mode(self):
        cfg = Config(prompt_mode="layout")
        assert "grounding" in cfg.prompt or "Convert" in cfg.prompt

    def test_prompt_describe_mode(self):
        cfg = Config(prompt_mode="describe")
        assert "Describe" in cfg.prompt or "describe" in cfg.prompt.lower()

    def test_prompt_parse_mode(self):
        cfg = Config(prompt_mode="parse")
        assert "Parse" in cfg.prompt or "parse" in cfg.prompt.lower()

    def test_prompt_rec_mode_contains_locate(self):
        cfg = Config(prompt_mode="rec", locate_target="A figure or graph")
        assert "Locate" in cfg.prompt or "locate" in cfg.prompt.lower()

    def test_prompt_rec_mode_includes_locate_target(self):
        cfg = Config(prompt_mode="rec", locate_target="un graphique")
        assert "un graphique" in cfg.prompt

    def test_prompt_unknown_mode_falls_back_to_plain(self):
        cfg = Config(prompt_mode="inexistant")
        assert cfg.prompt == Config.PROMPTS["plain"]

    def test_prompt_is_string(self):
        for mode in ["plain", "layout", "describe", "parse"]:
            cfg = Config(prompt_mode=mode)
            assert isinstance(cfg.prompt, str)


class TestConfigPathProperties:

    def test_images_path_returns_path_object(self):
        assert isinstance(Config(images_dir="./photos").images_path, Path)

    def test_images_path_value(self):
        assert Config(images_dir="/tmp/photos").images_path == Path("/tmp/photos")

    def test_output_path_returns_path_object(self):
        assert isinstance(Config(output_file="livre.md").output_path, Path)

    def test_output_path_value(self):
        assert Config(output_file="/tmp/sortie.md").output_path == Path("/tmp/sortie.md")


class TestConfigPartialOverride:

    def test_override_single_field(self):
        cfg = Config(max_tokens=1024)
        assert cfg.max_tokens == 1024
        assert cfg.model == "NexaAI/DeepSeek-OCR-GGUF"

    def test_override_quant(self):
        cfg = Config(quant="q8_0")
        assert cfg.quant == "q8_0"

    def test_override_multiple_fields(self):
        cfg = Config(max_tokens=512, verbose=True, quant="q8_0")
        assert cfg.max_tokens == 512
        assert cfg.verbose is True
        assert cfg.quant == "q8_0"
        assert cfg.temperature == 0.0

    def test_override_resume_false(self):
        assert Config(resume=False).resume is False

    def test_override_loop_params(self):
        cfg = Config(loop_check_every=100, loop_window_words=30, loop_divisor_threshold=0.5)
        assert cfg.loop_check_every == 100
        assert cfg.loop_window_words == 30
        assert cfg.loop_divisor_threshold == 0.5

    def test_two_configs_are_independent(self):
        cfg1 = Config(max_tokens=1000)
        cfg2 = Config(max_tokens=2000)
        assert cfg1.max_tokens != cfg2.max_tokens


class TestConfigModelConfig:

    def test_to_model_config_returns_object(self):
        cfg = Config()
        result = cfg.to_model_config()
        assert result is not None

    def test_to_sampler_config_returns_object(self):
        cfg = Config()
        result = cfg.to_sampler_config()
        assert result is not None

    def test_to_model_config_uses_n_ctx(self):
        cfg = Config(n_ctx=4096)
        mc = cfg.to_model_config()
        assert mc.n_ctx == 4096

    def test_to_sampler_config_uses_temperature(self):
        cfg = Config(temperature=0.5)
        sc = cfg.to_sampler_config()
        assert sc.temperature == pytest.approx(0.5)
