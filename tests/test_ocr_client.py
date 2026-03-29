"""
test_ocr_client.py — Tests unitaires rigoureux pour ocr_client.py

Stratégie :
  - requests.post est TOUJOURS mocké : aucun appel réseau réel.
  - Les images utilisées sont de vrais fichiers JPEG/PNG minimaux (conftest).
  - On teste exhaustivement les chemins d'erreur (Timeout, HTTPError,
    ConnectionError, réponse vide, contenu vide, fichier absent).
  - On vérifie la structure exacte du payload envoyé au serveur.

Couvre :
  - _encode_image   : base64 correct, MIME correct pour chaque extension
  - ocr_image       : succès, tous les cas d'erreur, structure du payload,
                      acceptance Path ou str, strip du texte retourné
  - OCRError        : hiérarchie d'héritage
"""

import base64
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import Config
from ocr_client import OCRError, _encode_image, ocr_image

# ── Octets minimaux ───────────────────────────────────────────────────────────

JPEG_BYTES = bytes([0xFF, 0xD8, 0xFF, 0xD9])
PNG_BYTES  = bytes([0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A,
                    0x00, 0x00, 0x00, 0x00, 0x49, 0x45, 0x4E, 0x44])


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def cfg():
    return Config(port=19999, model="NexaAI/TestModel", request_timeout_s=10,
                  prompt_mode="markdown", max_tokens=512, temperature=0.0, log_file="")


@pytest.fixture
def jpeg_file(tmp_path):
    p = tmp_path / "page_001.jpg"
    p.write_bytes(JPEG_BYTES)
    return p


@pytest.fixture
def png_file(tmp_path):
    p = tmp_path / "page_001.png"
    p.write_bytes(PNG_BYTES)
    return p


def make_http_response(content: str, status_code: int = 200) -> MagicMock:
    """Construit un mock requests.Response avec une réponse OCR valide."""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = {
        "choices": [{"message": {"content": content}}]
    }
    resp.raise_for_status = MagicMock()
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(
            f"{status_code}", response=resp
        )
    return resp


# ── OCRError ──────────────────────────────────────────────────────────────────

class TestOCRError:

    def test_is_runtime_error_subclass(self):
        assert issubclass(OCRError, RuntimeError)

    def test_message_preserved(self):
        err = OCRError("image introuvable")
        assert "image introuvable" in str(err)

    def test_can_be_caught_as_runtime_error(self):
        with pytest.raises(RuntimeError):
            raise OCRError("test")


# ── _encode_image ─────────────────────────────────────────────────────────────

class TestEncodeImage:

    def test_returns_tuple_of_two_strings(self, jpeg_file):
        result = _encode_image(jpeg_file)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert all(isinstance(x, str) for x in result)

    def test_base64_is_decodable(self, jpeg_file):
        b64, _ = _encode_image(jpeg_file)
        decoded = base64.b64decode(b64)
        assert decoded == JPEG_BYTES

    def test_base64_matches_file_content(self, tmp_path):
        content = b"contenu_test_arbitraire"
        f = tmp_path / "test.jpg"
        f.write_bytes(content)
        b64, _ = _encode_image(f)
        assert base64.b64decode(b64) == content

    def test_mime_jpg(self, tmp_path):
        f = tmp_path / "img.jpg"
        f.write_bytes(JPEG_BYTES)
        _, mime = _encode_image(f)
        assert mime == "image/jpeg"

    def test_mime_jpeg(self, tmp_path):
        f = tmp_path / "img.jpeg"
        f.write_bytes(JPEG_BYTES)
        _, mime = _encode_image(f)
        assert mime == "image/jpeg"

    def test_mime_png(self, png_file):
        _, mime = _encode_image(png_file)
        assert mime == "image/png"

    def test_mime_webp(self, tmp_path):
        f = tmp_path / "img.webp"
        f.write_bytes(b"RIFF" + b"\x00" * 4 + b"WEBP")
        _, mime = _encode_image(f)
        assert mime == "image/webp"

    def test_unknown_extension_defaults_to_jpeg(self, tmp_path):
        f = tmp_path / "img.bmp"
        f.write_bytes(b"BM")
        _, mime = _encode_image(f)
        assert mime == "image/jpeg"

    def test_case_insensitive_extension(self, tmp_path):
        f = tmp_path / "img.JPG"
        f.write_bytes(JPEG_BYTES)
        _, mime = _encode_image(f)
        assert mime == "image/jpeg"


# ── ocr_image : succès ────────────────────────────────────────────────────────

class TestOcrImageSuccess:

    def test_returns_string(self, jpeg_file, cfg):
        resp = make_http_response("## Chapitre 1\n\nTexte.")
        with patch("ocr_client.requests.post", return_value=resp):
            result = ocr_image(jpeg_file, cfg)
        assert isinstance(result, str)

    def test_returns_correct_text(self, jpeg_file, cfg):
        expected = "## Titre\n\nParagraphe de texte."
        resp = make_http_response(expected)
        with patch("ocr_client.requests.post", return_value=resp):
            result = ocr_image(jpeg_file, cfg)
        assert result == expected

    def test_strips_leading_trailing_whitespace(self, jpeg_file, cfg):
        resp = make_http_response("  \n\nTexte avec espaces\n\n  ")
        with patch("ocr_client.requests.post", return_value=resp):
            result = ocr_image(jpeg_file, cfg)
        assert not result.startswith(" ")
        assert not result.endswith(" ")

    def test_accepts_path_object(self, jpeg_file, cfg):
        resp = make_http_response("Texte")
        with patch("ocr_client.requests.post", return_value=resp):
            result = ocr_image(jpeg_file, cfg)   # Path object
        assert result == "Texte"

    def test_accepts_string_path(self, jpeg_file, cfg):
        resp = make_http_response("Texte")
        with patch("ocr_client.requests.post", return_value=resp):
            result = ocr_image(str(jpeg_file), cfg)   # str
        assert result == "Texte"

    def test_works_with_png(self, png_file, cfg):
        resp = make_http_response("Texte depuis PNG")
        with patch("ocr_client.requests.post", return_value=resp):
            result = ocr_image(png_file, cfg)
        assert result == "Texte depuis PNG"


# ── ocr_image : structure du payload ─────────────────────────────────────────

class TestOcrImagePayload:

    def _get_payload(self, jpeg_file, cfg):
        resp = make_http_response("Texte")
        with patch("ocr_client.requests.post", return_value=resp) as mock_post:
            ocr_image(jpeg_file, cfg)
        _, kwargs = mock_post.call_args
        return kwargs["json"]

    def test_payload_contains_model(self, jpeg_file, cfg):
        payload = self._get_payload(jpeg_file, cfg)
        assert payload["model"] == cfg.model

    def test_payload_contains_max_tokens(self, jpeg_file, cfg):
        payload = self._get_payload(jpeg_file, cfg)
        assert payload["max_completion_tokens"] == cfg.max_tokens

    def test_payload_temperature_is_zero(self, jpeg_file, cfg):
        payload = self._get_payload(jpeg_file, cfg)
        assert payload["temperature"] == 0.0

    def test_payload_stream_is_false(self, jpeg_file, cfg):
        payload = self._get_payload(jpeg_file, cfg)
        assert payload["stream"] is False

    def test_payload_has_one_message(self, jpeg_file, cfg):
        payload = self._get_payload(jpeg_file, cfg)
        assert len(payload["messages"]) == 1

    def test_message_role_is_user(self, jpeg_file, cfg):
        payload = self._get_payload(jpeg_file, cfg)
        assert payload["messages"][0]["role"] == "user"

    def test_message_content_has_two_parts(self, jpeg_file, cfg):
        payload = self._get_payload(jpeg_file, cfg)
        content = payload["messages"][0]["content"]
        assert len(content) == 2

    def test_first_content_part_is_image(self, jpeg_file, cfg):
        payload = self._get_payload(jpeg_file, cfg)
        first = payload["messages"][0]["content"][0]
        assert first["type"] == "image_url"
        assert "image_url" in first
        assert "url" in first["image_url"]

    def test_image_url_is_data_uri(self, jpeg_file, cfg):
        payload = self._get_payload(jpeg_file, cfg)
        url = payload["messages"][0]["content"][0]["image_url"]["url"]
        assert url.startswith("data:")
        assert ";base64," in url

    def test_image_url_contains_correct_mime(self, jpeg_file, cfg):
        payload = self._get_payload(jpeg_file, cfg)
        url = payload["messages"][0]["content"][0]["image_url"]["url"]
        assert "image/jpeg" in url

    def test_image_url_base64_matches_file(self, jpeg_file, cfg):
        payload = self._get_payload(jpeg_file, cfg)
        url = payload["messages"][0]["content"][0]["image_url"]["url"]
        b64_part = url.split(";base64,")[1]
        decoded = base64.b64decode(b64_part)
        assert decoded == jpeg_file.read_bytes()

    def test_second_content_part_is_text(self, jpeg_file, cfg):
        payload = self._get_payload(jpeg_file, cfg)
        second = payload["messages"][0]["content"][1]
        assert second["type"] == "text"

    def test_text_contains_prompt(self, jpeg_file, cfg):
        payload = self._get_payload(jpeg_file, cfg)
        text_part = payload["messages"][0]["content"][1]["text"]
        assert text_part == cfg.prompt

    def test_url_targets_correct_endpoint(self, jpeg_file, cfg):
        resp = make_http_response("Texte")
        with patch("ocr_client.requests.post", return_value=resp) as mock_post:
            ocr_image(jpeg_file, cfg)
        called_url = mock_post.call_args[0][0]
        assert str(cfg.port) in called_url
        assert "/v1/chat/completions" in called_url

    def test_request_uses_configured_timeout(self, jpeg_file, cfg):
        resp = make_http_response("Texte")
        with patch("ocr_client.requests.post", return_value=resp) as mock_post:
            ocr_image(jpeg_file, cfg)
        _, kwargs = mock_post.call_args
        assert kwargs.get("timeout") == cfg.request_timeout_s


# ── ocr_image : gestion des erreurs ──────────────────────────────────────────

class TestOcrImageErrors:

    def test_raises_ocr_error_if_file_not_found(self, tmp_path, cfg):
        with pytest.raises(OCRError, match="introuvable"):
            ocr_image(tmp_path / "inexistant.jpg", cfg)

    def test_raises_ocr_error_on_timeout(self, jpeg_file, cfg):
        with patch("ocr_client.requests.post", side_effect=requests.Timeout):
            with pytest.raises(OCRError, match="Timeout"):
                ocr_image(jpeg_file, cfg)

    def test_timeout_message_contains_filename(self, jpeg_file, cfg):
        with patch("ocr_client.requests.post", side_effect=requests.Timeout):
            with pytest.raises(OCRError, match=jpeg_file.name):
                ocr_image(jpeg_file, cfg)

    def test_timeout_message_contains_timeout_value(self, jpeg_file, cfg):
        with patch("ocr_client.requests.post", side_effect=requests.Timeout):
            with pytest.raises(OCRError, match=str(cfg.request_timeout_s)):
                ocr_image(jpeg_file, cfg)

    def test_raises_ocr_error_on_http_error_400(self, jpeg_file, cfg):
        resp = make_http_response("", status_code=400)
        with patch("ocr_client.requests.post", return_value=resp):
            with pytest.raises(OCRError, match="HTTP"):
                ocr_image(jpeg_file, cfg)

    def test_raises_ocr_error_on_http_error_500(self, jpeg_file, cfg):
        resp = make_http_response("", status_code=500)
        with patch("ocr_client.requests.post", return_value=resp):
            with pytest.raises(OCRError, match="HTTP"):
                ocr_image(jpeg_file, cfg)

    def test_raises_ocr_error_on_connection_error(self, jpeg_file, cfg):
        with patch("ocr_client.requests.post", side_effect=requests.ConnectionError):
            with pytest.raises(OCRError, match="joindre"):
                ocr_image(jpeg_file, cfg)

    def test_connection_error_message_contains_port(self, jpeg_file, cfg):
        with patch("ocr_client.requests.post", side_effect=requests.ConnectionError):
            with pytest.raises(OCRError, match=str(cfg.port)):
                ocr_image(jpeg_file, cfg)

    def test_raises_ocr_error_on_empty_choices(self, jpeg_file, cfg):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"choices": []}
        with patch("ocr_client.requests.post", return_value=resp):
            with pytest.raises(OCRError, match="vide"):
                ocr_image(jpeg_file, cfg)

    def test_raises_ocr_error_on_whitespace_only_content(self, jpeg_file, cfg):
        resp = make_http_response("    \n\n    ")  # strip() → ""
        with patch("ocr_client.requests.post", return_value=resp):
            with pytest.raises(OCRError, match="vide"):
                ocr_image(jpeg_file, cfg)

    def test_raises_ocr_error_on_missing_choices_key(self, jpeg_file, cfg):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {}  # pas de clé "choices"
        with patch("ocr_client.requests.post", return_value=resp):
            with pytest.raises(OCRError):
                ocr_image(jpeg_file, cfg)

    def test_raises_ocr_error_on_missing_content_key(self, jpeg_file, cfg):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {
            "choices": [{"message": {}}]  # pas de clé "content"
        }
        with patch("ocr_client.requests.post", return_value=resp):
            with pytest.raises(OCRError, match="vide"):
                ocr_image(jpeg_file, cfg)

    def test_does_not_catch_keyboard_interrupt(self, jpeg_file, cfg):
        """KeyboardInterrupt ne doit pas être transformé en OCRError."""
        with patch("ocr_client.requests.post", side_effect=KeyboardInterrupt):
            with pytest.raises(KeyboardInterrupt):
                ocr_image(jpeg_file, cfg)

    def test_http_error_message_contains_status_code(self, jpeg_file, cfg):
        resp = make_http_response("", status_code=503)
        with patch("ocr_client.requests.post", return_value=resp):
            with pytest.raises(OCRError, match="503"):
                ocr_image(jpeg_file, cfg)
