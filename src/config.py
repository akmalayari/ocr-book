"""
config.py — Central configuration for the OCR pipeline
"""

import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import overload

from dotenv import load_dotenv

load_dotenv()

# Variables renamed when the OCR_ namespace was introduced. The old spellings
# are still honoured so existing .env files keep working; `find_legacy_env()`
# reports them so main.py can offer to migrate the file.
LEGACY_ENV_VARS: dict[str, str] = {
    "OCR_LLAMA_SERVER_PATH":         "LLAMA_SERVER_PATH",
    "OCR_MODEL_PATH":                "MODEL_PATH",
    "OCR_MMPROJ_PATH":               "MMPROJ_PATH",
    "OCR_OBSIDIAN_VAULT_ROOT":       "OBSIDIAN_VAULT_ROOT",
    "OCR_OBSIDIAN_VAULT_PATH":       "OBSIDIAN_VAULT_PATH",
    "OCR_OBSIDIAN_VAULT_FIGURES_DIR": "OBSIDIAN_VAULT_FIGURES_DIR",
}


@overload
def _env_str(name: str) -> str | None: ...
@overload
def _env_str(name: str, default: str) -> str: ...


def _env_str(name: str, default: str | None = None) -> str | None:
    """
    Reads a string from the environment, falling back to the pre-OCR_ name.

    The legacy spelling is only consulted when the new one is unset, so a
    stale variable can never silently win over an explicit one.
    """
    raw = os.environ.get(name, "").strip()
    if not raw:
        legacy = LEGACY_ENV_VARS.get(name)
        raw = os.environ.get(legacy, "").strip() if legacy else ""
    return raw or default


def find_legacy_env() -> list[tuple[str, str, bool]]:
    """
    Returns the legacy variables currently set in the environment as
    (legacy_name, new_name, shadowed), where `shadowed` means the new name is
    also set and therefore the legacy value is being ignored.
    """
    found = []
    for new, legacy in LEGACY_ENV_VARS.items():
        if os.environ.get(legacy, "").strip():
            found.append((legacy, new, bool(os.environ.get(new, "").strip())))
    return found


def migrate_env_file(names: list[tuple[str, str]], path: str | Path = ".env") -> list[tuple[str, str]]:
    """
    Renames the given (legacy_name, new_name) keys in a .env file, in place.

    Only the assignment lines of those exact keys are rewritten; comments,
    blank lines, ordering and every other variable are left byte-identical.
    Returns the pairs actually renamed.
    """
    path = Path(path)
    if not path.exists():
        return []

    rename = dict(names)
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    renamed: list[tuple[str, str]] = []

    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, sep, value = stripped.partition("=")
        new = rename.get(key.strip())
        if new:
            indent = line[: len(line) - len(stripped)]
            lines[i] = f"{indent}{new}{sep}{value}"
            renamed.append((key.strip(), new))

    if renamed:
        path.write_text("".join(lines), encoding="utf-8")
    return renamed


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


def _env_opt_bool(name: str) -> bool | None:
    """
    Reads a bool from the environment (1/true/yes/on vs 0/false/no/off).

    Returns None when unset, which callers treat as "leave the decision to
    llama-server" rather than as False.
    """
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return None
    if raw in ("1", "true", "yes", "on"):
        return True
    if raw in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"Invalid boolean for {name}: {raw!r}")


@dataclass
class Config:
    # ── llama-server ─────────────────────────────────────────────────────────
    # Set via environment variables, CLI arguments, or edit this file directly.
    llama_server_path: str | None = _env_str("OCR_LLAMA_SERVER_PATH")
    model_path: str | None        = _env_str("OCR_MODEL_PATH")
    mmproj_path: str | None       = _env_str("OCR_MMPROJ_PATH")
    server_base_port: int  = _env_int("OCR_SERVER_BASE_PORT", 8080)  # 8080, 8081, … (one per server)
    server_timeout: int    = _env_int("OCR_SERVER_TIMEOUT", 60)      # seconds before declaring the server dead

    # ── llama-server parameters (tuning) ─────────────────────────────────────
    # None = flag not passed, letting llama-server apply its own default. Since
    # b6xxx it auto-fits unset arguments to device memory (`--fit on`), so
    # hardcoding these would opt out of that fitting and usually make things
    # worse. Set them in .env (OCR_* variables, see .env.example) only when
    # tuning for a specific machine; CLI flags still win over both.
    n_gpu_layers: int | None = _env_opt_int("OCR_N_GPU_LAYERS")   # llama.cpp: auto
    n_batch: int | None      = _env_opt_int("OCR_N_BATCH")        # llama.cpp: 2048
    n_ubatch: int | None     = _env_opt_int("OCR_N_UBATCH")       # llama.cpp: 512
    n_threads: int | None    = _env_opt_int("OCR_N_THREADS")      # llama.cpp: -1 (auto)
    prio: int | None         = _env_opt_int("OCR_PRIO")           # llama.cpp: 0 (normal)
    kv_offload: bool | None  = _env_opt_bool("OCR_KV_OFFLOAD")    # llama.cpp: enabled

    # Always passed: these are deliberate, not hardware tuning.
    # n_ctx caps VRAM use and per-slot budget on weaker hardware, so it stays
    # explicit; None = auto (n_parallel * 2048), resolved in __post_init__.
    n_ctx: int | None     = None
    max_tokens: int       = _env_int("OCR_MAX_TOKENS", 4096)
    temperature: float    = 0.0    # 0 = deterministic; llama.cpp defaults to 0.8
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
    vault_root: str | None               = _env_str("OCR_OBSIDIAN_VAULT_ROOT")
    vault_path: str                     = _env_str("OCR_OBSIDIAN_VAULT_PATH", "Documents/OCR")
    vault_figures_dir: str              = _env_str("OCR_OBSIDIAN_VAULT_FIGURES_DIR", "Files/OCR")
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
        """Resolve and validate the OCR executable and model paths."""
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
                "  Env: OCR_LLAMA_SERVER_PATH, OCR_MODEL_PATH, OCR_MMPROJ_PATH\n"
                "  Or edit src/config.py directly."
            )

        server_value = os.path.expandvars(os.path.expanduser(self.llama_server_path))
        server_path = Path(server_value)
        path_like = (
            server_path.is_absolute()
            or "/" in server_value
            or "\\" in server_value
            or server_path.is_file()
        )
        resolved_server = str(server_path.resolve()) if path_like and server_path.is_file() else None
        if not path_like:
            resolved_server = shutil.which(server_value)

        problems = []
        if not resolved_server:
            hint = (
                " Windows .exe paths cannot be used on Linux."
                if os.name != "nt" and server_value.lower().endswith(".exe")
                else ""
            )
            problems.append(f"llama-server executable not found: {server_value!r}.{hint}")
        elif os.name != "nt" and not os.access(resolved_server, os.X_OK):
            problems.append(
                f"llama-server is not executable: {resolved_server!r}. "
                f"Run: chmod +x {resolved_server!r}"
            )
        else:
            self.llama_server_path = resolved_server

        for attr, label in (("model_path", "model"), ("mmproj_path", "mmproj")):
            value = os.path.expandvars(os.path.expanduser(getattr(self, attr)))
            path = Path(value)
            if not path.is_file():
                problems.append(f"{label} file not found: {value!r}")
            else:
                setattr(self, attr, str(path.resolve()))

        if problems:
            raise ValueError("Invalid OCR configuration:\n  - " + "\n  - ".join(problems))

    @property
    def images_path(self) -> Path:
        return Path(self.images_dir)

    @property
    def output_path(self) -> Path:
        return Path(self.output_file)

    @property
    def figures_path(self) -> Path:
        return Path(self.figures_dir)
