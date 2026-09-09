"""
apply_paddlex_patch_parallel.py — Parallelizes VLM intra-page calls (global pool).

Prerequisite: apply_paddlex_patch_otsl.py (OTSL patch) must have been applied first.

Principle:
    PaddleOCR collects blocks by pixel_key then processes them sequentially.
    This patch replaces the entire for pixel_key loop with a single global pool:
    all blocks from all pixel_keys are submitted simultaneously, workers
    pick continuously without restart between pixel_keys, and results are redistributed.

    Advantage vs per-pixel_key pool:
    - No pool restart between pixel_keys
    - Fast blocks (small texts) immediately free a worker
    - Better global load balancing

    PaddleX architecture (asyncio.run_coroutine_threadsafe on global event loop)
    is thread-safe: multiple threads can call predict() simultaneously.
    The worker count is read from OCR_N_PARALLEL at runtime. ocr-book exports
    the resolved Config value so the pool and llama-server always use the same
    concurrency, including when --n-parallel overrides .env.

Usage :
    python docs/dev/apply_paddlex_patch_parallel.py           # apply
    python docs/dev/apply_paddlex_patch_parallel.py --check   # check
    python docs/dev/apply_paddlex_patch_parallel.py --revert  # remove (back to OTSL patch)
"""

import argparse
import importlib.util
import re
import sys
import sysconfig
from pathlib import Path


def _paddlex_pipeline_path() -> Path:
    """Locate PaddleX in the active environment on Windows, Linux, or macOS."""
    spec = importlib.util.find_spec("paddlex")
    if spec and spec.submodule_search_locations:
        package_dir = Path(next(iter(spec.submodule_search_locations)))
    else:
        # Keep --help and the missing-package error useful before PaddleX exists.
        package_dir = Path(sysconfig.get_paths()["purelib"]) / "paddlex"
    return package_dir / "inference/pipelines/paddleocr_vl/pipeline.py"


TARGET = _paddlex_pipeline_path()

# Expected state: result of apply_paddlex_patch_otsl.py (sequential loop with OTSL)
ORIGINAL = """\
        for pixel_key in batch_dict_by_pixel:
            min_pixels, max_pixels = pixel_key
            kwargs = {
                "use_cache": True,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                **vlm_kwargs,
            }
            images = batch_dict_by_pixel[pixel_key]["images"]
            queries = batch_dict_by_pixel[pixel_key]["queries"]
            batch_results = []
            for image, query in zip(images, queries):
                try:
                    result = list(
                        self.vl_rec_model.predict(
                            [{"image": image, "query": query}],
                            skip_special_tokens=False if has_spotting else True,
                            **kwargs,
                        )
                    )[0]
                except Exception as _vlm_err:
                    err_msg = str(_vlm_err)
                    otsl_start = err_msg.find("<fcel>")
                    if otsl_start != -1:
                        # OTSL content echoed back by llama-server; convert directly
                        result = {"result": err_msg[otsl_start:]}
                    else:
                        result = {"result": ""}
                batch_results.append(result)
            del images, queries
            batch_dict_by_pixel[pixel_key]["vlm_results"] = batch_results"""

# Global pool: all blocks from all pixel_keys submitted in a single pass.
# The runtime environment is synchronized with -np by src/pipeline.py.
PATCHED_BODY = """\
        def _infer_block(args):
            _img, _qry, _kw = args
            try:
                return list(
                    self.vl_rec_model.predict(
                        [{"image": _img, "query": _qry}],
                        skip_special_tokens=False if has_spotting else True,
                        **_kw,
                    )
                )[0]
            except Exception as _vlm_err:
                _err_msg = str(_vlm_err)
                _otsl = _err_msg.find("<fcel>")
                if _otsl != -1:
                    # OTSL content echoed back by llama-server; convert directly
                    return {"result": _err_msg[_otsl:]}
                return {"result": ""}

        # Collect all blocks in pixel_key order
        _all_tasks = []
        _key_counts = []
        for pixel_key in batch_dict_by_pixel:
            min_pixels, max_pixels = pixel_key
            _kw = {
                "use_cache": True,
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                **vlm_kwargs,
            }
            _imgs = batch_dict_by_pixel[pixel_key]["images"]
            _qrys = batch_dict_by_pixel[pixel_key]["queries"]
            for _img, _qry in zip(_imgs, _qrys):
                _all_tasks.append((_img, _qry, _kw))
            _key_counts.append((pixel_key, len(_imgs)))

        import concurrent.futures as _cf
        with _cf.ThreadPoolExecutor(max_workers=_VLM_PARALLEL) as _pool:
            _all_results = list(_pool.map(_infer_block, _all_tasks))

        # Redistribute results by pixel_key
        _idx = 0
        for pixel_key, _n in _key_counts:
            batch_dict_by_pixel[pixel_key]["vlm_results"] = _all_results[_idx:_idx + _n]
            _idx += _n"""


PATCHED = """\
        import os as _os
        _VLM_PARALLEL = max(1, int(_os.environ.get("OCR_N_PARALLEL", "1")))

""" + PATCHED_BODY

# Previous versions embedded the worker count in the installed PaddleX file.
# Match any positive legacy value so apply/revert works across 2, 3, 4, or a
# locally tuned value without depending on the current .env.
LEGACY_PATCHED_RE = re.compile(
    r"        _VLM_PARALLEL = (?P<workers>[1-9]\d*)  "
    r"# must match -np in llama-server\n\n"
    + re.escape(PATCHED_BODY)
)


def _legacy_patched(text: str) -> re.Match[str] | None:
    return LEGACY_PATCHED_RE.search(text)


def status(text: str) -> str:
    if PATCHED in text:
        return "patched"
    legacy = _legacy_patched(text)
    if legacy:
        return f"legacy-patched ({legacy.group('workers')} workers)"
    if ORIGINAL in text:
        return "original"
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check",  action="store_true", help="Check without modifying.")
    parser.add_argument("--revert", action="store_true", help="Remove parallel patch (back to OTSL patch).")
    args = parser.parse_args()

    if not TARGET.exists():
        print(f"[ERROR] File not found: {TARGET}")
        sys.exit(1)

    text = TARGET.read_text(encoding="utf-8")
    state = status(text)
    print(f"File  : {TARGET}")
    print(f"State : {state}")

    if args.check:
        sys.exit(0 if state == "patched" else 1)

    if args.revert:
        if state == "original":
            print("Already at original state (OTSL patch only), nothing to do.")
            return
        patched_block = PATCHED if state == "patched" else None
        if state.startswith("legacy-patched"):
            patched_block = _legacy_patched(text).group(0)
        if patched_block is None:
            print("[ERROR] Unknown state, manual modification required.")
            sys.exit(1)
        TARGET.write_text(text.replace(patched_block, ORIGINAL), encoding="utf-8")
        print("Parallel patch removed — back to OTSL patch only.")
        print("OCR_N_PARALLEL was not changed; set it to 1 when running without the patch.")
        return

    if state == "patched":
        print("Already patched, nothing to do.")
        return
    if state.startswith("legacy-patched"):
        legacy = _legacy_patched(text)
        TARGET.write_text(text.replace(legacy.group(0), PATCHED), encoding="utf-8")
        print(f"Legacy parallel patch ({legacy.group('workers')} workers) upgraded.")
        print("Worker count now follows OCR_N_PARALLEL at runtime (default: 1).")
        return
    if state != "original":
        print("[ERROR] Unknown state. Verify that apply_paddlex_patch_otsl.py was applied first.")
        sys.exit(1)
    TARGET.write_text(text.replace(ORIGINAL, PATCHED), encoding="utf-8")
    print("Parallel patch (global pool) applied.")
    print("Worker count follows OCR_N_PARALLEL at runtime (default: 1; test 2 first).")


if __name__ == "__main__":
    main()
