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
    llama-server must be launched with -np N >= VLM_PARALLEL.

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
# VLM_PARALLEL must match -np in llama-server (src/pipeline.py).
PATCHED = """\
        _VLM_PARALLEL = 4  # must match -np in llama-server

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


def _patched_parallel() -> str:
    """Returns the worker count declared in PATCHED, so messages can't drift."""
    match = re.search(r"_VLM_PARALLEL\s*=\s*(\d+)", PATCHED)
    return match.group(1) if match else "?"


def status(text: str) -> str:
    if PATCHED in text:
        return "patched"
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
        if state != "patched":
            print("[ERROR] Unknown state, manual modification required.")
            sys.exit(1)
        TARGET.write_text(text.replace(PATCHED, ORIGINAL), encoding="utf-8")
        print("Parallel patch removed — back to OTSL patch only.")
        return

    if state == "patched":
        print("Already patched, nothing to do.")
        return
    if state != "original":
        print("[ERROR] Unknown state. Verify that apply_paddlex_patch_otsl.py was applied first.")
        sys.exit(1)
    TARGET.write_text(text.replace(ORIGINAL, PATCHED), encoding="utf-8")
    print("Parallel patch (global pool) applied.")
    print(f"Make sure llama-server runs with -np {_patched_parallel()} "
          "(OCR_N_PARALLEL in .env, VLM_PARALLEL in this patch).")


if __name__ == "__main__":
    main()
