"""
apply_paddlex_patch_otsl.py — Applies (or checks) the paddlex per-region VLM patch.

See paddlex_patch_otsl.md for full description.

Usage :
    python docs/dev/apply_paddlex_patch_otsl.py           # apply patch
    python docs/dev/apply_paddlex_patch_otsl.py --check   # check without modifying
    python docs/dev/apply_paddlex_patch_otsl.py --revert  # restore original
"""

import argparse
import importlib.util
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

ORIGINAL = """\
            images = batch_dict_by_pixel[pixel_key]["images"]
            queries = batch_dict_by_pixel[pixel_key]["queries"]
            batch_results = list(
                self.vl_rec_model.predict(
                    [
                        {
                            "image": image,
                            "query": query,
                        }
                        for image, query in zip(images, queries)
                    ],
                    skip_special_tokens=False if has_spotting else True,
                    **kwargs,
                )
            )
            del images, queries
            batch_dict_by_pixel[pixel_key]["vlm_results"] = batch_results"""

PATCHED = """\
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

# Short signatures used for detection — robust to surrounding restructuring (e.g. parallel patch applied on top)
PATCHED_SIGNATURE = 'find("<fcel>")'
ORIGINAL_SIGNATURE = "batch_results = list("


def status(text: str) -> str:
    if PATCHED_SIGNATURE in text:
        return "patched"
    if ORIGINAL_SIGNATURE in text:
        return "original"
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check",  action="store_true", help="Check without modifying.")
    parser.add_argument("--revert", action="store_true", help="Restore original.")
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
            print("Already at original state, nothing to do.")
            return
        if state != "patched":
            print("[ERROR] Unknown state, manual modification required.")
            sys.exit(1)
        TARGET.write_text(text.replace(PATCHED, ORIGINAL), encoding="utf-8")
        print("Patch removed.")
        return

    # Apply patch
    if state == "patched":
        print("Already patched, nothing to do.")
        return
    if state != "original":
        print("[ERROR] Unknown state, manual modification required.")
        sys.exit(1)
    TARGET.write_text(text.replace(ORIGINAL, PATCHED), encoding="utf-8")
    print("Patch applied.")


if __name__ == "__main__":
    main()
