"""
apply_paddlex_patch_parallel.py — Parallélise les appels VLM intra-page (pool global).

Prérequis : apply_paddlex_patch_otsl.py (patch OTSL) doit avoir été appliqué en premier.

Principe :
    PaddleOCR collecte les blocs par pixel_key puis les traite séquentiellement.
    Ce patch remplace la boucle entière for pixel_key par un pool global unique :
    tous les blocs de toutes les pixel_keys sont soumis simultanément, les workers
    pickent en continu sans restart entre pixel_keys, et les résultats sont redistribués.

    Avantage vs pool par pixel_key :
    - Pas de redémarrage de pool entre pixel_keys
    - Les blocs rapides (petits textes) libèrent immédiatement un worker
    - Meilleur équilibrage de charge global

    L'architecture PaddleX (asyncio.run_coroutine_threadsafe sur event loop global)
    est thread-safe : plusieurs threads peuvent appeler predict() simultanément.
    llama-server doit être lancé avec -np N >= VLM_PARALLEL.

Usage :
    python docs/dev/apply_paddlex_patch_parallel.py           # applique
    python docs/dev/apply_paddlex_patch_parallel.py --check   # vérifie
    python docs/dev/apply_paddlex_patch_parallel.py --revert  # retire (retour patch OTSL)
"""

import argparse
import sys
from pathlib import Path

TARGET = (
    Path(sys.prefix)
    / "Lib/site-packages/paddlex/inference/pipelines/paddleocr_vl/pipeline.py"
)

# État attendu : résultat de apply_paddlex_patch_otsl.py (boucle séquentielle avec OTSL)
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

# Pool global : tous les blocs de toutes les pixel_keys soumis en une seule passe.
# VLM_PARALLEL doit correspondre à -np dans llama-server (src/pipeline.py).
PATCHED = """\
        _VLM_PARALLEL = 3  # doit correspondre à -np dans llama-server

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

        # Collecter tous les blocs dans l'ordre des pixel_keys
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

        # Redistribuer les résultats par pixel_key
        _idx = 0
        for pixel_key, _n in _key_counts:
            batch_dict_by_pixel[pixel_key]["vlm_results"] = _all_results[_idx:_idx + _n]
            _idx += _n"""


def status(text: str) -> str:
    if PATCHED in text:
        return "patched"
    if ORIGINAL in text:
        return "original"
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check",  action="store_true", help="Vérifie sans modifier.")
    parser.add_argument("--revert", action="store_true", help="Retire le patch parallèle (retour patch OTSL).")
    args = parser.parse_args()

    if not TARGET.exists():
        print(f"[ERREUR] Fichier introuvable : {TARGET}")
        sys.exit(1)

    text = TARGET.read_text(encoding="utf-8")
    state = status(text)
    print(f"Fichier  : {TARGET}")
    print(f"État     : {state}")

    if args.check:
        sys.exit(0 if state == "patched" else 1)

    if args.revert:
        if state == "original":
            print("Déjà à l'état original (patch OTSL seul), rien à faire.")
            return
        if state != "patched":
            print("[ERREUR] État inconnu, modification manuelle requise.")
            sys.exit(1)
        TARGET.write_text(text.replace(PATCHED, ORIGINAL), encoding="utf-8")
        print("Patch parallèle retiré — retour au patch OTSL seul.")
        return

    if state == "patched":
        print("Déjà patché, rien à faire.")
        return
    if state != "original":
        print("[ERREUR] État inconnu. Vérifier que apply_paddlex_patch_otsl.py a été appliqué en premier.")
        sys.exit(1)
    TARGET.write_text(text.replace(ORIGINAL, PATCHED), encoding="utf-8")
    print("Patch parallèle (pool global) appliqué.")
    print("Assure-toi que llama-server tourne avec -np 3 (VLM_PARALLEL dans le patch).")


if __name__ == "__main__":
    main()
