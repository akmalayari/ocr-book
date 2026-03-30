# Refactorisation — Passage à nexaai.VLM

## Contexte

Le serveur REST `nexa serve` retournait systématiquement HTTP 500 sur les requêtes
multimodales sous Windows. La solution : appel Python direct via `nexaai.VLM`, qui
charge DeepSeek-OCR-GGUF nativement via `nexa_bridge.dll`. Voir `docs/dev/approche_nexa_vlm.md`
pour l'explication technique détaillée.

## Fichiers modifiés

### Créés

| Fichier | Rôle |
|---------|------|
| `src/patch.py` | Monkey-patch `ProfileData.from_c_struct` — absorbe le `UnicodeDecodeError` sur `stop_reason` (bug `nexa_bridge.dll` Windows). Importé en premier dans `main.py`. |
| `src/preprocess.py` | Binarisation adaptative (`adaptiveThreshold` GAUSSIAN_C, blockSize=31, C=10). Retourne un fichier JPEG temporaire. |

### Supprimés

| Fichier | Raison |
|---------|--------|
| `src/server.py` | Gérait le cycle de vie de `nexa serve` (subprocess, polling, arrêt). Obsolète : plus de serveur REST. |

### Réécrits

**`src/ocr_client.py`**
- Avant : encodage base64 de l'image + requête HTTP POST vers `/v1/chat/completions`
- Après : `VlmChatMessage` + `apply_chat_template` + `GenerationConfig(image_paths=[...])` + `vlm.generate()`
- Signature : `ocr_image(image_path, vlm, cfg) -> tuple[str, dict]` — le dict contient `{"total_latency": float}`
- Applique `preprocess_image()` si `cfg.preprocess_mode == "binarize"`

### Modifiés

**`src/config.py`**
- Supprimé : `port`, `server_timeout_s`, `request_timeout_s`
- Ajouté : `n_ctx`, `n_threads`, `n_gpu_layers`, `n_batch`, `preprocess_mode`
- Ajouté : méthode `to_model_config()` → `nexaai.nexa_sdk.types.ModelConfig`

**`src/pipeline.py`**
- Supprimé : `with nexa_server(cfg):`
- Ajouté : chargement VLM singleton avant la boucle — `VLM.from_(cfg.model, config=cfg.to_model_config())`
- `ocr_image()` reçoit maintenant `vlm` en argument
- Métriques de latence transmises à `stats.record_success()`

**`src/progress.py`**
- `record_success(elapsed, chars, latency)` — nouveau paramètre `latency`
- `latencies: list[float]` — stockage par image
- `avg_latency` property
- `log_page()` affiche la latence VLM par image
- `log_summary()` affiche la latence moyenne

**`src/main.py`**
- `import patch` placé en première ligne (avant tout import nexaai)
- Supprimé : `--timeout`
- Ajouté : `--preprocess` (choices: none/binarize, défaut: binarize)

## Pré-traitement retenu

Tests comparatifs dans `draft/preprocess_test.py` :

| Variante | Boucle évitée | Mots (page1) | Temps |
|----------|:---:|---:|---:|
| original | ✗ | — | — |
| exposure_pillow | ✓ | ~830 | ~25s |
| binarize_adaptive | ✓ | ~1000 | ~19s |

`binarize_adaptive` retenu : meilleure couverture des zones sombres (plis), plus rapide,
moins d'hallucinations.
