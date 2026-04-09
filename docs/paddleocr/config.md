# PaddleOCR-VL — Configuration et paramètres

## Config YAML de référence

Fichier : `paddlex/configs/pipelines/PaddleOCR-VL-1.5.yaml`

```yaml
use_doc_preprocessor: False      # orientation + unwarping désactivés
use_layout_detection: True
use_chart_recognition: False     # graphiques traités comme images
use_seal_recognition: False
format_block_content: False      # formatage extra dans .json() désactivé
merge_layout_blocks: True
markdown_ignore_labels:
  - number       # numéros de page
  - footnote
  - header
  - header_image
  - footer
  - footer_image
  - aside_text

SubModules:
  LayoutDetection:
    model_name: PP-DocLayoutV3
    threshold: 0.3
    layout_nms: True
  VLRecognition:
    model_name: PaddleOCR-VL-1.5-0.9B
    genai_config:
      backend: native             # remplacé par llama-cpp-server dans notre config
```

## Paramètres du constructeur `PaddleOCRVL(...)`

| Paramètre | Default | Description |
|---|---|---|
| `vl_rec_backend` | `"native"` | Backend VLM. Valeurs : `native`, `llama-cpp-server`, `vllm-server`, `sglang-server`, `mlx-vlm-server`, `fastdeploy-server` |
| `vl_rec_server_url` | — | URL du serveur (ex: `http://127.0.0.1:8080/v1`) |
| `vl_rec_api_model_name` | — | Nom du modèle pour le serveur (ex: `"paddleocr"`) |
| `use_layout_detection` | `True` | Détection de layout. `False` = fallback simple, qualité dégradée |
| `use_chart_recognition` | `False` | Reconnaissance spécialisée des graphiques |
| `use_doc_orientation_classify` | `False` | Correction de rotation (nécessite modèle supplémentaire) |
| `use_doc_unwarping` | `False` | Correction de courbure de page |
| `format_block_content` | `False` | Formatage HTML des blocs dans `.json()` |
| `merge_layout_blocks` | `True` | Fusion des blocs adjacents de même type |
| `markdown_ignore_labels` | voir YAML | Labels de blocs ignorés dans la sortie markdown |

## Paramètres de `predict()` / `predict_iter()`

| Paramètre | Default | Description |
|---|---|---|
| `max_new_tokens` | — | Nombre max de tokens générés par le VLM |
| `temperature` | — | Température d'échantillonnage |
| `top_p` | — | Top-p sampling |
| `repetition_penalty` | — | Pénalité de répétition |
| `min_pixels` | — | Pixels min pour le vision encoder (non supporté avec `llama-cpp-server`) |
| `max_pixels` | `28×28×3600 = 2 822 400` | Pixels max pour le vision encoder. **Non supporté avec `llama-cpp-server`** — ignoré silencieusement, warning dans les logs |
| `prompt_label` | — | Label de prompt personnalisé |
| `layout_threshold` | `0.3` | Seuil de confiance pour la détection de layout |
| `layout_shape_mode` | `"auto"` | Mode de forme pour le layout |
| `use_layout_detection` | hérite du constructeur | Override par appel |
| `format_block_content` | hérite du constructeur | Override par appel |
| `vlm_extra_args` | — | Arguments supplémentaires passés au VLM |

## Paramètres de `save_to_markdown()`

| Paramètre | Default | Description |
|---|---|---|
| `pretty` | `True` | Ajoute styles inline HTML. `False` = HTML brut sans styles. Voir `output.md` |
| `show_formula_number` | `False` | Affiche les numéros de formule |

## Notre configuration (projet)

Définie dans `src/config.py` et appliquée dans `src/pipeline.py` :

```python
PaddleOCRVL(
    vl_rec_backend="llama-cpp-server",
    vl_rec_server_url="http://127.0.0.1:8080/v1",
    vl_rec_api_model_name="paddleocr",
)
# use_layout_detection=True (défaut)
# use_doc_preprocessor=False (défaut YAML)
```

llama-server lancé avec :
```
-m <model.gguf> --mmproj <mmproj.gguf>
-c 6144 -ngl 99 -b 512 -ub 512 -t 4 --prio 2 --temp 0.0 -kvo
```
