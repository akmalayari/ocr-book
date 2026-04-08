# Architecture — ocr-livre (version PaddleOCR)

Pipeline CLI Python qui OCRise un livre (photos de pages) en Markdown via PaddleOCR-VL-1.5 servi localement par llama-server.

---

## Vue d'ensemble

```
photos/          →  pipeline  →  output/livre.md
                                  output/figures/<page>/
                                  output/ocr_report.md
```

Flux d'exécution :

```
main.py
  └── pipeline.run_pipeline(cfg)
        ├── images.collect_images(cfg)          # liste des images triées
        ├── _start_server(cfg)                  # subprocess llama-server
        ├── _wait_for_server(cfg)               # polling /health
        ├── PaddleOCRVL(...)                    # pipeline principal
        ├── PaddleOCRVL(use_layout_detection=False)  # fallback
        │
        └── pour chaque image :
              ├── ocr_client.ocr_image(img, pipeline, cfg)
              │     ├── pipeline.predict(image)      # layout + OCR
              │     ├── save_to_markdown(save_path)  # écrit figures/page/page.md + imgs/
              │     └── retourne (markdown_text, {total_latency})
              │
              ├── postprocess.clean_page(text, cfg)
              ├── postprocess.format_page_block(page_id, text)
              └── stats.record_success(...)
```

---

## Modules

### `main.py`
Point d'entrée CLI (argparse). Parse les arguments, construit `Config`, appelle `run_pipeline`. Options principales : `--images`, `--out`, `--no-layout`, `--no-resume`, `--rename`.

### `config.py`
Dataclass `Config` — toutes les valeurs par défaut en un seul endroit.

Groupes de paramètres :
- **Chemins llama-server** : `llama_server_path`, `model_path`, `mmproj_path`, `server_url`, `server_port`, `server_timeout`
- **Tuning llama-server** : `n_ctx`, `n_gpu_layers`, `n_batch`, `n_ubatch`, `n_threads`, `prio`, `flash_attention`, `kv_offload`, `reasoning`, `temperature`
- **PaddleOCR** : `use_layout_detection`
- **Images** : `images_dir`, `extensions`, `rename_prefix`
- **Sortie** : `output_file`, `figures_dir`, `resume`
- **Post-traitement** : `postprocess`, `remove_isolated_page_numbers`, `rejoin_hyphenated_words`, `collapse_blank_lines`
- **Logging** : `log_file`, `report_file`, `verbose`

### `pipeline.py`
Orchestration complète. Responsabilités :
1. Démarrer llama-server en subprocess avec les paramètres de `Config`
2. Attendre que le serveur réponde sur `/health`
3. Instancier deux pipelines PaddleOCRVL : principal (avec layout) et fallback (sans layout)
4. Boucler sur les images, appeler `ocr_image`, post-traiter, écrire le Markdown incrémentalement
5. Arrêter llama-server dans le bloc `finally` (garantit la libération GPU)

Stratégie de fallback : si `ocr_image` échoue avec le pipeline principal (layout detection), retry avec le pipeline fallback (sans layout). Si les deux échouent, enregistre une erreur et continue.

### `ocr_client.py`
OCR d'une image unique. Interface : `ocr_image(image_path, pipeline, cfg) → (text, metrics)`.

Étapes internes :
1. `pipeline.predict(image_path)` — layout detection (ppdoclayout) + OCR VLM par région
2. `save_to_markdown(save_path)` — écrit `figures/<page>/<page>.md` + crops dans `figures/<page>/imgs/`
3. `read_text()` — charge le Markdown généré
4. Retourne `(text, {"total_latency": float})` — latence mesurée sur l'ensemble predict+save+read

### `postprocess.py`
Nettoyage léger du Markdown généré par PaddleOCR :
- Suppression des numéros de page isolés
- Réassemblage des mots coupés en fin de ligne (`condi-\ntion` → `condition`)
- Réduction des lignes vides excessives (3+ → 2 max)

Fonctions utilitaires pour le fichier de sortie :
- `format_page_block(page_id, text)` — encadre chaque page avec `<!-- Page xxx -->`
- `extract_done_pages(output_text)` — lit ces marqueurs pour la reprise

### `images.py`
- `collect_images(cfg)` — liste et trie les images par nom dans `cfg.images_dir`
- `rename_images(...)` — renomme les images en `page_001.jpg`, `page_002.jpg`…

### `progress.py`
Dataclass `Stats` — accumule les métriques de run et génère le rapport final.

Métriques collectées : temps OCR par page, temps post-traitement, temps total, caractères, erreurs.

Rapport Markdown écrit dans `output/ocr_report.md`.

---

## Stack d'inférence

| Composant | Rôle |
|-----------|------|
| **llama-server** (llama.cpp, Vulkan) | Inférence VLM (PaddleOCR-VL-1.5 GGUF F16) |
| **paddleocr** (depuis repo git) | Orchestration : layout detection → routing prompts → appels VLM |
| **paddlepaddle CPU** | Layout detection (ppdoclayout) |
| **paddlex[ocr]** | Sous-pipeline tableaux (OTSL → HTML via `convert_otsl_to_html`) |
| **openai** | Client HTTP pour le backend `llama-cpp-server` de paddleocr |

Patch requis sur paddlex : `docs/dev/apply_paddlex_patch.py` — gestion des erreurs VLM par région (tableaux complexes). Voir `docs/dev/paddlex_patch.md`.

---

## Format de sortie

PaddleOCR génère du HTML embarqué dans Markdown (compatible Obsidian) :

```markdown
<!-- Page page_001 -->

Texte courant paragraphe...

<table border=1>...</table>

<div><img src="imgs/page_001_fig_0.png" /></div>
```

Les crops de figures sont sauvegardés dans `output/figures/<page>/imgs/`.

---

## Reprise (`--resume`)

Chaque page traitée est encadrée par `<!-- Page <id> -->` dans le fichier de sortie. Au démarrage, `extract_done_pages()` lit ces marqueurs — les pages déjà présentes sont skippées. Désactiver avec `--no-resume`.

---

## Tuning performances

Paramètres llama-server configurables dans `Config` (ou via le code) :

| Paramètre | Défaut | Effet |
|-----------|--------|-------|
| `n_gpu_layers` | 99 | Couches déchargées sur GPU (Vulkan) |
| `n_batch` / `n_ubatch` | 1024 | Taille des batches — impacte le throughput |
| `flash_attention` | True | Flash Attention — réduit la mémoire KV |
| `kv_offload` | True | Offload du KV cache sur CPU |
| `n_ctx` | 4096 | Contexte max — réduire si pages courtes |
