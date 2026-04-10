# Architecture — ocr-livre (version PaddleOCR)

Pipeline CLI Python qui OCRise un livre (photos de pages) en Markdown via PaddleOCR-VL-1.5 servi localement par llama-server.

---

## Vue d'ensemble

```
photos/          →  pipeline  →  output/livre.md  (ou vault_root/vault_path/livre.md en mode obsidian)
                                  output/figures/<page>/
                                  output/parts/<page>.part
                                  output/ocr_report.md
```

Flux d'exécution :

```
main.py
  └── pipeline.run_pipeline(cfg)
        ├── images.collect_images(cfg)                 # liste des images triées
        ├── _start_server(cfg, port) × n_servers       # n subprocesses llama-server
        ├── _wait_for_server(url, timeout)             # polling /health pour chaque serveur
        ├── PaddleOCRVL(...) × n_servers               # un pipeline par serveur
        │
        └── ThreadPoolExecutor(max_workers=n_servers) — pour chaque image :
              ├── ocr_client.ocr_image(img, pipeline, cfg)
              │     ├── pipeline.predict(image)            # layout + OCR
              │     ├── save_to_markdown(save_path)        # écrit figures/page/page.md + imgs/
              │     └── retourne (markdown_text, {total_latency})
              │
              ├── postprocess.extract_page_number(text)
              ├── postprocess.clean_page(text, cfg)
              ├── postprocess.strip_table_styles(text)
              ├── fix_image_paths / fix_image_paths_obsidian
              ├── postprocess.format_page_block(page_id, text, page_number)
              └── écriture dans output/parts/<page_id>.part
        │
        ├── Combinaison des parts dans l'ordre d'entrée → output/livre.md
        ├── postprocess.apply_header_detection(text, cfg.header_patterns)  # si configuré
        ├── obsidian.migrate_figures(cfg)               # si mode obsidian
        └── stats.write_report(...)

  Sur timeout page :
        ├── restart_servers()                           # kill + relaunch tous les serveurs
        └── retry avec pipeline fallback (use_layout_detection=False)
```

---

## Modules

### `main.py`
Point d'entrée CLI (argparse). Parse les arguments, construit `Config`, appelle `run_pipeline`.

Options principales :
- `--images`, `--out` — chemins
- `--no-layout`, `--no-resume`, `--no-postprocess` — comportement OCR
- `--mode [base|obsidian]` — mode de sortie
- `--postprocess-only` — avec `--mode obsidian` : postprocess sur le `.md` existant sans relancer l'OCR
- `--migrate` — copier les figures vers le vault sans lancer l'OCR
- `--rename`, `--rename-only [START]`, `--rename-prefix` — renommage des images
- `--chapters NOM…` — sous-dossiers à traiter dans l'ordre fourni
- `--dir-level` — ordre par dossier (alpha dossiers > alpha sous-dossiers > images par date)
- `--dry-run`, `--verbose`

### `config.py`
Dataclass `Config` — toutes les valeurs par défaut en un seul endroit.

Groupes de paramètres :

- **Chemins llama-server** : `llama_server_path`, `model_path`, `mmproj_path`, `server_base_port`, `server_timeout`
- **Tuning llama-server** : `n_ctx`, `n_gpu_layers`, `n_batch`, `n_ubatch`, `n_threads`, `prio`, `kv_offload`, `temperature`, `max_tokens`
- **Parallélisme** : `n_servers` (serveurs parallèles), `n_parallel` (slots intra-page, requiert patch), `page_timeout` (secondes max par page, 0 = désactivé)
- **PaddleOCR** : `use_layout_detection`
- **Images** : `images_dir`, `extensions`, `rename_prefix`, `image_files` (liste explicite, court-circuite `images_dir`)
- **Sortie** : `output_file`, `figures_dir`, `resume`
- **Mode** : `mode` (`"base"` | `"obsidian"`)
- **Obsidian** : `vault_root`, `vault_path`, `vault_figures_dir`
- **Post-traitement** : `postprocess`, `remove_isolated_page_numbers`, `rejoin_hyphenated_words`, `collapse_blank_lines`, `header_patterns`
- **Logging** : `log_file`, `report_file`, `verbose`

### `pipeline.py`
Orchestration complète. Responsabilités :

1. Démarrer `n_servers` llama-server en subprocess (ports `server_base_port`, `server_base_port+1`, …)
2. Attendre que chaque serveur réponde sur `/health`
3. Instancier `n_servers` pipelines PaddleOCRVL dans une queue
4. Traiter les pages en parallèle via `ThreadPoolExecutor(max_workers=n_servers)`
5. Écrire chaque page dans `output/parts/<page_id>.part` (atomique)
6. En cas de timeout (`OCRTimeout`) : redémarrer tous les serveurs, retry avec pipeline fallback (sans layout)
7. Combiner les parts dans l'ordre d'entrée en fin de run
8. Appliquer `apply_header_detection` si `cfg.header_patterns`
9. Appeler `obsidian.migrate_figures` si mode obsidian
10. Arrêter tous les serveurs dans le bloc `finally`

### `ocr_client.py`
OCR d'une image unique. Interface : `ocr_image(image_path, pipeline, cfg) → (text, metrics)`.

Étapes internes :
1. `pipeline.predict(image_path)` — layout detection (ppdoclayout) + OCR VLM par région
2. `save_to_markdown(save_path)` — écrit `figures/<page>/<page>.md` + crops dans `figures/<page>/imgs/`
3. `read_text()` — charge le Markdown généré
4. Retourne `(text, {"total_latency": float})` — latence mesurée sur predict+save+read

Lève `OCRTimeout` si `cfg.page_timeout` est dépassé (capturé dans `pipeline.py`).

### `postprocess.py`
Nettoyage du Markdown généré par PaddleOCR.

- `clean_page(text, cfg, no_layout)` — suppression numéros de page isolés, réassemblage mots coupés, réduction lignes vides ; en mode no_layout, supprime aussi les boucles de génération
- `strip_table_styles(text)` — supprime les styles inline CSS des `<table>` et `<td>/<th>` générés par PaddleOCR, centre les tableaux
- `extract_page_number(text)` — extrait le numéro de page imprimé depuis les N premières/dernières lignes ; retourne `(label, texte_nettoyé)` où label vaut `None`, `"42"` ou `"42-43"`
- `apply_header_detection(text, header_patterns)` — ajoute les headers markdown selon les patterns regex configurés, avec heuristiques anti-faux-positifs
- `fix_image_paths(text, page_id, figures_rel)` — corrige les chemins `src="imgs/..."` en chemins relatifs depuis le dossier du fichier de sortie (mode base)
- `format_page_block(page_id, text, page_number)` — encadre chaque page avec `<!-- Page xxx (p. NN) -->`
- `format_error_block(page_id, error)` — bloc d'erreur pour une page échouée
- `extract_done_pages(output_text)` — lit les marqueurs `<!-- Page ... -->` pour la reprise (ancienne méthode, remplacée par parts)

### `obsidian.py`
Utilitaires pour l'export Obsidian.

- `prompt_if_needed(cfg)` — invite interactive pour `vault_root`, `vault_path`, `vault_figures_dir` si non configurés
- `fix_image_paths_obsidian(text, vault_figures_dir)` — convertit `<img src="imgs/...">` en wikilinks `![[vault_figures_dir/...]]` ; supprime les `<div>` wrappers
- `migrate_figures(cfg, page_ids, dry_run)` — copie les crops `output/figures/*/imgs/*` vers `vault_root/vault_figures_dir/` (structure aplatie, skip si déjà présent)
- `postprocess_file(cfg)` — applique le postprocess complet (`clean_page`, `strip_table_styles`, conversion img → wikilinks, `apply_header_detection`) sur un `.md` existant sans relancer l'OCR

### `images.py`
- `collect_images(cfg)` — liste et trie les images dans `cfg.images_dir` ; si `cfg.image_files` est fourni, l'utilise directement ; détecte les noms dupliqués
- `rename_images(folder, extensions, prefix, dry_run, start)` — renomme les images en `page_001.jpg`, `page_002.jpg`…
- `has_image_subdirs(folder, extensions)` — retourne True si le dossier contient des sous-dossiers avec des images
- `copy_from_subdirs(folder, extensions, chapters, prefix, start, dry_run, dir_level)` — copie les images des sous-dossiers vers le dossier parent avec numérotation séquentielle ; `chapters` permet de choisir les sous-dossiers dans l'ordre ; `dir_level` trie par dossier puis date

### `progress.py`
Dataclass `Stats` — accumule les métriques de run et génère le rapport final.

Métriques collectées : temps OCR par page, temps post-traitement, temps total, caractères, erreurs, pages fallback (no_layout), pages skippées.

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

Patches requis sur paddlex :
- `docs/dev/apply_paddlex_patch_otsl.py` — gestion des erreurs VLM par région (tableaux complexes). Voir `docs/dev/paddlex_patch_otsl.md`.
- `docs/dev/apply_paddlex_patch_parallel.py` — parallélisme intra-page (`n_parallel`). Requis si `n_parallel > 1`. Voir `docs/dev/paddlex_patch_parallel.md`.

---

## Format de sortie

PaddleOCR génère du HTML embarqué dans Markdown (compatible Obsidian) :

```markdown
<!-- Page page_001 (p. 42) -->

Texte courant paragraphe...

<table align="center" border=1>...</table>

![[Files/OCR/page_001_fig_0.png]]  (mode obsidian)
<img src="figures/page_001/imgs/page_001_fig_0.png" />  (mode base)
```

Les crops de figures sont sauvegardés dans `output/figures/<page>/imgs/`.

---

## Reprise (`--resume`)

Chaque page traitée est écrite dans `output/parts/<page_id>.part`. Au démarrage, le pipeline liste les fichiers `.part` existants — les pages déjà présentes sont skippées. En fin de run, les parts sont combinées dans l'ordre d'entrée. Désactiver avec `--no-resume` (supprime les parts existants).

---

## Mode Obsidian (`--mode obsidian`)

Quand `--mode obsidian` est activé :
1. `vault_root`, `vault_path`, `vault_figures_dir` sont demandés si non configurés dans `Config`
2. Le fichier de sortie est écrit dans `vault_root/vault_path/livre.md`
3. Les `<img src="imgs/...">` sont convertis en wikilinks `![[vault_figures_dir/...]]`
4. En fin de run, les crops sont copiés vers `vault_root/vault_figures_dir/` (`migrate_figures`)

Options dérivées :
- `--postprocess-only` — applique le postprocess sur le `.md` existant sans relancer l'OCR
- `--migrate` — copie uniquement les figures vers le vault

---

## Tuning performances

Paramètres llama-server configurables dans `Config` :

| Paramètre | Défaut | Effet |
|-----------|--------|-------|
| `n_servers` | 1 | Nombre de llama-server parallèles (une page par serveur simultanément) |
| `n_gpu_layers` | 99 | Couches déchargées sur GPU (Vulkan) |
| `n_batch` / `n_ubatch` | 512 | Taille des batches |
| `kv_offload` | True | Offload du KV cache sur CPU |
| `n_ctx` | 6144 | Contexte max (2048 tokens/slot × n_parallel=3) |
| `n_parallel` | 3 | Slots parallèles intra-page (requiert patch paddlex) |
| `page_timeout` | 120 | Secondes max par page avant abandon et redémarrage serveur (0 = désactivé) |
