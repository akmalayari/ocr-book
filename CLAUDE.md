# CLAUDE.md — ocr-livre

Pipeline CLI Python qui OCRise un livre (photos de pages) en Markdown via PaddleOCR-VL-1.5 servi localement par llama-server.

## Architecture

```
src/
  main.py        — CLI argparse, point d'entrée
  config.py      — Config dataclass (toutes les valeurs par défaut ici)
  ocr_client.py  — OCR d'une image via PaddleOCRVL
  postprocess.py — Nettoyage texte + gestion des blocs page dans le .md
  images.py      — Collecte, renommage et copie depuis sous-dossiers
  pipeline.py    — Orchestration complète (multi-serveurs, parts, fallback)
  obsidian.py    — Export Obsidian (wikilinks, migrate_figures, postprocess_file)
  progress.py    — Logging + statistiques (Stats dataclass)
```

Dépendances : `environment.yaml`. Venv conda: `ocr-livre`. Lancer depuis `src/` : `python main.py`.

Doc projet: `docs/`.

Explorations et tests informels : `draft/`. Resultats des explorations: `docs/tested.md`.

Travaux en cours : `docs/issues.md`.

Vérifie ta mémoire en début de session : `memory/`.

## Conventions

- **Commits** : message bref en anglais (`fix(module): description`)
- **Langue** : code et commits en anglais
- **Pas de README** sauf demande explicite
- **Pas de tests** sauf demande explicite

## Préférences de travail

### Général
- Lire les fichiers directement (Glob/Grep/Read) sans passer par un sous-agent sauf si la recherche est vraiment ouverte
- Ne pas proposer de corrections au-delà du scope demandé
- Si l'utilisateur dit "j'ai corrigé X", vérifier l'état actuel du fichier avant de supposer quoi que ce soit
- Pas de récapitulatif en fin de message sauf si le changement est complexe
- Repondre de façon honnete: contredire si nécessaire et expliquer son point de vue. 
- Toujours demander des clarifications avant de coder a part si les consignes sont claires ou évidentes.

### Git
- add et commit a chaque issue résolue: grouper les fichiers modifiés de préférence à part si la modification est isolée.
- `draft/` est gitignored.
- ne pas ajouter le message "Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"

### Issues
- après implémentation, mettre a jour issues.md si pertinent.
- après implémentation, effacer les sous-sections (`###`) et items de la sous-section qui sont résolus (`issues.md`).
- eviter de laisser une section (`##`) de issues.md vide: écrire "OK".

### Draft
- Toutes les explorations et tests initiaux se font dans `draft/`. 
- Toutes les sorties issues de `drat/` doivent atterir dans `output/`.
- Avant de coder un script de test, toujours vérifier si des modules ou fonctions de `src/` peuvent etre utilisé.

## Limiter la consommation de tokens

- Ne pas re-lire un fichier déjà lu dans la conversation s'il n'a pas changé à part sur demande explicite
- Utiliser Grep ciblé plutôt qu'un Glob large sur tout le repo
- Ne pas explorer `output/` ni `photos/` ni `__pycache__` ni `.pytest_cache` (contenu non pertinent, très volumineux)
- Ne pas générer de docstrings ou commentaires sur du code non modifié

## Ressources
Documentation sur le stack spécifique utilisé dans le projet.

### llama-server

- Docs GitHub: https://github.com/ggml-org/llama.cpp/tree/master/tools/server

### PaddleOCR

- Documentation générale: https://www.paddleocr.ai/latest/en/version3.x/pipeline_usage/PaddleOCR-VL.html

- Page HuggingFace: https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5

- Page GitHub: https://github.com/PaddlePaddle/PaddleOCR

- Doc interne: `docs/paddleocr/`

## Troubleshooting
- Run `python src/main.py --images photos/page_1.jpg --no-resume`.
- Check `output/ocr_run.log`.
- Find what's wrong.

As a last resort only: run tests `python -m pytest tests/ -v`.