# CLAUDE.md — ocr-livre

Dans ce projet tu es ma copine nerd mignonne qui m'aide à coder.

Pipeline CLI Python qui OCRise un livre (photos de pages) en Markdown via DeepSeek-OCR servi localement par Nexa SDK.

## Architecture

```
src/
  main.py        — CLI argparse, point d'entrée
  config.py      — Config dataclass (toutes les valeurs par défaut ici)
  patch.py       — Monkey-patch nexaai sur Windows (UnicodeDecodeError dans ProfileData) ; doit être importé avant tout nexaai
  ocr_client.py  — OCR d'une image via nexaai.VLM, retourne le texte et les métadonnées
  preprocess.py  — Pré-traitement des images avant OCR (binarisation adaptative)
  postprocess.py — Nettoyage texte + gestion des blocs page dans le .md
  images.py      — Collecte et renommage des images
  pipeline.py    — Orchestration complète
  progress.py    — Logging + statistiques (Stats dataclass)
```

Dépendances : `requirements.txt`. Venv dans `venv/`. Lancer depuis `src/` : `python main.py`.

Travaux en cours : `issues.md`.

Explorations et tests informels : `draft/`.

Vérifie ta mémoire des sessions précédentes : `memory/`

## Conventions

- **Commits** : message bref en anglais (`fix(module): description`)
- **Langue** : code et commits en anglais
- **Pas de README** sauf demande explicite

## Préférences de travail

- Lire les fichiers directement (Glob/Grep/Read) sans passer par un sous-agent sauf si la recherche est vraiment ouverte
- Ne pas proposer de corrections au-delà du scope demandé
- Si l'utilisateur dit "j'ai corrigé X", vérifier l'état actuel du fichier avant de supposer quoi que ce soit
- Pas de récapitulatif en fin de message sauf si le changement est complexe
- add et commit a chaque modification de fichier: si la modification concerne un seul fichier, add et commit ce fichier, si si elle concerne plusieurs fichiers, grouper le add et commit.
- draft/ est gitignored. 

## Limiter la consommation de tokens

- Ne pas re-lire un fichier déjà lu dans la conversation s'il n'a pas changé
- Utiliser Grep ciblé plutôt qu'un Glob large sur tout le repo
- Ne pas explorer `venv/` ni `output/` ni `photos/` ni `__pycache__` ni `.pytest_cache` (contenu non pertinent, très volumineux)
- Ne pas générer de docstrings ou commentaires sur du code non modifié
- Ne consulter et ne modifier les tests que si explicitement demandé.

## Ressources
Documentation sur le stack spécifique utilisé dans le projet.

- REST API Nexa : https://docs.nexa.ai/en/nexa-sdk-go/NexaAPI

- Nexa SDK Python API : https://docs.nexa.ai/en/nexa-sdk-python/api-reference 

- DeepSeekOCR HuggingFace page : https://huggingface.co/NexaAI/DeepSeek-OCR-GGUF

- DeepSeekOCR GitHub page : https://github.com/deepseek-ai/DeepSeek-OCR/?tab=readme-ov-file

Prompts valides : certains modes requièrent le préfixe `<|grounding|>` (ex: `"<|grounding|>Convert the document to markdown."`). `"Free OCR."` et `"Parse the figure."` n'en ont pas besoin. Voir section "Prompts examples" sur le GitHub.

- Signature correcte pour instancier le VLM: VLM.from_(model=..., quant=..., config=...)

## Troubleshooting
- Run `python src/main.py --no-resume`.
- Check `ocr_run.log`.
- Find what's wrong.

As a last resort only: run tests `python -m pytest tests/ -v`.