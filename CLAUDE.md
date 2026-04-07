# CLAUDE.md — ocr-livre

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

Travaux en cours : `docs/issues.md`.

Explorations et tests informels : `draft/`. 

Vérifie ta mémoire en début de session : `memory/`

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
- Repondre de façon honnete sans brosser dans le sens du poil ni contredire sans raison.

### Git
- add et commit a chaque issue résolue: grouper les fichiers modifiés de préférence à part si la modification est isolée.
- `draft/` est gitignored.
- ne pas ajouter le message "Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>"

### Issues
- après implémentation, mettre a jour issues.md si pertinent.
- après implémentation, effacer les sous-sections (`###`) et items de la sous-section qui sont résolus (`issues.md`).
- eviter de laisser une section (`##`) de issues.md vide: écrire "OK".

### Draft
- Toutes les explorations et tests intiaux se font dans `draft/`. 
- Toutes les sorties issues de `drat/` doivent atterir dans `output/`.
- Avant de coder un script de test, toujours vérifier si des modules ou fonctions de `src/` peuvent etre utilisé.

## Limiter la consommation de tokens

- Ne pas re-lire un fichier déjà lu dans la conversation s'il n'a pas changé à part sur demande explicite
- Utiliser Grep ciblé plutôt qu'un Glob large sur tout le repo
- Ne pas explorer `venv/` ni `output/` ni `photos/` ni `__pycache__` ni `.pytest_cache` (contenu non pertinent, très volumineux)
- Ne pas générer de docstrings ou commentaires sur du code non modifié

## Ressources
Documentation sur le stack spécifique utilisé dans le projet.

### Nexa

- Reference API Python  : https://docs.nexa.ai/en/nexa-sdk-python/api-reference 

- Signature correcte pour instancier le VLM: VLM.from_(model=..., quant=..., config=...)

### DeepSeekOCR

- HuggingFace page : https://huggingface.co/NexaAI/DeepSeek-OCR-GGUF

- GitHub page : https://github.com/deepseek-ai/DeepSeek-OCR/?tab=readme-ov-file

- Prompts valides : certains modes requièrent le préfixe `<|grounding|>` (ex: `"<|grounding|>Convert the document to markdown."`). `"Free OCR."` et `"Parse the figure."` n'en ont pas besoin. Voir section "Prompts examples" sur le GitHub.


## Troubleshooting
- Run `python src/main.py --no-resume`.
- Check `output/ocr_run.log`.
- Find what's wrong.

As a last resort only: run tests `python -m pytest tests/ -v`.