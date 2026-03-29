# CLAUDE.md — ocr-livre

Pipeline CLI Python qui OCRise un livre (photos de pages) en Markdown via DeepSeek-OCR servi localement par Nexa SDK.

## Architecture

```
src/
  main.py        — CLI argparse, point d'entrée
  config.py      — Config dataclass (toutes les valeurs par défaut ici)
  server.py      — Démarrage/arrêt serveur Nexa (context manager)
  ocr_client.py  — Requête HTTP vers le serveur, retourne le texte OCR
  postprocess.py — Nettoyage texte + gestion des blocs page dans le .md
  images.py      — Collecte et renommage des images
  pipeline.py    — Orchestration complète
  progress.py    — Logging + statistiques (Stats dataclass)
```

Dépendances : `requirements.txt`. Venv dans `venv/`. Lancer depuis `src/` : `python main.py`.

Améliorations différées : `issues.md`.

## Conventions

- **Commits** : un fichier par commit, message bref en français (`fix(module): description`)
- **Langue** : code et commits en français
- **Pas de README** sauf demande explicite

## Préférences de travail

- Lire les fichiers directement (Glob/Grep/Read) sans passer par un sous-agent sauf si la recherche est vraiment ouverte
- Ne pas proposer de corrections au-delà du scope demandé
- Si l'utilisateur dit "j'ai corrigé X", vérifier l'état actuel du fichier avant de supposer quoi que ce soit
- Réponses courtes ; pas de récapitulatif en fin de message sauf si le changement est complexe

## Limiter la consommation de tokens

- Ne pas re-lire un fichier déjà lu dans la conversation s'il n'a pas changé
- Utiliser Grep ciblé plutôt qu'un Glob large sur tout le repo
- Ne pas explorer `venv/` (contenu non pertinent, très volumineux)
- Ne pas générer de docstrings ou commentaires sur du code non modifié
