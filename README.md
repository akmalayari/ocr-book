# book_ocr — Pipeline OCR livre → Markdown

Digitalise un livre entier en Markdown à partir de photos de pages,
en utilisant **DeepSeek-OCR** via le serveur local **Nexa SDK**.

---

## Prérequis

- Nexa CLI installé et modèle téléchargé :
  ```bash
  nexa pull NexaAI/DeepSeek-OCR-GGUF
  ```
- Python 3.11+
- Dépendances :
  ```bash
  pip install -r requirements.txt
  ```

---

## Structure du projet

```
ocr-livre/
├── main.py          # Point d'entrée CLI
├── config.py        # Configuration centrale (dataclass)
├── server.py        # Démarrage/arrêt du serveur Nexa (context manager)
├── ocr_client.py    # Envoi des images au serveur, récupération OCR
├── postprocess.py   # Nettoyage du texte (césures, numéros de page…)
├── images.py        # Découverte et tri des images, renommage
├── pipeline.py      # Orchestration complète du pipeline
├── requirements.txt
└── README.md
```

---

## Utilisation rapide

```bash
# Pipeline par défaut (photos dans ./photos, sortie livre.md)
python main.py

# Spécifier les dossiers
python main.py --images ./mes_photos --out mon_livre.md

# OCR texte brut (sans mise en forme Markdown)
python main.py --mode plain

# Recommencer depuis le début (ignore le fichier existant)
python main.py --no-resume

# Logs détaillés
python main.py --verbose
```

---

## Renommage des images

Si vos photos ont des noms incohérents (IMG_2024.jpg, DSC_042.jpg…),
renommez-les d'abord avec un padding numérique uniforme :

```bash
# Prévisualiser sans modifier
python main.py --rename-only --dry-run

# Renommer effectivement
python main.py --rename-only --rename-prefix page
# → page_001.jpg, page_002.jpg, page_003.jpg, …
```

---

## Reprise automatique

Si le pipeline est interrompu (Ctrl+C, coupure, crash), relancez simplement :

```bash
python main.py
```

Les pages déjà traitées (marquées `<!-- Page page_XXX -->` dans le .md) sont
automatiquement ignorées.

---

## Options complètes

```
--images PATH       Dossier des photos          (défaut: ./photos)
--out FILE          Fichier Markdown de sortie  (défaut: livre.md)
--model MODEL       Modèle Nexa                 (défaut: NexaAI/DeepSeek-OCR-GGUF)
--port PORT         Port serveur Nexa           (défaut: 18181)
--mode MODE         markdown | plain | figure   (défaut: markdown)
--max-tokens N      Tokens max par page         (défaut: 4096)
--timeout N         Timeout par image (s)       (défaut: 180)
--no-resume         Recommencer depuis le début
--verbose           Logs DEBUG
--rename-only       Renommer les images sans OCR
--rename-prefix P   Préfixe renommage           (défaut: page)
--dry-run           Simuler --rename-only sans modifier
```

---

## Codes de retour

| Code | Signification                          |
|------|----------------------------------------|
| 0    | Succès total                           |
| 1    | Erreur fatale (serveur, config…)       |
| 2    | Terminé avec des erreurs sur certaines pages |
