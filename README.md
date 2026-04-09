# ocr-livre — Pipeline OCR livre → Markdown

Digitalise un livre entier en Markdown à partir de photos de pages,
en utilisant **PaddleOCR-VL-1.5** via **llama-server** (inférence locale).

---

## Prérequis

- [miniforge](https://github.com/conda-forge/miniforge) ou Anaconda
- [llama-server](https://github.com/ggerganov/llama.cpp) (Vulkan recommandé sur Windows)
- Modèle GGUF : [PaddleOCR-VL-1.5-GGUF](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5)

---

## Installation

```bash
python setup.py
conda activate ocr-livre
```

Voir [docs/SETUP.md](docs/SETUP.md) pour le détail.

---

## Structure du projet

```
ocr-livre/
├── src/
│   ├── main.py          # Point d'entrée CLI
│   ├── config.py        # Configuration centrale (dataclass)
│   ├── ocr_client.py    # OCR d'une image via PaddleOCRVL
│   ├── postprocess.py   # Nettoyage du texte OCR
│   ├── images.py        # Collecte et renommage des images
│   ├── pipeline.py      # Orchestration complète
│   └── progress.py      # Logging et statistiques
├── docs/
│   ├── architecture/    # Documentation architecture
│   ├── dev/             # Patches et notes de développement
│   ├── SETUP.md         # Instructions d'installation
│   ├── tested.md        # Résultats des expérimentations
│   └── issues.md        # Travaux en cours
├── photos/              # Images source (une par page)
├── output/              # Markdown généré + logs + figures
├── environment.yml      # Dépendances conda
└── setup.py             # Script d'installation automatisé
```

---

## Utilisation

Lancer depuis `src/` :

```bash
# Pipeline par défaut (photos dans ./photos, sortie output/livre.md)
python main.py

# Spécifier les dossiers
python main.py --images ./mes_photos --out output/mon_livre.md

# Sans layout detection
python main.py --no-layout

# Recommencer depuis le début
python main.py --no-resume

# Logs détaillés
python main.py --verbose
```

---

## Renommage des images

```bash
# Prévisualiser sans modifier
python main.py --rename --dry-run

# Renommer effectivement (→ page_001.jpg, page_002.jpg, …)
python main.py --rename --rename-prefix page
```

---

## Reprise automatique

Si le pipeline est interrompu, relancez simplement :

```bash
python main.py
```

Les pages déjà traitées sont automatiquement ignorées.

---

## Options complètes

```
--images PATH         Dossier des photos            (défaut: ./photos)
--out FILE            Fichier Markdown de sortie    (défaut: output/livre.md)
--no-layout           Désactiver layout detection
--no-resume           Recommencer depuis le début
--no-postprocess      Sortie brute sans nettoyage
--verbose             Logs DEBUG
--rename              Renommer les images avant OCR
--rename-prefix P     Préfixe renommage             (défaut: page)
--dry-run             Simuler --rename sans modifier
```

---

## Codes de retour

| Code | Signification                                |
|------|----------------------------------------------|
| 0    | Succès total                                 |
| 1    | Erreur fatale                                |
| 2    | Terminé avec des erreurs sur certaines pages |
