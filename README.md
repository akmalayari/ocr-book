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
│   ├── obsidian.py      # Export Obsidian (wikilinks, migration)
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

## Export Obsidian

En mode `obsidian`, le pipeline :
- convertit les figures en wikilinks `![[Files/image.jpg]]`
- sauvegarde le `.md` directement dans le vault
- copie les figures vers `vault_path/vault_figures_dir/`

Configurer `vault_path` et `vault_figures_dir` dans `config.py`, puis :

```bash
# OCR complet + export obsidian
python main.py --mode obsidian

# Ré-appliquer le postprocess obsidian sans relancer l'OCR
python main.py --mode obsidian --postprocess-only

# Migrer les figures vers le vault uniquement
python main.py --migrate
```

---

## Renommage des images

```bash
# Prévisualiser sans modifier
python main.py --rename --dry-run

# Renommer effectivement (→ page_001.jpg, page_002.jpg, …)
python main.py --rename

# Renommer sans lancer l'OCR
python main.py --rename-only

# Traiter des sous-dossiers par chapitre
python main.py --rename-only --chapters "Chapitre 1" "Chapitre 2"
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
--images PATH              Dossier des photos                (défaut: ./photos)
--out FILE                 Fichier Markdown de sortie        (défaut: output/livre.md)
--mode {base,obsidian}     Mode de sortie                    (défaut: base)
--no-layout                Désactiver layout detection
--no-resume                Recommencer depuis le début
--no-postprocess           Sortie brute sans nettoyage
--postprocess-only         Postprocess obsidian sans OCR     (requiert --mode obsidian)
--migrate                  Copier les figures vers le vault  (requiert vault_path configuré)
--dry-run                  Simuler sans modifier
--verbose                  Logs DEBUG
--rename                   Renommer les images avant OCR
--rename-only [N]          Renommer sans lancer l'OCR        (N = numéro de départ)
--rename-prefix P          Préfixe renommage                 (défaut: page)
--chapters NOM…            Sous-dossiers à traiter (dans l'ordre)
--dir-level                Ordre par dossier pour --rename
```

---

## Codes de retour

| Code | Signification                                |
|------|----------------------------------------------|
| 0    | Succès total                                 |
| 1    | Erreur fatale                                |
| 2    | Terminé avec des erreurs sur certaines pages |
