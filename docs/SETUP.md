# Setup — ocr-livre (PaddleOCR version)

## Installation de l'environnement

```bash
# Créer l'env conda depuis environment.yml
conda env create -f environment.yml

# Activer l'env
conda activate ocr-livre-paddle
```

## Appliquer le patch paddlex

Le patch est nécessaire pour gérer les erreurs VLM par région (voir [paddlex_patch.md](dev/paddlex_patch.md)).

```bash
# Appliquer le patch
python docs/dev/apply_paddlex_patch.py

# (Optionnel) Vérifier l'état
python docs/dev/apply_paddlex_patch.py --check

# (Optionnel) Restaurer l'original
python docs/dev/apply_paddlex_patch.py --revert
```

## Lancer le pipeline

```bash
python src/main.py --help
python src/main.py <photos_dir>
```

## Dépannage

- **paddlex file not found** : Vérifier que l'env est activée (`conda activate ocr-livre-paddle`)
- **Patch fails** : L'état de paddlex peut être "unknown" si la version diffère. Voir [apply_paddlex_patch.py](dev/apply_paddlex_patch.py) pour détails
