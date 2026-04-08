# Setup — ocr-livre (PaddleOCR version)

## Installation de l'environnement

### Option 1 : Script automatique (recommandé)

```bash
python setup.py
conda activate ocr-livre
```

### Option 2 : Manuel

```bash
# Créer l'env conda depuis environment.yml
conda env create -f environment.yml

# Activer l'env
conda activate ocr-livre

# Installer PaddleOCR depuis le repo git (version dev avec llama-server compatibility)
pip install git+https://github.com/PaddlePaddle/PaddleOCR.git

# Appliquer le patch paddlex
python docs/dev/apply_paddlex_patch.py
```

## Lancer le pipeline

```bash
python src/main.py --help
python src/main.py <photos_dir>
```

## Dépannage

- **paddlex file not found** : Vérifier que l'env est activée (`conda activate ocr-livre-paddle`)
- **Patch fails** : L'état de paddlex peut être "unknown" si la version diffère. Voir [apply_paddlex_patch.py](dev/apply_paddlex_patch.py) pour détails
