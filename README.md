# ocr-livre — Pipeline OCR livre → Markdown

Digitalise un livre entier en Markdown à partir de photos de pages,
en utilisant **DeepSeek-OCR** via **Nexa SDK** (inférence locale).

---

## Prérequis

- Python 3.11+
- Dépendances :
  ```bash
  pip install -r requirements.txt
  ```

---

## Structure du projet

```
ocr-livre/
├── src/
│   ├── main.py          # Point d'entrée CLI
│   ├── config.py        # Configuration centrale (dataclass)
│   ├── patch.py         # Monkey-patch nexaai (Windows)
│   ├── ocr_client.py    # OCR d'une image via nexaai.VLM
│   ├── preprocess.py    # Pré-traitement des images
│   ├── postprocess.py   # Nettoyage du texte OCR
│   ├── images.py        # Collecte et renommage des images
│   ├── pipeline.py      # Orchestration complète
│   └── progress.py      # Logging et statistiques
├── photos/              # Images source (une par page)
├── output/              # Markdown généré + logs
├── requirements.txt
└── README.md
```

---

## Utilisation

Lancer depuis `src/` :

```bash
# Pipeline par défaut (photos dans ./photos, sortie output/livre.md)
python main.py

# Spécifier les dossiers
python main.py --images ./mes_photos --out output/mon_livre.md

# Changer la quantization (q8_0 plus rapide, bf16 plus précis)
python main.py --quant q8_0

# Mode OCR avec mise en forme spatiale (défaut: plain)
python main.py --mode layout

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
--model MODEL         Modèle Nexa                   (défaut: NexaAI/DeepSeek-OCR-GGUF)
--quant {q8_0,bf16}   Quantization                  (défaut: bf16)
--mode MODE           plain | layout | describe | parse | rec:<cible>
--max-tokens N        Tokens max par page           (défaut: 4096)
--preprocess MODE     none | binarize               (défaut: binarize)
--no-resume           Recommencer depuis le début
--verbose             Logs DEBUG
--rename              Renommer les images avant OCR
--rename-prefix P     Préfixe renommage             (défaut: page)
--dry-run             Simuler --rename sans modifier
```

---

## Codes de retour

| Code | Signification                               |
|------|---------------------------------------------|
| 0    | Succès total                                |
| 1    | Erreur fatale                               |
| 2    | Terminé avec des erreurs sur certaines pages |
