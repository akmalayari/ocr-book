# Images

## Collecte (`collect_images`)

Retourne la liste triée alphabétiquement des images dans le dossier `--images` (défaut : `./photos`).

Extensions supportées : `.jpg`, `.jpeg`, `.png`, `.webp`, `.pdf`.

Le tri est alphabétique sur le nom de fichier — les images doivent donc être nommées avec un padding numérique cohérent (`page_001.jpg`, `page_002.jpg`, …). Si ce n'est pas le cas, utiliser `--rename` avant l'OCR.

`--images` peut pointer vers un fichier unique ou un dossier.

## Renommage (`--rename` / `--rename-only`)

Renomme les images par date de création (`st_birthtime` sur Windows, fallback `st_mtime`), avec padding numérique uniforme.

- `--rename` : renomme puis enchaîne sur le pipeline OCR.
- `--rename-only [START]` : renomme sans lancer l'OCR. `START` est le numéro de départ (défaut : 1).

```
python src/main.py --rename
python src/main.py --rename --rename-prefix scan   # → scan_001.jpg, scan_002.jpg, …
python src/main.py --rename-only                   # renommage seul, départ à 1
python src/main.py --rename-only 15               # renommage seul, départ à 15
python src/main.py --rename --dry-run             # affiche les renommages, pas d'OCR
```

| Argument | Défaut | Description |
|---|---|---|
| `--rename` | off | Renomme avant OCR |
| `--rename-only [START]` | off | Renomme sans OCR (START = numéro de départ) |
| `--rename-prefix` | `page` | Préfixe des nouveaux noms |
| `--dry-run` | off | Affiche les renommages sans les effectuer ni lancer l'OCR |

Résultat : `{prefix}_{numéro paddé}{extension}` — padding minimum 3 chiffres.

## Sous-dossiers

Si le dossier `--images` contient des sous-dossiers avec des images, le pipeline passe automatiquement en mode copie : les images sont copiées vers le dossier parent avec une numérotation séquentielle.

La détection et la copie sont **récursives** : les sous-sous-dossiers sont inclus.

### Ordre de tri par défaut

Sans `--dir-level`, toutes les images de chaque sous-dossier (récursivement) sont triées par date de création globalement.

### `--dir-level` : ordre par niveau de dossier

Avec `--dir-level`, l'ordre respecte la hiérarchie des dossiers :

1. Sous-dossiers de premier niveau triés en ordre naturel (`leçon 2` avant `leçon 10`)
2. Sous-sous-dossiers triés en ordre naturel
3. Images dans chaque dossier triées par date de création

```
python src/main.py --rename-only --dir-level
python src/main.py --rename-only --dir-level --dry-run
```

### `--chapters` : sélection et ordre des sous-dossiers

Permet de choisir quels sous-dossiers traiter et dans quel ordre.

```
python src/main.py --rename-only --chapters "Leçon 1" "Leçon 3"
```

## Structure de dossiers recommandée

Pour un livre découpé en parties et chapitres, la structure idéale est :

```
photos/
  01 - Partie 1/
    01 - Chapitre 1/
      IMG_0001.jpg
      IMG_0002.jpg
      …
    02 - Chapitre 2/
      IMG_0010.jpg
      …
  02 - Partie 2/
    …
```

Workflow complet :

```
# 1. Vérifier l'ordre avant de renommer
python src/main.py --rename-only --dir-level --dry-run

# 2. Renommer
python src/main.py --rename-only --dir-level

# 3. Lancer l'OCR
python src/main.py
```

Le préfixe numérique dans les noms de dossiers (`01 -`, `02 -`) garantit l'ordre alphabétique même sans `--dir-level`. Avec des noms purement textuels (`Chapitre 1`, `Chapitre 10`), `--dir-level` est nécessaire pour le tri naturel.
