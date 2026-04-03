# Images

## Collecte (`collect_images`)

Retourne la liste triée alphabétiquement des images dans le dossier `--images` (défaut : `./photos`).

Extensions supportées : `.jpg`, `.jpeg`, `.png`, `.webp`, `.pdf`.

Le tri est alphabétique sur le nom de fichier — les images doivent donc être nommées avec un padding numérique cohérent (`page_001.jpg`, `page_002.jpg`, …). Si ce n'est pas le cas, utiliser `--rename` avant l'OCR.

`--images` peut pointer vers un fichier unique ou un dossier.

## Renommage (`--rename`)

Renomme les images du dossier par date de création (`st_birthtime` sur Windows, fallback `st_mtime`), puis enchaîne sur le pipeline OCR.

```
python src/main.py --rename
python src/main.py --rename --rename-prefix scan   # → scan_001.jpg, scan_002.jpg, …
python src/main.py --rename --dry-run              # affiche les renommages, pas d'OCR
```

| Argument | Défaut | Description |
|---|---|---|
| `--rename` | off | Renomme avant OCR |
| `--rename-prefix` | `page` | Préfixe des nouveaux noms |
| `--dry-run` | off | Affiche les renommages sans les effectuer ni lancer l'OCR |

Résultat : `{prefix}_{numéro paddé}{extension}` — padding minimum 3 chiffres.
