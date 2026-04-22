# Images

## Collection (`collect_images`)

Returns the alphabetically sorted list of images in the `--images` folder (default: `./photos`).

Supported extensions: `.jpg`, `.jpeg`, `.png`, `.webp`, `.pdf`.

Sorting is alphabetical on the filename — images must therefore be named with consistent numeric padding (`page_001.jpg`, `page_002.jpg`, …). If not, use `--rename` before OCR.

`--images` can point to a single file or a folder.

## Renaming (`--rename` / `--rename-only`)

Renames images by creation date (`st_birthtime` on Windows, fallback `st_mtime`), with uniform numeric padding.

- `--rename`: renames then continues to the OCR pipeline.
- `--rename-only [START]`: renames without running OCR. `START` is the starting number (default: 1).

```
python src/main.py --rename
python src/main.py --rename --rename-prefix scan   # → scan_001.jpg, scan_002.jpg, …
python src/main.py --rename-only                   # rename only, start at 1
python src/main.py --rename-only 15               # rename only, start at 15
python src/main.py --rename --dry-run             # print renames, no OCR
```

| Argument | Default | Description |
|---|---|---|
| `--rename` | off | Rename before OCR |
| `--rename-only [START]` | off | Rename without OCR (START = starting number) |
| `--rename-prefix` | `page` | Prefix for new names |
| `--dry-run` | off | Print renames without performing them or running OCR |

Result: `{prefix}_{padded number}{extension}` — minimum 3-digit padding.

## Subfolders

If the `--images` folder contains subfolders with images, the pipeline automatically switches to copy mode: images are copied to the parent folder with sequential numbering.

Detection and copying are **recursive**: sub-subfolders are included.

### Default Sort Order

Without `--dir-level`, all images from each subfolder (recursively) are sorted globally by creation date.

### `--dir-level`: Folder-level Order

With `--dir-level`, the order respects the folder hierarchy:

1. First-level subfolders sorted in natural order (`lesson 2` before `lesson 10`)
2. Sub-subfolders sorted in natural order
3. Images in each folder sorted by creation date

```
python src/main.py --rename-only --dir-level
python src/main.py --rename-only --dir-level --dry-run
```

### `--chapters`: Subfolder Selection and Order

Allows choosing which subfolders to process and in what order.

```
python src/main.py --rename-only --chapters "Lesson 1" "Lesson 3"
```

## Recommended Folder Structure

For a book split into parts and chapters, the ideal structure is:

```
photos/
  01 - Part 1/
    01 - Chapter 1/
      IMG_0001.jpg
      IMG_0002.jpg
      …
    02 - Chapter 2/
      IMG_0010.jpg
      …
  02 - Part 2/
    …
```

Full workflow:

```
# 1. Check order before renaming
python src/main.py --rename-only --dir-level --dry-run

# 2. Rename
python src/main.py --rename-only --dir-level

# 3. Run OCR
python src/main.py
```

The numeric prefix in folder names (`01 -`, `02 -`) guarantees alphabetical order even without `--dir-level`. With purely textual names (`Chapter 1`, `Chapter 10`), `--dir-level` is needed for natural sorting.
