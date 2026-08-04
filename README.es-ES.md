

# ocr-book — Pipeline de OCR para Libros → Markdown

Digitaliza un libro completo en Markdown a partir de fotos de páginas, PDFs o EPUBs,
utilizando **PaddleOCR-VL-1.5** a través de **llama-server** (inferencia local).

---

## Prerrequisitos

- [miniforge](https://github.com/conda-forge/miniforge) o Anaconda
- [llama-server](https://github.com/ggerganov/llama.cpp) (se recomienda Vulkan en Windows)
- Modelo GGUF: [PaddleOCR-VL-1.5-GGUF](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5)

---

## Instalación

```bash
python setup.py
conda activate ocr-livre
```

Luego, configura las rutas a `llama-server` y a los modelos. La forma más fácil es copiar `.env.example` a `.env` y editarlo, pero también puedes usar variables de entorno o argumentos de CLI; consulta [docs/SETUP.md](docs/SETUP.md) para ver todas las opciones.

```bash
cp .env.example .env
# Edita .env y configura LLAMA_SERVER_PATH, MODEL_PATH y MMPROJ_PATH
```

---

## Estructura del Proyecto

```
ocr-livre/
├── src/
│   ├── main.py          # Punto de entrada CLI
│   ├── config.py        # Configuración central (dataclass)
│   ├── ocr_client.py    # OCR de una imagen vía PaddleOCRVL
│   ├── postprocess.py   # Limpieza del texto OCR
│   ├── obsidian.py      # Exportación a Obsidian (wikilinks, migración)
│   ├── images.py        # Recopilación y renombrado de imágenes
│   ├── pipeline.py      # Orquestación completa
│   ├── progress.py      # Registro de actividad y estadísticas
│   ├── pdf.py           # Procesamiento de PDFs (extracción de texto o renderizado → OCR)
│   └── epub.py          # Extracción de EPUB (basada en Pandoc)
├── docs/
│   ├── architecture/    # Documentación de arquitectura
│   ├── dev/             # Parches y notas de desarrollo
│   ├── SETUP.md         # Instrucciones de instalación
│   ├── tested.md        # Resultados de experimentos
│   └── issues.md        # Trabajo en progreso
├── photos/              # Imágenes de origen (una por página)
├── output/              # Markdown generado + registros + figuras
├── environment.yml      # Dependencias de Conda
└── setup.py             # Script de instalación automatizado
```

---

## Uso

Ejecutar desde la raíz del proyecto:

```bash
# Pipeline predeterminado (fotos en ./photos, salida en output/book.md)
python main.py

# Especificar carpetas
python main.py --images ./my_photos --out output/my_book.md

# Entrada PDF
python main.py --images ./book.pdf --out output/book.md

# Entrada EPUB
python main.py --images ./book.epub --out output/book.md

# Sin detección de diseño (layout)
python main.py --no-layout

# Reiniciar desde el principio
python main.py --no-resume

# Registros detallados
python main.py --verbose

# Tablas densas: aumenta el contexto si las tablas se truncan
python main.py --n-ctx 12288 --n-parallel 3
```

---

## Ejemplo

Una foto tomada con un teléfono de una página de un libro de texto (gráficos, tablas y texto denso) convertida a Markdown limpio con un solo comando.

![OCR before/after](docs/assets/ocr.png)

*Izquierda: foto original de la página. Derecha: Markdown extraído renderizado.*

---

## Procesamiento de PDFs

Los PDFs se clasifican automáticamente como **basados en texto** (capa de texto nativa) o **basados en imágenes** (escaneados).

- **Basado en texto**: extrae el texto nativamente con `pymupdf`, detecta figuras con el modelo de diseño, sin OCR VLM.
- **Basado en imágenes**: renderiza las páginas como imágenes y luego ejecuta el pipeline normal de OCR.

Elige el método de extracción explícitamente:

```bash
python main.py --images ./book.pdf --method text         # rápido, solo texto nativo
python main.py --images ./book.pdf --method docling      # extracción estructurada
python main.py --images ./book.pdf --method paddleocrvl  # mejor calidad, más lento
```

---

## Extracción de EPUB

Los EPUB se convierten a Markdown mediante Pandoc, extrayendo automáticamente las figuras incrustadas.

```bash
python main.py --images ./book.epub --out output/book.md
```

---

## Exportación a Obsidian

En el modo `obsidian`, el pipeline:
- convierte las figuras a wikilinks `![[Files/image.jpg]]`
- guarda el `.md` directamente en el vault
- copia las figuras a `vault_path/vault_figures_dir/`

Configura `vault_path` y `vault_figures_dir` en `config.py`, luego:

```bash
# OCR completo + exportación a Obsidian
python main.py --mode obsidian

# Reaplicar postprocesamiento de Obsidian sin volver a ejecutar el OCR
python main.py --mode obsidian --postprocess-only

# Migrar solo las figuras al vault
python main.py --migrate
```

---

## Renombrado de Imágenes

```bash
# Vista previa sin modificar
python main.py --rename --dry-run

# Renombrar realmente (→ page_001.jpg, page_002.jpg, …)
python main.py --rename

# Renombrar sin ejecutar OCR
python main.py --rename-only

# Procesar subcarpetas por capítulo
python main.py --rename-only --chapters "Chapter 1" "Chapter 2"
```

---

## Reanudación Automática

Si el pipeline se interrumpe, simplemente vuelve a ejecutarlo:

```bash
python main.py
```

Las páginas ya procesadas se omiten automáticamente.

---

## Opciones Completas

```
--images PATH              Carpeta de fotos, PDF o EPUB       (predeterminado: ./photos)
--out FILE                 Archivo Markdown de salida         (predeterminado: output/book.md)
--llama-server PATH        Ruta al ejecutable llama-server    (env: LLAMA_SERVER_PATH)
--model PATH               Ruta al modelo .gguf               (env: MODEL_PATH)
--mmproj PATH              Ruta a mmproj .gguf                (env: MMPROJ_PATH)
--mode {base,obsidian}     Modo de salida                     (predeterminado: base)
--method {text,docling,paddleocrvl}  Método de extracción PDF (predeterminado: paddleocrvl)
--no-layout                Desactivar detección de diseño
--no-resume                Reiniciar desde el principio
--no-postprocess           Salida cruda sin limpieza
--postprocess-only         Postprocesamiento Obsidian sin OCR  (requiere --mode obsidian)
--migrate                  Copiar figuras al vault            (requiere vault_path configurado)
--dry-run                  Simular sin modificar
--verbose                  Registros DEBUG
--rename                   Renombrar imágenes antes del OCR
--rename-only [N]          Renombrar sin ejecutar OCR         (N = número inicial)
--rename-prefix P          Prefijo de renombrado              (predeterminado: page)
--chapters NAME…           Subcarpetas a procesar (en orden)
--dir-level                Orden a nivel de carpeta para --rename
--max-tokens N             Tokens máximos generados por página (predeterminado: 4096)
--n-ctx N                  Tamaño de caché KV (ventana de contexto) (predeterminado: 6144)
--n-parallel N             Ranuras paralelas intra-página     (predeterminado: 3)
```

---

## Códigos de Salida

| Código | Significado                                      |
|--------|--------------------------------------------------|
| 0      | Éxito total                                      |
| 1      | Error fatal                                      |
| 2      | Finalizado con errores en algunas páginas        |
