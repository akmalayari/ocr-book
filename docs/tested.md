# Tested — What has been experimented in the project

Reference pages used for testing: `page_1` to `page_9`.
- page_1: simple text, poor lighting
- page_2: dense text, normal lighting
- page_3: text + text table
- page_4: text + numeric table
- page_5: text + table + graph, blurry image
- page_6: same content as page_5, sharp image (reference version for graph tests)
- page_7: chapter start, text only, sharp image
- page_8: text only, sharp image
- page_9: chapter end, text only, sharp image

---

## Experiment Summary

| Topic | Result |
|---|---|
| **DeepSeek-OCR + nexaai stack** | Abandoned — migration to PaddleOCR VL 1.5 |
| **DeepSeek-OCR prompts** (layout, plain, rec…) | Abandoned with the model |
| **Image preprocessing** (binarize, sauvola, nlmeans, sesr…) | Abandoned — PaddleOCR works on raw image |
| **Inter-page parallelization** (`n_servers > 1`) | Abandoned on APU — Vulkan GPU serialization |
| **HTTP Streaming for -np 4** | Abandoned — incomplete text, no gain |
| **OTSL Patch** (`apply_paddlex_patch_otsl.py`) | Retained — VLM errors per region on complex tables |
| **Intra-page parallelism patch** (`apply_paddlex_patch_parallel.py`, -np 3) | Retained — ~30% gain (60s → 43s/page) |
| **PaddleOCR VL 1.5 + llama-server** | Retained — current stack |
| **`page_timeout` + no-layout fallback** | Retained — protection against llama-server loops |

---

## DeepSeek-OCR (abandoned — migration to PaddleOCR VL 1.5)

> "Retained" statuses in this section are relative to the DeepSeek-OCR era only.
> This model was entirely replaced by PaddleOCR VL 1.5 on 2026-04-08.

### Inference Stack

#### REST API `nexa serve`
**Status: abandoned.**
Systematically returned HTTP 500 on multimodal requests under Windows. Nexa's REST server cannot handle the GGUF + mmproj format of DeepSeek-OCR.

#### `nexaai.VLM` Direct Python
**Status: retained (DeepSeek-OCR era).**
Loads both model files (GGUF + mmproj) via `nexa_bridge.dll`. Bypasses the REST server entirely. Interface: `VlmChatMessage` + `apply_chat_template` + `vlm.generate(GenerationConfig(image_paths=[...]))`.

Nexaai uses llama-cpp under the hood — DeepSeek-OCR can run on GPU via Vulkan.

**Limit:** on Windows, `nexa_bridge.dll` returns a corrupted `stop_reason` (invalid UTF-8 byte `0xc0`) in profiling metadata. Worked around by a monkey-patch in `src/patch.py` — the OCR text itself is intact.

---

### Model Quantization

| Quantization | Speed | Quality |
|---|---|---|
| Q8_0 | ~20s/page | immediate loops on difficult pages |
| BF16 | ~50s/page | more faithful than F16 on difficult passages |
| F16  | ~50s/page | less faithful than F16 — hallucinations and rephrasings on difficult passages |

**Status: BF16 retained (DeepSeek-OCR era).** BF16 vs F16 comparison (2026-04-02, `compare.py` sentence mode, page_1): BF16 faithfully retranscribes several passages where F16 hallucinates or rephrases. BF16 remains imperfect (see Bug #1 and #5).

---

### Prompts

Tested on pages 1–6 with `preprocess=binarize` (`draft/test_prompts.py`).

| Mode | Prompt | Result |
|---|---|---|
| `plain` | `"Free OCR."` | clean raw text on simple pages, loops on dense pages |
| `layout` | `"<\|grounding\|>Convert the document to markdown."` | adds grounding boxes, fixes page_2 loop but loops on tables (`<tr>`/`<td>`) |
| `describe` | `"Describe this image in detail."` | description in English, regardless of document language |
| `parse` | `"Parse the figure."` | fine analysis of visual elements in English |
| `rec` | `"Locate <\|ref\|>{target}<\|/ref\|> in the image."` | returns bbox(s) matching target. See dedicated section below. |
| `classic` | (removed) | one grounding box per phrase, systematically exceeds `n_ctx` |

**Status:** `layout` retained (DeepSeek-OCR era). Slightly better accuracy than `plain`. Allows retrieving bboxes for image processing in the second pass.

**`repetition_penalty`** tested — apparently no effect on generation loops.

#### `rec` Mode Behavior (2026-04-03, page_6, BF16, binarize)

`Locate <|ref|>{target}<|/ref|> in the image.`

| Prompt | Result |
|---|---|
| `Locate <\|ref\|>A figure or graph<\|/ref\|> in the image.` | 1 exact bbox for the graph, stops cleanly (5.8s) |
| `Locate <\|ref\|>A figure or graph<\|/ref\|> in the image. Describe it.` | identical — post-bbox instruction ignored |
| `Locate <\|ref\|>A figure or graph<\|/ref\|> in the image. Parse it.` | identical — post-bbox instruction ignored |
| `Locate <\|ref\|>every element<\|/ref\|> in the image.` | all page bboxes (same result as `layout`) |
| `Locate <\|ref\|>everything<\|/ref\|> in the image.` | all page bboxes |
| `Locate <\|ref\|>pliure du livre<\|/ref\|> in the image.` | all page bboxes |
| `Locate <\|ref\|>image<\|/ref\|> in the image.` | all page bboxes + `<td>` loop on table |

**Observed behavior:** when the model does not find the requested element (or the target is too generic), it returns all page bboxes. When it finds a specific element, it returns only its bbox and stops. Any instruction added after `<|det|>` is ignored — two-pass is mandatory to get a region's content.

**Retained usage:** `Locate <|ref|>A figure or graph<|/ref|> in the image.` for quick test, but not needed in the pipeline since `layout` already finds bboxes with text included.

#### Grounding Label Vocabulary (layout and rec modes)

Observed on pages 1–6 (BF16, binarize):

| Label | Content |
|---|---|
| `text` | regular text block |
| `title` | title |
| `sub_title` | subtitle / heading |
| `table` | table (including figures misclassified on blurry image) |
| `table_caption` | table caption |
| `table_footnote` | table footnote |
| `image` | figure / graph (correct label on sharp image) |
| `image_caption` | figure caption |

**Image quality effect on classification:** page_5 (blurry) → graph classified `table`, hallucination `<td>30</td>` in loop. Page_6 (sharp, same content) → graph correctly classified `image`, empty content.

**Content of `image` regions:** always empty — the model detects and delimits the region but generates no description. See section below for two-pass crop results.

#### Second Pass on Graph Crop (2026-04-05, page_6, BF16, `draft/test_two_pass.py`)

| Combination | Result |
|---|---|
| `parse` + raw | structured table, table footnotes missing |
| `parse` + binarize | structured table, table footnotes present but treated as table rows |
| `describe` + raw | general description + interpretation noise |
| `describe` + binarize | general description + interpretation noise |

**Verdict: `parse` + `binarize` retained (DeepSeek-OCR era)** for second pass on `image` regions. Known limitation: table footnotes are included in the table instead of being separated.

#### Custom Prompts Tested (2026-04-03, BF16, binarize)

| Prompt | Result |
|---|---|
| `"Describe this image in detail in french."` | description in English (language instruction ignored) |
| `"Describe this image in detail in the language of the document."` | description in English |
| `"Décrit cette image en détail."` | description in English |
| `"Décrit cette image en détail en français."` | description in English |
| `"What is the language of the document?"` | answers `"pt"` then OCRs the page |
| `"Figure or text?"` | loops on title |
| `"Does this document contain a figure?"` | describes document in English |
| `"Does this document contain a figure? Yes \| No"` | writes titles then a short incoherent paragraph in French |
| `"If this is a figure or illustration, describe it. Otherwise, Free OCR."` | describes image in English even on text page |
| `"Transcribe the text exactly as it appears."` | unstable and disorderly OCR, lower quality than `"Free OCR."` |
| `"Is there a figure in this document?"` | pages 1–3: general description ; page_4: detailed table description ; page_5: loop |
| `"OCR only the text, ignore any figures."` | ignores figures and tables, but also skips some text columns (partial behavior — see issues Feature 1) |
| `"<\|grounding\|>Describe this image in detail."` | exact equivalent of `layout` (grounding boxes + region classification) — tested page_6 (43.5s) |
| `"<\|grounding\|>Parse the figure."` | **terminal freeze** — avoid |

**Conclusion:** the model systematically ignores language instructions. Classification prompts (yes/no, figure?) do not produce usable structured responses. `"OCR only the text, ignore any figures."` is the only prompt that actually filters figures, but incompletely.

---

### Image Preprocessing

> Explorations conducted with DeepSeek-OCR. PaddleOCR works directly on raw image —
> no preprocessing is applied in the current pipeline.

#### Original Image (no preprocessing) — `"none"` mode
**Status: retained as reference (DeepSeek-OCR era).** Initially abandoned (loops in Q8_0 + `plain` prompt on page_1). Re-evaluated in BF16 + `layout` prompt (2026-04-06/07): no loops on pages 4, 5, 6, 9. Best text score on blurry image (94.9% on page_5). Only config without loops on all tested pages with light preprocessing.

**Limit:** does not detect figure on sharp page_6 (graph ignored or absorbed into text).

#### Pillow Exposure Boost (contrast ×1.8 + brightness ×1.2)
**Status: abandoned.** Avoids loops but produces fewer words (~830 vs ~1000 for binarize_adaptive), slower (~25s), more hallucinations.

#### CLAHE (adaptive contrast equalization, OpenCV LAB)
**Status: abandoned.** Tested visually (`draft/viz_preprocess2.py`) and in OCR (`draft/preprocess_test.py`). Produces ~1519 words but with loops at end of generation. No clear accuracy improvement.

#### Otsu Binarization (GaussianBlur(5,5) + global Otsu threshold)
**Status: abandoned.** Loops immediately, massive hallucinations (page_1: repetitions of "Ils sont des mots utilisés dans la société").

#### EqualizeHist alone / EqualizeHist + adaptive binarize
**Status: abandoned.** Tested visually (`draft/viz_preprocess2.py`). EqualizeHist amplifies background noise and lighting gradients — counterproductive before adaptive binarization.

#### bg_divide + adaptive binarize
**Status: not retained.** Tested visually (`draft/viz_preprocess2.py`). Normalizes illumination by dividing by estimated background (GaussianBlur 101×101). Visually interesting on pages with very uneven lighting, but not tested in OCR.

#### Adaptive Binarization (GAUSSIAN_C, blockSize=31, C=10)
**Status: replaced by blockSize=31, C=15.** Causes generation loops on blurry or noisy pages.

**Advantages:** fast, removes background, robust to local lighting variations.
**Limits:** C=10 erases soft strokes on blurry images → HTML `<td>` loops.

#### GaussianBlur(5,5) + adaptive binarize (blockSize=31, C=15)
**Status: retained (DeepSeek-OCR era).** Grid test blockSize ∈ {21,31} × C ∈ {10,15} on page_5 (noisy, Laplacian=58) and page_6 (sharp, Laplacian=134) via `draft/test_binarize_grid.py` + `draft/compare_grid.py`.

- C=10 (old default): HTML `<tr><td>` loops on blurry/noisy pages. Loop detection by word frequency insufficient for this type of loop — addition of `_has_char_repeat` in `ocr_client.py`.
- C=15: no loops on page_5 or page_6. blockSize=31 ≥ blockSize=21 on noisy page (91% vs 85% word-level similarity vs reference page_6).

**Limits:** degraded accuracy on very noisy pages (page_5).

#### Morphological Operations after Binarization (opening/closing)
**Status: abandoned.** Tested on pages 1, 2, 5, 6 via `draft/test_morpho.py` (kernels 2×2 and 3×3, opening, closing, open+close, close+open). Text remains fragmented on noisy pages — morphological operations do not recover strokes erased by adaptive binarization.

#### Sauvola Binarization alone (scikit-image, `threshold_sauvola`)
**Status: abandoned.** Renders text well in general, but erases text in low local variance zones (fold, binding shadow) → the `AND` variant below is retained instead.

#### fastNlMeansDenoising + adaptive binarize — `"nlmeans"` mode
**Status: OCR phase completed on pages 4, 5, 6, 9.** Tested visually on pages 4, 5, 9 via `draft/test_nlmeans.py`. Configs evaluated: `nlmeans_5`, `nlmeans_10`, `nlmeans_15`, `nlm5_median`, `nlm5_open`, `nlm5_and`, `nlm10_and`, `nlm10_bgdiv`, `nlm10_bgdiv_and`, `median_adaptive`, `median_and`.

**Visual observations:**
- `nlmeans_5`: residual granules on some pages, text sharper than baseline.
- `nlmeans_10`: good denoising/stroke preservation balance.
- `nlm5_median` (nlmeans h=5 + medianBlur(3) + adaptive): granules removed, promising.
- `nlm5_and`: very good on page_9, granules on other pages.
- `nlm10_and`: good results, consistent across pages.
- `median_and` (medianBlur(3) + AND(Sauvola, adaptive)): promising, fast.
- `nlm10_bgdiv`, `nlm10_bgdiv_and`: not retained for OCR after visualization.
- `nlm5_open` (MORPH_OPEN post-binarization): not retained.

**Configs retained for OCR phase:** `median_and`, `nlm5_median`, `nlmeans_10`, `nlm10_and`, `nlm5_and`.

**OCR Results (2026-04-06, `draft/test_nlmeans.py --ocr`, pages 4/5/6/9, BF16, layout):**

- **page_4**: all configs loop. `median_and`, `nlm5_median`, `nlm10_and` approximately reproduce table before looping; `nlmeans_10` and `nlm5_and` loop without useful result.
- **page_9**: `nlm5_median` and `nlm10_and` loop immediately. `nlm10` pseudo-loop (number sequence — false negative of loop detection). `median_and` and `nlm5_and` do not loop, accuracy apparently subpar.
- **page_5** (noisy): `nlm5_and` → 4 words (image bbox only, failure). `median_and`, `nlm5_median`, `nlmeans_10`, `nlm10_and` → 820–958 words, no loop. Accuracy evaluated — see section below.
- **page_6** (sharp, same content as page_5): all configs without loop, 944–1005 words. Authentic reference: `photos/md/page_6.md` (accuracy 100%).

**Loop detection:** `page_4_median_and` looped without being detected because `<|det|>[[x,y,…]]<|/det|>` coordinates change per block, diluting the `repeated/n_unique` ratio. Fix: remove `<|det|>` blocks before frequency analysis in `_is_looping` (`src/ocr_client.py`).

**Integration:** `src/preprocess.py::nlmeans_binarize()`, `nlmeans_h: int = 15` in `Config`.

#### medianBlur(3) alone + adaptive
Tested visually in `draft/test_nlmeans.py`. The `median_and` variant (with AND Sauvola) is retained for OCR.

#### AND(Sauvola w=51 k=0.3, adaptive binarize) — `"sauvola"` mode
**Status: retained (DeepSeek-OCR era).** Tested on pages 1, 2, 5, 6 via `draft/test_sauvola_patch.py` + full pipeline (`--preprocess sauvola`).

`bitwise_and(sauvola(gray, w=51, k=0.3), adaptive_binarize(gray))` — preserves text pixels detected by either → corrects Sauvola's text loss in fold, improves accuracy on noisy pages.

**Estimated accuracy (page_5):** `0.98 × 0.95 = 93%` (vs `0.98 × 0.92 = 90%` for baseline). Calculation: `sim(sauvola_page5, baseline_page6) = 95%` and `sim(baseline_page5, baseline_page6) = 92%`, with `sim(page6_31_15, page6_31_10) ≈ 98%` as proxy for reference accuracy.

**Full pipeline (2026-04-05, `--preprocess sauvola`):**
- Loops on pages 4, 9, 10.
- Page 4: approximately retranscribes large table (best attempt to date), loops on table footnotes.
- Page 9: retranscribes landmarks without looping, loops on bibliography start.
- Page 10: loops in middle of text.
- Page 5: complete failure on graph (very blurry, 2nd pass).

**Full pipeline (2026-04-05, `--preprocess binarize`):**
- Loops on pages 4, 5 (2nd pass), 9.
- Page 4: loops on large table without retranscribing information.
- Page 5: loops only on 2nd pass (figure), main text OK.
- Page 9: loops immediately ("un, " in loop).
- Page 10: first transcription of Lorenz curve, very poor quality.

**Integration:** `src/preprocess.py::sauvola_binarize()`, `preprocess_mode="sauvola"` in `Config`, `--preprocess sauvola` in CLI.

#### Comparative Preprocessing Evaluation — page_5 / page_6 (2026-04-06)

5 configs × 2 pages via `compare_ocr.py` (diff=sentence, score=word). Reference: `photos/md/page_6.md`.
Full report: `output/rapports/preprocess_p5_p6.md`.

**page_5 — blurry (Laplacian=57.97):**

| Config | Text % | Fig % | Global % |
|--------|---------|-------|----------|
| none | **94.9%** | 16.9% | **92.3%** |
| sauvola_and | 93.9% | 16.7% | 92.0% |
| nlmeans_and | 93.1% | 17.0% | 90.8% |
| median_and | 92.2% | **38.0%** | 91.2% |
| blur_adaptive | 92.0% | 16.8% | 90.1% |

**page_6 — sharp (Laplacian=134.19):**

| Config | Text % | Fig detected | Fig % | Global % |
|--------|---------|:------------:|-------|----------|
| blur_adaptive | **96.3%** | yes | 94.9% | **96.3%** |
| none | 95.7% | **no** | — | 96.1% |
| nlmeans_and | 96.1% | yes | 92.8% | 95.8% |
| median_and | 96.0% | yes | 92.8% | 95.6% |
| sauvola_and | 94.6% | yes | **96.6%** | 94.4% |

**Conclusions:**
- On blurry image, `none` gives best text score — preprocessing degrades already soft strokes.
- On sharp image, `blur_adaptive` is the best balanced config. `none` systematically misses the figure.
- Figure remains intractable on blurry image regardless of config (max 38%).
- **Idea:** conditional preprocessing based on Laplacian — `blur_adaptive` if > threshold, `none` otherwise.

#### Light Preprocessings without Binarization (2026-04-07, `draft/test_preprocess.py`, pages 4/5/6/9, BF16, layout)

Tested hypothesis: DeepSeek-OCR being trained on natural photos, light filters preserving the photo look are preferable to binarization. Scripts: `draft/test_preprocess.py` (OCR), `draft/realesrgan_sesr.py` (SR generation).

Full report: `output/rapports/preprocess_legers_analyse.md`.

**fastNlMeansDenoising alone — `"nlmeans"` mode**
**Status: retained, pipeline default (DeepSeek-OCR era).** `h = nlmeans_k × noise_level`. Loops on noisy page_4 (noise=5.5) — only problematic page. No loops on pages 5, 6, 9, 10 and clean variants. Pure text accuracy ~99% on clean images (p56c_nlmeans=99.3%). Preprocessing cost ~2-3s.

**bilateralFilter(d=9, σ=75) — `"bilateral"` mode**
**Status: abandoned.** Loops on page_6 (sharp, 31 words) and page_4. The "cartoon" effect (very smooth zones + very sharp edges) disturbs the model on sharp images. Opposite of expected behavior.

**SESR-M7 x2 (AMD NPU, 256×256 tiles) → resize original — `"sesr"` mode**
**Status: retained, available as option (DeepSeek-OCR era).** ~7s/image on NPU. Loops on noisy page_4 (noise=5.5) and on page_4_clean on page-bottom symbols (main text complete). Pure text accuracy slightly better than nlmeans on clean images (p56c_sesr=98.8%, p6_sesr=99.2%). Integrated in `src/sesr.py`.

| Config | p5 text % | p6 text % | p56c text % |
|--------|:---:|:---:|:---:|
| none | 98.8% | 98.7% | 97.9% |
| nlmeans | 98.9% | 98.8% | 99.3% |
| sesr | 98.7% | 99.2% | 98.8% |

Results 2026-04-07, `compare_ocr.py` text-only component mode, reference `photos/md/page_6_text.md`, report `output/rapports/global_report_5-6.md`.

**RealESRGAN x4 (AMD NPU, 128×128 tiles) → resize original — `"esrgan"` mode**
**Status: abandoned.** ~137s/image. Zero or marginal gain vs `none`. Prohibitive cost/benefit ratio.

**Architectural Pivot (2026-04-07)**
Tables and figures treated as crops (referenced by image path, not retranscribed). Accuracy score calculated on **pure text only**. Objective: >99% text on clean image. Clean images available: page_4, page_5, page_6, page_9, page_10.

#### Unsharp Mask (standard: `img + alpha*(img - blurred)`)
**Status: abandoned.** Tested in `draft/test_unsharp.py`. Amplifies high frequencies — adds dark granules, degrades binarization on most configs. Ineffective on focus blur (high frequency information is physically lost).

#### Inverted Unsharp Mask (`blurred - alpha*(img - blurred)`)
**Status: abandoned.** No improvement on binarization, introduces artifacts on some configs.

#### page-dewarp (lmmx, `pip install page-dewarp[jax]`)
**Status: abandoned.**

Our images are double pages (two pages per photo). page-dewarp is designed for single pages — it does not correctly detect contours or text lines on double-page images and produces degraded results.

Workaround attempt: cut image in two halves (mid-width), dewarp each half separately, recombine. Problem: page-dewarp crops output according to its page contour detection, which cuts edges of artificial half-images and makes recombination incoherent.

---

## PaddleOCR VL 1.5 (current stack, 2026-04-07)

Model: PaddleOCR-VL-1.5, 0.9B parameters, GGUF F16 format.
Scripts: `draft/test_paddle.py`, `draft/compare_ocr.py`.

### Retained Stack

- **llama-server** (llama-b8683-bin-win-vulkan-x64, Vulkan) — VLM inference
- **paddleocr** from main repo (not PyPI 3.4.0 — `llama-cpp-server` backend absent from release) — layout + prompt routing orchestration
- **paddlepaddle CPU** — layout detection (ppdoclayout)
- **Python 3.10** (conda env `ocr-livre`) — paddlepaddle incompatible with Python 3.11+
- Required paddlex patch: `docs/dev/apply_paddlex_patch_otsl.py` (see `docs/dev/paddlex_patch_otsl.md`)

### Accuracy Results (pure text, reference `page_6_text.md`)

| Page | Condition | Text % | Notes |
|------|-----------|---------|-------|
| page_6 | clean | 100% | only diff: header level (`##` vs `###`) |
| page_5-6 | clean | 100% | same |
| page_5 | noisy | 99.9% | only real error: "croisance" instead of "croissance" |
| page_4 | noisy | 97% vs clean version | complex table: digits with some errors, table footnotes ignored |
| page_9 | noisy vs clean | 99.5% | minor errors on noisy version |

Global (~98%): differences on accents, formatting or easily interpretable errors.

### Speed

35–40s/image vs 45–55s for DeepSeek-OCR BF16. ~25% gain.

### Behavior on Complex Tables

PaddleOCR uses a specialized pipeline for tables:
1. ppdoclayout extracts cells via traditional OCR
2. Content is encoded in OTSL (`<fcel>col<fcel>col<nl>...`)
3. VLM receives OTSL for HTML reconstruction

With the `llama-cpp-server` backend, llama-server cannot parse OTSL as image → 500 error. The `paddlex_patch_otsl.md` patch intercepts this error per region, extracts OTSL from the error message, and converts it directly via `convert_otsl_to_html()`.

### Output Format

Embedded HTML in Markdown:
- Tables: `<table><tr><td>...</td></tr></table>` (raw HTML without styles — `pretty=False` retained, see below)
- Figures: `<img src="imgs/..." />` (crop saved locally)
- Superscripts: `<sup>er</sup>`

No DeepSeek tokens (`<|ref|>`, `<|det|>`).

### Speed and Optimizations (2026-04-08/09)

Baseline speed measured: ~60s/page. Bottleneck = raw generation speed (~36 tok/s under Vulkan) + idle between blocks (GPU alternates layout detection → HTTP call → idle).

**`n_parallel=2` alone (llama-server `-np 2`, full pages in parallel)** : tested, **abandoned**. Increases total time (128s + 85s vs 55s + 55s). Vulkan GPU contention — both requests fight for the entire GPU. Context divided between slots: -np 2 with -c 4096 → 2048 tokens/slot → long blocks truncated (HTTP 400).

**PIL Resize before predict (`--max-image-size 1500`)** : tested, **abandoned**. Slightly faster but strongly degrades OCR quality. Cause: resize was applied **before** layout detection. Source image: 4080×3072.

**`max_pixels` (PaddleOCR parameter)** : not applicable for `llama-cpp-server` (only `vllm-server`). Silently ignored. Internal default: `28 × 28 × 3600 = 2,822,400` pixels.

**`save_to_markdown(pretty=True)`** : **retained**. Inline styles are added by PaddleOCR in post-processing, not generated by the VLM. Strip `<td>`/`<th>` styles in `postprocess.py`.

**Intra-page parallelization (global ThreadPoolExecutor pool, 2026-04-09)** : **retained**, `docs/dev/apply_paddlex_patch_parallel.py`.

Principle: PaddleOCR processes blocks sequentially. The patch submits all blocks from all pixel_keys simultaneously to a global pool, workers pick continuously without restart.

| Config | Time/page |
|---|---|
| sequential (baseline) | ~60s |
| -np 2, 2 workers | ~49s |
| -np 3, 3 workers | ~46s |
| -np 4, 4 workers | crash (vision encoder Vulkan saturated) |
| -np 6, 6 workers | hang |
| -np 3, 3 workers, **-c 6144** (2048/slot) | **~43.6s** — **retained** |

Gain: ~35 min on 150 pages. Retained config: `-np 3 -c 6144` (2048/slot). Reduction from `-c 12288` → `-c 6144`: additional ~2.5s/page gain without observed truncation.

**Inter-page parallelization — 2 llama-servers (2026-04-09)** : **abandoned** (still available via `n_servers > 1` in `Config`, but useless on APU). `src/pipeline.py` modified to launch N servers on distinct ports (8080, 8081…) and process pages in parallel via `ThreadPoolExecutor`.

Results on 3 test pages (pages 1–3):
- Pages 1 and 2 (simultaneous): ~102s and ~105s each
- Page 3 (freed server): ~55s
- Average throughput: ~53s/page — zero gain vs sequential (56s/page baseline)

Cause: on APU (Ryzen AI 9 HX370), GPU and RAM are physically the same LPDDR5X. Under Vulkan/Windows, command queues from two distinct processes are **serialized by the driver** — no true inter-process parallelism. Each server gets ~50% of GPU, becoming ~2× slower. Total throughput is identical.

Conceptually tested variant (1 GPU + 1 CPU): abandoned without implementation. CPU inference ~5–10× slower than GPU, same memory bus → throughput limited by CPU server.

**Conclusion:** on single GPU, continuous batching intra-page (1 server, -np 3) remains the only effective optimization. Multi-server approach brings nothing on APU.

**Final retained config (2026-04-09)** : 1 server, -np 3, -c 6144. Measured speed on `photos/test/`: **43.4s/page text, 37s/page with graph**.

Updated pipeline architecture in this version:
- Pages written to `output/parts/<page_id>.part` (no lock, robust resume to crashes)
- Combination in input order at end of run
- Retry ×1 in `ocr_client.py` on empty output, MD not generated or empty MD
- Fallback `PaddleOCRVL` (layout → no-layout) removed at this stage (covered by OTSL patch)

### Page Timeout + No-layout Fallback (post 2026-04-09)

`page_timeout = 120s`: `pipeline.predict()` runs in a monitoring thread. If timeout is exceeded, `OCRTimeout` is raised.

On `OCRTimeout` in `pipeline.py`:
1. All servers are killed and relaunched (`restart_servers()`)
2. Page is reprocessed with fallback pipeline `use_layout_detection=False`
3. If fallback also fails (`OCRError`): block `<!-- Page page_xxx — ERROR -->` + continue

**Motivation:** some pages trigger internal llama-server generation loops not detected by `ocr_client.py` (HTTP call never returns). Neither retry nor OTSL patch covers this case. Timeout + restart is the only clean recovery without blocking the run.

### HTTP Streaming to Unblock -np 4 (2026-04-09)
**Status: abandoned.** `docs/obsolete/apply_paddlex_patch_streaming.py`.

Principle: patch `GenAIClient.create_chat_completion` (genai.py) to use `stream=True` and release an asyncio semaphore after the first content token (= end of prefill). Would allow 3 simultaneous prefills max while keeping 4 slots in generation.

Results:
- `-np 3` + streaming: no gain (expected — 3 slots available, 4th worker queues server-side)
- `-np 4` + streaming: incomplete text, no time gain

Probable cause of incomplete text: llama-server sends a `role: assistant` chunk (empty content) before prefill ends. Initial version released semaphore on `stream.__anext__()`, i.e. on this empty chunk, letting 4 prefills through simultaneously and corrupting generations. Corrected version (wait for first non-empty token): same result — incomplete text, no gain. Root cause not identified, probably a llama-server limitation of 4 simultaneous slots in streaming under Vulkan.

### Verdict

**PaddleOCR VL 1.5 retained and integrated** — superior to DeepSeek-OCR BF16 on all criteria:
accuracy, speed, absence of loops. Resolves Bug 4 (loops), Improvement 1 (speed) and Improvement 2 (alternative model).

Main pipeline (`src/`) migrated to PaddleOCR (2026-04-08): `ocr_client.py` rewritten, `pipeline.py` simplified (preprocess/figure/nexaai removed), `config.py` cleaned.

---

### `markdown_ignore_labels`

`PaddleOCRVL(...)` constructor parameter — list of block labels excluded from markdown output.

**Retained config:**
```python
markdown_ignore_labels=["header_image", "footer", "footer_image"]
```

| Label removed from list (included) | Observed Effect |
|---|---|
| `number` | printed page number recovered → extracted in `<!-- Page page_2 (p. 42-43) -->` via `extract_page_number()` |
| `header` | running header recovered (ex: "Lesson 5") — tested pages 2 and 9, no particular noise |
| `footnote` | footnotes recovered — tested, ignored as noise without value |
| `aside_text` | marginal text recovered — retained (potentially useful content) |

Labels always ignored: `header_image`, `footer`, `footer_image`.

---

### Generation Parameters

| Parameter | Tested Value | Effect |
|---|---|---|
| `max_tokens` | 4096 | current value — 2048 cut some long pages |
| `temperature` | 0.0 | deterministic, retained |
