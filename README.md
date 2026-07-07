# Handwritten Documents OCR Retrieval in LanceDB

This repository demonstrates how to use [LanceDB](https://docs.lancedb.com/) as the storage and
retrieval layer using handwritten documents and OCR-extracted text for document search.
It uses [this Kaggle Doctor Handwriting Recognition Dataset](https://www.kaggle.com/datasets/mamun1113/doctors-handwritten-prescription-bd-dataset)
The dataset is stored under the `data/` directory in this repo (download the dataset from Kaggle and place it in this location)

The main idea is that the handwritten PNG images of doctor's notes are stored natively as raw image bytes in LanceDB.
OCR, structured extraction features, embeddings, search results, and validation labels all live in the same table.
During OCR, the model receives the image bytes read back from LanceDB, and it writes back its predicted text into the same LanceDB table.

The table below shows how we clearly separate the storage stage from the retrieval stage, both of which involve LanceDB.

![Retrieval strategy diagram](assets/retrieval-strategy.svg)

## Setup

All scripts have been tested with `uv`, so this is the recommended way to set up the Python environment:

```bash
uv sync
cp .env.example .env
# add OPENAI_API_KEY=... to .env
```

If you prefer, you can also run the code with standard `venv` and `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# add OPENAI_API_KEY=... to .env
```

If you're not using `uv`, use standard `python *.py` commands to run the scripts below.

## Pipeline

The stages are run in the following order, via the `uv` commands listed below:

```bash
uv run src/00_inspect_data.py
uv run src/01_ingest_images.py
uv run src/02_ocr.py
uv run src/03_extract.py
uv run src/04_embed_index.py
uv run src/05_search.py "antibiotic medication" "fever and pain medication"
uv run src/06_rerank.py "antibiotic for infection"
uv run src/07_evaluate.py
```

We ingest the full Testing split from the source data: 780 data rows. The CSV has 781 lines including
the header. Use `--limit 10` on any stage for quick local debugging.

> [!NOTE]
> `src/01_ingest_images.py` compacts the table after the initial local LanceDB ingestion, by calling `table.optimize()`. The OCR script in `src/02_ocr.py` makes to five concurrent model API calls for OCR and bulk-writes results back to LanceDB in batches of 25 rows, so the batch size for ingestion must be a clean multiple of the concurrency setting.

### LanceDB-specific code

The LanceDB-specific pieces of the codebase are intentionally separated into logical components and concentrated in a few files:

- `src/db.py` defines the `DoctorHandwriting` LanceDB schema, including the raw `image: bytes`
  column, validation labels, OCR/extraction fields, and 384-dimensional `vector` column. It also
  centralizes the local DB connection, table creation/opening, bounded reads, and merge updates.
- `src/01_ingest_images.py` is the first LanceDB write path. It reads the Testing split PNG files as
  bytes, stores them directly in the table, adds validation-only labels, and runs `table.optimize()`
  once after initial ingestion.
- `src/02_ocr.py` and `src/03_extract.py` show how downstream stages read bounded rows from the same
  table and write computed columns back with batched `merge_insert` updates.
- `src/04_embed_index.py` embeds `searchable_summary`, writes the `vector` column, and builds the
  LanceDB scalar, full-text, and vector indexes.
- `src/05_search.py` is the baseline dense retrieval path using `table.search(query_vector)` with
  explicit `select(...)` and `limit(...)`.
- `src/06_rerank.py` starts from LanceDB dense candidates, then applies optional late-interaction
  reranking outside the table.
- `src/07_evaluate.py` reads bounded Testing rows from LanceDB and publishes baseline OCR/extraction
  metrics against the validation labels.

## Interfacing with VLMs

OCR and structured extraction are implemented with DSPy:

- OCR uses `dspy.Image` and `dspy.LM("openai/gpt-5.4-mini")` (several common model
providers are supported and can be swapped out as necessary).
- Extraction uses a DSPy signature over `ocr_text`.
- `OPENAI_API_KEY` is read from `.env` or the environment.

The OCR stage is expected to make live model calls. A `--mock-from-labels` flag exists only for quick
developer smoke tests of the storage/indexing path; do not use it for reported OCR quality.

### Why DSPy?

There are many reasonable ways to do structured extraction from OCR text, including native
structured-output endpoints from LLM/VLM providers. This demo uses DSPy because the OCR and extraction
steps become composable programs rather than one-off prompts. This matters because the prompts, field
descriptions, and overall program can be optimized with algorithms like
[GEPA](https://arxiv.org/abs/2507.19457), instead of relying only on manual prompt engineering.

That is also why the dataset's source splits are useful. The Testing split stays reserved for the
demo evaluation path, while the Training split can become a labeled optimization pool. First, run the
unoptimized OCR/extraction pipeline and publish its Testing-split baseline with:

```bash
uv run src/07_evaluate.py
```

That produces `outputs/baseline_results.md`, including a table the one shown below.

Use the following metric key to understand the results in the table:

- `Exact OCR match`: the model output exactly matches `MEDICINE_NAME`, including casing and punctuation.
- `Normalized OCR match`: both strings match after lowercasing and removing non-alphanumeric
  characters. This is the more forgiving word-level accuracy number.
- `Avg edit distance`: average Levenshtein distance after normalization. It is the number of
  single-character insertions, deletions, or substitutions needed to turn OCR text into the ground truth label.
- `Median edit distance`: the middle edit distance. A median of `1.00` means at least half the
  samples are within one character edit, even though many still fail exact/normalized matching.
- `Human review flagged`: rows where the OCR output differed enough from the label to merit review.
- `Extraction coverage`: rows where the structured extraction stage populated all expected fields.

| Run | OCR model | Split | Rows | Exact OCR match | Normalized OCR match | Avg edit distance | Median edit distance | Human review flagged | Extraction coverage |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unoptimized baseline | `gpt-5.4-mini` | Testing | 780 | 27.8% | 35.1% | 1.60 | 1.00 | 43.5% | 100.0% |

These results (and the edit distances) show that most extractions are off by just ~1-2 characters, so the low accuracy
numbers can be improved with a better model, or a better description of the extraction task.

Prompt optimization can help with this: a GEPA optimization pass over Training examples and compare
the optimized program against the same held-out Testing rows. With frameworks like DSPy, the OCR/extraction program
can be optimized to squeeze more quality out of a smaller, cheaper model. For document pipelines, that
can make the difference between a neat prototype and a system that scales to more images without
spending heavily on LLM tokens.

## Retrieval

`src/04_embed_index.py` embeds `searchable_summary` with
`sentence-transformers/all-MiniLM-L6-v2`, writes the 384-dimensional vector column, and builds:

- scalar indexes on `id`, `category`, and `needs_human_review`
- a full-text index on `searchable_summary`
- a cosine vector index on `vector`

If the sentence-transformers model is unavailable locally, the dense embedding path can fall back to
deterministic feature-hashing embeddings so ingestion/indexing demos remain runnable. Use
`--strict-embeddings` to fail instead.

`src/06_rerank.py` uses a ColBERT-style late-interaction pass by default with
`colbert-ir/colbertv2.0`: it encodes query and candidate text into token vectors, then scores each
candidate with MaxSim over the dense-search candidate set. Search and reranking scripts export
decoded image previews under `outputs/`.

### Example Retrieval Results

```bash
uv run src/05_search.py "antibiotic medication" --limit 3
```

```text
Query: antibiotic medication
Image previews: outputs/search/antibiotic-medication
01. distance=0.3693 id=test_00614 label=Omastin generic=Fluconazole ocr='Amoxicin' normalized='Amoxicin' category=medication review=True
02. distance=0.3708 id=test_00222 label=Diflu generic=Fluconazole ocr='Diffl' normalized='Diffl' category=medication review=True
03. distance=0.3780 id=test_00178 label=Candinil generic=Fluconazole ocr='Cemdinol' normalized='Cemdinol' category=medication review=True
```

Dense search finds medication-relevant rows, but the noisy OCR text pushes all three into human review.

```bash
uv run src/06_rerank.py "antibiotic for infection" --candidate-limit 10 --limit 3
```

```text
Query: antibiotic for infection
Dense top results:
  01. distance=0.4501 test_00614 Omastin (Fluconazole)
  02. distance=0.4543 test_00177 Candinil (Fluconazole)
  03. distance=0.4632 test_00175 Candinil (Fluconazole)
Late-interaction reranked results:
  01. late=2.2198 dense_rank=9 id=test_00699 label=Romycin generic=Azithromycin Dihydrate (Ophthalmic) ocr='Ramycin'
  02. late=2.2084 dense_rank=4 id=test_00698 label=Romycin generic=Azithromycin Dihydrate (Ophthalmic) ocr='Rnycin'
  03. late=2.1979 dense_rank=8 id=test_00694 label=Romycin generic=Azithromycin Dihydrate (Ophthalmic) ocr='RMYCIN'
```

Late interaction reranking promotes candidates with stronger token-level evidence for the query.

## Use LanceDB Enterprise for Production

Note that LanceDB OSS is fine for this local, reproducible proof-of-concept.

For real-world datasets, [LanceDB Enterprise](https://docs.lancedb.com/enterprise) is recommended as the natural
production path for larger document-intelligence workloads because it comes with the following benefits:
- More scalable retrieval due to separation of compute from storage
- Unbounded storage (larger, higher resolution images) leading to petabyte scale and beyond
- Scalable feature engineering with [Geneva](https://docs.lancedb.com/geneva)
- Access control, governance, operational monitoring, and reliability controls

Workflows like these in healthcare or other highly regulated spaces should **always** include
human validation (ideally with prompt optimization for better performance on the task),
auditability, approved datasets and models, security review, and appropriate compliance checks.
