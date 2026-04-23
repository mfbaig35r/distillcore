# distillcore

Universal document processing: extract, chunk, enrich, embed, validate.

distillcore takes any document (PDF, text, markdown) and runs it through an intelligent pipeline — extracting text, classifying the document, breaking it into structured sections, chunking for RAG, enriching chunks with LLM-generated metadata, generating embeddings, and validating coverage at every stage.

Works as a **Python library** or a **standalone FastMCP server**. Domain-neutral by default, with pluggable presets for specialized domains (legal built-in).

## Install

```bash
# Core (text extraction + OpenAI for LLM/embeddings)
pip install distillcore

# With PDF support
pip install distillcore[pdf]

# With MCP server
pip install distillcore[mcp]

# Everything
pip install distillcore[all,mcp]
```

## Quickstart

### Process text

```python
from distillcore import process_text, DistillConfig

result = process_text(
    "Introduction\n\nThis report covers Q4 results...\n\nConclusion\n\nWe recommend...",
    config=DistillConfig(),
)

for chunk in result.chunks:
    print(f"[{chunk.chunk_index}] {chunk.topic} ({chunk.relevance})")
    print(f"  {chunk.text[:80]}...")
    print(f"  embedding: {len(chunk.embedding)}d")
```

### Process a file

```python
from distillcore import process_document

result = process_document("report.pdf")
print(f"Type: {result.document.metadata.document_type}")
print(f"Chunks: {len(result.chunks)}")
print(f"Validation: {result.validation.passed}")
```

### Skip LLM calls

```python
from distillcore import process_text, DistillConfig, EmbeddingConfig

# Custom embedder — no OpenAI needed
def my_embedder(texts):
    return [[0.0] * 384 for _ in texts]

result = process_text(
    "Your text here...",
    config=DistillConfig(
        enrich_chunks=False,
        embedding=EmbeddingConfig(embed_fn=my_embedder),
    ),
)
```

### Persist and search

```python
from distillcore import process_text, DistillConfig
from distillcore.storage import Store

result = process_text("Document content...", config=DistillConfig())

store = Store()  # ~/.distillcore/store.db
doc_id = store.save(result)

# Cosine similarity search
results = store.search(query_embedding=[0.1, 0.2, ...], top_k=5)
```

### Domain presets

```python
from distillcore import process_document, DistillConfig, load_preset

# Legal domain — extracts case numbers, attorneys, court orders
result = process_document(
    "motion.pdf",
    config=DistillConfig(domain=load_preset("legal")),
)
print(result.document.metadata.extra)
# {"case_number": "2024-CV-001", "court": "Superior Court", ...}
```

### Custom domain

```python
from distillcore import DomainConfig, DistillConfig, process_text

medical = DomainConfig(
    name="medical",
    classification_prompt="You are a medical document analyst. Extract...",
    enrichment_prompt="Tag each chunk with medical concepts...",
)

result = process_text("Patient presents with...", config=DistillConfig(domain=medical))
```

## MCP Server

Run as a standalone FastMCP server:

```bash
pip install distillcore[mcp]
distillcore
```

### Tools

| Tool | Description |
|------|-------------|
| `distill_file` | Process a document file through the full pipeline |
| `distill_text` | Process raw text (skips extraction) |
| `distill_chunks_only` | Chunk text without LLM calls |
| `distill_validate` | Validate coverage between text and chunks |
| `distill_search` | Semantic search across stored documents |
| `distill_list_documents` | List stored documents |
| `distill_get_document` | Get a document and its chunks |

Use `store=True` on `distill_file` / `distill_text` to persist results for later search.

## Pipeline stages

```
extract → classify → structure → chunk → enrich → embed → validate
```

1. **Extract** — pull text from PDF (with OCR fallback), TXT, or MD files
2. **Classify** — LLM identifies document type, title, and domain-specific metadata
3. **Structure** — LLM breaks the document into hierarchical sections
4. **Chunk** — section-aware splitting with paragraph-boundary overlap
5. **Enrich** — LLM tags each chunk with topic, key concepts, and relevance
6. **Embed** — generate vector embeddings (OpenAI default, swappable)
7. **Validate** — coverage checks at each stage (structuring, chunking, end-to-end)

Every LLM stage degrades gracefully — if the API key is missing or a call fails, the pipeline continues with fallback values.

## Configuration

```python
from distillcore import DistillConfig, ChunkConfig, EmbeddingConfig, DomainConfig

config = DistillConfig(
    # LLM
    openai_api_key="sk-...",       # or set OPENAI_API_KEY env var
    openai_model="gpt-4o",

    # Chunking
    chunk=ChunkConfig(
        target_tokens=500,
        overlap_chars=200,
    ),

    # Embedding
    embedding=EmbeddingConfig(
        model="text-embedding-3-small",
        embed_fn=None,             # custom callable overrides OpenAI
    ),

    # Domain
    domain=DomainConfig(),         # or load_preset("legal")

    # Feature flags
    enrich_chunks=True,
    enable_ocr=True,

    # Validation thresholds
    structuring_coverage_threshold=0.95,
    chunking_coverage_threshold=0.98,
    end_to_end_coverage_threshold=0.93,

    # Progress callback
    on_progress=lambda stage, data: print(f"{stage}: {data}"),
)
```

## API reference

### Pipeline

| Function | Description |
|----------|-------------|
| `process_document(path, config?, format?, embed?)` | Full pipeline from file |
| `process_text(text, config?, filename?, embed?)` | Full pipeline from text |
| `extract(path, format?)` | Extract text only |

### Models

| Model | Description |
|-------|-------------|
| `ProcessingResult` | Complete pipeline output (document + chunks + validation) |
| `Document` | Metadata + sections + transcript turns + full text |
| `DocumentChunk` | Text chunk with enrichment fields and optional embedding |
| `ValidationReport` | Coverage metrics and warnings |
| `Section` | Hierarchical document section |

### Storage

| Method | Description |
|--------|-------------|
| `Store(path?)` | Create/open SQLite store |
| `store.save(result)` | Persist a ProcessingResult, returns document_id |
| `store.search(embedding, top_k?)` | Cosine similarity search |
| `store.get_document(id)` | Retrieve document metadata |
| `store.get_chunks(id)` | Retrieve chunks for a document |
| `store.list_documents(type?, limit?)` | List stored documents |
| `store.delete_document(id)` | Delete document and chunks |
| `store.stats()` | Aggregate store statistics |

### Utilities

| Function | Description |
|----------|-------------|
| `compute_coverage(original, derived)` | Word-level coverage metric (0-1) |
| `find_missing_segments(original, derived)` | Find gaps in coverage |
| `safe_parse(json_str)` | Parse JSON with truncation repair |
| `load_preset(name)` | Load a domain preset ("generic", "legal") |
| `register_extractor(extractor)` | Register a custom file extractor |

## License

MIT
