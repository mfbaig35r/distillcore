# distillcore

Universal document processing: extract, chunk, enrich, embed, validate.

distillcore takes any document (PDF, DOCX, HTML, text, markdown) and runs it through an intelligent pipeline — extracting text, classifying the document, breaking it into structured sections, chunking for RAG, enriching chunks with LLM-generated metadata, generating embeddings, and validating coverage at every stage.

Works as a **Python library** or a **standalone FastMCP server**. Supports sync and async pipelines, batch processing, optional SQLite persistence with cosine search, and 4 embedding providers. Domain-neutral by default, with pluggable presets for specialized domains (legal built-in).

## Install

```bash
# Core (text extraction + OpenAI for LLM/embeddings)
pip install distillcore

# With PDF support
pip install distillcore[pdf]

# With all document formats
pip install distillcore[all]  # pdf, docx, html

# With MCP server
pip install distillcore[mcp]

# Fully offline embeddings (sentence-transformers)
pip install distillcore[local]

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

### Async pipeline

```python
import asyncio
from distillcore import process_document_async, process_text_async

result = asyncio.run(process_document_async("report.pdf"))
```

### Batch processing

```python
from distillcore import process_batch_sync

# Process 50 files, 5 at a time
results = process_batch_sync(
    ["doc1.pdf", "doc2.docx", "doc3.html", ...],
    max_concurrent=5,
)

for r in results:
    print(f"{r.document.metadata.source_filename}: {len(r.chunks)} chunks")
```

Or async:

```python
from distillcore import process_batch

results = await process_batch(
    paths,
    max_concurrent=5,
    on_result=lambda src, res: print(f"Done: {src}"),
)
```

Failed files don't crash the batch — each gets a `ProcessingResult` with `passed=False`.

### Embedding providers

```python
from distillcore import DistillConfig, EmbeddingConfig
from distillcore.embedding import openai_embedder, ollama_embedder

# OpenAI (default)
config = DistillConfig(embedding=EmbeddingConfig(
    embed_fn=openai_embedder("text-embedding-3-large"),
))

# Ollama (local, no API key, no pip deps)
config = DistillConfig(embedding=EmbeddingConfig(
    embed_fn=ollama_embedder("nomic-embed-text"),
))

# Sentence-transformers (fully offline) — pip install distillcore[local]
from distillcore.embedding import local_embedder
config = DistillConfig(embedding=EmbeddingConfig(
    embed_fn=local_embedder("all-MiniLM-L6-v2"),
))

# Cohere — pip install distillcore[cohere]
from distillcore.embedding import cohere_embedder
config = DistillConfig(embedding=EmbeddingConfig(
    embed_fn=cohere_embedder("embed-english-v3.0"),
))
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

Tenant isolation is supported — pass `tenant_id` to scope access:

```python
store.save(result, tenant_id="user_123")
store.search(query_embedding, tenant_id="user_123")  # only sees this tenant's docs
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

### Skip LLM calls entirely

```python
from distillcore import process_text, DistillConfig, EmbeddingConfig

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

## MCP Server

Run as a standalone FastMCP server:

```bash
pip install distillcore[mcp]
distillcore
```

### Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI API key for LLM + embeddings | |
| `DISTILLCORE_STORE` | SQLite store path | `~/.distillcore/store.db` |
| `DISTILLCORE_TENANT_ID` | Tenant isolation for multi-user | |
| `DISTILLCORE_ALLOWED_DIRS` | Colon-separated allowed file paths | unrestricted |
| `DISTILLCORE_EMBEDDING_MODEL` | Embedding model for search | `text-embedding-3-small` |

### Tools

| Tool | Description |
|------|-------------|
| `distill_file` | Process a document file through the full pipeline |
| `distill_text` | Process raw text (skips extraction) |
| `distill_batch` | Process multiple files concurrently |
| `distill_chunks_only` | Chunk text without LLM calls |
| `distill_validate` | Validate coverage between text and chunks |
| `distill_search` | Semantic search across stored documents |
| `distill_list_documents` | List stored documents |
| `distill_get_document` | Get a document and its chunks |

Use `store=True` on `distill_file` / `distill_text` / `distill_batch` to persist results for later search.

## Pipeline stages

```
extract -> classify -> structure -> chunk -> enrich -> embed -> validate
```

1. **Extract** — pull text from PDF (with OCR fallback), DOCX, HTML, TXT, or MD files
2. **Classify** — LLM identifies document type, title, and domain-specific metadata
3. **Structure** — LLM breaks the document into hierarchical sections
4. **Chunk** — section-aware splitting with paragraph-boundary overlap
5. **Enrich** — LLM tags each chunk with topic, key concepts, and relevance
6. **Embed** — generate vector embeddings (OpenAI, Ollama, local, or Cohere)
7. **Validate** — coverage checks at each stage (structuring, chunking, end-to-end)

Every LLM stage degrades gracefully — if the API key is missing or a call fails, the pipeline continues with fallback values.

## Supported formats

| Format | Extension | Extra |
|--------|-----------|-------|
| Plain text | `.txt`, `.text` | included |
| Markdown | `.md`, `.markdown` | included |
| PDF | `.pdf` | `distillcore[pdf]` |
| Word | `.docx` | `distillcore[docx]` |
| HTML | `.html`, `.htm` | `distillcore[html]` |

Custom extractors can be registered for any format:

```python
from distillcore import register_extractor

class MyExtractor:
    formats = ["xml"]
    def extract(self, source, config=None):
        ...

register_extractor(MyExtractor())
```

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

    # Security
    allowed_dirs=None,             # restrict file access (list of paths)

    # Validation thresholds
    structuring_coverage_threshold=0.95,
    chunking_coverage_threshold=0.98,
    end_to_end_coverage_threshold=0.93,

    # Progress callback
    on_progress=lambda stage, data: print(f"{stage}: {data}"),
)
```

## API reference

### Pipeline (sync)

| Function | Description |
|----------|-------------|
| `process_document(path, config?, format?, embed?)` | Full pipeline from file |
| `process_text(text, config?, filename?, embed?)` | Full pipeline from text |
| `extract(path, format?)` | Extract text only |

### Pipeline (async + batch)

| Function | Description |
|----------|-------------|
| `process_document_async(path, config?, format?, embed?)` | Async full pipeline from file |
| `process_text_async(text, config?, filename?, embed?)` | Async full pipeline from text |
| `process_batch(sources, config?, max_concurrent?, on_result?)` | Concurrent batch processing |
| `process_batch_sync(sources, **kwargs)` | Sync wrapper for batch |

### Embedding providers

| Factory | Deps | API key? |
|---------|------|----------|
| `openai_embedder(model, api_key)` | included | yes |
| `ollama_embedder(model, base_url)` | included | no |
| `local_embedder(model, device)` | `distillcore[local]` | no |
| `cohere_embedder(model, api_key, input_type)` | `distillcore[cohere]` | yes |

### Models

| Model | Description |
|-------|-------------|
| `ProcessingResult` | Complete pipeline output (document + chunks + validation) |
| `BatchResult` | Batch summary (total, succeeded, failed, results) |
| `Document` | Metadata + sections + transcript turns + full text |
| `DocumentChunk` | Text chunk with enrichment fields and optional embedding |
| `ValidationReport` | Coverage metrics and warnings |
| `Section` | Hierarchical document section |

### Storage

| Method | Description |
|--------|-------------|
| `Store(path?)` | Create/open SQLite store |
| `store.save(result, tenant_id?)` | Persist a ProcessingResult, returns document_id |
| `store.search(embedding, top_k?, tenant_id?)` | Cosine similarity search |
| `store.get_document(id, tenant_id?)` | Retrieve document metadata |
| `store.get_chunks(id, tenant_id?)` | Retrieve chunks for a document |
| `store.list_documents(type?, limit?, tenant_id?)` | List stored documents |
| `store.delete_document(id, tenant_id?)` | Delete document and chunks |
| `store.stats()` | Aggregate store statistics |

### Utilities

| Function | Description |
|----------|-------------|
| `compute_coverage(original, derived)` | Word-level coverage metric (0-1) |
| `find_missing_segments(original, derived)` | Find gaps in coverage |
| `safe_parse(json_str)` | Parse JSON with truncation repair |
| `load_preset(name)` | Load a domain preset ("generic", "legal") |
| `register_extractor(extractor)` | Register a custom file extractor |

## Security

distillcore includes several security features for production use:

- **Path traversal protection** — `allowed_dirs` config restricts file access to specified directories
- **Prompt injection hardening** — untrusted document content is isolated with delimiters, LLM output fields are sanitized
- **Tenant isolation** — optional `tenant_id` scoping on all Store operations
- **Config validation** — `config.validate()` warns early if API key is missing
- **Graceful degradation** — no stage failure crashes the pipeline

## License

MIT
