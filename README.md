# distillcore

Universal document processing: extract, chunk, enrich, embed, validate.

distillcore takes any document (PDF, DOCX, HTML, text, markdown) and runs it through an intelligent 7-stage pipeline — extracting text, classifying the document, breaking it into structured sections, chunking for RAG, enriching chunks with LLM-generated metadata, generating embeddings, and validating coverage at every stage.

Works as a **Python library** or a **standalone FastMCP server**. Supports sync and async pipelines, batch processing, optional SQLite persistence with cosine search, and 4 embedding providers. Domain-neutral by default, with pluggable presets for specialized domains (legal built-in).

**New in v0.7:** `openai` is now optional — standalone chunking, extraction, validation, and storage work without it. Install `distillcore[openai]` for LLM features.

## Install

```bash
# Core (chunking, extraction, validation, storage — no API key needed)
pip install distillcore

# With LLM features (classification, structuring, enrichment, OpenAI embeddings)
pip install distillcore[openai]

# With PDF support
pip install distillcore[pdf]

# With all document formats + OpenAI
pip install distillcore[all]

# With MCP server
pip install distillcore[mcp]

# Fully offline embeddings (sentence-transformers)
pip install distillcore[local]

# Everything
pip install distillcore[all,mcp,local]
```

## Quickstart

### Chunk text (no API key needed)

```python
from distillcore import chunk, estimate_tokens

chunks = chunk("Your document text here...", strategy="paragraph", target_tokens=500)

for i, c in enumerate(chunks):
    print(f"[{i}] {estimate_tokens(c)} tokens: {c[:80]}...")
```

Four strategies: `"paragraph"` (default), `"sentence"`, `"fixed"`, `"llm"` (requires API key).

```python
# Sentence boundaries
chunks = chunk(text, strategy="sentence", target_tokens=300)

# Fixed sliding window with overlap
chunks = chunk(text, strategy="fixed", target_tokens=500, overlap_tokens=50)

# LLM-driven semantic chunking
chunks = chunk(text, strategy="llm", api_key="sk-...", target_tokens=500)

# Async version
from distillcore import achunk
chunks = await achunk(text, strategy="paragraph")
```

### Process a file (full pipeline)

```python
from distillcore import process_document

result = process_document("report.pdf")
print(f"Type: {result.document.metadata.document_type}")
print(f"Chunks: {len(result.chunks)}")
print(f"Coverage: {result.validation.end_to_end_coverage:.1%}")
```

### Process raw text

```python
from distillcore import process_text, DistillConfig

result = process_text(
    "Introduction\n\nThis report covers Q4 results...\n\nConclusion\n\nWe recommend...",
    config=DistillConfig(openai_api_key="sk-..."),
)

for chunk in result.chunks:
    print(f"[{chunk.chunk_index}] {chunk.topic} ({chunk.relevance})")
```

### Async pipeline

```python
from distillcore import process_document_async

result = await process_document_async("report.pdf")
```

### Batch processing

```python
from distillcore import process_batch_sync

results = process_batch_sync(
    ["doc1.pdf", "doc2.docx", "doc3.html"],
    max_concurrent=5,
)

for r in results:
    print(f"{r.document.metadata.source_filename}: {len(r.chunks)} chunks")
```

Or async with callbacks:

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

# OpenAI (requires distillcore[openai])
from distillcore.embedding import openai_embedder
config = DistillConfig(embedding=EmbeddingConfig(
    embed_fn=openai_embedder("text-embedding-3-large"),
))

# Ollama (local, no API key, no pip deps)
from distillcore.embedding import ollama_embedder
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
from distillcore import Store

store = Store()  # ~/.distillcore/store.db
doc_id = store.save(result)

# Cosine similarity search
results = store.search(query_embedding=[0.1, 0.2, ...], top_k=5)
```

Tenant isolation:

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

### Without LLM (zero API calls)

```python
from distillcore import process_text, DistillConfig, DomainConfig

result = process_text(
    "Your text here...",
    config=DistillConfig(domain=DomainConfig(), enrich_chunks=False),
    embed=False,
)
# Chunking and validation still work — no API key needed
```

## MCP Server

Run as a standalone FastMCP server:

```bash
pip install distillcore[mcp,openai]
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

## Pipeline stages

```
extract -> classify -> structure -> chunk -> enrich -> embed -> validate
```

1. **Extract** — pull text from PDF (with OCR fallback), DOCX, HTML, TXT, or MD files
2. **Classify** — LLM identifies document type, title, and domain-specific metadata
3. **Structure** — LLM breaks the document into hierarchical sections (boundary-based with page ranges)
4. **Chunk** — section-aware splitting with 4 strategies: paragraph, sentence, fixed, or LLM-driven
5. **Enrich** — LLM tags each chunk with topic, key concepts, and relevance
6. **Embed** — generate vector embeddings (OpenAI, Ollama, local, or Cohere)
7. **Validate** — coverage checks at each stage (structuring 95%, chunking 98%, end-to-end 93%)

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
    # LLM (requires distillcore[openai])
    openai_api_key="sk-...",       # or set OPENAI_API_KEY env var
    openai_model="gpt-4o",

    # Chunking
    chunk=ChunkConfig(
        target_tokens=500,
        overlap_chars=200,
        max_tokens=1000,
        min_tokens=50,             # merge small chunks
        strategy="auto",           # "auto", "paragraph", "sentence", "fixed", "llm"
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

### Standalone chunking

| Function | Description |
|----------|-------------|
| `chunk(text, strategy?, target_tokens?, ...)` | Split text into chunks |
| `achunk(text, ...)` | Async version of chunk |
| `estimate_tokens(text, tokenizer?)` | Estimate token count |

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
| `openai_embedder(model, api_key)` | `distillcore[openai]` | yes |
| `ollama_embedder(model, base_url)` | included | no |
| `local_embedder(model, device)` | `distillcore[local]` | no |
| `cohere_embedder(model, api_key, input_type)` | `distillcore[cohere]` | yes |

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

- **Path traversal protection** — `allowed_dirs` config restricts file access to specified directories
- **Prompt injection hardening** — untrusted document content is isolated with `--- BEGIN/END UNTRUSTED ---` sentinels, with explicit "ignore instructions" directives
- **Tenant isolation** — optional `tenant_id` scoping on all Store operations
- **Config validation** — `config.validate()` warns early if API key is missing
- **Graceful degradation** — no stage failure crashes the pipeline

## License

MIT
