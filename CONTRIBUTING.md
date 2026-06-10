# Contributing to distillcore

Thanks for your interest in contributing. This guide covers the dev setup and the conventions we follow.

## Prerequisites

- Python 3.11, 3.12, or 3.13
- [uv](https://docs.astral.sh/uv/) for dependency management

## Dev setup

```bash
git clone https://github.com/mfbaig35r/distillcore.git
cd distillcore
uv sync --extra dev --extra pdf --extra mcp --extra docx --extra html
```

The `dev` extra includes `openai`, so LLM-backed tests work out of the box.

## Running tests

```bash
uv run python -m pytest tests/ -v --tb=short
```

The full suite runs in ~12s and must pass on all supported Python versions in CI.

## Linting

```bash
uv run ruff check src/ tests/
```

CI runs the same command. Formatting is not enforced separately, but keep new code consistent with the surrounding style.

## Type checking

```bash
uv run mypy src/distillcore
```

mypy is configured with `disallow_untyped_defs = true`. New public functions need type annotations.

## Code style

- Ruff rules: `E`, `F`, `I` (errors, pyflakes, isort)
- Line length: 100
- Target: Python 3.11

## PR guidelines

- One concern per PR. Refactors, fixes, and new features in separate PRs.
- Tests required for new behavior. Add a regression test for bug fixes.
- **Async parity**: if you add a sync function in the pipeline, add the async counterpart (or explain in the PR why not). The shared helpers in `pipeline/_shared.py` exist to keep both paths in sync.
- Update `CHANGELOG.md` under `## [Unreleased]` with a one-line entry.
- Run `ruff check` and the test suite locally before pushing.

## Commit messages

Format: `type: description`

Types we use:
- `feat:` new functionality
- `fix:` bug fix
- `refactor:` no behavior change
- `docs:` documentation only
- `test:` tests only
- `chore:` build, deps, version bumps

Example: `fix: strip embedding arrays from MCP responses`

## Where things live

- `src/distillcore/extractors/` — format-specific extractors (PDF, DOCX, HTML, etc.)
- `src/distillcore/pipeline/` — stage implementations (classify, structure, chunk, enrich, embed)
- `src/distillcore/chunking.py` — standalone chunking API (`chunk()`, `achunk()`)
- `src/distillcore/validation/` — coverage and lossless-join checks
- `src/distillcore/presets/` — domain presets (generic, legal)
- `src/distillcore/server.py` — FastMCP tool registrations
- `tests/` — one test file per module under test

## Releasing (maintainers)

1. Bump version in `pyproject.toml`
2. Move `## [Unreleased]` entries to a dated version section in `CHANGELOG.md`
3. Commit, tag (`git tag v0.x.y`), push tag
4. The publish workflow pushes to PyPI via trusted publishers
