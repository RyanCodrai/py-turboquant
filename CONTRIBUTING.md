# Contributing

Thanks for your interest in turbovec. This file covers the conventions a contributor needs to know before opening a PR. For build and benchmark details, see the [README](README.md).

## Workflow

1. **Open an issue** describing the change and your proposed approach.
2. **Discuss** — leave space for a 👍 or design back-and-forth before writing code. The issue is where the design conversation lives.
3. **Open a PR** referencing the issue with `Closes #N`.

This applies to features, refactors, behaviour changes, and anything touching public API, on-disk format, or recall. The narrow exceptions that can skip the issue step:

- Typo / wording fixes
- One-line obvious bug fixes
- Documentation-only PRs

Everything else — including "I think this is small" — wants an issue first. The cost of writing one is low; the cost of building something that doesn't fit the project is high.

## Development setup

- **Rust** 1.70 or later
- **Python** 3.9 or later
- **maturin** for the Python extension (`pip install maturin`)
- **Linux only:** `libopenblas-dev` and `pkg-config` (macOS uses Accelerate, no extra deps; Windows wheels are produced in CI — see `.github/workflows/release-pypi.yml`)

## Building and testing

Rust core:

```bash
cargo test -p turbovec --release
```

Python extension (editable install via maturin) and tests:

```bash
cd turbovec-python
maturin develop --release
pytest tests/
```

The Python suite includes four framework integrations (LangChain, LlamaIndex, Haystack, Agno). They're skipped via `pytest.importorskip` unless the corresponding extras are installed. To run the full suite:

```bash
pip install -e ".[langchain,llama-index,haystack,agno]"
pytest tests/
```

## Benchmarks

See the [Running benchmarks](README.md#running-benchmarks) section of the README. Speed benchmarks for the README's headline figures are run on a specific x86 host so the numbers stay comparable across releases; ARM and other architectures are useful for spot-checking but aren't the source of the published numbers.

## Commit and PR conventions

- **One logical change per PR.** Refactors get their own PR, separate from feature work.
- **Commit messages:** short imperative title, body explaining *why* (the *what* is in the diff). Multi-line bodies should preserve formatting — use a HEREDOC if writing from the shell.
- **PRs reference their issue** with `Closes #N` and include a test plan.
- **`Co-Authored-By:` trailers** are fine on commits where Claude or another tool collaborated — leave them in place.

## Integration contributions

If you're adding or modifying an integration (LangChain, LlamaIndex, Haystack, Agno, or a new framework), structurally compare against the canonical in-tree reference store (`InMemoryVectorStore`, `SimpleVectorStore`, `InMemoryDocumentStore` etc.) for that framework. The wrappers should match the reference's surface and idioms — that's the bar for a drop-in replacement.

## Release process

Releases are tag-triggered, not automatic on merge:

- `v0.X.Y` tag → [`.github/workflows/release-crates.yml`](.github/workflows/release-crates.yml) publishes the Rust crate to crates.io.
- `py-v0.X.Y` tag → [`.github/workflows/release-pypi.yml`](.github/workflows/release-pypi.yml) builds wheels for Linux x86_64/aarch64, macOS aarch64, Windows x64, plus an sdist, and publishes to PyPI.

The Rust crate and Python package version independently. See [CHANGELOG.md](CHANGELOG.md) for the cadence so far.

## Code of conduct

This project doesn't have a formal CoC. The expectation is professional collaboration: clear, kind, and on-topic.
