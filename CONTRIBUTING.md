# Contributing to turbovec

Thank you for your interest in contributing! This guide covers everything you need to know to get started.

## Getting Started

### Prerequisites

- **Rust** 1.70+ (`rustup update`)
- **OpenBLAS** development libraries
  - macOS: included with the Accelerate framework (automatic)
  - Linux: `sudo apt-get install libopenblas-dev`
- **Python** 3.9+ with `pip`
- **maturin** for building Python bindings: `pip install maturin`

### Build and Test

**Rust tests:**
```bash
cargo test --all
```

**Python tests:**
```bash
cd turbovec-python
pip install -e .
pytest
```

**Full build (Rust + Python):**
```bash
cargo build --release
cd turbovec-python && maturin develop && pytest
```

### Benchmarking

If your PR changes performance, run the existing benchmarks and report the delta in your PR description:

```bash
cargo bench
```

Significant regressions (>5%) will require discussion before merge.

## Branch Strategy

```
upstream/main  ← your feature branch  ← PR target
     ↑
   your fork
```

1. Fork the repository.
2. Create a feature branch from `upstream/main`.
3. Make your changes.
4. Open a PR targeting `upstream/main`.

## Versioning

- **Rust crate**: `Cargo.toml` version — bump on any Rust-facing change.
- **Python package**: `turbovec-python/pyproject.toml` version — bump on any Python-facing change.

If a change affects both, bump both versions.

## What to Work On

Check the [open issues](https://github.com/RyanCodrai/turbovec/issues) for good starting points. The [PR backlog](./04_PR_BACKLOG.md) also lists triaged candidates.

**Before starting large PRs**, open an issue to discuss the approach — this saves time and avoids duplicated work.

## Code Style

- Rust: follow `cargo fmt` defaults (`cargo fmt` before committing)
- Python: follow `black` formatting (`pip install black && black .`)
- Add tests for any new behavior or bug fix

## Reporting Issues

When opening an issue, include:
- `turbovec` version (`pip show turbovec` or `cargo show turbovec`)
- Rust version (`rustc --version`)
- OS and architecture
- Minimal reproduction case
