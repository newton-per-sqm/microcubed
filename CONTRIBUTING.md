# Contributing

## Development environment

Install Python 3.12 or newer, a Rust toolchain with rustfmt and Clippy, and uv.
From the repository root:

```bash
uv sync --locked --extra dev --extra docs --extra examples
uv run --locked --extra dev pre-commit install
```

Maturin builds the extension during installation. After changing Rust code,
rebuild it before running Python tests:

```bash
uv sync --locked --extra dev --extra docs --extra examples --reinstall-package microcubed
```

## Checks

```bash
uv run --locked --extra dev pre-commit run --all-files
cargo test --locked
cargo clippy --locked --all-targets -- -D warnings
uv run --locked --extra dev pytest
uv run --locked --extra dev mypy
uv run --locked --extra examples python tools/notebooks.py --check --execute
uv run --locked --extra docs sphinx-build -W --keep-going -b html docs docs/_build/html
uv build
```

GitHub Actions tests Python 3.12–3.14 on Linux, macOS, and Windows, checks
Rust and Python, builds distributions, and compiles the example notebooks to
HTML. It also rebuilds a wheel from the source distribution and tests it outside
the checkout. Build artifacts are available from the workflow run.

## Examples and documentation

Write documentation in Markdown using MyST fenced directives for Sphinx features.
The API reference also uses Markdown with `automodule` directives; no generated
reStructuredText pages are required. The Furo theme supports light and dark modes. MyST-NB executes the canonical
notebooks from `examples/` during documentation builds, including their outputs
and plots. Keep edits in `examples/`; `docs/_notebooks/` contains generated copies.
Plain Markdown code fences are displayed without execution. For executable
Markdown pages, use MyST-NB notebook front matter and `{code-cell} python` fences.

Keep examples small, deterministic, and independent of downloaded data or
external simulation programs. Use Jupyter notebooks in `examples/`, clear their
outputs before committing, and include assertions for meaningful numerical
properties. Notebook cells tagged `readme` are the source of README snippets:

```bash
uv run --locked --extra examples python tools/notebooks.py
uv run --locked --extra examples python tools/notebooks.py --check --execute
```

The second command executes every notebook with a fresh kernel and writes
executed notebooks and HTML to `build/examples/`. Exceptions fail the command.
The README synchronization check also runs in pre-commit and CI.

## Changes and releases

Keep changes focused and add regression tests for numerical or API changes.
Explain the problem, resulting behavior, and validation in pull requests.
Update `uv.lock` with `uv lock` when Python dependencies change, and retain
`Cargo.lock` for reproducible Rust builds.

The old `ThreadedArrangement` process-pool helper has been removed. Use
`get_backend("rust").Arrangement` for parallel evaluation. Research-specific
OVF/CSV data are no longer part of the examples. The optional Ubermag/OOMMF
comparison lives in `examples/optional/`; see `docs/examples.md` for execution
instructions. Core examples do not require an external solver.

Before a release, update the version in `Cargo.toml`, refresh both lockfiles,
and review `CHANGELOG.md`. The workflow validates distribution artifacts;
package publication is a separate maintainer step.
