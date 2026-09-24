# Examples

The core Jupyter notebooks are self-contained: no downloaded datasets or external
micromagnetic solvers are needed. GitHub Actions executes their cells and exports
HTML and executed notebooks as the `examples` workflow artifact. The getting-started,
calculation, and visualization guides also execute their Markdown code cells.

The pages below include executed code, numerical results, and plots. Sphinx
runs the canonical notebooks from `examples/` in fresh kernels; execution errors
fail the documentation build.

```{toctree}
:maxdepth: 1

_notebooks/single_cuboid
_notebooks/arrangement
_notebooks/shape
_notebooks/backends
_notebooks/compare_ubermag
```

From a checkout with Python and Rust installed:

```bash
uv sync --locked --extra examples
uv run --locked --extra examples python tools/notebooks.py --check --execute
```

Open `build/examples/*.html` to view executed results. The tagged notebook cells
also generate the examples in the repository README; run
`python tools/notebooks.py` after editing those cells.

The single-cuboid example includes component line profiles, analytical gradient
profiles, a plane map, and a 3D vector field with source geometry. The arrangement
example adds individual cuboid contributions to show how their fields superpose,
alongside its plane map and 3D view.

## Optional Ubermag/OOMMF comparison

The comparison notebook uses Ubermag's `discretisedfield`, `micromagneticmodel`,
and `oommfc` packages. Install only when needed:

```bash
uv sync --locked --extra examples --extra comparison --extra docs
# For a local Tcl installation; alternatively configure oommfc's runner.
export OOMMFTCL=/absolute/path/to/oommf.tcl
uv run --locked --extra examples --extra comparison python tools/notebooks.py --check --execute --include-optional
MICROCUBED_RUN_OOMMF=1 uv run --locked --extra docs --extra comparison sphinx-build -E -W -b html docs docs/_build/html
```

Ordinary builds render the comparison source without running OOMMF. The
optional comparison workflow executes it and publishes docs with its results.
No Ubermag packages are required by the core installation or standard CI jobs.
