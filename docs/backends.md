# Calculation backends

Microcubed contains two interchangeable calculation backends:

- `numpy` is the portable pure-Python implementation;
- `rust` uses the compiled, parallel Rust extension bundled in Microcubed.

By default Microcubed selects `rust` when the compiled extension is available
and transparently falls back to `numpy` otherwise (equivalent to `auto`).

Both backends share the same geometry, field, gradient, plotting, and shape
decomposition API. There is no separate Rust package or public low-level
module to install.

## Installation

Binary Microcubed wheels include the Rust extension:

```bash
python -m pip install microcubed
```

For development, build the mixed Python/Rust project with Maturin:

```bash
python -m pip install maturin
maturin develop --release --manifest-path Cargo.toml
```

The NumPy backend remains usable directly from a source checkout even when the
Rust extension has not been compiled.

## Explicit backend namespaces

Selecting a namespace does not mutate process-wide state and is recommended
for libraries and comparisons:

```python
import microcubed

numpy_backend = microcubed.get_backend("numpy")
rust_backend = microcubed.get_backend("rust")

numpy_magnet = numpy_backend.Magnet([100, 100, 50], [0, 0, 0], [0, 0, 8e5])
rust_magnet = rust_backend.Magnet([100, 100, 50], [0, 0, 0], [0, 0, 8e5])
```

The namespace also provides shape decomposition:

```python
shape = rust_backend.cuboidize(
    [[0, 0], [100, 0], [100, 40], [40, 40], [40, 100], [0, 100]],
    t=20,
    delta=5,
    mag=[0, 0, 8e5],
)
```

`auto` is a selector rather than a backend name: it chooses `rust` when the
compiled extension is importable and otherwise returns `numpy`.

```python
backend = microcubed.get_backend("auto")
print(backend.name)  # "rust" or "numpy"
print(microcubed.available_backends())
```

## Process-wide selection

Applications using one backend consistently can switch the top-level classes
before constructing objects:

```python
import microcubed

microcubed.set_backend("rust")
magnet = microcubed.Magnet([100, 100, 50], [0, 0, 0], [0, 0, 8e5])
print(microcubed.backend_name())  # rust
```

The initial backend can also be selected before import:

```bash
MICROCUBED_BACKEND=rust python calculation.py
```

Names imported earlier with `from microcubed import Magnet` remain bound to
the class active at import time.

## Compatibility contract

For both backends:

- `Bfield` returns `(3, N)` in tesla;
- `dBfield` returns `(3, 3, N)` with derivative axis first;
- `Hfield`, `dHfield`, direct-call sampling, and plotting are identical;
- magnet geometry arrays use the `(3, 1)`, `(3, 2)`, and `(3, 8)` conventions;
- single-magnet interior and boundary points are masked with `NaN`;
- arrangements, transformations, convex hulls, generators, and `from_shape`
  share one implementation; and
- stable NumPy evaluation replaces a Rust result only at exceptional exterior
  points where direct corner expressions produce a removable non-finite form.

Random exterior-point tests require both implementations to agree within
`rtol=1e-9` and `atol=1e-13` in field units.

## Choosing a backend

The Rust backend is most useful for many evaluation points or large
arrangements, where parallel loops amortise conversion overhead. NumPy can be
faster for very small arrays. Benchmark the actual magnet count, point count,
and hardware used by the application; backend selection does not change the
physical model or numerical units.
