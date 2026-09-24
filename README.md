[![CI](https://github.com/newton-per-sqm/microcubed/actions/workflows/ci.yml/badge.svg)](https://github.com/newton-per-sqm/microcubed/actions/workflows/ci.yml)

# Microcubed

Calculate 3D magnetic stray fields and analytical gradients for uniformly
magnetized, axis-aligned cuboids and arrangements. Microcubed provides NumPy
and compiled, parallel Rust backends behind the same Python API, plus polygon
decomposition and Matplotlib plots.

The field equations follow Ravaud and Lemarquand,
[Magnetic Field Produced by a Parallelepipedic Magnet of Various and Uniform Polarization](https://doi.org/10.2528/PIER09091704).
Magnetization is an input; Microcubed does not solve magnetic equilibrium or dynamics.

## Installation

Python **3.12 or newer** is required. Building from source also requires a Rust
toolchain (Cargo and rustc) and a platform linker/C compiler.

```bash
git clone https://github.com/newton-per-sqm/microcubed.git
cd microcubed
python -m pip install .
```

The Maturin build includes the Rust extension. At runtime `auto` selects Rust
when available and otherwise falls back to NumPy. Select a backend explicitly
with `microcubed.get_backend("numpy")` or `microcubed.get_backend("rust")`.

Use one consistent length unit for size, position, and observation points.
Magnetization is in A/m, `Bfield` returns T, and `dBfield` returns T per length
unit. Points are columns in a `(3, N)` array. Evaluate outside the magnets;
the analytical exterior-field model does not describe their internal field.

## Examples

These snippets are generated from the tagged cells in the Jupyter notebooks.
GitHub Actions builds the extension, executes every core notebook, checks numerical
assertions, and uploads executed notebooks and HTML under the `examples` artifact.

```bash
uv sync --locked --extra examples
uv run --locked --extra examples python tools/notebooks.py --check --execute
```

Open the `.ipynb` files in a Jupyter-compatible editor using `.venv` as the
Python environment. Generated HTML is written to `build/examples/`.

<!-- examples:start -->

### Superposition and field maps

Build a small array by translating one cuboid. The arrangement field is the sum of its members. Sample a plane below the magnets, safely outside all material.

[Open notebook](examples/arrangement.ipynb)

```python
import numpy as np

from microcubed import Arrangement, Magnet

cube = Magnet([80, 80, 40], [0, 0, 0], [0, 0, 8e5])
magnets = [cube.moved_to([x, 0, 0]) for x in (-150, 0, 150)]
array = Arrangement(magnets)
points = np.array([[0, 0, -100], [100, 25, -100]]).T
field = array.Bfield(points)
```

```python
import matplotlib.pyplot as plt

fig, ax = array.plot_2d(x=(-300, 300, 61), y=(-200, 200, 41), z=-100, component="z")
for index, cuboid in enumerate(array):
    ax.plot(
        *cuboid.chull_points("xy"),
        color="black",
        linewidth=1.3,
        label="Projected cuboids" if index == 0 else None,
    )
ax.plot(*array.chull_points("xy"), "w--", linewidth=2, label="Projected convex hull")
ax.set(xlabel="x (nm)", ylabel="y (nm)", title="Bz (T) at z = -100 nm", aspect="equal")
ax.legend(loc="upper right", fontsize=8, facecolor="#555555", labelcolor="white", framealpha=0.95)
fig.tight_layout()
plt.show()
```

### Compare NumPy and Rust

Choose backend namespaces explicitly without changing global state. This notebook requires the compiled Rust extension and checks both fields and gradients at exterior points.

[Open notebook](examples/backends.ipynb)

```python
import numpy as np

from microcubed import get_backend

points = np.array([[0, 0, -150], [80, 30, -120], [-90, 50, 160]]).T
results = {}
for name in ("numpy", "rust"):
    backend = get_backend(name)
    magnet = backend.Magnet([100, 80, 40], [0, 0, 0], [2e5, 1e5, 8e5])
    results[name] = (magnet.Bfield(points), magnet.dBfield(points))

for reference, compiled in zip(results["numpy"], results["rust"]):
    np.testing.assert_allclose(compiled, reference, rtol=1e-9, atol=1e-13)
print("Fields and gradients agree.")
```

### Resolving a polygon boundary

A concave polygon with slanted edges makes rasterization error visible. All lengths
are in nm. `delta` limits the raster-cell size, not the size of the final cuboids:
merging adjacent occupied cells into larger cuboids preserves the rasterized geometry
exactly. Refining `delta` improves the staircase approximation along oblique edges.
The previous axis-aligned L-shape could be represented exactly by two large cuboids;
a small cuboid count alone does not imply a coarse approximation.

[Open notebook](examples/shape.ipynb)

```python
import numpy as np

from microcubed import cuboidize

polygon = np.array([(0, 0), (120, 15), (95, 65), (55, 45), (35, 115), (-15, 80)])
thickness = 20
magnetization = [0, 0, 8e5]
delta = 0.5
shape = cuboidize(polygon, t=thickness, delta=delta, mag=magnetization)
field = shape.Bfield([50, 50, -60])
print(f"Raster spacing: {delta} nm; merged cuboids: {len(shape)}")
print("Field at (50, 50, -60) nm (T):", field.ravel())
```

```python
import matplotlib.pyplot as plt

fig, ax = shape.plot_2d(x=(-40, 145, 201), y=(-25, 140, 201), z=-60, component="z")
for index, cuboid in enumerate(shape):
    ax.plot(
        *cuboid.chull_points("xy"),
        color="black",
        linewidth=0.3,
        alpha=0.35,
        label="Projected cuboids" if index == 0 else None,
    )
ax.plot(*np.vstack([polygon, polygon[0]]).T, color="#ff9500", linewidth=1.5, label="Input polygon")
for index, boundary in enumerate(shape.union_boundary("xy")):
    ax.plot(*boundary, color="#00ffff", linewidth=1.5, label="Union boundary" if index == 0 else None)
ax.plot(*shape.chull_points("xy"), "w--", linewidth=1.7, label="Projected convex hull")
ax.set(xlabel="x (nm)", ylabel="y (nm)", title="Bz (T) at z = -60 nm; delta = 0.5 nm", aspect="equal")
ax.legend(loc="upper right", fontsize=7, facecolor="#555555", labelcolor="white", framealpha=0.95)
fig.tight_layout()
plt.show()
```

### Single cuboid

Calculate the field and its analytical gradient outside a uniformly magnetized cube. All lengths here are in nm, magnetization is in A/m, fields are in T, and gradients are in T/nm.

[Open notebook](examples/single_cuboid.ipynb)

```python
import numpy as np

from microcubed import Magnet

cube = Magnet(size=[100, 100, 100], center=[0, 0, 0], magnetization=[0, 0, 8e5])
points = np.array([[0, 0, -150], [80, 0, -150]]).T
field = cube.Bfield(points)  # (3, N): Bx, By, Bz
gradient = cube.dBfield(points)  # (3, 3, N): derivative axis, field axis, point
print(field)
```

```python
import matplotlib.pyplot as plt

fig, ax = cube.plot_2d(x=(-250, 250, 61), y=(-250, 250, 61), z=-150, component="z")
ax.plot(*cube.chull_points("xy"), "w--", linewidth=2, label="Projected convex hull")
ax.set(xlabel="x (nm)", ylabel="y (nm)", title="Bz (T) at z = -150 nm", aspect="equal")
ax.legend(loc="upper right", fontsize=8, facecolor="#555555", labelcolor="white", framealpha=0.95)
fig.tight_layout()
plt.show()
```

<!-- examples:end -->

## Optional solver comparison

The [Ubermag/OOMMF comparison](examples/optional/compare_ubermag.ipynb) compares
fields and gradients at identical exterior points. Install the `comparison`
extra and an OOMMF runner to execute it; neither is needed for Microcubed itself.
See the [example guide](docs/examples.md) for setup and documentation builds
with comparison results.

## Documentation and development

See the [user guide](docs/index.md), [backend guide](docs/backends.md), and
[contributing guide](CONTRIBUTING.md) for development and validation commands.
Python code lives in `src/microcubed/`; Rust kernels live in `src/rust/`.

Microcubed is distributed under the [MIT license](LICENSE.txt).
