---
file_format: mystnb
kernelspec:
  name: python3
  display_name: Python 3
  language: python
---

# Installation and getting started

## Requirements

Microcubed requires Python 3.12 or newer. Its runtime dependencies are NumPy,
SciPy, and Matplotlib.

## Installation

From a local checkout:

```bash
cd microcubed
python -m pip install .
```

For development and testing:

```bash
python -m pip install -e ".[dev]"
pytest
```

From a checkout, the locked environment can be used reproducibly:

```bash
uv run --frozen --extra dev pytest
```

Binary distributions include the integrated Rust backend. See {doc}`backends`
for selection and source-build details:

```bash
python -m pip install microcubed
```

## A first magnet

```{code-cell} python
import numpy as np
from microcubed import Magnet

cube = Magnet(
    size=[100, 100, 100],
    center=[0, 0, 0],
    magnetization=[0, 0, 8e5],
)

points = np.array(
    [
        [0, 0, -75],
        [50, 0, -100],
        [100, 30, -150],
    ]
).T

B = cube.Bfield(points)
print("Field shape:", B.shape)
print("Field (T):\n", np.round(B, 6))
```

Each column index identifies one point; the first axis contains `Bx`, `By`,
and `Bz`.

## Unit convention

Microcubed does not implement a unit system. Three rules apply:

1. `size`, `center`, and `points` must use the same length unit.
2. `magnetization` is specified in A/m.
3. `Bfield` returns tesla; `dBfield` returns tesla per chosen length unit.

For geometry expressed in nanometres, the gradient is therefore in T/nm. For
geometry expressed in metres, it is in T/m. `Hfield` and `dHfield` similarly
return A/m and A/m per length unit.

```{warning}
Do not mix length units. A magnet defined in nanometres and points supplied in
metres produce formally computable but physically incorrect results.
```

## Point input

The following inputs are reshaped to `(3, N)`:

```{code-cell} python
for points in ([0, 0, -150], [[0, 10], [0, 0], [-150, -150]]):
    print(cube.Bfield(points))
```

A single `Magnet` masks points inside and on the boundary of the cuboid with
`NaN`. For an `Arrangement`, no automatic interior check is performed for
performance reasons; callers must exclude those points themselves.

## Next steps

- {doc}`theory`: model and assumptions
- {doc}`calculations`: fields and gradients
- {doc}`visualization`: 1D, 2D, and 3D visualisation
