# Microcubed

Microcubed computes the magnetostatic stray field of uniformly magnetised,
axis-aligned cuboids and arrangements of such cuboids. The field expressions
are analytical; no spatial volume grid or micromagnetic solver is required.

The library is particularly useful for:

- permanent magnets with cuboid geometry,
- segmented approximations of more complex bodies,
- fast parameter sweeps outside magnets,
- field gradients for force, sensor, and magnetometry applications,
- reproducible comparisons with finite-difference programs such as OOMMF.

```{important}
Microcubed does not solve a magnetic equilibrium problem. Magnetisation,
position, and geometry are inputs. Domain formation, hysteresis, exchange,
anisotropy, and magnetisation dynamics are outside the model.
```

## Documentation

```{toctree}
:maxdepth: 2
:caption: User guide

getting-started
theory
magnets-arrangements
calculations
visualization
examples
backends
numerics-performance
references
```

```{toctree}
:maxdepth: 2
:caption: Project

API reference <api-reference>
Changelog <changelog>
Contributing <contributing>
Authors <authors>
License <license>
```

## Quick example

```python
from microcubed import Magnet

magnet = Magnet(
    size=[500, 300, 50],  # nm
    center=[0, 0, 0],  # nm
    magnetization=[0, 0, 1.0e6],  # A/m
)

B = magnet.Bfield([0, 0, -150])
dB = magnet.dBfield([0, 0, -150])

print(B.shape)  # (3, 1), T
print(dB.shape)  # (3, 3, 1), T/nm
```

## Indices

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
