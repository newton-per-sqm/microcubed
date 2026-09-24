# Numerics, stability, and performance

## Vectorized corner evaluation

For each magnet, Microcubed evaluates all eight corner contributions together
using NumPy arrays. In a local benchmark with 10,000 evaluation points, this
reduced execution time relative to a Python loop over the corners as follows:

| Quantity | Corner loop | Vectorized | Speed-up |
|---|---:|---:|---:|
| `Bfield` | 7.49 ms | 4.09 ms | 1.83× |
| `dBfield` | 16.80 ms | 7.85 ms | 2.14× |

These values are not a hardware-independent performance guarantee. They record
the scale and setup of one benchmark; the CPU, NumPy version, array layout, and
point distribution all affect the result.

## Stable evaluation on extended edge lines

The individual terms in the analytical corner formulas contain expressions
such as $\log(r+z)$. On a negatively extended edge line, two terms can tend to
$-\infty$ although their difference remains finite. Microcubed evaluates such
logarithms pairwise as a stable logarithm of a quotient.

At the same exceptional positions, the analytical gradient can contain
removable `0/0` forms. Non-finite columns outside the magnet are replaced
locally by a symmetric finite difference of the stable field implementation.
Regular points continue to use the analytical derivatives.

## Result cache

`Arrangement.Bfield` and `Arrangement.dBfield` use an LRU cache holding up to
20 point arrays. A cache key includes the array shape, data type, and a SHA-256
digest of its contents.

This is useful when exactly the same points are evaluated repeatedly, for
example while redrawing a plot. For a single small calculation, hashing the
input adds some overhead.

## Summing many magnets

An arrangement uses linear superposition:

$$
\mathbf B_\mathrm{total}(\mathbf r)
=\sum_{m=1}^{N_m}\mathbf B_m(\mathbf r).
$$

Execution time therefore scales approximately with $N_mN_p$, the number of
magnets times the number of points. For very large problems, consider:

- reducing the sampling-grid resolution;
- evaluating only the required spatial region;
- using coarser magnet segments for non-cuboidal bodies;
- using the integrated Rust backend for parallel calculation;
- reusing identical point arrays to benefit from caching; and
- processing the points in chunks when memory is limited.

## Memory use and chunking

Vectorization creates temporary arrays spanning eight corners and all
evaluation points. `dBfield` requires more intermediate arrays than `Bfield`.
For millions of points, process the data in chunks:

```python
import numpy as np


def field_in_chunks(source, points, chunk_size=100_000):
    return np.hstack(
        [source.Bfield(points[:, start : start + chunk_size]) for start in range(0, points.shape[1], chunk_size)]
    )
```

## Convergence checks for scientific use

Publication-quality calculations should include at least the following
convergence checks:

1. **Sampling resolution:** refine the plotting or measurement grid.
2. **Geometry segmentation:** reduce `dx` for non-cuboidal bodies.
3. **Surface distance:** report errors as a function of distance from the
   magnet surface.
4. **Gradients:** compare the analytical gradient with finite differences at
   several step sizes.
5. **Reference solver:** halve the OOMMF cell size at least twice and estimate
   the observed convergence order.

A single favorable error metric is not a substitute for these studies.
