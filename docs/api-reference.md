# API reference

This reference documents the supported public API. Parameter and return-value
sections use the [NumPy docstring convention](https://numpydoc.readthedocs.io/)
and are rendered by Sphinx Napoleon. Backend implementation classes and private
helpers are intentionally omitted. The top-level aliases select a backend
automatically; the Rust implementation is the documented default.

## Magnets and arrangements

### `microcubed.Magnet`

```{eval-rst}
.. autoclass:: microcubed.backends.rust.RustMagnet
   :members: Bfield, dBfield, Hfield, dHfield, union_boundary, plot_1d, plot_2d, plot_3d, from_bbox, moved_by, moved_to, mirrored, overlapping, touching
   :no-index:
```

### `microcubed.Arrangement`

```{eval-rst}
.. autoclass:: microcubed.backends.rust.RustArrangement
   :members: Bfield, dBfield, Hfield, dHfield, union_boundary, plot_1d, plot_2d, plot_3d, from_shape, from_voronoi_grains, from_circle, from_ellipsis, from_rounded_rectangle, from_superellipsis, from_right_triangle, moved_by, moved_to, mirrored
   :no-index:
```

## Backend selection

```{eval-rst}
.. autofunction:: microcubed.get_backend
```

```{eval-rst}
.. autofunction:: microcubed.available_backends
```

```{eval-rst}
.. autofunction:: microcubed.set_backend
```

```{eval-rst}
.. autofunction:: microcubed.backend_name
```

## Shape decomposition and grains

```{eval-rst}
.. autofunction:: microcubed.cuboidize
```

```{eval-rst}
.. autofunction:: microcubed.cuboidize_voronoi
```

```{eval-rst}
.. autoclass:: microcubed.VoronoiGrains
   :members: dimension, grain_count, cell_centers, grain_measures, equivalent_diameters
```

```{eval-rst}
.. autofunction:: microcubed.generate_voronoi_grains
```

```{eval-rst}
.. autofunction:: microcubed.generate_voronoi_grains_from_shape
```

```{eval-rst}
.. autofunction:: microcubed.geometry.rasterize_shape
```

```{eval-rst}
.. autofunction:: microcubed.geometry.rectangle_union_boundary
```

## Sampling and visualisation

The following functions are also re-exported from `microcubed`.

```{eval-rst}
.. autofunction:: microcubed.viz.sample_field
```

```{eval-rst}
.. autofunction:: microcubed.viz.plot_1d
```

```{eval-rst}
.. autofunction:: microcubed.viz.plot_2d
```

```{eval-rst}
.. autofunction:: microcubed.viz.plot_3d
```

## Units and coordinate helpers

```{eval-rst}
.. autofunction:: microcubed.utils.spherical2cartesian
```

```{eval-rst}
.. autofunction:: microcubed.utils.cartesian2spherical
```

```{eval-rst}
.. autofunction:: microcubed.utils.range2array
```
