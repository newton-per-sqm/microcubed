# Magnets and arrangements

## `Magnet`

```python
from microcubed import Magnet

magnet = Magnet(
    size=[100, 50, 20],
    center=[10, 0, -5],
    magnetization=[0, 8e5, 0],
)
```

All three inputs are stored internally as column vectors with shape `(3, 1)`.
Zero or negative dimensions are not allowed.

Important properties:

| Property | Meaning |
|---|---|
| `size` | edge lengths `(sx, sy, sz)` |
| `center` | cuboid centre |
| `magnetization` | magnetisation in A/m |
| `volume` | volume in the third power of the length unit |
| `bbox` | bounding box with shape `(3, 2)` |
| `corners` | eight corners with shape `(3, 8)` |
| `polarization` | internal field prefactor |

Constructing a magnet from two bounding-box points:

```python
magnet = Magnet.from_bbox(
    p1=[-50, -25, -10],
    p2=[50, 25, 10],
    magnetization=[0, 8e5, 0],
)
```

## Geometric operations

Operations return new objects and do not modify the original:

```python
moved = magnet.moved_by([100, 0, 0])
centered = magnet.moved_to([0, 0, 0])
mirrored = magnet.mirrored(x=0, z=10)
```

Testing contact and volume overlap:

```python
magnet.touching(moved)
magnet.overlapping(moved)
```

Convex hulls are available in 3D and as plane projections:

```python
xy_outline = magnet.chull_points("xy")
```

## `Arrangement`

Fields from multiple magnets are added by superposition:

```python
from microcubed import Arrangement, Magnet

prototype = Magnet([100, 100, 50], [0, 0, 0], [0, 0, 1e6])
arrangement = Arrangement(
    [
        prototype.moved_to([-150, 0, 0]),
        prototype,
        prototype.moved_to([150, 0, 0]),
    ]
)

B = arrangement.Bfield([0, 0, -200])
```

With `validate=True`, construction checks for overlapping cuboids. The check
scales quadratically with the number of magnets. It can be disabled for
geometries loaded from a previously verified grid:

```python
arrangement = Arrangement(magnets, validate=False)
```

Arrangements support iteration, length, indexing, slices, and `+`/`-` with
magnets and other arrangements.

## Geometry generators

`Arrangement` can approximate selected 2D cross-sections with cuboid strips:

| Constructor | Cross-section |
|---|---|
| `from_circle` | circle |
| `from_ellipsis` | ellipse |
| `from_rounded_rectangle` | rounded rectangle |
| `from_superellipsis` | superellipse |
| `from_right_triangle` | right triangle |

Example:

```python
ellipse = Arrangement.from_ellipsis(
    size=[400, 200, 30],
    center=[0, 0, 0],
    mag=[0, 0, 8e5],
    dx=5,
)
```

`dx` controls strip width and therefore geometry error, runtime, and memory.
Scientific use requires a convergence study over multiple `dx` values.

## Arbitrary 2D shapes

`cuboidize` rasterizes an arbitrary two-dimensional shape, merges occupied
cells into large rectangles, and extrudes those rectangles by a thickness
`t`. The result is a normal `Arrangement` and can immediately be used for
field and gradient calculations:

```python
from microcubed import cuboidize

polygon = [
    (-200, -100),
    (200, -100),
    (200, 0),
    (50, 0),
    (50, 150),
    (-200, 150),
]

magnet = cuboidize(
    polygon,
    t=30,
    delta=5,
    mag=[0, 0, 8e5],
    z=15,
)

B = magnet.Bfield([0, 0, -100])
```

The equivalent class constructor is
`Arrangement.from_shape(shape, t, delta, mag, ...)`.

Supported shape representations are:

- an `(N, 2)` array or sequence of polygon vertices;
- a `matplotlib.path.Path`; or
- a callable `shape(x, y)` returning a Boolean mask. A callable requires
  `bounds=(xmin, ymin, xmax, ymax)`.

For example, an annulus can be represented without constructing polygon
vertices:

```python
def annulus(x, y):
    radius_squared = x**2 + y**2
    return (radius_squared <= 200**2) & (radius_squared >= 80**2)


ring = cuboidize(
    annulus,
    t=20,
    delta=(5, 5),
    mag=[1e6, 0, 0],
    bounds=(-200, -200, 200, 200),
)
```

`delta` is either a scalar or `(dx, dy)` and specifies the maximum sampling
cell size. A cell belongs to the approximation when its centre is inside the
shape. The occupied cells are then partitioned by repeatedly selecting the
largest remaining rectangle. The produced cuboids are non-overlapping and
preserve the rasterized area exactly.

The greedy rectangle partition is deterministic and usually reduces the
number of magnets substantially compared with one cuboid per occupied cell.
It does not guarantee the global minimum number for every possible binary
shape; finding that optimum is a combinatorial covering problem. The geometric
boundary remains a centre-sampled approximation, so scientific calculations
should be repeated with progressively smaller `delta`.

See {doc}`examples` for a self-contained polygon decomposition notebook.

## Voronoi grain structures

Voronoi grains can be generated in rectangular 2D or cuboid 3D domains. The
requested `grain_size` is interpreted as an equivalent circle or sphere
diameter and is converted automatically into a seed count:

```python
from microcubed import cuboidize_voronoi, generate_voronoi_grains

grains = generate_voronoi_grains(
    size=(100, 60, 20),
    grain_size=10,
    cell_size=2,
    origin=(-50, -30, -10),
    seed=42,
)
magnet = cuboidize_voronoi(grains, mag=[0, 0, 8e5])

print(grains.grain_count)
print(grains.equivalent_diameters)
```

For a polygon, `Path`, or callable 2D geometry, use
`generate_voronoi_grains_from_shape`. This first rasterizes the full shape,
creates one global Voronoi tessellation inside its occupied mask, and only
then merges each grain into cuboids. This ordering avoids artificial grain
boundaries at geometry-decomposition seams.

## Parallel calculation

Use `get_backend("rust").Arrangement` for the integrated parallel implementation.
See {doc}`backends` for selection and comparisons with NumPy.

## Union boundary versus convex hull

`union_boundary(plane="xy", atol=None)` is available on both `Magnet` and
`Arrangement`, with either numerical backend. It returns the boundary of the
**union of projected cuboid footprints**, retaining concavities, disconnected
pieces, and holes. Shared internal edges disappear; overlapping footprints merge.
A convex hull instead spans concavities and gaps.

```python
for boundary in arrangement.union_boundary("xy"):
    ax.plot(*boundary, color="cyan", label="Material boundary")
ax.plot(*arrangement.chull_points("xy"), "k--", label="Convex hull")
```

Each returned array has shape `(2, N)` and repeats its first vertex at the end.
Outer rings run counterclockwise and hole rings clockwise. Corner-touching
components have separate rings; an empty arrangement returns `[]`. The list is
not a hierarchy associating holes with their enclosing components.

Supported planes are `xy`, `xz`, and `yz`. Like `chull_points`, reversed aliases
such as `yx` use canonical axis order (x, then y). This is a projection, not a
slice: cuboids separated in z can still have overlapping XY footprints.

By default, coordinates within 32 machine epsilons times the largest absolute
projected coordinate are snapped together to remove floating-point seams.
Use `atol=0` for exact coordinate comparisons or supply an absolute tolerance
in the geometry's length unit. Features smaller than the snapping tolerance may
collapse; use a tolerance well below any gap or feature you need to resolve.
The sweep keeps one coordinate-coverage vector in memory rather than allocating
a dense 2D raster; its worst-case running time is quadratic in the cuboid count.

See the [polygon example](_notebooks/shape.ipynb) for input-polygon, union-boundary,
and convex-hull overlays. Refining the source raster improves the union boundary's
approximation of the input polygon; it does not eliminate a convex hull's bridges.
