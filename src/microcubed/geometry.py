"""Rasterization and cuboid decomposition of two-dimensional shapes."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from numbers import Real

import numpy as np
from matplotlib.path import Path
from scipy.spatial import cKDTree

Bounds = tuple[float, float, float, float]
Shape2D = Sequence[Sequence[float]] | np.ndarray | Path | Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class VoronoiGrains:
    """A rasterized two- or three-dimensional Voronoi tessellation.

    Array axes are ordered like image/volume data: ``(y, x)`` in 2D and
    ``(z, y, x)`` in 3D. Coordinate columns and the values in ``origin``,
    ``size``, and ``cell_size`` remain in Cartesian ``(x, y[, z])`` order.
    Cells outside an optional domain mask have label ``-1``.
    """

    labels: np.ndarray
    seeds: np.ndarray
    origin: np.ndarray
    size: np.ndarray
    cell_size: np.ndarray

    @property
    def dimension(self) -> int:
        """Spatial dimension of the tessellation (2 or 3)."""
        return self.seeds.shape[1]

    @property
    def grain_count(self) -> int:
        """Number of represented grains."""
        return self.seeds.shape[0]

    @property
    def cell_centers(self) -> tuple[np.ndarray, ...]:
        """Return one-dimensional cell-center coordinates in Cartesian order."""
        counts = np.asarray(self.labels.shape[::-1])
        return tuple(
            lower + (np.arange(count) + 0.5) * step for lower, count, step in zip(self.origin, counts, self.cell_size)
        )

    @property
    def grain_measures(self) -> np.ndarray:
        """Area (2D) or volume (3D) represented by every grain."""
        occupied = self.labels[self.labels >= 0]
        counts = np.bincount(occupied, minlength=self.grain_count)
        return counts * np.prod(self.cell_size)

    @property
    def equivalent_diameters(self) -> np.ndarray:
        """Circle- or sphere-equivalent diameter of every rasterized grain."""
        if self.dimension == 2:
            return 2 * np.sqrt(self.grain_measures / np.pi)
        return np.cbrt(6 * self.grain_measures / np.pi)


def _validated_vector(value, name: str, dimension: int | None = None) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(-1)
    if dimension is not None and vector.size == 1:
        vector = np.repeat(vector, dimension)
    if vector.size not in {2, 3} or (dimension is not None and vector.size != dimension):
        expected = "a scalar or one value per dimension" if dimension is not None else "two or three values"
        raise ValueError(f"{name} must contain {expected}")
    if not np.all(np.isfinite(vector)) or np.any(vector <= 0):
        raise ValueError(f"{name} must contain positive finite values")
    return vector


def generate_voronoi_grains(
    size: Sequence[float] | np.ndarray,
    grain_size: Real,
    cell_size: Real | Sequence[float] | None = None,
    *,
    origin: Sequence[float] | np.ndarray | None = None,
    mask: np.ndarray | None = None,
    seed: int | np.random.Generator | None = None,
) -> VoronoiGrains:
    """Generate rasterized Voronoi grains in a 2D rectangle or 3D cuboid.

    ``grain_size`` is the target equivalent circle (2D) or sphere (3D)
    diameter. It determines the seed count automatically. ``cell_size`` is
    the maximum raster spacing and defaults to one fifth of ``grain_size``.
    The returned spacing may be slightly smaller so that the domain is tiled
    exactly. An optional Boolean ``mask`` restricts the occupied cells; this
    makes the function directly compatible with :func:`rasterize_shape`.

    Seeds are sampled without replacement from occupied cell centers. Thus
    every requested grain is represented even on comparatively coarse grids.
    """
    domain_size = _validated_vector(size, "size")
    dimension = domain_size.size
    if not isinstance(grain_size, Real) or not np.isfinite(grain_size) or grain_size <= 0:
        raise ValueError("grain_size must be a positive finite scalar")
    maximum_spacing = _validated_vector(
        float(grain_size) / 5 if cell_size is None else cell_size,
        "cell_size",
        dimension,
    )

    if origin is None:
        domain_origin = np.zeros(dimension)
    else:
        domain_origin = np.asarray(origin, dtype=float).reshape(-1)
        if domain_origin.size != dimension or not np.all(np.isfinite(domain_origin)):
            raise ValueError("origin must contain one finite value per dimension")

    counts = np.maximum(1, np.ceil(domain_size / maximum_spacing).astype(int))
    actual_spacing = domain_size / counts
    axes = [
        lower + (np.arange(count) + 0.5) * step for lower, count, step in zip(domain_origin, counts, actual_spacing)
    ]
    mesh = np.meshgrid(*axes, indexing="ij")
    points = np.column_stack([coordinate.ravel() for coordinate in mesh])
    array_shape = tuple(counts[::-1])

    if mask is None:
        domain_mask = np.ones(array_shape, dtype=bool)
    else:
        domain_mask = np.asarray(mask, dtype=bool)
        if domain_mask.shape != array_shape:
            raise ValueError(f"mask must have shape {array_shape}")
        if not np.any(domain_mask):
            raise ValueError("mask must contain at least one occupied cell")

    # Align Cartesian mesh points with conventional image/volume array axes.
    point_grid = points.reshape(tuple(counts) + (dimension,)).transpose(
        tuple(range(dimension - 1, -1, -1)) + (dimension,)
    )
    occupied_points = point_grid[domain_mask]
    target_measure = np.pi * float(grain_size) ** dimension / (4 if dimension == 2 else 6)
    occupied_measure = occupied_points.shape[0] * np.prod(actual_spacing)
    grain_count = min(occupied_points.shape[0], max(1, round(occupied_measure / target_measure)))

    rng = seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)
    seed_indices = rng.choice(occupied_points.shape[0], size=grain_count, replace=False)
    seeds = occupied_points[seed_indices]
    labels = np.full(array_shape, -1, dtype=np.int64)
    labels[domain_mask] = cKDTree(seeds).query(occupied_points)[1]
    return VoronoiGrains(labels, seeds, domain_origin, domain_size, actual_spacing)


def _validated_spacing(delta: Real | Sequence[float]) -> tuple[float, float]:
    spacing = np.asarray(delta, dtype=float).reshape(-1)
    if spacing.size == 1:
        spacing = np.repeat(spacing, 2)
    if spacing.size != 2 or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("delta must be a positive scalar or a pair (dx, dy)")
    return float(spacing[0]), float(spacing[1])


def _validated_bounds(bounds: Bounds) -> Bounds:
    values = np.asarray(bounds, dtype=float).reshape(-1)
    if values.size != 4 or not np.all(np.isfinite(values)):
        raise ValueError("bounds must contain four finite values (xmin, ymin, xmax, ymax)")
    xmin, ymin, xmax, ymax = map(float, values)
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("bounds must have positive width and height")
    return xmin, ymin, xmax, ymax


def _axis_edges(lower: float, upper: float, maximum_step: float) -> np.ndarray:
    cells = max(1, int(np.ceil((upper - lower) / maximum_step)))
    return np.linspace(lower, upper, cells + 1)


def rasterize_shape(
    shape: Shape2D,
    delta: Real | Sequence[float],
    *,
    bounds: Bounds | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a 2D shape at cell centers on a grid no coarser than ``delta``.

    ``shape`` can be polygon vertices, a :class:`matplotlib.path.Path`, or a
    callable ``shape(x, y)`` returning a Boolean mask. Callable shapes require
    explicit ``(xmin, ymin, xmax, ymax)`` bounds.
    """
    dx, dy = _validated_spacing(delta)

    if isinstance(shape, Path):
        path = shape
        shape_bounds = tuple(path.get_extents().extents)
    elif callable(shape):
        path = None
        if bounds is None:
            raise ValueError("bounds are required for callable shapes")
        shape_bounds = bounds
    else:
        vertices = np.asarray(shape, dtype=float)
        if vertices.ndim != 2 or vertices.shape[0] < 3 or vertices.shape[1] != 2:
            raise ValueError("polygon vertices must have shape (N, 2) with N >= 3")
        if not np.all(np.isfinite(vertices)):
            raise ValueError("polygon vertices must be finite")
        # ``contains_points`` implicitly closes a polygon. Passing
        # ``closed=True`` would turn the final supplied vertex into a dummy
        # CLOSEPOLY vertex and could therefore discard a real corner.
        path = Path(vertices)
        shape_bounds = (vertices[:, 0].min(), vertices[:, 1].min(), vertices[:, 0].max(), vertices[:, 1].max())

    xmin, ymin, xmax, ymax = _validated_bounds(shape_bounds if bounds is None else bounds)
    x_edges = _axis_edges(xmin, xmax, dx)
    y_edges = _axis_edges(ymin, ymax, dy)
    x_centers = (x_edges[:-1] + x_edges[1:]) / 2
    y_centers = (y_edges[:-1] + y_edges[1:]) / 2
    xx, yy = np.meshgrid(x_centers, y_centers, indexing="xy")

    if path is not None:
        points = np.column_stack((xx.ravel(), yy.ravel()))
        mask = path.contains_points(points, radius=np.finfo(float).eps).reshape(xx.shape)
    else:
        mask = np.asarray(shape(xx, yy), dtype=bool)
        try:
            mask = np.broadcast_to(mask, xx.shape).copy()
        except ValueError as error:
            raise ValueError("callable shape must return a mask broadcastable to the sampling grid") from error

    if not np.any(mask):
        raise ValueError("shape contains no sampled cells; reduce delta or adjust bounds")
    return x_edges, y_edges, mask


def generate_voronoi_grains_from_shape(
    shape: Shape2D,
    grain_size: Real,
    cell_size: Real | Sequence[float] | None = None,
    *,
    bounds: Bounds | None = None,
    seed: int | np.random.Generator | None = None,
) -> VoronoiGrains:
    """Generate 2D Voronoi grains clipped to an arbitrary rasterized shape.

    Shape representations and boundary sampling follow
    :func:`rasterize_shape`. Grain IDs outside the shape are ``-1``.
    """
    if not isinstance(grain_size, Real) or not np.isfinite(grain_size) or grain_size <= 0:
        raise ValueError("grain_size must be a positive finite scalar")
    raster_spacing = float(grain_size) / 5 if cell_size is None else cell_size
    x_edges, y_edges, mask = rasterize_shape(shape, raster_spacing, bounds=bounds)
    return generate_voronoi_grains(
        [x_edges[-1] - x_edges[0], y_edges[-1] - y_edges[0]],
        grain_size,
        [x_edges[1] - x_edges[0], y_edges[1] - y_edges[0]],
        origin=[x_edges[0], y_edges[0]],
        mask=mask,
        seed=seed,
    )


def _largest_rectangle(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return the largest true rectangle as ``(row0, row1, column0, column1)``."""
    heights = np.zeros(mask.shape[1], dtype=int)
    best = (0, 0, 0, 0)
    best_key = (0, 0, 0, 0, 0)

    for row, occupied in enumerate(mask):
        heights = np.where(occupied, heights + 1, 0)
        stack: list[tuple[int, int]] = []
        for column, height in enumerate(np.append(heights, 0)):
            start = column
            while stack and stack[-1][1] > height:
                start, previous_height = stack.pop()
                candidate = (row + 1 - previous_height, row + 1, start, column)
                area = previous_height * (column - start)
                key = (area, -candidate[0], -candidate[2], candidate[1], candidate[3])
                if key > best_key:
                    best_key = key
                    best = candidate
            if not stack or stack[-1][1] < height:
                stack.append((start, int(height)))
    return best


def rectangle_union_boundary(rectangles: np.ndarray, *, atol: float | None = None) -> list[np.ndarray]:
    """Trace oriented boundary rings of axis-aligned rectangles.

    Input rows are ``(xmin, ymin, xmax, ymax)``. A sweep along compressed x
    coordinates keeps only one y coverage vector in memory. Coordinates closer
    than ``atol`` are snapped together to remove floating-point seams; the default
    is 32 machine epsilons times the largest absolute input coordinate.
    """
    rectangles = np.asarray(rectangles, dtype=float).reshape(-1, 4)
    if atol is not None and (not np.isfinite(atol) or atol < 0):
        raise ValueError("atol must be finite and non-negative")
    if not np.isfinite(rectangles).all():
        raise ValueError("Footprint coordinates must be finite")
    if not len(rectangles):
        return []
    if np.any(rectangles[:, 2:] <= rectangles[:, :2]):
        raise ValueError("Footprints must have positive width and height")
    if atol is None:
        atol = 32 * np.finfo(float).eps * np.max(np.abs(rectangles))

    def compress(values):
        unique, inverse = np.unique(values, return_inverse=True)
        representatives = [unique[0]]
        labels = np.zeros(len(unique), dtype=int)
        for i, value in enumerate(unique[1:], 1):
            if value - representatives[-1] > atol:
                representatives.append(value)
            labels[i] = len(representatives) - 1
        return np.asarray(representatives), labels[inverse].reshape(values.shape)

    xs, xi = compress(rectangles[:, [0, 2]])
    ys, yi = compress(rectangles[:, [1, 3]])
    events = [[] for _ in xs]
    for (x0, x1), (y0, y1) in zip(xi, yi):
        if x0 == x1 or y0 == y1:
            continue  # Explicit snapping can collapse features smaller than atol.
        events[x0].append((y0, y1, 1))
        events[x1].append((y0, y1, -1))

    edges = set()
    coverage = np.zeros(len(ys) - 1, dtype=int)
    for x, changes in enumerate(events):
        previous = coverage > 0
        for y0, y1, change in changes:
            coverage[y0:y1] += change
        current = coverage > 0
        # Orient every edge with occupied material on its left.
        for y in np.flatnonzero(previous != current):
            lower, upper = (x, int(y)), (x, int(y) + 1)
            edges.add((upper, lower) if current[y] else (lower, upper))
        if x + 1 < len(xs):
            transitions = np.diff(np.r_[False, current, False].astype(int))
            for y in np.flatnonzero(transitions):
                left, right = (x, int(y)), (x + 1, int(y))
                edges.add((left, right) if transitions[y] == 1 else (right, left))

    outgoing = {}
    for start, end in edges:
        outgoing.setdefault(start, []).append(end)

    def direction(start, end):
        dx, dy = end[0] - start[0], end[1] - start[1]
        return 0 if dx > 0 else 1 if dy > 0 else 2 if dx < 0 else 3

    # The leftmost continuation keeps corner-touching components as separate rings.
    preference = {1: 0, 0: 1, 3: 2, 2: 3}
    successor = {}
    for start, end in edges:
        incoming = direction(start, end)
        successor[(start, end)] = (
            end,
            min(outgoing[end], key=lambda point: preference[(direction(end, point) - incoming) % 4]),
        )

    rings = []
    while edges:
        first = min(edges)
        edge = first
        points = []
        while True:
            edges.remove(edge)
            points.append(edge[0])
            edge = successor[edge]
            if edge == first:
                break
        # Remove collinear intermediate vertices without changing the boundary.
        corners = [
            point
            for i, point in enumerate(points)
            if direction(points[i - 1], point) != direction(point, points[(i + 1) % len(points)])
        ]
        corners.append(corners[0])
        rings.append(np.array([[xs[x], ys[y]] for x, y in corners]).T)
    return rings


def merge_mask_to_rectangles(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Greedily partition a Boolean mask into large non-overlapping rectangles."""
    remaining = np.asarray(mask, dtype=bool).copy()
    if remaining.ndim != 2:
        raise ValueError("mask must be two-dimensional")

    rectangles = []
    while np.any(remaining):
        rectangle = _largest_rectangle(remaining)
        row0, row1, column0, column1 = rectangle
        if row0 == row1 or column0 == column1:  # pragma: no cover - defensive invariant
            raise RuntimeError("failed to find a rectangle in a non-empty mask")
        rectangles.append(rectangle)
        remaining[row0:row1, column0:column1] = False
    return rectangles


def merge_mask_to_boxes(mask: np.ndarray) -> list[tuple[int, int, int, int, int, int]]:
    """Partition a 3D Boolean mask into non-overlapping axis-aligned boxes.

    Each z slice is first compressed with :func:`merge_mask_to_rectangles`.
    Equal rectangles in consecutive slices are then merged along z. The
    result is deterministic and exactly preserves the input voxels.
    """
    occupied = np.asarray(mask, dtype=bool)
    if occupied.ndim != 3:
        raise ValueError("mask must be three-dimensional")

    boxes: list[tuple[int, int, int, int, int, int]] = []
    active: dict[tuple[int, int, int, int], int] = {}
    for layer, layer_mask in enumerate(occupied):
        rectangles = merge_mask_to_rectangles(layer_mask)
        current = set(rectangles)
        for rectangle, layer0 in active.items():
            if rectangle not in current:
                row0, row1, column0, column1 = rectangle
                boxes.append((layer0, layer, row0, row1, column0, column1))
        active = {rectangle: active.get(rectangle, layer) for rectangle in rectangles}

    for rectangle, layer0 in active.items():
        row0, row1, column0, column1 = rectangle
        boxes.append((layer0, occupied.shape[0], row0, row1, column0, column1))
    return boxes


__all__ = [
    "VoronoiGrains",
    "generate_voronoi_grains",
    "generate_voronoi_grains_from_shape",
    "merge_mask_to_boxes",
    "merge_mask_to_rectangles",
    "rasterize_shape",
]
