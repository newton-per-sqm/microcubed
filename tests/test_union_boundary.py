from __future__ import annotations

import numpy as np
import pytest

from microcubed import get_backend
from microcubed.geometry import rectangle_union_boundary


def signed_area(loop):
    # Translate first to avoid cancellation for small shapes far from the origin.
    x, y = loop - loop[:, :1]
    return np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) / 2


@pytest.fixture(params=["numpy", "rust"])
def backend(request):
    return get_backend(request.param)


def arrangement(backend, rectangles):
    return backend.Arrangement(
        [
            backend.Magnet([x1 - x0, y1 - y0, 1], [(x0 + x1) / 2, (y0 + y1) / 2, 0], [0, 0, 1])
            for x0, y0, x1, y1 in rectangles
        ],
        validate=False,
    )


@pytest.mark.parametrize(
    "plane,indices", [("xy", [0, 1]), ("yx", [0, 1]), ("XZ", [0, 2]), ("zx", [0, 2]), ("yz", [1, 2]), ("zy", [1, 2])]
)
def test_single_cuboid_projection(backend, plane, indices):
    magnet = backend.Magnet([2, 4, 6], [10, -5, 3], [0, 0, 1])
    (loop,) = magnet.union_boundary(plane)
    assert loop.shape == (2, 5)
    np.testing.assert_array_equal(loop[:, 0], loop[:, -1])
    np.testing.assert_allclose(loop.min(axis=1), magnet.bbox[indices, 0])
    np.testing.assert_allclose(loop.max(axis=1), magnet.bbox[indices, 1])
    assert signed_area(loop) == np.prod(magnet.size.ravel()[indices])


@pytest.mark.parametrize(
    "rectangles,areas",
    [
        ([(0, 0, 2, 1), (0, 1, 1, 2)], [3]),  # Concave L, no hull bridge.
        ([(0, 0, 2, 2), (1, 1, 3, 3)], [7]),  # Overlap.
        ([(0, 0, 3, 3), (1, 1, 2, 2), (0, 0, 3, 3)], [9]),  # Containment and duplicates.
        ([(0, 0, 1, 1), (1, 0, 2, 1)], [2]),  # Shared edge removed.
        ([(0, 0, 1, 1), (1, 1, 2, 2)], [1, 1]),  # Corner contact.
        ([(0, 1, 1, 2), (1, 0, 2, 1)], [1, 1]),
        ([(0, 0, 1, 1), (2, 0, 3, 1)], [1, 1]),  # Gap preserved.
        ([(0, 0, 3, 1), (0, 2, 3, 3), (0, 1, 1, 2), (2, 1, 3, 2)], [-1, 9]),  # Hole.
    ],
)
def test_union_topology(backend, rectangles, areas):
    loops = arrangement(backend, rectangles).union_boundary()
    np.testing.assert_allclose(sorted(map(signed_area, loops)), areas)
    for loop in loops:
        np.testing.assert_array_equal(loop[:, 0], loop[:, -1])
        steps = np.diff(loop)
        assert np.all(np.count_nonzero(steps, axis=0) == 1)
    if areas == [3]:
        assert loops[0].shape == (2, 7)
        assert any(np.array_equal(point, [1, 1]) for point in loops[0].T)


def test_projection_merges_cuboids_separated_in_depth(backend):
    cube = backend.Magnet([2, 2, 2], [0, 0, 0], [0, 0, 1])
    model = backend.Arrangement([cube, cube.moved_by([0, 0, 10])])
    assert len(model.union_boundary("xy")) == 1
    assert len(model.union_boundary("xz")) == 2


def test_empty_and_invalid_arguments(backend):
    model = backend.Arrangement([], validate=False)
    assert model.union_boundary() == []
    for plane in (None, "xyz", "bad", 1):
        with pytest.raises(ValueError, match="plane"):
            model.union_boundary(plane)
    for tolerance in (-1, np.inf, np.nan):
        with pytest.raises(ValueError, match="atol"):
            model.union_boundary(atol=tolerance)


def test_tolerance_removes_roundoff_seams_but_preserves_real_gaps():
    rectangles = [[0, 0, 1, 1], [1 + 1e-15, 0, 2, 1]]
    assert len(rectangle_union_boundary(rectangles)) == 1
    assert len(rectangle_union_boundary(rectangles, atol=0)) == 2
    assert len(rectangle_union_boundary([[0, 0, 1, 1], [1.001, 0, 2, 1]])) == 2
    assert rectangle_union_boundary([[0, 0, 1, 1]], atol=2) == []


def test_seeded_unions_match_independent_cell_area():
    rng = np.random.default_rng(7)
    for _ in range(40):
        rectangles = []
        occupied = np.zeros((10, 10), dtype=bool)
        for _ in range(15):
            x0, x1 = sorted(rng.choice(11, 2, replace=False))
            y0, y1 = sorted(rng.choice(11, 2, replace=False))
            rectangles.append((x0, y0, x1, y1))
            occupied[x0:x1, y0:y1] = True
        loops = rectangle_union_boundary(rectangles, atol=0)
        assert sum(map(signed_area, loops)) == occupied.sum()
        # Every boundary segment separates an occupied and an empty unit cell.
        for loop in loops:
            for start, end in zip(loop.T[:-1], loop.T[1:]):
                length = int(np.abs(end - start).sum())
                step = (end - start) / length
                normal = np.array([-step[1], step[0]])
                for i in range(length):
                    midpoint = start + (i + 0.5) * step
                    cells = [np.floor(midpoint + sign * 0.25 * normal).astype(int) for sign in (1, -1)]
                    values = [bool(occupied[tuple(c)]) if np.all((c >= 0) & (c < 10)) else False for c in cells]
                    assert values == [True, False]
