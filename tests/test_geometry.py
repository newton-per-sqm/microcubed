from __future__ import annotations

import numpy as np
import pytest
from matplotlib.path import Path

from microcubed import (
    Arrangement,
    cuboidize,
    cuboidize_voronoi,
    generate_voronoi_grains,
    generate_voronoi_grains_from_shape,
)
from microcubed.geometry import merge_mask_to_boxes, merge_mask_to_rectangles, rasterize_shape

MAGNETIZATION = [1e5, 2e5, 3e5]


def rasterized_area(arrangement):
    return sum(magnet.size[0, 0] * magnet.size[1, 0] for magnet in arrangement)


def test_rectangle_becomes_one_cuboid():
    polygon = [[-2, -1], [2, -1], [2, 1], [-2, 1]]
    arrangement = cuboidize(polygon, t=0.5, delta=0.25, mag=MAGNETIZATION, z=3)

    assert isinstance(arrangement, Arrangement)
    assert len(arrangement) == 1
    np.testing.assert_allclose(arrangement[0].size.ravel(), [4, 2, 0.5])
    np.testing.assert_allclose(arrangement[0].center.ravel(), [0, 0, 3])
    np.testing.assert_allclose(arrangement[0].magnetization.ravel(), MAGNETIZATION)
    assert np.isfinite(arrangement.Bfield([0, 0, -5])).all()
    assert np.isfinite(arrangement.dBfield([0, 0, -5])).all()


def test_concave_polygon_uses_fewer_cuboids_than_cells():
    # An L-shaped polygon on a 3 x 3 grid contains five cells and has an exact
    # partition into two rectangles.
    polygon = [[0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [0, 3]]
    x_edges, y_edges, mask = rasterize_shape(polygon, delta=1)
    arrangement = Arrangement.from_shape(polygon, t=2, delta=1, mag=MAGNETIZATION)

    assert x_edges.size == y_edges.size == 4
    assert mask.sum() == 5
    assert len(arrangement) == 2
    assert len(arrangement) < mask.sum()
    assert rasterized_area(arrangement) == mask.sum()
    assert not arrangement[0].overlapping(arrangement[1])


def test_path_and_anisotropic_delta():
    path = Path.unit_circle()
    x_edges, y_edges, mask = rasterize_shape(path, delta=(0.5, 0.25))
    arrangement = Arrangement.from_shape(path, t=1, delta=(0.5, 0.25), mag=MAGNETIZATION)

    assert np.max(np.diff(x_edges)) <= 0.5
    assert np.max(np.diff(y_edges)) <= 0.25
    assert rasterized_area(arrangement) == pytest.approx(mask.sum() * 0.5 * 0.25)


def test_callable_shape_with_hole_and_bounds_override():
    def annulus(x, y):
        radius_squared = x**2 + y**2
        return (radius_squared <= 4) & (radius_squared >= 1)

    x_edges, y_edges, mask = rasterize_shape(annulus, 0.5, bounds=(-2, -2, 2, 2))
    arrangement = cuboidize(annulus, 1, 0.5, MAGNETIZATION, bounds=(-2, -2, 2, 2))
    assert not mask[3:5, 3:5].any()
    assert rasterized_area(arrangement) == pytest.approx(mask.sum() * 0.25)

    # Explicit bounds may crop a polygon or Path deliberately.
    _, _, cropped = rasterize_shape(Path.unit_circle(), 0.25, bounds=(0, 0, 1, 1))
    assert cropped.any()
    assert not cropped.all()
    assert x_edges.size == y_edges.size == 9


def test_callable_scalar_mask_is_broadcast():
    x_edges, y_edges, mask = rasterize_shape(lambda x, y: True, 1, bounds=(0, 0, 2, 3))
    np.testing.assert_array_equal(mask, np.ones((3, 2), dtype=bool))
    assert x_edges.size == 3
    assert y_edges.size == 4


def test_rectangle_merging_preserves_arbitrary_mask():
    mask = np.array(
        [
            [1, 1, 1, 0],
            [1, 1, 1, 0],
            [1, 0, 1, 1],
        ],
        dtype=bool,
    )
    reconstructed = np.zeros_like(mask)
    rectangles = merge_mask_to_rectangles(mask)
    for row0, row1, column0, column1 in rectangles:
        assert not reconstructed[row0:row1, column0:column1].any()
        reconstructed[row0:row1, column0:column1] = True
    np.testing.assert_array_equal(reconstructed, mask)
    assert merge_mask_to_rectangles(np.zeros((2, 3), dtype=bool)) == []


@pytest.mark.parametrize("delta", [0, -1, np.inf, (1, 0), (1, 2, 3)])
def test_invalid_delta(delta):
    with pytest.raises(ValueError, match="delta"):
        rasterize_shape([[0, 0], [1, 0], [0, 1]], delta)


@pytest.mark.parametrize(
    "vertices",
    [
        [0, 1, 2],
        [[0, 0], [1, 0]],
        [[0, 0, 1], [1, 0, 1], [0, 1, 1]],
        [[0, 0], [1, 0], [np.nan, 1]],
        [[0, 0], [0, 1], [0, 2]],
    ],
)
def test_invalid_polygon(vertices):
    with pytest.raises(ValueError, match="vertices|bounds"):
        rasterize_shape(vertices, 0.5)


@pytest.mark.parametrize(
    "bounds",
    [
        (0, 0, 1),
        (0, 0, np.inf, 1),
        (1, 0, 0, 1),
        (0, 1, 1, 0),
    ],
)
def test_invalid_bounds(bounds):
    with pytest.raises(ValueError, match="bounds"):
        rasterize_shape(lambda x, y: True, 1, bounds=bounds)


def test_callable_validation():
    with pytest.raises(ValueError, match="bounds are required"):
        rasterize_shape(lambda x, y: True, 1)
    with pytest.raises(ValueError, match="broadcastable"):
        rasterize_shape(lambda x, y: np.ones((3, 3)), 1, bounds=(0, 0, 2, 2))
    with pytest.raises(ValueError, match="no sampled cells"):
        rasterize_shape(lambda x, y: False, 1, bounds=(0, 0, 2, 2))


def test_mask_and_extrusion_validation():
    with pytest.raises(ValueError, match="two-dimensional"):
        merge_mask_to_rectangles(np.ones(3))

    polygon = [[0, 0], [1, 0], [0, 1]]
    for thickness in (0, -1, np.inf, [1]):
        with pytest.raises(ValueError, match="t must"):
            Arrangement.from_shape(polygon, thickness, 0.1, MAGNETIZATION)
    for z in (np.inf, [0]):
        with pytest.raises(ValueError, match="z must"):
            Arrangement.from_shape(polygon, 1, 0.1, MAGNETIZATION, z=z)


def test_automatic_2d_voronoi_grain_count_and_reproducibility():
    grains = generate_voronoi_grains([20, 10], grain_size=4, cell_size=1, origin=[-10, -5], seed=12)
    repeated = generate_voronoi_grains([20, 10], grain_size=4, cell_size=1, origin=[-10, -5], seed=12)

    expected_count = round(20 * 10 / (np.pi * 2**2))
    assert grains.dimension == 2
    assert grains.labels.shape == (10, 20)
    assert grains.grain_count == expected_count
    assert np.unique(grains.labels).size == expected_count
    np.testing.assert_array_equal(grains.labels, repeated.labels)
    np.testing.assert_array_equal(grains.seeds, repeated.seeds)
    np.testing.assert_allclose(grains.origin, [-10, -5])
    np.testing.assert_allclose(grains.cell_size, [1, 1])
    assert grains.grain_measures.sum() == pytest.approx(200)
    assert np.all(grains.equivalent_diameters > 0)


def test_automatic_3d_voronoi_grains_and_default_resolution():
    grains = generate_voronoi_grains([12, 8, 4], grain_size=3, seed=3)
    expected_count = round(12 * 8 * 4 / (np.pi * 3**3 / 6))

    assert grains.dimension == 3
    assert grains.labels.ndim == 3
    assert grains.labels.shape == tuple((np.ceil(np.array([12, 8, 4]) / 0.6).astype(int))[::-1])
    assert grains.grain_count == expected_count
    assert np.unique(grains.labels).size == expected_count
    assert grains.seeds.shape == (expected_count, 3)
    assert len(grains.cell_centers) == 3
    assert grains.grain_measures.sum() == pytest.approx(12 * 8 * 4)


def test_voronoi_grains_are_clipped_to_arbitrary_shape():
    polygon = [[0, 0], [6, 0], [6, 2], [2, 2], [2, 6], [0, 6]]
    grains = generate_voronoi_grains_from_shape(polygon, grain_size=2, cell_size=1, seed=4)
    _, _, mask = rasterize_shape(polygon, 1)

    np.testing.assert_array_equal(grains.labels >= 0, mask)
    assert np.all(grains.seeds[:, 0] < 6)
    assert np.all(grains.seeds[:, 1] < 6)
    assert grains.grain_measures.sum() == pytest.approx(mask.sum())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"size": [1]}, "size"),
        ({"size": [1, -1]}, "size"),
        ({"size": [1, 1], "grain_size": 0}, "grain_size"),
        ({"size": [1, 1], "cell_size": [1, 1, 1]}, "cell_size"),
        ({"size": [1, 1], "origin": [0]}, "origin"),
        ({"size": [2, 2], "mask": np.ones((2, 3))}, "mask"),
        ({"size": [2, 2], "mask": np.zeros((2, 2))}, "mask"),
    ],
)
def test_voronoi_validation(kwargs, message):
    parameters = {"grain_size": 1, "cell_size": 1, **kwargs}
    with pytest.raises(ValueError, match=message):
        generate_voronoi_grains(**parameters)


def test_merge_mask_to_boxes_preserves_volume():
    mask = np.zeros((3, 4, 5), dtype=bool)
    mask[:2, :2, :3] = True
    mask[1:, 3, 2:] = True
    reconstructed = np.zeros_like(mask)
    boxes = merge_mask_to_boxes(mask)
    for layer0, layer1, row0, row1, column0, column1 in boxes:
        assert not reconstructed[layer0:layer1, row0:row1, column0:column1].any()
        reconstructed[layer0:layer1, row0:row1, column0:column1] = True
    np.testing.assert_array_equal(reconstructed, mask)
    assert merge_mask_to_boxes(np.zeros((2, 2, 2), dtype=bool)) == []
    with pytest.raises(ValueError, match="three-dimensional"):
        merge_mask_to_boxes(np.ones((2, 2)))


def test_cuboidize_2d_voronoi_shape_with_per_grain_magnetization():
    polygon = [[0, 0], [6, 0], [6, 2], [2, 2], [2, 6], [0, 6]]
    grains = generate_voronoi_grains_from_shape(polygon, grain_size=2, cell_size=1, seed=5)

    def magnetizations(seeds):
        result = np.zeros((len(seeds), 3))
        result[:, 0] = np.arange(len(seeds)) + 1
        return result

    arrangement = cuboidize_voronoi(grains, magnetizations, thickness=2, z=3)
    assert isinstance(arrangement, Arrangement)
    assert sum(magnet.volume for magnet in arrangement) == pytest.approx(np.count_nonzero(grains.labels >= 0) * 2)
    assert all(magnet.center[2, 0] == 3 for magnet in arrangement)
    assert {magnet.magnetization[0, 0] for magnet in arrangement} == set(range(1, grains.grain_count + 1))


def test_cuboidize_3d_voronoi_preserves_volume():
    grains = generate_voronoi_grains([4, 3, 2], grain_size=2, cell_size=1, origin=[-2, -1, 4], seed=8)
    arrangement = Arrangement.from_voronoi_grains(grains, [0, 0, 1e5])

    assert sum(magnet.volume for magnet in arrangement) == pytest.approx(24)
    np.testing.assert_allclose(arrangement.bbox[:, 0], [-2, -1, 4])
    np.testing.assert_allclose(arrangement.bbox[:, 1], [2, 2, 6])
    assert arrangement.valid


def test_voronoi_cuboid_validation():
    grains_2d = generate_voronoi_grains([2, 2], 1, 1, seed=1)
    grains_3d = generate_voronoi_grains([2, 2, 2], 1, 1, seed=1)
    with pytest.raises(ValueError, match="thickness"):
        Arrangement.from_voronoi_grains(grains_2d, [1, 2, 3])
    with pytest.raises(ValueError, match="omitted"):
        Arrangement.from_voronoi_grains(grains_3d, [1, 2, 3], thickness=1)
    with pytest.raises(ValueError, match="one finite vector"):
        Arrangement.from_voronoi_grains(grains_2d, np.ones((2, 3)), thickness=1)
    with pytest.raises(TypeError, match="VoronoiGrains"):
        Arrangement.from_voronoi_grains(np.ones((2, 2)), [1, 2, 3], thickness=1)
