from __future__ import annotations

import numpy as np
import pytest
from scipy.constants import mu_0

import microcubed.base as base_module
from microcubed import Arrangement, Magnet
from microcubed.base import custom_warning


@pytest.fixture
def magnet() -> Magnet:
    return Magnet([2, 4, 6], [1, 2, 3], [4e5, -2e5, 1e5])


@pytest.fixture
def separated_magnets() -> list[Magnet]:
    return [
        Magnet([2, 2, 2], [0, 0, 0], [1e5, 0, 0]),
        Magnet([2, 2, 2], [2, 0, 0], [0, 1e5, 0]),
        Magnet([2, 2, 2], [4, 0, 0], [0, 0, 1e5]),
    ]


def test_magnet_protocol_and_field_helpers(magnet):
    same = Magnet(magnet.size, magnet.center, magnet.magnetization)
    point = np.array([20.0, -10.0, 30.0])

    assert magnet == same
    assert hash(magnet) == hash(same)
    assert magnet != magnet.moved_by([1, 0, 0])
    np.testing.assert_allclose(magnet.Hfield(point), magnet.Bfield(point) / mu_0)
    np.testing.assert_allclose(magnet.dHfield(point), magnet.dBfield(point) / mu_0)

    assert magnet(0, 0, 20, "B").shape == (3,)
    assert magnet(0, 0, 20, "Bfield").shape == (3,)
    assert magnet((-2, 2, 4), 0, 20, "H").shape == (3, 4)
    assert magnet((-2, 2, 4), 0, 20, "dB").shape == (3, 3, 4)
    assert magnet((-2, 2, 4), 0, 20, "dH").shape == (3, 3, 4)


def test_human_readable_representations_and_warning_format(magnet, separated_magnets):
    magnet_name = type(magnet).__name__
    assert str(magnet).startswith(f"{magnet_name}(size=")
    assert repr(magnet) == str(magnet)
    short = Arrangement(separated_magnets)
    assert magnet_name in str(short)
    assert repr(short) == str(short)

    long = Arrangement([separated_magnets[0].moved_by([3 * index, 0, 0]) for index in range(11)])
    assert str(long) == f"{type(long).__name__}(11 magnets)"
    assert custom_warning("message", RuntimeWarning, "example.py") == "example.py: RuntimeWarning - message\n"


def test_magnet_geometry_and_transformations(magnet):
    expected_bbox = np.array([[0, 2], [0, 4], [0, 6]])
    np.testing.assert_allclose(magnet.bbox, expected_bbox)
    assert magnet.volume == 48

    mirrored = magnet.mirrored(x=10, y=-1, z=4)
    np.testing.assert_allclose(mirrored.center.ravel(), [19, -4, 5])
    np.testing.assert_allclose(magnet.moved_to([0, 0, 0]).center.ravel(), [0, 0, 0])

    touching = magnet.moved_by([2, 0, 0])
    separated = magnet.moved_by([2.01, 0, 0])
    assert not magnet.overlapping(touching)
    assert magnet.touching(touching)
    assert not magnet.touching(separated)

    from_bbox = Magnet.from_bbox([3, 4, 5], [-1, 0, -1], magnet.magnetization)
    np.testing.assert_allclose(from_bbox.size.ravel(), [4, 4, 6])
    np.testing.assert_allclose(from_bbox.center.ravel(), [1, 2, 2])


@pytest.mark.parametrize(
    "plane, dimensions", [(None, 3), ("xy", 2), ("yx", 2), ("yz", 2), ("zy", 2), ("xz", 2), ("zx", 2)]
)
def test_magnet_convex_hulls(magnet, plane, dimensions):
    hull = magnet.chull(plane)
    assert hull.points.shape[1] == dimensions
    points = magnet.chull_points(plane)
    assert points.shape[0] == dimensions
    np.testing.assert_allclose(points[:, 0], points[:, -1])


def test_magnet_rejects_invalid_geometry_operations(magnet):
    with pytest.raises(ValueError, match="not supported"):
        magnet.chull("bad")
    with pytest.raises(TypeError, match="another Magnet"):
        magnet.overlapping(object())
    with pytest.raises(TypeError, match="another Magnet"):
        magnet.touching(object())


def test_arrangement_collection_protocol_and_arithmetic(separated_magnets):
    first = Arrangement(separated_magnets[:2])
    last = Arrangement(separated_magnets[2:])

    assert len(first) == 2
    assert list(iter(first)) == separated_magnets[:2]
    assert first[0] == separated_magnets[0]
    assert len(first[:1]) == 1
    assert first == Arrangement(separated_magnets[:2])
    assert hash(first) == hash(Arrangement(separated_magnets[:2]))

    assert len(first + last) == 3
    assert len(first + separated_magnets[2]) == 3
    assert len(first.__radd__(last)) == 3
    assert len(separated_magnets[2] + first) == 3
    assert len((first + last) - last) == 2
    assert len((first + last) - separated_magnets[1]) == 2

    with pytest.raises(TypeError, match="Indexing"):
        _ = first[1.5]
    with pytest.raises(TypeError, match="Cannot add"):
        _ = first + 1
    with pytest.raises(TypeError, match="Cannot add"):
        first.__radd__(1)
    with pytest.raises(TypeError, match="Cannot subtract"):
        _ = first - 1


def test_arrangement_validation_bounds_and_closed_state(separated_magnets, monkeypatch):
    arrangement = Arrangement(separated_magnets)
    np.testing.assert_allclose(arrangement.center.ravel(), [2, 0, 0])
    np.testing.assert_allclose(arrangement.bbox, [[-1, 5], [-1, 1], [-1, 1]])
    assert arrangement.valid
    assert arrangement.closed

    open_arrangement = Arrangement([separated_magnets[0], separated_magnets[2]])
    assert not open_arrangement.closed

    unchecked = Arrangement(separated_magnets, validate=False)
    assert unchecked.valid

    with pytest.raises(ValueError, match="overlap"):
        Arrangement([separated_magnets[0], separated_magnets[0]])

    calls = []

    def passthrough(iterable, *args, **kwargs):
        calls.append((args, kwargs))
        return iterable

    monkeypatch.setattr(base_module, "tqdm", passthrough)
    monkeypatch.setattr(Arrangement, "tqdm_threshold", 0)
    with_progress = Arrangement(separated_magnets)
    assert with_progress.valid
    assert with_progress.closed
    assert len(calls) == 2


@pytest.mark.parametrize("plane", [None, "xy", "yz", "xz"])
def test_arrangement_hulls_and_transforms(separated_magnets, plane):
    arrangement = Arrangement(separated_magnets)
    hull = arrangement.chull(plane)
    assert hull.points.size > 0
    assert len(arrangement.chull_points(plane, individual=True)) > 0
    assert arrangement.chull_points(plane).shape[-1] > 1

    moved = arrangement.moved_by([1, 2, 3])
    np.testing.assert_allclose(moved.center, arrangement.center + np.array([[1], [2], [3]]))
    np.testing.assert_allclose(arrangement.mirrored(x=0).center.ravel(), [-2, 0, 0])
    np.testing.assert_allclose(arrangement.moved_to([0, 0, 0]).center.ravel(), [0, 0, 0])


def test_arrangement_rejects_unknown_hull_plane(separated_magnets):
    with pytest.raises(ValueError, match="not supported"):
        Arrangement(separated_magnets).chull("bad")


def test_circle_and_ellipse_generators():
    magnetization = [1e5, 0, 0]
    circle = Arrangement.from_circle(2, 1, mag=magnetization, dx=1)
    moved_circle = Arrangement.from_circle(2, 1, center=[3, 4, 5], mag=magnetization, dx=1)
    assert len(circle) == 4
    np.testing.assert_allclose(moved_circle.center.ravel(), [3, 4, 5])

    ellipse_x = Arrangement.from_ellipsis([2, 4, 1], mag=magnetization, dx=1)
    ellipse_y = Arrangement.from_ellipsis([4, 2, 1], center=[1, 2, 3], mag=magnetization, dx=1)
    assert len(ellipse_x) == len(ellipse_y) == 2
    np.testing.assert_allclose(ellipse_y.center.ravel(), [1, 2, 3])


def test_rounded_rectangle_generator_and_validation():
    magnetization = [1e5, 0, 0]
    rounded = Arrangement.from_rounded_rectangle([6, 4, 1], mag=magnetization, edge_radius=1, dx=0.5)
    moved = Arrangement.from_rounded_rectangle([6, 4, 1], center=[1, 2, 3], mag=magnetization, edge_radius=1, dx=0.5)
    assert len(rounded) == 5
    np.testing.assert_allclose(moved.center.ravel(), [1, 2, 3])
    with pytest.raises(ValueError, match="too large"):
        Arrangement.from_rounded_rectangle([4, 2, 1], mag=magnetization, edge_radius=2)


def test_superellipse_generator_and_validation():
    magnetization = [1e5, 0, 0]
    wide = Arrangement.from_superellipsis([2, 4, 1], mag=magnetization, n=4, dx=1)
    long = Arrangement.from_superellipsis([4, 2, 1], center=[1, 0, 0], mag=magnetization, n=4, dx=1)
    assert len(wide) == len(long) == 2
    np.testing.assert_allclose(long.center.ravel(), [1, 0, 0])
    with pytest.raises(ValueError, match="greater than 0"):
        Arrangement.from_superellipsis([4, 2, 1], mag=magnetization, n=0)


@pytest.mark.parametrize("size", [[2, 4, 1], [4, 2, 1]])
@pytest.mark.parametrize("corner", ["lb", "rb", "rt", "lt"])
def test_right_triangle_generator_orientations(size, corner):
    triangle = Arrangement.from_right_triangle(size, center=[1, 2, 3], mag=[1e5, 0, 0], corner=corner, dx=1)
    np.testing.assert_allclose(triangle.center.ravel(), [1, 2, 3])
    assert len(triangle) == 2


def test_right_triangle_rejects_unknown_corner():
    with pytest.raises(ValueError, match="not recognized"):
        Arrangement.from_right_triangle([2, 4, 1], mag=[1e5, 0, 0], corner="north", dx=1)
