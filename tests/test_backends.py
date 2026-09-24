from __future__ import annotations

import numpy as np
import pytest

import microcubed
import microcubed.backends as backend_module


@pytest.fixture
def backends():
    return microcubed.get_backend("numpy"), microcubed.get_backend("rust")


def test_backend_discovery_names_and_errors(backends):
    numpy_backend, rust_backend = backends
    assert microcubed.get_backend("rust").name == "rust"
    assert microcubed.get_backend("auto").name == "rust"
    assert rust_backend.name == "rust"
    assert numpy_backend.Magnet.backend == "numpy"
    assert numpy_backend.Arrangement.backend == "numpy"
    assert microcubed.available_backends() == ("numpy", "rust")
    for unknown in ("gpu", "python", "nanocubed"):
        with pytest.raises(ValueError, match="Unknown backend"):
            microcubed.get_backend(unknown)


def test_auto_and_discovery_fall_back_when_rust_is_unavailable(monkeypatch):
    def unavailable():
        raise ImportError("missing")

    monkeypatch.setattr(backend_module, "_rust_backend", unavailable)
    assert backend_module.get_backend("auto").name == "numpy"
    assert backend_module.available_backends() == ("numpy",)
    with pytest.raises(ImportError, match="missing"):
        backend_module.get_backend("rust")


def test_process_wide_backend_selection_is_reversible():
    original = microcubed.backend_name()
    original_magnet, original_arrangement = microcubed.Magnet, microcubed.Arrangement
    try:
        selected = microcubed.set_backend("numpy" if original == "rust" else "rust")
        assert selected.name != original
        assert microcubed.backend_name() != original
    finally:
        microcubed.set_backend(original)
    assert microcubed.Magnet is original_magnet
    assert microcubed.Arrangement is original_arrangement


def test_rust_magnet_matches_numpy_api_and_values(backends):
    numpy_backend, rust_backend = backends
    parameters = ([2.0, 3.0, 4.0], [0.2, -0.3, 0.4], [1e5, -2e5, 3e5])
    numpy_magnet = numpy_backend.Magnet(*parameters)
    rust_magnet = rust_backend.Magnet(*parameters)
    rng = np.random.default_rng(42)
    points = rng.uniform(5, 20, (3, 200))

    assert rust_magnet.backend == "rust"
    np.testing.assert_allclose(rust_magnet.size, numpy_magnet.size)
    np.testing.assert_allclose(rust_magnet.center, numpy_magnet.center)
    np.testing.assert_allclose(rust_magnet.magnetization, numpy_magnet.magnetization)
    np.testing.assert_allclose(rust_magnet.bbox, numpy_magnet.bbox)
    np.testing.assert_allclose(rust_magnet.corners, numpy_magnet.corners)
    np.testing.assert_allclose(rust_magnet.Bfield(points), numpy_magnet.Bfield(points), rtol=1e-9, atol=1e-13)
    np.testing.assert_allclose(rust_magnet.dBfield(points), numpy_magnet.dBfield(points), rtol=1e-9, atol=1e-13)
    np.testing.assert_allclose(rust_magnet.Hfield(points), numpy_magnet.Hfield(points), rtol=1e-9, atol=1e-7)
    np.testing.assert_allclose(rust_magnet.dHfield(points), numpy_magnet.dHfield(points), rtol=1e-9, atol=1e-7)


def test_rust_magnet_masks_interior_and_stabilizes_edge_lines(backends):
    numpy_backend, rust_backend = backends
    numpy_magnet = numpy_backend.Magnet([2, 2, 2], [0, 0, 0], [1e5, 2e5, 3e5])
    rust_magnet = rust_backend.Magnet([2, 2, 2], [0, 0, 0], [1e5, 2e5, 3e5])
    points = np.array([[0, 1, 1, 1], [0, 1, 1, -1], [0, -2, 2, 2]], dtype=float)

    assert np.isnan(rust_magnet.Bfield(points)[:, 0]).all()
    assert np.isnan(rust_magnet.dBfield(points)[:, :, 0]).all()
    np.testing.assert_allclose(rust_magnet.Bfield(points[:, 1:]), numpy_magnet.Bfield(points[:, 1:]))
    np.testing.assert_allclose(rust_magnet.dBfield(points[:, 1:]), numpy_magnet.dBfield(points[:, 1:]))
    assert np.isfinite(rust_magnet.Bfield(points[:, 1:])).all()
    assert np.isfinite(rust_magnet.dBfield(points[:, 1:])).all()


def test_rust_arrangement_matches_numpy_and_supports_mixed_magnets(backends):
    numpy_backend, rust_backend = backends
    specs = [
        ([2, 3, 4], [-3, 0, 0], [1e5, 2e5, 0]),
        ([2, 2, 2], [3, 0, 0], [-1e5, 0, 3e5]),
    ]
    numpy_magnets = [numpy_backend.Magnet(*spec) for spec in specs]
    rust_magnets = [rust_backend.Magnet(*spec) for spec in specs]
    numpy_arrangement = numpy_backend.Arrangement(numpy_magnets)
    rust_arrangement = rust_backend.Arrangement(rust_magnets)
    mixed_arrangement = rust_backend.Arrangement(numpy_magnets)
    points = np.array([[0, 1, -1], [0, 5, -5], [10, 20, 30]], dtype=float)

    assert len(rust_arrangement) == len(numpy_arrangement)
    np.testing.assert_allclose(rust_arrangement.bbox, numpy_arrangement.bbox)
    for candidate in (rust_arrangement, mixed_arrangement):
        np.testing.assert_allclose(candidate.Bfield(points), numpy_arrangement.Bfield(points), rtol=1e-9, atol=1e-13)
        np.testing.assert_allclose(candidate.dBfield(points), numpy_arrangement.dBfield(points), rtol=1e-9, atol=1e-13)
        assert candidate.Bfield(points.copy()) is candidate.Bfield(points)


def test_rust_arrangement_fallback_and_empty_shapes(backends):
    numpy_backend, rust_backend = backends
    specs = [([2, 2, 2], [0, 0, 0], [1e5, 2e5, 3e5])]
    numpy_arrangement = numpy_backend.Arrangement([numpy_backend.Magnet(*specs[0])])
    rust_arrangement = rust_backend.Arrangement([rust_backend.Magnet(*specs[0])])
    edge_points = np.array([[1, 1, 1], [1, 1, -1], [-2, 2, 2]], dtype=float)

    np.testing.assert_allclose(rust_arrangement.Bfield(edge_points), numpy_arrangement.Bfield(edge_points))
    np.testing.assert_allclose(rust_arrangement.dBfield(edge_points), numpy_arrangement.dBfield(edge_points))
    assert np.isfinite(rust_arrangement.Bfield(edge_points)).all()
    assert np.isfinite(rust_arrangement.dBfield(edge_points)).all()

    empty = rust_backend.Arrangement([], validate=False)
    np.testing.assert_array_equal(empty.Bfield([[1], [2], [3]]), np.zeros((3, 1)))
    np.testing.assert_array_equal(empty.dBfield([[1], [2], [3]]), np.zeros((3, 3, 1)))


def test_backend_cuboidize_and_visualization_api(backends):
    _, rust_backend = backends
    polygon = [[0, 0], [3, 0], [3, 1], [1, 1], [1, 3], [0, 3]]
    arrangement = rust_backend.cuboidize(polygon, 2, 1, [0, 0, 1e5])
    assert isinstance(arrangement, rust_backend.Arrangement)
    assert len(arrangement) == 2
    assert np.isfinite(arrangement.Bfield([5, 5, 5])).all()

    direct = rust_backend.cuboidize(polygon, 2, 1, [0, 0, 1e5])
    coordinates, field = microcubed.sample_field(direct, x=(-2, 4, 3), y=0, z=-5)
    assert tuple(len(axis) for axis in coordinates) == (3, 1, 1)
    assert field.shape == (3, 3, 1, 1)


def test_rust_transformations_generators_and_call_syntax(backends):
    _, rust_backend = backends
    magnet = rust_backend.Magnet([2, 2, 2], [1, 2, 3], [1e5, 0, 0])
    moved = magnet.moved_by([1, 0, 0])
    mirrored = magnet.mirrored(x=0, y=0, z=0)
    assert isinstance(moved, rust_backend.Magnet)
    assert isinstance(mirrored, rust_backend.Magnet)
    np.testing.assert_allclose(mirrored.center.ravel(), [-1, -2, -3])
    assert magnet((-2, 2, 3), 0, -5, "B").shape == (3, 3)
    assert magnet((-2, 2, 3), 0, -5, "dB").shape == (3, 3, 3)

    ellipse = rust_backend.Arrangement.from_ellipsis([4, 2, 1], mag=[0, 1e5, 0], dx=1)
    assert isinstance(ellipse, rust_backend.Arrangement)
    assert all(isinstance(item, rust_backend.Magnet) for item in ellipse)
