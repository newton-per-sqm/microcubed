from __future__ import annotations

import numpy as np

from microcubed import Arrangement, Magnet
from microcubed.backends.numpy import HashableArray, numpy_cache


def test_hashable_array_identity_includes_shape_dtype_and_values():
    base = HashableArray(np.array([[1.0, 2.0]]))
    equal = HashableArray(np.array([[1.0, 2.0]]))
    transposed = HashableArray(np.array([[1.0], [2.0]]))
    integer = HashableArray(np.array([[1, 2]]))

    assert base == equal
    assert hash(base) == hash(equal)
    assert base != transposed
    assert base != integer
    assert base != object()


def test_numpy_cache_reuses_equal_arrays_and_distinguishes_arguments():
    class Calculator:
        def __init__(self):
            self.calls = 0

        @numpy_cache(maxsize=4)
        def calculate(self, array, scale=1, *, offset=0):
            self.calls += 1
            return array * scale + offset

    calculator = Calculator()
    first = calculator.calculate([1, 2, 3], 2, offset=1)
    second = calculator.calculate(np.array([1.0, 2.0, 3.0]), 2, offset=1)
    third = calculator.calculate([1, 2, 3], 3, offset=1)

    np.testing.assert_allclose(first.ravel(), [3, 5, 7])
    assert second is first
    assert calculator.calls == 2
    assert not np.array_equal(first, third)


def test_numeric_magnet_masks_interior_and_boundary_points():
    magnet = Magnet([2, 2, 2], [0, 0, 0], [1e5, 2e5, 3e5])
    points = np.array([[0, 1, 1.01], [0, 0, 0], [0, 0, 0]])

    prepared = magnet.prepare_points(points)
    assert np.isnan(prepared[:, :2]).all()
    assert np.isfinite(prepared[:, 2]).all()
    assert np.isnan(magnet.Bfield(points)[:, :2]).all()
    assert np.isnan(magnet.dBfield(points)[:, :, :2]).all()
    assert np.isfinite(magnet.Bfield(points)[:, 2]).all()


def test_arrangement_field_is_superposition_and_cache_is_content_based():
    magnets = [
        Magnet([2, 2, 2], [-2, 0, 0], [1e5, 0, 0]),
        Magnet([2, 2, 2], [2, 0, 0], [0, 2e5, 0]),
    ]
    arrangement = Arrangement(magnets)
    points = np.array([[0.0, 0.5], [0, 0], [5, 6]])

    expected_field = sum(magnet.Bfield(points) for magnet in magnets)
    expected_gradient = sum(magnet.dBfield(points) for magnet in magnets)
    field = arrangement.Bfield(points)
    gradient = arrangement.dBfield(points)
    np.testing.assert_allclose(field, expected_field)
    np.testing.assert_allclose(gradient, expected_gradient)
    assert arrangement.Bfield(points.copy()) is field
    assert arrangement.dBfield(points.copy()) is gradient
    assert arrangement.prepare_points([1, 2, 3]).shape == (3, 1)
