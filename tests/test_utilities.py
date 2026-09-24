import numpy as np
import pytest

from microcubed.utils import cartesian2spherical, memsize, range2array, spherical2cartesian


def test_spherical2cartesian():
    # z axis is the origin for both angles: polar and azimuthal
    assert np.allclose(spherical2cartesian(1, 0, 0), [0, 0, 1])
    # polar angle is changing mostly the z coordinate
    assert np.allclose(spherical2cartesian(1, np.pi, 0), [0, 0, -1])
    # azimuthal angle is changing only the x and y coordinate
    assert np.allclose(spherical2cartesian(1, 0, np.pi), [0, 0, 1])
    assert np.allclose(spherical2cartesian(1, np.pi, np.pi), [0, 0, -1])
    # polar angle pi/2 is leveling the vector in the xy plane
    assert np.allclose(spherical2cartesian(1, np.pi / 2, 0), [1, 0, 0])
    assert np.allclose(spherical2cartesian(1, np.pi / 2, np.pi / 2), [0, 1, 0])
    assert np.allclose(spherical2cartesian(1, np.pi / 2, np.pi), [-1, 0, 0])
    assert np.allclose(spherical2cartesian(1, np.pi / 2, -np.pi / 2), [0, -1, 0])


def test_cartesian2spherical():
    # z axis is the origin for both angles: polar and azimuthal
    assert np.allclose(cartesian2spherical(0, 0, 1), [1, 0, 0])
    # polar angle is changing mostly the z coordinate
    assert np.allclose(cartesian2spherical(0, 0, -1), [1, np.pi, 0])
    # polar angle pi/2 is leveling the vector in the xy plane
    assert np.allclose(cartesian2spherical(1, 0, 0), [1, np.pi / 2, 0])
    assert np.allclose(cartesian2spherical(0, 1, 0), [1, np.pi / 2, np.pi / 2])
    assert np.allclose(cartesian2spherical(-1, 0, 0), [1, np.pi / 2, np.pi])
    assert np.allclose(cartesian2spherical(0, -1, 0), [1, np.pi / 2, -np.pi / 2])


def test_memsize_reports_kibibytes():
    array = np.zeros(256, dtype=np.float64)
    assert memsize(array) == 2
    assert memsize([1, 2, 3]) == np.asarray([1, 2, 3]).nbytes / 1024


def test_range2array_accepts_all_documented_specs():
    np.testing.assert_allclose(range2array((0, 1, 3)), [0, 0.5, 1])
    assert range2array(2) == 2
    assert range2array(np.float64(2.5)) == 2.5
    np.testing.assert_array_equal(range2array([1, 2]), [1, 2])
    array = np.array([3, 4])
    np.testing.assert_array_equal(range2array(array), array)
    with pytest.raises(TypeError, match="Can't handle"):
        range2array({1, 2})
