import numpy as np
import pytest
from scipy.constants import mu_0

from microcubed import Magnet


@pytest.fixture
def cube(size=None, center=None, mag=None):
    size = size or [1, 1, 1]
    center = center or [0, 0, 0]
    mag = mag or [1e6, 0, 0]
    return Magnet(size, center, mag)


def test_initialization():
    size = [1, 1, 1]
    nsize = [-1, -1, -1]
    center = [0, 0, 0]
    mag = [1e6, 0, 0]

    with pytest.raises(ValueError):
        Magnet(nsize, center, mag)

    cube = Magnet(size, center, mag)

    size = np.asarray(size).reshape(3, 1)
    center = np.asarray(center).reshape(3, 1)
    mag = np.asarray(mag).reshape(3, 1)

    # position and size
    assert np.allclose(cube.size, size)
    assert np.allclose(cube.center, center)
    assert np.allclose(cube.volume, np.prod(size))
    assert np.allclose(cube.bbox, np.hstack([center - size / 2, center + size / 2]))
    assert np.allclose(cube.moved_to([1, 1, 1]).center, np.asarray([1, 1, 1]).reshape(3, 1))
    assert np.allclose(cube.moved_by([1, 1, 1]).center, np.asarray([1, 1, 1]).reshape(3, 1))

    # magnetization and polarisation
    assert np.allclose(cube.magnetization, mag)
    assert np.allclose(cube.polarization, -np.asarray(mag) * mu_0 / (4 * np.pi))

    # equality
    assert cube == Magnet(size, center, mag)
    assert cube == cube.mirrored(x=0, y=0, z=0)
    assert cube != cube.moved_to([1, 1, 1])

    # touching & overlapping
    assert cube.touching(cube)
    assert cube.overlapping(cube)

    with pytest.raises(TypeError):
        cube.touching(1)

    with pytest.raises(TypeError):
        cube.overlapping(1)

    # corners
    assert np.allclose(
        cube.corners.T,
        np.asarray(
            [
                [0.5, 0.5, 0.5],
                [0.5, 0.5, -0.5],
                [0.5, -0.5, 0.5],
                [0.5, -0.5, -0.5],
                [-0.5, 0.5, 0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [-0.5, -0.5, -0.5],
            ],
        ),
    )
