"""Module for stray-field calculations of a cuboid magnet arrangement.

The used equations for magnetic field calculations are based and got derived
from the paper of Ravaud and Lemarquand (Nov. 2009): "Magnetic Field Produced by
a Parallelepipedic Magnet of Various and Uniform Polarization", HAL Open Science,
URI: https://hal.science/hal-00430854, PIER 98, 207-219, 2009

(c) 2023 - 2025 Pascal Muster, MIT License
"""

import os
from importlib.metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = "microcubed"
    __version__ = version(dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

from microcubed.backends import available_backends, get_backend
from microcubed.geometry import VoronoiGrains, generate_voronoi_grains, generate_voronoi_grains_from_shape

_active_backend = None


def set_backend(name: str = "auto"):
    """Set the implementation used by top-level ``Magnet`` and ``Arrangement``.

    Parameters
    ----------
    name : {"auto", "numpy", "rust"}, default="auto"
        Backend to select. ``"auto"`` chooses Rust when its extension is
        importable and otherwise chooses NumPy.

    Returns
    -------
    microcubed.backends.Backend
        The selected backend namespace.
    """
    global Arrangement, Magnet, _active_backend
    _active_backend = get_backend(name)
    Magnet = _active_backend.Magnet
    Arrangement = _active_backend.Arrangement
    return _active_backend


def backend_name() -> str:
    """Return the name of the active top-level calculation backend.

    Returns
    -------
    str
        Either ``"numpy"`` or ``"rust"``.
    """
    return _active_backend.name


set_backend(os.getenv("MICROCUBED_BACKEND", "auto"))


def cuboidize(shape, t, delta, mag, **kwargs):
    """Approximate an extruded two-dimensional shape with cuboids.

    Parameters
    ----------
    shape : array-like, matplotlib.path.Path, or callable
        Polygon vertices, a Matplotlib path, or a callable Boolean mask.
    t : float
        Extrusion thickness.
    delta : float or tuple of float
        Maximum raster-cell spacing.
    mag : array-like
        Magnetization vector in A/m.

    Returns
    -------
    Arrangement
        A non-overlapping arrangement representing the rasterized shape.
    """
    return Arrangement.from_shape(shape, t, delta, mag, **kwargs)


def cuboidize_voronoi(grains, mag, **kwargs):
    """Convert rasterized Voronoi grains into a compact cuboid arrangement.

    Parameters
    ----------
    grains : VoronoiGrains
        Two- or three-dimensional rasterized grains.
    mag : array-like or callable
        Shared, per-grain, or seed-dependent magnetization in A/m.

    Returns
    -------
    Arrangement
        Cuboids representing the occupied grain cells.
    """
    return Arrangement.from_voronoi_grains(grains, mag, **kwargs)


def sample_field(*args, **kwargs):
    from microcubed.viz import sample_field as _sample_field

    return _sample_field(*args, **kwargs)


def plot_1d(*args, **kwargs):
    from microcubed.viz import plot_1d as _plot_1d

    return _plot_1d(*args, **kwargs)


def plot_2d(*args, **kwargs):
    from microcubed.viz import plot_2d as _plot_2d

    return _plot_2d(*args, **kwargs)


def plot_3d(*args, **kwargs):
    from microcubed.viz import plot_3d as _plot_3d

    return _plot_3d(*args, **kwargs)


__all__ = [
    "Arrangement",
    "Magnet",
    "VoronoiGrains",
    "available_backends",
    "backend_name",
    "cuboidize",
    "cuboidize_voronoi",
    "generate_voronoi_grains",
    "generate_voronoi_grains_from_shape",
    "get_backend",
    "plot_1d",
    "plot_2d",
    "plot_3d",
    "sample_field",
    "set_backend",
]
