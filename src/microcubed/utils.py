"""Module for stray-field calculations of a cuboid magnet arrangement.

The used equations for magnetic field calculations are based and got derived
from the paper of Ravaud and Lemarquand (Nov. 2009): "Magnetic Field Produced by
a Parallelepipedic Magnet of Various and Uniform Polarization", HAL Open Science,
URI: https://hal.science/hal-00430854, PIER 98, 207-219, 2009

(c) 2023 - 2025 Pascal Muster, MIT License
"""

from __future__ import annotations

import numpy as np


def spherical2cartesian(r, theta, phi) -> tuple[np.ndarray]:
    """Converts spherical coordinates to cartesian coordinates.

    Parameters
    ----------

    r : float
        Distance from origin.
    theta: float
        Polar angle.
    phi: float
        Azimuthal angle.

    Returns
    -------
    tuple
        (x, y, z)
        (x-coordinate, y-coordinate, z-coordinate)
    """
    x = r * np.sin(theta) * np.cos(phi)
    y = r * np.sin(theta) * np.sin(phi)
    z = r * np.cos(theta)
    return x, y, z


def cartesian2spherical(x, y, z) -> tuple[np.ndarray]:
    """Converts cartesian coordinates to spherical coordinates.

    Parameters
    ----------

    x : float
        x-coordinate.
    y : float
        y-coordinate.
    z : float
        z-coordinate.

    Returns
    -------
    tuple
        (r, theta, phi)
        (distance from origin, polar angle, azimuthal angle)
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return r, theta, phi


def memsize(array: np.ndarray) -> float:
    """Returns the size of the array in MiB.

    Parameters
    ----------
    array : np.ndarray
        Numpy array.

    Returns
    -------
    int
        Size of the array in kibibytes (kiB).
    """
    return np.asarray(array).nbytes / 1024


def range2array(_range) -> float | np.ndarray:
    """Convert a range description into an array"""
    if isinstance(_range, tuple):
        values = np.linspace(*_range, endpoint=True)
    elif isinstance(_range, (int, float, np.number)):
        values = _range
    elif isinstance(_range, (list, np.ndarray)):
        values = np.asarray(_range)
    else:
        raise TypeError(f"Can't handle type for range {_range}")
    return values
