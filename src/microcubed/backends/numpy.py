"""Module for stray-field calculations of a cuboid magnet arrangement.

The used equations for magnetic field calculations are based and got derived
from the paper of Ravaud and Lemarquand (Nov. 2009): "Magnetic Field Produced by
a Parallelepipedic Magnet of Various and Uniform Polarization", HAL Open Science,
URI: https://hal.science/hal-00430854, PIER 98, 207-219, 2009

(c) 2023 - 2025 Pascal Muster, MIT License
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache, wraps
from hashlib import sha256

import numpy as np

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is an optional progress enhancement

    def tqdm(iterable, *args, **kwargs):
        return iterable


from microcubed.backends.equations import Bfield, dBfield
from microcubed.base import BasicArrangement, BasicMagnet


class HashableArray:
    """Class for hashing numpy arrays. Used for lru caching."""

    def __init__(self, array: np.ndarray) -> None:
        self.values = array
        contiguous = np.ascontiguousarray(array)
        digest = sha256(contiguous.view(np.uint8)).hexdigest()
        self.hashed = (array.shape, array.dtype.str, digest)

    def __hash__(self) -> int:
        return hash(self.hashed)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, HashableArray) and other.hashed == self.hashed


def numpy_cache(*args, **kwargs):
    """Create a cache wrapper for numpy ndarrays.

    Parameters
    ----------
    See functools.lru_cache for more information.

    Returns
    -------
    Callable
        Wrapper function for caching numpy ndarrays.
        As cache input, one numpy.ndarray is foreseen!
    """

    def outer_wrapper(function: Callable):
        """Wrapper for caching numpy inputs.

        Parameters
        ----------
        function : Callable
            The function to be evaluated and cached.
        """

        @lru_cache(*args, **kwargs)
        def cached_wrapper(
            instance: object,
            shell: HashableArray,
            function_args: tuple,
            function_kwargs: tuple,
        ) -> np.ndarray:
            """This function is calling the real calculation function.
            Its output is cached by the lru_cache decorator.

            Parameters
            ----------
            instance : object
                Instance of the class the function is called from.
            shell : HashableArray
                Hashable array input to be calculated and cached.

            Returns
            -------
            np.ndarray
                Output of the function.
            """
            return function(instance, shell.values, *function_args, **dict(function_kwargs))

        @wraps(function)
        def inner_wrapper(instance: object, array: np.ndarray, *function_args, **function_kwargs) -> Callable:
            """This function turns the array into an hashable object.
            This is necessary for making numpy inputs cacheable.

            Parameters
            ----------
            instance : object
                Instance of the class the function is called from.
            array : np.ndarray
                Array input to be hashed, calculated and cached.

            Returns
            -------
            Callable
                The cached wrapper callable.
            """
            shell = HashableArray(np.asarray(array, dtype=np.double).reshape(3, -1))
            return cached_wrapper(instance, shell, function_args, tuple(sorted(function_kwargs.items())))

        return inner_wrapper

    return outer_wrapper


class NumericMagnet(BasicMagnet):
    """Class for basic numerical calculations of the stray field of a cuboid magnet."""

    backend = "numpy"

    def prepare_points(self, points: np.ndarray | list):
        """Check and reshape point array

        If some points are inside the cuboid magnet, mask them with np.nan.
        """
        points = np.asarray(points, dtype=np.double).reshape(3, -1)
        inside_magnet = np.all(np.abs(points - self.center) <= self.size / 2, axis=0)
        return np.where(inside_magnet, np.nan, points)  # mask points inside the magnet

    def Bfield(self, points: np.ndarray) -> np.ndarray:
        """Calculate the stray field components of a cuboid magnet.

        Notes
        ---------
        Equations taken from the paper of R. Ravaud et al (2009).
        It's assumed the surrounding medium is vacuum/air with µr = 1.

        Parameters
        ----------
        points : np.ndarray
            Array of points to calculate the stray field at.
            Shape should be equivalent to (3, N).

        Returns
        -------
        np.ndarray
            Magnetic field [Bx, By, Bz] with shape (3, N).
        """
        # B, shape (3, N) [(Bx,By,Bz), N]
        return Bfield(self.size, self.center, self.polarization, self.prepare_points(points))

    def dBfield(self, points: np.ndarray) -> np.ndarray:
        """Calculate all derivatives of the magnetic field of a cuboid magnet.

        Notes
        ---------
        Equations derived from the paper of R. Ravaud et al (2009).
        It's assumed the surrounding medium is vacuum/air with µr = 1.

        Parameters
        ----------
        points : np.ndarray
            Array of points to calculate the stray field at.z
            Shape should be equivalent to (3, N).

        Returns
        -------
        np.ndarray
            All derivatives d(x,y,z)[Bx, By, Bz] of the magnetic field components.
            Has shape (3, 3, N): [(dx, dy, dz), (Bx, By, Bz), N].
        """
        return dBfield(self.size, self.center, self.polarization, self.prepare_points(points))


class NumericArrangement(BasicArrangement):
    """Class for basic numerical calculations of the stray field of
    a cuboid magnet arrangement.
    """

    Magnet = NumericMagnet
    """Corresponding magnet class."""

    backend = "numpy"

    tqdm_threshold = 20
    """Threshold for using tqdm progress bar."""

    use_tqdm = True
    """Toggle tqdm progress bar on/off for all class instances."""

    leave_tqdm = False
    """Whether to leave the tqdm progress bar after finishing the calculation."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # if we prepare the point array in the arrangement class
        # we can save some time by not preparing the points in the magnet class
        self.use_tqdm &= len(self) >= self.tqdm_threshold

    def prepare_points(self, points: np.ndarray | list):
        """Reshape point array without checking if p is inside any cuboid magnet!"""
        return np.asarray(points, dtype=np.double).reshape(3, -1)

    @numpy_cache(maxsize=20)
    def Bfield(self, points: np.ndarray) -> np.ndarray:
        """Calculate the magnetic field at the given points.

        Parameters
        ----------
        p : np.ndarray
            Array of points in shape (3, N) [auto reshaped]

        Returns
        -------
        np.ndarray
            Array of magnetic field components in shape (3, N)
        """
        points = self.prepare_points(points)
        result = np.zeros((3, points.shape[1]), dtype=np.double)
        for magnet in tqdm(
            self,
            desc=f"Calculating B(r) [for {points.shape[1]:g} points]",
            leave=self.leave_tqdm,
            disable=(not self.use_tqdm),
        ):
            result += Bfield(magnet.size, magnet.center, magnet.polarization, points)
        return result

    @numpy_cache(maxsize=20)
    def dBfield(self, points: np.ndarray) -> np.ndarray:
        """Calculate the magnetic field gradients at the given points.

        Parameters
        ----------
        p : np.ndarray
            Array of points in shape (3, N) [auto reshaped]

        Returns
        -------
        np.ndarray
            Array of the magnetic field gradient components in shape (3, 3, N)
        """
        points = self.prepare_points(points)
        result = np.zeros((3, 3, points.shape[1]), dtype=np.double)
        for magnet in tqdm(
            self,
            desc=f"Calculating dB(r) [for {points.shape[1]:g} points]",
            leave=self.leave_tqdm,
            disable=(not self.use_tqdm),
        ):
            result += dBfield(magnet.size, magnet.center, magnet.polarization, points)
        return result
