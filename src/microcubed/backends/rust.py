"""Rust-backed implementations of the Microcubed numerical API."""

from __future__ import annotations

import numpy as np

from microcubed.backends.equations import Bfield as numpy_bfield
from microcubed.backends.equations import dBfield as numpy_dbfield
from microcubed.backends.numpy import NumericArrangement, NumericMagnet, numpy_cache

try:
    from microcubed._rust import RustyArrangement, RustyMagnet
except ImportError as error:  # pragma: no cover - exercised only without the optional package
    raise ImportError(
        "The compiled Microcubed Rust extension is unavailable. Install a binary "
        "Microcubed wheel or build the project with maturin."
    ) from error


def _raw_magnet(magnet: NumericMagnet) -> RustyMagnet:
    return RustyMagnet(
        magnet.size.ravel().tolist(),
        magnet.center.ravel().tolist(),
        magnet.magnetization.ravel().tolist(),
    )


class RustMagnet(NumericMagnet):
    """Cuboid magnet evaluated by the integrated parallel Rust field kernel.

    Parameters
    ----------
    size, center, magnetization : array-like
        Three Cartesian components in the project's length unit and A/m.
    """

    backend = "rust"

    def __init__(self, size, center, magnetization):
        super().__init__(size, center, magnetization)
        self._rust = _raw_magnet(self)

    def Bfield(self, points: np.ndarray) -> np.ndarray:
        """Evaluate the exterior magnetic flux density.

        Parameters
        ----------
        points : array-like
            Observation points with shape ``(3, N)``.

        Returns
        -------
        numpy.ndarray
            Flux density ``(Bx, By, Bz)`` in tesla with shape ``(3, N)``.
        """
        prepared = self.prepare_points(points)
        result = np.asarray(self._rust.bfield(prepared))
        finite_input = np.all(np.isfinite(prepared), axis=0)
        fallback = finite_input & ~np.all(np.isfinite(result), axis=0)
        if np.any(fallback):
            result[:, fallback] = numpy_bfield(
                self.size,
                self.center,
                self.polarization,
                prepared[:, fallback],
            )
        return result

    def dBfield(self, points: np.ndarray) -> np.ndarray:
        """Evaluate the exterior magnetic-flux-density gradient.

        Parameters
        ----------
        points : array-like
            Observation points with shape ``(3, N)``.

        Returns
        -------
        numpy.ndarray
            Gradient with shape ``(3, 3, N)``. The first axis is the
            derivative direction and the second is the field component.
        """
        prepared = self.prepare_points(points)
        result = np.asarray(self._rust.dbfield(prepared)).reshape(3, 3, -1)
        finite_input = np.all(np.isfinite(prepared), axis=0)
        fallback = finite_input & ~np.all(np.isfinite(result), axis=(0, 1))
        if np.any(fallback):
            result[:, :, fallback] = numpy_dbfield(
                self.size,
                self.center,
                self.polarization,
                prepared[:, fallback],
            )
        return result


class RustArrangement(NumericArrangement):
    """Arrangement summed by the integrated parallel Rust field kernel.

    Parameters
    ----------
    magnets : sequence of Magnet, optional
        Cuboids included in the superposition.
    validate : bool, default=True
        Check that cuboids do not overlap.
    """

    Magnet = RustMagnet
    backend = "rust"

    def __init__(self, magnets=None, validate: bool = True):
        super().__init__(magnets, validate=validate)
        self._rust = RustyArrangement(
            [magnet._rust if isinstance(magnet, RustMagnet) else _raw_magnet(magnet) for magnet in self]
        )

    @numpy_cache(maxsize=20)
    def Bfield(self, points: np.ndarray) -> np.ndarray:
        """Evaluate the summed magnetic flux density at observation points.

        Parameters
        ----------
        points : array-like
            Observation points with shape ``(3, N)``.

        Returns
        -------
        numpy.ndarray
            Flux density in tesla with shape ``(3, N)``.
        """
        prepared = self.prepare_points(points)
        result = np.asarray(self._rust.bfield(prepared))
        fallback = ~np.all(np.isfinite(result), axis=0)
        if np.any(fallback):
            result[:, fallback] = sum(
                numpy_bfield(magnet.size, magnet.center, magnet.polarization, prepared[:, fallback]) for magnet in self
            )
        return result

    @numpy_cache(maxsize=20)
    def dBfield(self, points: np.ndarray) -> np.ndarray:
        """Evaluate the summed magnetic-flux-density gradient.

        Parameters
        ----------
        points : array-like
            Observation points with shape ``(3, N)``.

        Returns
        -------
        numpy.ndarray
            Gradient in field units per length with shape ``(3, 3, N)``.
        """
        prepared = self.prepare_points(points)
        result = np.asarray(self._rust.dbfield(prepared)).reshape(3, 3, -1)
        fallback = ~np.all(np.isfinite(result), axis=(0, 1))
        if np.any(fallback):
            result[:, :, fallback] = sum(
                numpy_dbfield(magnet.size, magnet.center, magnet.polarization, prepared[:, fallback]) for magnet in self
            )
        return result


__all__ = ["RustArrangement", "RustMagnet"]
