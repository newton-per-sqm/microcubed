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
    """Microcubed magnet using the integrated Rust field kernel."""

    backend = "rust"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rust = _raw_magnet(self)

    def Bfield(self, points: np.ndarray) -> np.ndarray:
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
    """Microcubed arrangement summed by the integrated Rust kernel."""

    Magnet = RustMagnet
    backend = "rust"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rust = RustyArrangement(
            [magnet._rust if isinstance(magnet, RustMagnet) else _raw_magnet(magnet) for magnet in self]
        )

    @numpy_cache(maxsize=20)
    def Bfield(self, points: np.ndarray) -> np.ndarray:
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
        prepared = self.prepare_points(points)
        result = np.asarray(self._rust.dbfield(prepared)).reshape(3, 3, -1)
        fallback = ~np.all(np.isfinite(result), axis=(0, 1))
        if np.any(fallback):
            result[:, :, fallback] = sum(
                numpy_dbfield(magnet.size, magnet.center, magnet.polarization, prepared[:, fallback]) for magnet in self
            )
        return result


__all__ = ["RustArrangement", "RustMagnet"]
