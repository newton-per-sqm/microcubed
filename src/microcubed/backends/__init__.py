"""Backend discovery and selection for Microcubed."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Backend:
    """Classes and helpers belonging to one calculation backend."""

    name: str
    Magnet: type
    Arrangement: type

    def cuboidize(self, shape, t, delta, mag, **kwargs):
        return self.Arrangement.from_shape(shape, t, delta, mag, **kwargs)

    def cuboidize_voronoi(self, grains, mag, **kwargs):
        return self.Arrangement.from_voronoi_grains(grains, mag, **kwargs)


def _numpy_backend() -> Backend:
    from microcubed.backends.numpy import NumericArrangement, NumericMagnet

    return Backend("numpy", NumericMagnet, NumericArrangement)


def _rust_backend() -> Backend:
    from microcubed.backends.rust import RustArrangement, RustMagnet

    return Backend("rust", RustMagnet, RustArrangement)


def get_backend(name: str = "auto") -> Backend:
    """Return a backend without changing the process-wide default classes."""
    normalized = name.lower()
    if normalized == "numpy":
        return _numpy_backend()
    if normalized == "rust":
        return _rust_backend()
    if normalized == "auto":
        try:
            return _rust_backend()
        except ImportError:
            return _numpy_backend()
    raise ValueError(f"Unknown backend {name!r}; expected numpy, rust, or auto")


def available_backends() -> tuple[str, ...]:
    """Return the calculation backends importable in the current environment."""
    try:
        _rust_backend()
    except ImportError:
        return ("numpy",)
    return ("numpy", "rust")


__all__ = ["Backend", "available_backends", "get_backend"]
