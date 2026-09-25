"""Module for stray-field calculations of a cuboid magnet arrangement.

The used equations for magnetic field calculations are based and got derived
from the paper of Ravaud and Lemarquand (Nov. 2009): "Magnetic Field Produced by
a Parallelepipedic Magnet of Various and Uniform Polarization", HAL Open Science,
URI: https://hal.science/hal-00430854, PIER 98, 207-219, 2009

(c) 2023 - 2025 Pascal Muster, MIT License
"""

from __future__ import annotations

from numbers import Real
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from mpl_toolkits.axes_grid1 import make_axes_locatable

from microcubed.utils import range2array

AxisName = Literal["x", "y", "z"]
RangeSpec = Real | tuple[float, float, int] | np.ndarray | list[float]
Component = AxisName | int | tuple[AxisName | int, AxisName | int] | Literal["magnitude"] | None

_AXES = ("x", "y", "z")


def _coordinate_arrays(x: RangeSpec, y: RangeSpec, z: RangeSpec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates = []
    for name, specification in zip(_AXES, (x, y, z)):
        values = np.atleast_1d(range2array(specification)).astype(float, copy=False)
        if values.ndim != 1 or values.size == 0:
            raise ValueError(f"{name} must describe a non-empty one-dimensional range")
        coordinates.append(values)
    return tuple(coordinates)


def _axis_index(axis: AxisName | int) -> int:
    if isinstance(axis, str):
        try:
            return _AXES.index(axis.lower())
        except ValueError as error:
            raise ValueError(f"Unknown component {axis!r}; expected x, y or z") from error
    if axis not in range(3):
        raise ValueError("Component index must be 0, 1 or 2")
    return axis


def sample_field(
    source,
    *,
    x: RangeSpec,
    y: RangeSpec,
    z: RangeSpec,
    what: str = "Bfield",
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray], np.ndarray]:
    """Evaluate a field on an x/y/z section.

    A scalar fixes an axis; ``(start, stop, count)`` or a one-dimensional
    array samples it. The returned field retains all three spatial axes,
    including length-one axes, which makes sections unambiguous.

    Parameters
    ----------
    source : Magnet or Arrangement
        Object providing the field method named by ``what``.
    x, y, z : float, tuple of float and int, or numpy.ndarray
        Fixed coordinate, uniform range, or explicit sample coordinates.
    what : str, default="Bfield"
        Field method to evaluate, such as ``"Bfield"`` or ``"dBfield"``.

    Returns
    -------
    tuple
        Cartesian coordinate arrays followed by a field array whose trailing
        three dimensions correspond to x, y, and z.
    """
    coordinates = _coordinate_arrays(x, y, z)
    meshes = np.meshgrid(*coordinates, indexing="ij")
    points = np.stack([mesh.ravel() for mesh in meshes])
    field = np.asarray(getattr(source, what)(points))
    return coordinates, field.reshape(*field.shape[:-1], *(len(values) for values in coordinates))


def _check_dimension(coordinates: tuple[np.ndarray, ...], dimension: int) -> list[int]:
    varying = [index for index, values in enumerate(coordinates) if values.size > 1]
    if len(varying) != dimension:
        raise ValueError(f"Expected exactly {dimension} varying coordinate range(s), got {len(varying)}")
    return varying


def _component_values(field: np.ndarray, component: Component, spatial_dimensions: int) -> tuple[np.ndarray, str]:
    field_rank = field.ndim - spatial_dimensions
    if component in (None, "magnitude"):
        axes = tuple(range(field_rank))
        return np.linalg.norm(field, axis=axes), "magnitude"
    if field_rank == 1:
        index = _axis_index(component)  # type: ignore[arg-type]
        return field[index], _AXES[index]
    if field_rank == 2 and isinstance(component, tuple) and len(component) == 2:
        derivative, field_axis = (_axis_index(item) for item in component)
        return field[derivative, field_axis], f"d{_AXES[field_axis]}/d{_AXES[derivative]}"
    raise ValueError("Gradient components must be selected as a pair, e.g. component=('x', 'z')")


def plot_1d(
    source,
    *,
    x: RangeSpec,
    y: RangeSpec,
    z: RangeSpec,
    what: str = "Bfield",
    component: Component = None,
    ax: Axes | None = None,
    **plot_kwargs,
) -> tuple[Figure, Axes]:
    """Plot a field along one varying coordinate section.

    Parameters
    ----------
    source : Magnet or Arrangement
        Field source to sample.
    x, y, z : float, tuple, or numpy.ndarray
        Exactly one coordinate must vary.
    what : str, default="Bfield"
        Field method to evaluate.
    component : str, int, tuple, or None, default=None
        Vector component, gradient component pair, or magnitude.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw into.

    Returns
    -------
    tuple of matplotlib.figure.Figure and matplotlib.axes.Axes
        Figure and axes containing the line plot.
    """
    coordinates, field = sample_field(source, x=x, y=y, z=z, what=what)
    (axis,) = _check_dimension(coordinates, 1)
    values, label = _component_values(field, component, spatial_dimensions=3)
    figure, axes = (ax.figure, ax) if ax is not None else plt.subplots()
    axes.plot(coordinates[axis], np.squeeze(values), **plot_kwargs)
    axes.set(xlabel=_AXES[axis], ylabel=f"{what} {label}")
    return figure, axes


def plot_2d(
    source,
    *,
    x: RangeSpec,
    y: RangeSpec,
    z: RangeSpec,
    what: str = "Bfield",
    component: Component = None,
    ax: Axes | None = None,
    colorbar: bool = True,
    **mesh_kwargs,
) -> tuple[Figure, Axes]:
    """Plot a scalar field component on a two-dimensional section.

    Parameters
    ----------
    source : Magnet or Arrangement
        Field source to sample.
    x, y, z : float, tuple, or numpy.ndarray
        Exactly two coordinates must vary.
    what : str, default="Bfield"
        Field method to evaluate.
    component : str, int, tuple, or None, default=None
        Vector component, gradient component pair, or magnitude.
    ax : matplotlib.axes.Axes, optional
        Existing axes to draw into.
    colorbar : bool, default=True
        Add a colorbar for the sampled values.

    Returns
    -------
    tuple of matplotlib.figure.Figure and matplotlib.axes.Axes
        Figure and axes containing the field map.
    """
    coordinates, field = sample_field(source, x=x, y=y, z=z, what=what)
    horizontal, vertical = _check_dimension(coordinates, 2)
    values, label = _component_values(field, component, spatial_dimensions=3)
    figure, axes = (ax.figure, ax) if ax is not None else plt.subplots()
    mesh_kwargs.setdefault("shading", "auto")
    mesh = axes.pcolormesh(
        coordinates[horizontal],
        coordinates[vertical],
        np.squeeze(values).T,
        **mesh_kwargs,
    )
    axes.set(xlabel=_AXES[horizontal], ylabel=_AXES[vertical], title=f"{what} {label}")
    if colorbar:
        # attach the colorbar via a divider so it always matches the axes height/width,
        # even after aspect ratio changes (e.g. ax.set_aspect("equal"))
        divider = make_axes_locatable(axes)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        figure.colorbar(mesh, cax=cax, label=f"{what} {label}")
    return figure, axes


def plot_3d(
    source,
    *,
    x: RangeSpec,
    y: RangeSpec,
    z: RangeSpec,
    what: str = "Bfield",
    component: Component = None,
    ax: Axes | None = None,
    max_points: int = 2_000,
    cmap: str = "viridis",
    **quiver_kwargs,
) -> tuple[Figure, Axes]:
    """Plot a three-dimensional vector field as magnitude-coloured arrows.

    Parameters
    ----------
    source : Magnet or Arrangement
        Field source to sample.
    x, y, z : float, tuple, or numpy.ndarray
        All three coordinates must vary.
    what : str, default="Bfield"
        Vector field method to evaluate.
    ax : matplotlib.axes.Axes, optional
        Existing three-dimensional axes to draw into.
    max_points : int, default=2000
        Maximum number of arrows after regular subsampling.

    Returns
    -------
    tuple of matplotlib.figure.Figure and matplotlib.axes.Axes
        Figure and axes containing the vector-field plot.
    """
    coordinates, field = sample_field(source, x=x, y=y, z=z, what=what)
    _check_dimension(coordinates, 3)
    if field.ndim != 4:
        raise ValueError("3D quiver plots require a vector field such as Bfield or Hfield")
    if max_points <= 0:
        raise ValueError("max_points must be greater than zero")

    values, label = _component_values(field, component, spatial_dimensions=3)
    meshes = np.meshgrid(*coordinates, indexing="ij")
    flat = [mesh.ravel() for mesh in meshes]
    vectors = [field[index].ravel() for index in range(3)]
    magnitudes = values.ravel()
    step = max(1, int(np.ceil(magnitudes.size / max_points)))
    selection = slice(None, None, step)

    figure = ax.figure if ax is not None else plt.figure()
    axes = ax if ax is not None else figure.add_subplot(projection="3d")
    norm = colors.Normalize(vmin=np.nanmin(magnitudes), vmax=np.nanmax(magnitudes))
    arrow_colors = plt.get_cmap(cmap)(norm(magnitudes[selection]))
    axes.quiver(
        *(array[selection] for array in (*flat, *vectors)),
        colors=arrow_colors,
        **quiver_kwargs,
    )
    axes.set(xlabel="x", ylabel="y", zlabel="z", title=f"{what} {label}")
    scalar_mappable = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    figure.colorbar(scalar_mappable, ax=axes, label=f"{what} {label}", shrink=0.7)
    return figure, axes


__all__ = ["plot_1d", "plot_2d", "plot_3d", "sample_field"]
