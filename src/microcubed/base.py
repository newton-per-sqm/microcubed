"""Module for stray-field calculations of a cuboid magnet arrangement.

The used equations for magnetic field calculations are based and got derived
from the paper of Ravaud and Lemarquand (Nov. 2009): "Magnetic Field Produced by
a Parallelepipedic Magnet of Various and Uniform Polarization", HAL Open Science,
URI: https://hal.science/hal-00430854, PIER 98, 207-219, 2009

(c) 2023 - 2025 Pascal Muster, MIT License
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterator
from functools import cached_property
from hashlib import sha256
from itertools import combinations, pairwise, product
from typing import Literal

import numpy as np
from scipy.constants import mu_0
from scipy.spatial import ConvexHull
from scipy.special import comb

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is an optional progress enhancement

    def tqdm(iterable, *args, **kwargs):
        return iterable


from microcubed.utils import range2array


def custom_warning(msg, category, file, *args, **kwargs):
    # ignore everything except the warning message
    return f"{file}: {category.__name__} - {msg!s}\n"


class BasicMagnet(ABC):
    """Class defining a cuboid magnet without any calculation methods."""

    def __init__(
        self,
        size: list | np.ndarray,
        center: list | np.ndarray,
        magnetization: list | np.ndarray,
    ) -> None:
        """Create a new cuboid magnet object.

        Parameters
        ----------
        size : np.ndarray
            Size of the magnet in x, y and z direction.
        center : np.ndarray
            Center of the magnet in x, y and z direction.
        mag : float, optional
            Magnetisation value in x, y and z direction.

        Raises
        ------
        ValueError
            If size is not positive.
        """
        if np.any(np.asarray(size) <= 0):
            raise ValueError("Size has to be positive.")

        self._size = np.asarray(size, order="C").reshape(3, 1)
        self._center = np.asarray(center, order="C").reshape(3, 1)
        self._magnetization = np.asarray(magnetization, order="C").reshape(3, 1)

    def __str__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"size={self.size.ravel()}, "
            f"center={self.center.ravel()}, "
            f"mag={self.magnetization.ravel()})"
        )  # pragma: no cover

    def __repr__(self) -> str:
        return str(self)  # pragma: no cover

    def __hash__(self) -> int:
        return hash(self.hashed)

    def __eq__(self, other) -> bool:
        return self.hashed == other.hashed

    def __call__(
        self,
        xrange: tuple[float, float, int] | float,
        yrange: tuple[float, float, int] | float,
        zrange: tuple[float, float, int] | float,
        what: Literal["Bfield", "Hfield", "dBfield", "dHfield"] = "Bfield",
    ):
        if len(what) <= 2:
            # assuming only H, B, dH or dB supplied
            what += "field"

        arrays = [range2array(element) for element in [xrange, yrange, zrange]]
        xx, yy, zz = np.meshgrid(*arrays, indexing="ij")

        shape = xx.shape
        points = np.asarray([xx.ravel(), yy.ravel(), zz.ravel()])

        if what.startswith("d"):
            results = np.squeeze(getattr(self, what)(points).reshape(3, 3, *shape))
        else:
            results = np.squeeze(getattr(self, what)(points).reshape(3, *shape))
        return results

    def plot_1d(self, **kwargs):
        """Plot a one-dimensional field section; see :func:`microcubed.viz.plot_1d`."""
        from microcubed.viz import plot_1d

        return plot_1d(self, **kwargs)

    def plot_2d(self, **kwargs):
        """Plot a two-dimensional field section; see :func:`microcubed.viz.plot_2d`."""
        from microcubed.viz import plot_2d

        return plot_2d(self, **kwargs)

    def plot_3d(self, **kwargs):
        """Plot a three-dimensional field section; see :func:`microcubed.viz.plot_3d`."""
        from microcubed.viz import plot_3d

        return plot_3d(self, **kwargs)

    @property
    def size(self):
        """Return the magnet size (write protected)."""
        return self._size

    @property
    def center(self):
        """Return the magnet center point (write protected)"""
        return self._center

    @property
    def magnetization(self):
        """Return the magnetization vector (write protected)"""
        return self._magnetization

    @cached_property
    def hashed(self):
        """Return an object corresponding hash."""
        return "".join(
            [
                sha256(self.size.view()).hexdigest(),
                sha256(self.center.view()).hexdigest(),
                sha256(self.magnetization.view()).hexdigest(),
            ]
        )

    @cached_property
    def polarization(self):
        """'Saturated' magnetic polarization vector of the cuboid magnet."""
        return -self.magnetization * mu_0 / (4 * np.pi)

    @cached_property
    def corners(self):
        """Return the corner locations this cuboid magnet instance."""
        corners = np.zeros((3, 8))
        for idx, (i, j, k) in enumerate(product(range(2), repeat=3)):
            signs = (-1) ** np.asarray([[i], [j], [k]])
            corners[:, idx] = (signs * self.size / 2 + self.center).ravel()

        return corners

    @cached_property
    def volume(self):
        """Return the volume of this cuboid magnet."""
        return np.prod(self.size)

    @cached_property
    def bbox(self):
        """Return the bbox of the cuboid magnet."""
        return np.hstack((self.center - self.size / 2, self.center + self.size / 2))

    def union_boundary(self, plane: str = "xy", *, atol: float | None = None) -> list[np.ndarray]:
        """Return closed boundary loops of the union of projected cuboid footprints.

        Parameters
        ----------
        plane : str
            Projection plane: ``xy``, ``xz`` or ``yz`` (case-insensitive).
            Reversed spellings are aliases with canonical axis order, as in
            :meth:`chull_points`; ``yx`` still returns x then y.
        atol : float, optional
            Absolute coordinate snapping tolerance in the geometry's length unit.
            By default, use 32 machine epsilons times the largest absolute projected
            coordinate to remove round-off seams. Set zero to disable snapping.
            Features smaller than this tolerance can collapse.

        Returns
        -------
        list of numpy.ndarray
            Closed ``(2, N)`` loops, suitable for ``ax.plot(*loop)``. Outer loops
            are counterclockwise; holes are clockwise. Disconnected or merely
            corner-touching components have separate loops. An empty arrangement
            returns an empty list. Loop ordering is not a component hierarchy.

        Notes
        -----
        This is a 2D projection, not a slice or a 3D surface. Footprints may
        overlap even when their cuboids are separated along the omitted axis.
        Unlike a convex hull, the union boundary retains concavities and holes.
        """
        from microcubed.geometry import rectangle_union_boundary

        planes = {"xy": (0, 1), "yx": (0, 1), "xz": (0, 2), "zx": (0, 2), "yz": (1, 2), "zy": (1, 2)}
        if not isinstance(plane, str) or plane.lower() not in planes:
            raise ValueError("plane must be xy, xz or yz (or a reversed alias)")
        axes = list(planes[plane.lower()])
        magnets = self if isinstance(self, BasicArrangement) else (self,)
        rectangles = [magnet.bbox[axes].T.ravel() for magnet in magnets]
        return rectangle_union_boundary(rectangles, atol=atol)

    def chull(self, plane: str | None = None) -> ConvexHull:
        """Return the 3D convex hull of the cuboid as ConvexHull object.

        Parameters
        ----------
        plane: str
            2D-plane to build up the sub-space convex hull onto.
            This defaults to None, therefore the 3D hull will be returned.

        Returns
        -------
        scipy.spatial.ConvexHull
        """
        if plane is None:
            return ConvexHull(self.corners.T)
        elif plane.lower() in {"xy", "yx"}:
            return ConvexHull(self.corners.T[:, :2])
        elif plane.lower() in {"yz", "zy"}:
            return ConvexHull(self.corners.T[:, 1:])
        elif plane.lower() in {"xz", "zx"}:
            return ConvexHull(self.corners.T[:, ::2])
        else:
            raise ValueError(f"Plane {plane} not supported.")

    def chull_points(self, plane: str | None = None) -> np.ndarray:
        """Return the convex hull of the cuboid as points of the vertices.

        Parameters
        ----------
        plane: str
            2D-plane to build up the sub-space convex hull onto.
            This defaults to None, therefore the 3D hull will be returned.

        Returns
        -------
        numpy.ndarray
        """

        chull = self.chull(plane)
        points, vertices = chull.points, chull.vertices
        return points[np.hstack((vertices, vertices[0]))].T

    def overlapping(self, other: BasicMagnet) -> bool:
        """Calculate the *axis-aligned* 'overlapping' between this and
        another cuboid magnet.

        Parameters
        ----------
        other: BasicMagnet
            Another magnet instance to check the overlapping to.

        Returns
        -------
        bool
        """
        if isinstance(other, BasicMagnet):
            return np.all((self.bbox[:, 0] < other.bbox[:, 1]) & (self.bbox[:, 1] > other.bbox[:, 0]))
        else:
            raise TypeError("Can only check overlap with another Magnet.")

    def touching(self, other: BasicMagnet) -> bool:
        """Calculate the *axis-aligned* 'touching' between this and another cuboid.

        Parameters
        ----------
        other: BasicMagnet
            Another magnet instance to check the touching to.

        Returns
        -------
        bool
        """
        if isinstance(other, BasicMagnet):
            return np.all((self.bbox[:, 0] <= other.bbox[:, 1]) & (self.bbox[:, 1] >= other.bbox[:, 0]))
        else:
            raise TypeError("Can only check touching with another Magnet.")

    def mirrored(self, x: float | None = None, y: float | None = None, z: float | None = None) -> BasicMagnet:
        """Create a new cuboid magnet instance mirrored across plane(s).

        Parameters
        ----------
        x: float | None
            Value of x axis to mirror against.
        y: float | None
            Value of y axis to mirror against.
        z: float | None
            Value of z axis to mirror against.

        Returns
        -------
        BasicMagnet
            Mirrored magnet.
        """
        cls = type(self)
        new_center = self.center.copy()
        if x is not None:
            new_center[0] = 2 * x - new_center[0]
        if y is not None:
            new_center[1] = 2 * y - new_center[1]
        if z is not None:
            new_center[2] = 2 * z - new_center[2]

        return cls(self.size, new_center, self.magnetization)

    def moved_by(self, diff: np.ndarray):
        """Create a new magnet with the center moved by the given vector.

        Parameters
        ----------
        center : np.ndarray
            New center position (3, 1).

        Returns
        -------
        BasicMagnet
            New Magnet with the center moved by the given vector.
        """
        cls = type(self)
        diff = np.asarray(diff).reshape(3, 1)
        return cls(self.size, self.center + diff, self.magnetization)

    def moved_to(self, center: np.ndarray):
        """Create a new magnet with the center moved to the given position.

        Parameters
        ----------
        center : np.ndarray
            New center position (3, 1).

        Returns
        -------
        BasicMagnet
            New Magnet with the center moved to the given position.
        """
        return self.moved_by(np.asarray(center).reshape(3, 1) - self.center)

    def Hfield(self, *args, **kwargs):
        """Calculate the magnetic field strength."""
        return self.Bfield(*args, **kwargs) / mu_0

    def dHfield(self, *args, **kwargs):
        """Calculate the magnetic field strength gradient."""
        return self.dBfield(*args, **kwargs) / mu_0

    @abstractmethod
    def Bfield(self, *args, **kwargs):
        """Calculate the magnetic flux density."""

    @abstractmethod
    def dBfield(self, *args, **kwargs):
        """Calculate the magnetic flux density gradient."""

    @classmethod
    def from_bbox(
        cls,
        p1: list | np.ndarray,
        p2: list | np.ndarray,
        magnetization: list | np.ndarray,
    ) -> BasicMagnet:
        """Create a Magnet from two points defining the bounding box.

        Parameters
        ----------
        p1 : np.ndarray
            First point of the bounding box.
        p2 : np.ndarray
            Second point of the bounding box.
        mag : np.ndarray, optional
            Magnetization of the bar magnet.

        Returns
        -------
        BasicMagnet
            Magnet with the given bounding box and saturation magnetization.
        """
        p1 = np.asarray(p1).reshape(3, 1)
        p2 = np.asarray(p2).reshape(3, 1)
        return cls(np.abs(p1 - p2), (p1 + p2) / 2, magnetization)


class BasicArrangement(BasicMagnet):
    """Arrangement of multiple magnets placed somewhere in a 3D room."""

    warnings.formatwarning = custom_warning
    """Custom warning format."""

    Magnet = BasicMagnet
    """Corresponding magnet class."""

    def __init__(self, magnets: list | None = None, validate: bool = True) -> None:
        """Create a new arrangement of magnets.

        An arrangement is a collection of magnets placed somewhere in a 3D room.
        The magnets are stored in a list. The arrangement can be validated whether
        the included magnets overlap or not. The validation is done by default.
        The arrangement can be used as an iterator to iterate over the included magnets.

        By using the `B(p)` or `dB(p)` methods, the magnetic field or the
        magnetic field gradient at a given point (or array of points, row-vectors)
        can be calculated. Its a simple summation of all field components of all
        included magnets.

        Parameters
        ----------
        magnets : list
            List of `Magnet` instances included in the arrangement.
        validate : bool, optional
            Validate the arrangement (no overlapping magnets), by default True.

        Raises
        ------
        ValueError
            The arrangement must contain at least one magnet.
        ValueError
            The arrangement is not valid if some cuboids overlap.
        """
        self._magnets = [] or magnets
        self.validate = validate

        if self.validate and not self.valid:
            raise ValueError("The arrangement is not valid. Some cuboids overlap.")

    def __add__(self, other) -> BasicArrangement:
        """Add two arrangements together, validate by the first arrangement."""
        if isinstance(other, BasicArrangement):
            # should be validated because additions can change validity
            return type(self)(self.magnets + other.magnets, self.validate)
        elif isinstance(other, self.Magnet):
            # should be validated because additions can change validity
            return type(self)(self.magnets + [other], self.validate)
        else:
            raise TypeError(f"Cannot add other objects of type {type(other)}.")

    def __radd__(self, other) -> BasicArrangement:
        """Add two arrangements together, validate by the second arrangement."""
        if isinstance(other, BasicArrangement):
            # should be validated because additions can change validity
            return type(self)(self.magnets + other.magnets, other.validate)
        elif isinstance(other, self.Magnet):
            # should be validated because additions can change validity
            return type(self)(self.magnets + [other], self.validate)
        else:
            raise TypeError(f"Cannot add other objects of type {type(other)}.")

    def __sub__(self, other) -> BasicArrangement:
        if isinstance(other, BasicArrangement):
            return type(self)(
                # has not to be validated, because removals are not changing validity
                [magnet for magnet in self if magnet not in other],
                validate=False,
            )
        elif isinstance(other, self.Magnet):
            return type(self)(
                # has not to be validated, because removals are not changing validity
                [magnet for magnet in self if magnet != other],
                validate=False,
            )
        else:
            raise TypeError(f"Cannot subtract other objects of type {type(other)}.")

    def __str__(self) -> str:
        if len(self) > 10:
            return f"{type(self).__name__}({len(self)} magnets)"  # pragma: no cover

        magnets = "".join([4 * " " + f"{magnet!s},\n" for magnet in self])
        return f"{type(self).__name__}(\n{magnets})"  # pragma: no cover

    def __repr__(self) -> str:
        return str(self)  # pragma: no cover

    def __getitem__(self, index) -> BasicArrangement | BasicMagnet:
        if isinstance(index, slice):
            # has not to be validated, because it is a subset copy
            return type(self)(self.magnets[index], validate=False)
        elif isinstance(index, int):
            return self.magnets[index]
        else:
            raise TypeError(f"Indexing with type {type(index)} not supported.")

    def __iter__(self) -> Iterator[BasicMagnet]:
        return iter(self.magnets)

    def __len__(self) -> int:
        return len(self.magnets)

    def __hash__(self) -> int:
        return hash(self.hashed)

    def __eq__(self, other: object) -> bool:
        return self.hashed == other.hashed

    @property
    def magnets(self):
        """List of magnets in the arrangement (as a property for write protection)."""
        return self._magnets

    @cached_property
    def hashed(self):
        return sha256("".join([magnet.hashed for magnet in self]).encode()).hexdigest()

    @cached_property
    def center(self):
        """Center of the arrangement (as property for write protection)."""
        return np.mean([magnet.center for magnet in self], axis=0)

    @cached_property
    def bbox(self):
        """Bounding box of the arrangement."""
        return np.vstack(
            (
                np.min([magnet.bbox[:, 0] for magnet in self], axis=0),
                np.max([magnet.bbox[:, 1] for magnet in self], axis=0),
            )
        ).T

    @cached_property
    def valid(self):
        """Check if the arrangement is valid and no intersections occur. Brute-Force."""
        if self.use_tqdm:
            desc = "Checking for intersections"
            total = int(comb(len(self), 2))
            return not any(m1.overlapping(m2) for m1, m2 in tqdm(combinations(self, 2), desc, total))
        else:
            return not any(m1.overlapping(m2) for m1, m2 in combinations(self, 2))

    @cached_property
    def closed(self):
        """Check whether nearest neighbors are touching each other.

        It is assumed the list provided is sorted by the nearest neighbours.
        More precisely, each magnet has to be the neighbour of the next one in the list.
        """
        if self.use_tqdm:
            desc = "Checking for a closed arrangement"
            total = len(self) - 1
            return all(m1.touching(m2) for m1, m2 in tqdm(pairwise(self), desc, total))
        else:
            return all(m1.touching(m2) for m1, m2 in pairwise(self))

    def chull(self, plane: str | None = None) -> ConvexHull:
        """Return the 3D convex hull of the cuboid as ConvexHull object.

        Parameters
        ----------
        plane: str
            2D-plane to build up the sub-space convex hull onto.
            This defaults to None, therefore the 3D hull will be returned.

        Returns
        -------
        scipy.spatial.ConvexHull
        """
        corners = np.hstack([magnet.corners for magnet in self]).T
        if plane is None:
            return ConvexHull(corners)
        elif plane.lower() in {"xy", "yx"}:
            return ConvexHull(corners[:, :2])
        elif plane.lower() in {"yz", "zy"}:
            return ConvexHull(corners[:, 1:])
        elif plane.lower() in {"xz", "zx"}:
            return ConvexHull(corners[:, ::2])
        else:
            raise ValueError(f"Plane {plane} not supported.")

    def chull_points(self, plane: str | None = None, individual: bool = False) -> np.ndarray:
        """Return the convex hull of the cuboid as points of the vertices.

        Parameters
        ----------
        plane: str
            2D-plane to build up the sub-space convex hull onto.
            This defaults to None, therefore the 3D hull will be returned.

        Returns
        -------
        numpy.ndarray
        """
        if not individual:
            return super().chull_points(plane)
        else:
            return [item for magnet in self for item in magnet.chull_points(plane)]

    def mirrored(self, x: float | None = None, y: float | None = None, z: float | None = None) -> BasicArrangement:
        """Create a mirrored copy of the arrangement."""
        return type(self)(
            [magnet.mirrored(x, y, z) for magnet in self],
            validate=False,  # has not to be validated, because it is a copy
        )

    def moved_by(self, diff: np.ndarray):
        """Returns a copy of the arrangement moved in its center by the given vector."""
        return type(self)(
            [magnet.moved_by(diff) for magnet in self],
            validate=False,
        )

    @classmethod
    def from_shape(
        cls,
        shape,
        t: float,
        delta: float | tuple[float, float],
        mag: list | np.ndarray,
        *,
        z: float = 0,
        bounds: tuple[float, float, float, float] | None = None,
    ) -> BasicArrangement:
        """Extrude and decompose a rasterized 2D shape into cuboid magnets.

        The shape can be polygon vertices, a ``matplotlib.path.Path``, or a
        callable ``shape(x, y)``. Callable shapes require explicit bounds.
        Occupied grid cells are greedily merged into large, non-overlapping
        rectangles before extrusion by thickness ``t``.
        """
        from microcubed.geometry import merge_mask_to_rectangles, rasterize_shape

        if not isinstance(t, (int, float, np.number)) or not np.isfinite(t) or t <= 0:
            raise ValueError("t must be a positive finite scalar")
        if not isinstance(z, (int, float, np.number)) or not np.isfinite(z):
            raise ValueError("z must be a finite scalar")

        x_edges, y_edges, mask = rasterize_shape(shape, delta, bounds=bounds)
        cuboids = []
        for row0, row1, column0, column1 in merge_mask_to_rectangles(mask):
            x0, x1 = x_edges[column0], x_edges[column1]
            y0, y1 = y_edges[row0], y_edges[row1]
            cuboids.append(
                cls.Magnet(
                    [x1 - x0, y1 - y0, t],
                    [(x0 + x1) / 2, (y0 + y1) / 2, z],
                    mag,
                )
            )
        return cls(cuboids, validate=False)

    @classmethod
    def from_voronoi_grains(
        cls,
        grains,
        mag,
        *,
        thickness: float | None = None,
        z: float = 0,
    ) -> BasicArrangement:
        """Convert rasterized 2D or 3D Voronoi grains to cuboid magnets.

        For 2D grains, ``thickness`` is required and ``z`` is the extrusion
        center. Three-dimensional grains already contain their z extent, so
        ``thickness`` must be omitted. ``mag`` can be one vector shared by all
        grains, one vector per grain, or a callable receiving all seed points
        and returning either form.
        """
        from microcubed.geometry import VoronoiGrains, merge_mask_to_boxes, merge_mask_to_rectangles

        if not isinstance(grains, VoronoiGrains):
            raise TypeError("grains must be a VoronoiGrains instance")
        if callable(mag):
            mag = mag(grains.seeds.copy())
        magnetizations = np.asarray(mag)
        if magnetizations.shape == (3,):
            magnetizations = np.broadcast_to(magnetizations, (grains.grain_count, 3))
        if magnetizations.shape != (grains.grain_count, 3) or not np.all(np.isfinite(magnetizations)):
            raise ValueError("mag must be a finite vector or one finite vector per grain")

        if grains.dimension == 2:
            if not isinstance(thickness, (int, float, np.number)) or not np.isfinite(thickness) or thickness <= 0:
                raise ValueError("thickness must be a positive finite scalar for 2D grains")
            if not isinstance(z, (int, float, np.number)) or not np.isfinite(z):
                raise ValueError("z must be a finite scalar")
        elif thickness is not None:
            raise ValueError("thickness must be omitted for 3D grains")

        cuboids = []
        dx, dy = grains.cell_size[:2]
        ox, oy = grains.origin[:2]
        for grain in range(grains.grain_count):
            grain_mask = grains.labels == grain
            if grains.dimension == 2:
                for row0, row1, column0, column1 in merge_mask_to_rectangles(grain_mask):
                    cuboids.append(
                        cls.Magnet(
                            [(column1 - column0) * dx, (row1 - row0) * dy, thickness],
                            [ox + (column0 + column1) * dx / 2, oy + (row0 + row1) * dy / 2, z],
                            magnetizations[grain],
                        )
                    )
            else:
                dz = grains.cell_size[2]
                oz = grains.origin[2]
                for layer0, layer1, row0, row1, column0, column1 in merge_mask_to_boxes(grain_mask):
                    cuboids.append(
                        cls.Magnet(
                            [(column1 - column0) * dx, (row1 - row0) * dy, (layer1 - layer0) * dz],
                            [
                                ox + (column0 + column1) * dx / 2,
                                oy + (row0 + row1) * dy / 2,
                                oz + (layer0 + layer1) * dz / 2,
                            ],
                            magnetizations[grain],
                        )
                    )
        return cls(cuboids, validate=False)

    @classmethod
    def from_circle(
        cls,
        radius: float,
        thickness: float,
        center: list | None = None,
        mag: list | None = None,
        dx: float = 1,
    ) -> BasicArrangement:
        # construct cuboid centers
        xc = np.arange(-radius + dx / 2, radius + dx / 2, dx)
        yc = np.zeros(xc.size)  # all same
        zc = np.zeros(xc.size)  # all same

        # construct cuboid sizes
        xs = np.ones(xc.size) * dx  # all same
        ys = 2 * np.sqrt(radius**2 - xc**2)  # perimeter
        zs = np.ones(xc.size) * thickness  # all same

        sizes = np.array([xs, ys, zs]).T
        centers = np.array([xc, yc, zc]).T

        cuboids = [cls.Magnet(sz, ct, mag) for sz, ct in zip(sizes, centers)]

        arrangement = cls(cuboids, validate=False)  # skip validation, function verified
        return arrangement if center is None else arrangement.moved_to(center)

    @classmethod
    def from_ellipsis(
        cls,
        size: list,
        center: list | None = None,
        mag: list | None = None,
        dx: float = 1,
    ) -> BasicArrangement:
        L, W, D = size

        if L <= W:
            # construct cuboid centers
            xc = np.arange(-L / 2, L / 2, dx) + dx / 2
            yc = np.zeros(xc.size)  # all same
            zc = np.zeros(xc.size)  # all same

            # construct cuboid sizes
            xs = np.ones(xc.size) * dx  # all same
            ys = W * np.sqrt(L**2 - 4 * xc**2) / L
            zs = np.ones(xc.size) * D  # all same

        else:
            # construct cuboid centers
            yc = np.arange(-W / 2, W / 2, dx) + dx / 2
            xc = np.zeros(yc.size)  # all same
            zc = np.zeros(yc.size)  # all same

            # construct cuboid sizes
            xs = L * np.sqrt(W**2 - 4 * yc**2) / W
            ys = np.ones(yc.size) * dx  # all same
            zs = np.ones(yc.size) * D  # all same

        sizes = np.array([xs, ys, zs]).T
        centers = np.array([xc, yc, zc]).T

        cuboids = [cls.Magnet(sz, ct, mag) for sz, ct in zip(sizes, centers)]

        arrangement = cls(cuboids, validate=False)  # skip validation, function verified
        return arrangement if center is None else arrangement.moved_to(center)

    @classmethod
    def from_rounded_rectangle(
        cls,
        size: list,
        center: list | None = None,
        mag: list | None = None,
        edge_radius: float = 0,
        dx: float = 1,
    ) -> BasicArrangement:
        (L, W, D), R = size, edge_radius

        if (R > L / 2) or (R > W / 2):
            raise ValueError("Edge radius is too large for the given rectangle.")

        # inner uninfluenced rectangle
        xc0, yc0, zc0 = 0, 0, 0
        xs0, ys0, zs0 = (L - 2 * R), W, D

        middle_cuboid = cls.Magnet([xs0, ys0, zs0], [xc0, yc0, zc0], mag)

        # rounded corners on left side
        xc = np.arange(-L / 2, -L / 2 + R, dx) + dx / 2
        yc = np.zeros(xc.size)  # all same
        zc = np.zeros(xc.size)  # all same

        xs = np.ones(xc.size) * dx  # all same
        ys = 2 * np.sqrt(R**2 - (xc + L / 2 - R) ** 2) + W - 2 * R
        zs = np.ones(xc.size) * D  # all same

        sizes = np.array([xs, ys, zs]).T
        centers = np.array([xc, yc, zc]).T

        corner_cuboids = [cls.Magnet(sz, ct, mag) for sz, ct in zip(sizes, centers)]
        corner_arrangement = cls(corner_cuboids, validate=False)  # left side
        corner_arrangement += corner_arrangement.mirrored(x=0)  # right side

        arrangement = corner_arrangement + middle_cuboid  # add middle cuboid
        return arrangement if center is None else arrangement.moved_to(center)

    @classmethod
    def from_superellipsis(
        cls,
        size: list,
        center: list | None = None,
        mag: list | None = None,
        n: int = 5,
        dx: float = 1,
    ) -> BasicArrangement:
        L, W, D = size

        if n <= 0:
            raise ValueError("Superellipse exponent 'n' must be greater than 0.")

        if L <= W:
            # construct cuboid centers
            xc = np.arange(-L / 2, L / 2, dx) + dx / 2
            yc = np.zeros(xc.size)  # all same
            zc = np.zeros(xc.size)  # all same

            # construct cuboid sizes
            xs = np.ones(xc.size) * dx  # all same
            ys = (W**n * (L**n - 2**n * np.abs(xc) ** n)) ** (1 / n) / L
            zs = np.ones(xc.size) * D  # all same
        else:
            # construct cuboid centers
            yc = np.arange(-W / 2, W / 2, dx) + dx / 2
            xc = np.zeros(yc.size)  # all same
            zc = np.zeros(yc.size)  # all same

            # construct cuboid sizes
            xs = (L**n * (W**n - 2**n * np.abs(yc) ** n)) ** (1 / n) / W
            ys = np.ones(yc.size) * dx  # all same
            zs = np.ones(yc.size) * D  # all same

        sizes = np.array([xs, ys, zs]).T
        centers = np.array([xc, yc, zc]).T

        cuboids = [cls.Magnet(sz, ct, mag) for sz, ct in zip(sizes, centers)]

        arrangement = cls(cuboids, validate=False)  # skip validation, function verified
        return arrangement if center is None else arrangement.moved_to(center)

    @classmethod
    def from_right_triangle(
        cls,
        size: list,
        center: list | None = None,
        mag: list | None = None,
        corner: str = "rb",
        dx: float = 1,
    ) -> BasicArrangement:
        L, W, D = size
        # L is always x, W is always y

        if L <= W:
            # construct cuboid centers
            xc = np.arange(0, L, dx) + dx / 2
            yc = W / (2 * L) * xc
            zc = np.zeros(xc.size)  # all same

            # construct cuboid sizes
            xs = np.ones(xc.size) * dx  # all same
            ys = W / L * xc
            zs = np.ones(xc.size) * D  # all same

            sizes = np.array([xs, ys, zs]).T
            centers = np.array([xc, yc, zc]).T

            # this is oriented with the right angle in the bottom right corner
            cuboids = [cls.Magnet(sz, ct, mag) for sz, ct in zip(sizes, centers)]
            arrangement = cls(cuboids, validate=False).moved_to([0, 0, 0])
            arrangement = arrangement.mirrored(x=0, y=0)

        else:
            # construct cuboid centers
            yc = np.arange(0, W, dx) + dx / 2
            xc = L / (2 * W) * yc
            zc = np.zeros(xc.size)  # all same

            # construct cuboid sizes
            ys = np.ones(xc.size) * dx  # all same
            xs = L / W * yc
            zs = np.ones(xc.size) * D  # all same

            sizes = np.array([xs, ys, zs]).T
            centers = np.array([xc, yc, zc]).T

            # this is oriented with the right angle in the top left corner
            cuboids = [cls.Magnet(sz, ct, mag) for sz, ct in zip(sizes, centers)]
            arrangement = cls(cuboids, validate=False).moved_to([0, 0, 0])

        # rotate to desired corner (origin = top left)
        if corner in {"lb"}:
            arrangement = arrangement.mirrored(y=0)
        elif corner in {"rb"}:
            arrangement = arrangement.mirrored(x=0, y=0)
        elif corner in {"rt"}:
            arrangement = arrangement.mirrored(x=0)
        elif corner not in {"lt"}:
            raise ValueError(f"Corner '{corner}' not recognized.")

        return arrangement if center is None else arrangement.moved_to(center)
