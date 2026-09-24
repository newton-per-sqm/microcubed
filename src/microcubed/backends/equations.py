"""Analytical exterior field and gradient of a uniformly magnetised cuboid.

The implementation follows the Coulombian closed-form solution of Ravaud and
Lemarquand, *Magnetic Field Produced by a Parallelepipedic Magnet of Various
and Uniform Polarization*, PIER 98, 207-219 (2009),
https://doi.org/10.2528/PIER09091704.

For each observation point the cuboid is represented by its eight corners.
The field and its Jacobian are alternating sums of logarithmic and ``atan2``
corner kernels. See ``docs/theory.md`` for equations, notation, and the
relationship between public magnetisation and internal polarisation.
"""

from __future__ import annotations

import numpy as np

_CORNER_OFFSETS = np.array(
    [
        [1.0, 1.0, 1.0, 1.0, -1.0, -1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, -1.0],
        [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
    ],
)[:, :, None]
_CORNER_SIGNS = np.array([-1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0])[:, None]


def edge(i: int, j: int, k: int, size: np.ndarray, center: np.ndarray):
    r"""Return corner $\mathbf e_{ijk}$ for binary corner indices.

    ``size`` and ``center`` have shape ``(3, 1)``. The returned corner is
    $\mathbf c + \tfrac12(({-1})^i s_x, ({-1})^j s_y, ({-1})^k s_z)^\mathsf T$.
    """
    return size / 2 * (-1.0) ** np.array([[i], [j], [k]]) + center


def sign(i: int, j: int, k: int):
    """Return the alternating corner-sum sign $(-1)^{i+j+k+1}$."""
    return (-1.0) ** (i + j + k + 1)


def Bijk(points: np.ndarray, edge: np.ndarray, polarization: np.ndarray):
    r"""Return the three-vector kernel contributed by one cuboid corner.

    This is the un-signed $\mathbf B_{ijk}(\mathbf r-\mathbf e_{ijk})$
    term. :func:`Bfield` applies the eight alternating corner signs. It is
    retained for testing and derivation work; production evaluation uses the
    vectorised corner implementation below.
    """
    dx, dy, dz = points - edge
    dr = np.hypot(np.hypot(dx, dy), dz)

    Bx = -polarization * np.asarray([-np.arctan2(dy * dz, dx * dr), np.log(dz + dr), np.log(dy + dr)])
    By = -polarization * np.asarray([np.log(dz + dr), -np.arctan2(dx * dz, dy * dr), np.log(dx + dr)])
    Bz = -polarization * np.asarray([np.log(dy + dr), np.log(dx + dr), -np.arctan2(dx * dy, dz * dr)])

    # summing up the sub-components in x, y, z on penultimate axis
    return np.sum([Bx, By, Bz], axis=1)


def dBijk(points: np.ndarray, edge: np.ndarray, polarization: np.ndarray):
    """Calculate one corner contribution to the field Jacobian.

    The off-diagonal derivatives are symmetric outside the magnet because the
    magnetostatic field is curl-free there. The implementation therefore
    evaluates one expression for each symmetric pair, for example
    ``dxBy = dyBx``.
    """
    dx, dy, dz = points - edge
    dr = np.hypot(np.hypot(dx, dy), dz)

    # Reusable factors from differentiating the logarithmic and atan2 kernels.
    dxz, dyz, dxy = dx * dz, dy * dz, dx * dy
    dxq, dyq, dzq, drq = dx**2, dy**2, dz**2, dr**2

    f_xr_yz = dxq * drq + dyz**2
    f_yr_xz = dyq * drq + dxz**2
    f_zr_xy = dzq * drq + dxy**2

    g_rzr = dr * dz + drq
    g_ryr = dr * dy + drq
    g_rxr = dr * dx + drq

    # Diagonal components of the Jacobian matrix.
    dxBx = -polarization * np.asarray(
        [
            dyz * (dxq + drq) / (dr * f_xr_yz),
            dx / g_rzr,
            dx / g_ryr,
        ]
    )
    dyBy = -polarization * np.asarray(
        [
            dy / g_rzr,
            dxz * (dyq + drq) / (dr * f_yr_xz),
            dy / g_rxr,
        ]
    )
    dzBz = -polarization * np.asarray(
        [
            dz / g_ryr,
            dz / g_rxr,
            dxy * (dzq + drq) / (dr * f_zr_xy),
        ]
    )

    # Off-diagonal components. Curl B vanishes in the current-free exterior.
    dxBy = dyBx = polarization * np.asarray(
        [
            dxz * (drq - dyq) / (dr * f_xr_yz),
            -dy / g_rzr,
            -1 / dr,
        ]
    )
    dxBz = dzBx = polarization * np.asarray(
        [
            dxy * (drq - dzq) / (dr * f_xr_yz),
            -1 / dr,
            -dz / g_ryr,
        ]
    )
    dyBz = dzBy = polarization * np.asarray(
        [
            -1 / dr,
            dxy * (drq - dzq) / (dr * f_yr_xz),
            -dz / g_rxr,
        ]
    )

    # Construct the full Jacobian and sum its magnetisation-vector terms.
    return np.sum(
        [
            [dxBx, dxBy, dxBz],
            [dyBx, dyBy, dyBz],
            [dzBx, dzBy, dzBz],
        ],
        axis=2,
    )


def _corner_displacements(size: np.ndarray, center: np.ndarray, points: np.ndarray):
    r"""Return $\mathbf r-\mathbf e_{ijk}$ for all corners, shape ``(3, 8, N)``."""
    corners = center[:, None, :] + size[:, None, :] * _CORNER_OFFSETS / 2
    return points[:, None, :] - corners


def _signed_corner_sum(values: np.ndarray) -> np.ndarray:
    """Apply the alternating corner sum along the corner axis."""
    with np.errstate(invalid="ignore"):
        return np.sum(_CORNER_SIGNS * values, axis=0)


def _signed_log_sum(displacements: np.ndarray, radii: np.ndarray, axis: int) -> np.ndarray:
    r"""Evaluate an alternating $\log(R+q)$ sum without removable singularities.

    Pairing corners that differ only along ``axis`` turns edge-line
    ``log(0)-log(0)`` expressions into their finite limiting log ratios.
    """
    pairs = (
        ((0, 4), (1, 5), (2, 6), (3, 7)),
        ((0, 2), (1, 3), (4, 6), (5, 7)),
        ((0, 1), (2, 3), (4, 5), (6, 7)),
    )[axis]
    coordinate = displacements[axis]
    result = np.zeros(displacements.shape[-1], dtype=np.result_type(displacements, float))

    with np.errstate(divide="ignore", invalid="ignore"):
        for first, second in pairs:
            first_term = radii[first] + coordinate[first]
            second_term = radii[second] + coordinate[second]
            scale = np.maximum(np.maximum(radii[first], radii[second]), 1.0)
            both_cancelled = (first_term <= np.finfo(float).eps * scale) & (second_term <= np.finfo(float).eps * scale)
            regular_ratio = np.log(first_term) - np.log(second_term)
            limit_ratio = np.log(radii[second] - coordinate[second]) - np.log(radii[first] - coordinate[first])
            result += _CORNER_SIGNS[first, 0] * np.where(both_cancelled, limit_ratio, regular_ratio)
    return result


def Bfield(size: np.ndarray, center: np.ndarray, polarization: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Evaluate the analytical exterior flux density for one cuboid.

    Parameters have shapes ``(3, 1)``, ``(3, 1)``, ``(3, 1)``, and ``(3, N)``
    for size, centre, internal polarisation, and points. The result has shape
    ``(3, N)`` in tesla. All eight Ravaud--Lemarquand corner terms are
    evaluated simultaneously; paired logarithms preserve finite edge limits.
    """
    displacements = _corner_displacements(size, center, points)
    dx, dy, dz = displacements
    dr = np.hypot(np.hypot(dx, dy), dz)
    px, py, pz = polarization[:, 0]

    with np.errstate(divide="ignore", invalid="ignore"):
        atan_x = _signed_corner_sum(np.arctan2(dy * dz, dx * dr))
        atan_y = _signed_corner_sum(np.arctan2(dx * dz, dy * dr))
        atan_z = _signed_corner_sum(np.arctan2(dx * dy, dz * dr))
        log_x = _signed_log_sum(displacements, dr, 0)
        log_y = _signed_log_sum(displacements, dr, 1)
        log_z = _signed_log_sum(displacements, dr, 2)

    return np.stack(
        [
            px * atan_x - py * log_z - pz * log_y,
            -px * log_z + py * atan_y - pz * log_x,
            -px * log_y - py * log_x + pz * atan_z,
        ]
    )


def dBfield(size: np.ndarray, center: np.ndarray, polarization: np.ndarray, points: np.ndarray) -> np.ndarray:
    r"""Evaluate the analytical exterior Jacobian $[\partial_i B_j]$.

    The result has shape ``(3, 3, N)`` with the derivative axis first. The
    named intermediates match the rational factors in ``docs/theory.md``:
    ``f_xr_yz = x^2R^2+(yz)^2`` and ``g_rzr = Rz+R^2`` (plus cyclic
    permutations). At isolated removable singularities, a centred field
    difference supplies a finite limiting value.
    """
    dx, dy, dz = _corner_displacements(size, center, points)
    dr = np.hypot(np.hypot(dx, dy), dz)
    px, py, pz = polarization[:, 0]

    dxz, dyz, dxy = dx * dz, dy * dz, dx * dy
    dxq, dyq, dzq, drq = dx**2, dy**2, dz**2, dr**2
    f_xr_yz = dxq * drq + dyz**2
    f_yr_xz = dyq * drq + dxz**2
    f_zr_xy = dzq * drq + dxy**2
    g_rzr = dr * dz + drq
    g_ryr = dr * dy + drq
    g_rxr = dr * dx + drq

    with np.errstate(divide="ignore", invalid="ignore"):
        j00 = -(px * dyz * (dxq + drq) / (dr * f_xr_yz) + py * dx / g_rzr + pz * dx / g_ryr)
        j11 = -(px * dy / g_rzr + py * dxz * (dyq + drq) / (dr * f_yr_xz) + pz * dy / g_rxr)
        j22 = -(px * dz / g_ryr + py * dz / g_rxr + pz * dxy * (dzq + drq) / (dr * f_zr_xy))
        j01 = px * dxz * (drq - dyq) / (dr * f_xr_yz) - py * dy / g_rzr - pz / dr
        j02 = px * dxy * (drq - dzq) / (dr * f_xr_yz) - py / dr - pz * dz / g_ryr
        j12 = -px / dr + py * dxy * (drq - dzq) / (dr * f_yr_xz) - pz * dz / g_rxr

    j00, j01, j02, j11, j12, j22 = (_signed_corner_sum(component) for component in (j00, j01, j02, j11, j12, j22))
    result = np.array([[j00, j01, j02], [j01, j11, j12], [j02, j12, j22]])

    bad_points = np.any(~np.isfinite(result), axis=(0, 1)) & np.all(np.isfinite(points), axis=0)
    if np.any(bad_points):
        selected = points[:, bad_points]
        scale = max(float(np.max(size)), float(np.max(np.abs(selected - center))), 1.0)
        step = np.cbrt(np.finfo(float).eps) * scale
        for derivative_axis in range(3):
            offset = np.zeros_like(selected)
            offset[derivative_axis] = step
            result[derivative_axis][:, bad_points] = (
                Bfield(size, center, polarization, selected + offset)
                - Bfield(size, center, polarization, selected - offset)
            ) / (2 * step)
    return result
