from __future__ import annotations

from itertools import product

import numpy as np

from microcubed.backends.equations import Bfield, Bijk, dBfield, dBijk, edge, sign


def test_vectorized_equations_match_corner_reference():
    rng = np.random.default_rng(42)
    size = np.array([2.0, 3.0, 4.0]).reshape(3, 1)
    center = np.array([0.2, -0.3, 0.4]).reshape(3, 1)
    polarization = np.array([0.1, -0.2, 0.3]).reshape(3, 1)
    points = rng.uniform(5, 10, (3, 100))

    reference = sum(
        sign(i, j, k) * Bijk(points, edge(i, j, k, size, center), polarization)
        for i, j, k in product(range(2), repeat=3)
    )
    gradient_reference = sum(
        sign(i, j, k) * dBijk(points, edge(i, j, k, size, center), polarization)
        for i, j, k in product(range(2), repeat=3)
    )

    np.testing.assert_allclose(Bfield(size, center, polarization, points), reference, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(dBfield(size, center, polarization, points), gradient_reference, rtol=1e-10, atol=1e-14)
