# Physical model

## Magnetostatic assumptions

Microcubed describes an axis-aligned cuboid with constant magnetisation

$$
\mathbf M=(M_x,M_y,M_z)^\mathsf T.
$$

Outside the magnet, free currents, time-dependent fields, and media differing
from vacuum are excluded. Consequently,

$$
\nabla\times\mathbf H=0,\qquad
\nabla\cdot\mathbf B=0,\qquad
\mathbf B=\mu_0\mathbf H
$$

hold in the exterior domain. The field follows from the scalar magnetic
potential of the Coulomb model. The implemented closed-form expressions are
based on Ravaud and Lemarquand, who give all three field components for a
uniformly and arbitrarily polarised parallelepiped
([Ravaud and Lemarquand, 2009](https://doi.org/10.2528/PIER09091704)).

## Geometry and corner sum

Let the cuboid centre be $\mathbf c$, its size be
$\mathbf s=(s_x,s_y,s_z)$, and the observation point be $\mathbf r$. The eight
corners are

$$
\mathbf e_{ijk}=\mathbf c+
\frac{1}{2}\begin{pmatrix}(-1)^i s_x\\(-1)^j s_y\\(-1)^k s_z\end{pmatrix},
\qquad i,j,k\in\{0,1\}.
$$

The total field is an alternating sum of corner terms:

$$
\mathbf B(\mathbf r)=
\sum_{i,j,k=0}^{1}(-1)^{i+j+k+1}
\mathbf B_{ijk}(\mathbf r-\mathbf e_{ijk}).
$$

The components of $\mathbf B_{ijk}$ consist of logarithms and `atan2` terms.
Microcubed evaluates all eight corners in a vectorised pass. Paired logarithms
are combined stably so that removable expressions of the form
`log(0) - log(0)` on extended edge lines do not produce `NaN`.

More explicitly, let $(x,y,z)=\mathbf r-\mathbf e_{ijk}$,
$R=\sqrt{x^2+y^2+z^2}$, and $\mathbf p=(p_x,p_y,p_z)^\mathsf T$ denote the
internal polarisation. Before applying the alternating corner sign, the field
kernel implemented by both the NumPy and Rust backends is

$$
\begin{aligned}
B_x &= p_x\operatorname{atan2}(yz,xR)-p_y\ln(z+R)-p_z\ln(y+R),\\
B_y &=-p_x\ln(z+R)+p_y\operatorname{atan2}(xz,yR)-p_z\ln(x+R),\\
B_z &=-p_x\ln(y+R)-p_y\ln(x+R)+p_z\operatorname{atan2}(xy,zR).
\end{aligned}
$$

These expressions follow by differentiating the Coulombian scalar potential
of the six uniformly charged faces. Integrating each face analytically leaves
the logarithmic and inverse-tangent terms above; inclusion--exclusion of the
opposite face limits produces the eight-corner alternating sum.

## Magnetisation, polarisation, and sign

The public input is $\mathbf M$ in A/m. Internally, Microcubed uses the factor

$$
-\frac{\mu_0}{4\pi}\mathbf M.
$$

The returned `Bfield` is magnetic flux density in tesla. `Hfield` is computed
as $\mathbf B/\mu_0$.

## Field gradient

The gradient is returned as the Jacobian matrix ordered as

$$
[\partial_i B_j]
=
\begin{pmatrix}
\partial_xB_x & \partial_xB_y & \partial_xB_z\\
\partial_yB_x & \partial_yB_y & \partial_yB_z\\
\partial_zB_x & \partial_zB_y & \partial_zB_z
\end{pmatrix}.
$$

In NumPy, use

```python
dB[derivative_axis, field_component, point]
```

For example, `dB[0, 2]` is $\partial_xB_z$.

In the current-free exterior domain, $\nabla\times\mathbf B=0$, so the
gradient matrix is symmetric. In addition, $\nabla\cdot\mathbf B=0$ implies a
vanishing trace. These identities provide useful consistency checks, but they
do not replace a convergence study near material boundaries.

For the differentiated kernels, the implementation introduces the reusable
abbreviations

$$
F_{xR,yz}=x^2R^2+(yz)^2,
\qquad
G_{Rz}=Rz+R^2,
$$

with cyclic permutations for the other axes. For example, one corner's
$\partial_x B_x$ contribution is

$$
-p_x\frac{yz(x^2+R^2)}{R F_{xR,yz}}
-p_y\frac{x}{G_{Rz}}
-p_z\frac{x}{G_{Ry}}.
$$

The remaining diagonal components follow by cyclic permutation. The
off-diagonal entries are obtained by differentiating the same kernels and are
filled symmetrically, e.g. $\partial_xB_y=\partial_yB_x$, in the current-free
exterior. This is why `dBfield` returns the analytical Jacobian rather than a
finite difference of sampled field values.

## Scope and limitations

The model is exact within its assumptions:

- axis-aligned cuboid,
- spatially constant magnetisation in each cuboid,
- linear exterior medium with $\mu_r=1$,
- magnetostatic state,
- evaluation outside magnetic material.

The model does not include:

- feedback of the stray field on $\mathbf M$,
- domains, exchange, or anisotropy,
- hysteresis or magnetisation dynamics,
- temperature-dependent material parameters,
- finite relative permeability of the magnetic material,
- interaction with soft-magnetic bodies.

These tasks require a micromagnetic or finite-element solver. Microcubed can
subsequently evaluate the field of a prescribed, piecewise-constant
magnetisation.
