//! Analytical Ravaud--Lemarquand kernels for one uniformly magnetised cuboid.
//!
//! For a corner displacement `(dx, dy, dz)` and radius `r`, the field is the
//! alternating sum over eight corners of logarithmic and `atan2` terms. The
//! public Python API passes `polarization = -mu0 * magnetization / (4 pi)`;
//! details and the derivation are documented in `docs/theory.md`.

/// Calculate the exterior magnetic flux density at one point.
///
/// The result is in the unit implied by `polarization`. Each loop iteration
/// evaluates one signed corner contribution to the analytical cuboid field.
pub fn calc_bfield(
    size: &[f64; 3],
    center: &[f64; 3],
    polarization: &[f64; 3],
    pt: &[f64; 3],
) -> (f64, f64, f64) {
    let mut bx: f64 = 0.0;
    let mut by: f64 = 0.0;
    let mut bz: f64 = 0.0;

    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                let sign: f64 = (-1.0f64).powi(i + j + k + 1);
                let edge: [f64; 3] = [
                    size[0] / 2.0 * (-1.0f64).powi(i) + center[0],
                    size[1] / 2.0 * (-1.0f64).powi(j) + center[1],
                    size[2] / 2.0 * (-1.0f64).powi(k) + center[2],
                ];

                let dx: f64 = pt[0] - edge[0];
                let dy: f64 = pt[1] - edge[1];
                let dz: f64 = pt[2] - edge[2];
                let dr: f64 = (dx * dx + dy * dy + dz * dz).sqrt();

                bx += sign
                    * (polarization[0] * (dy * dz).atan2(dx * dr)
                        - polarization[1] * (dz + dr).ln()
                        - polarization[2] * (dy + dr).ln());

                by += sign
                    * (-polarization[0] * (dz + dr).ln()
                        + polarization[1] * (dx * dz).atan2(dy * dr)
                        - polarization[2] * (dx + dr).ln());

                bz += sign
                    * (-polarization[0] * (dy + dr).ln() - polarization[1] * (dx + dr).ln()
                        + polarization[2] * (dx * dy).atan2(dz * dr));
            }
        }
    }

    (bx, by, bz)
}

/// Calculate the analytical exterior field Jacobian `[partial_i B_j]`.
///
/// The rational factors use the derivation abbreviations
/// `f_xr_yz = dx^2 r^2 + (dy dz)^2` and `g_rzr = r dz + r^2` (and cyclic
/// permutations). The off-diagonal terms are symmetric because the exterior
/// field is curl-free.
pub fn calc_dbfield(
    size: &[f64; 3],
    center: &[f64; 3],
    polarization: &[f64; 3],
    pt: &[f64; 3],
) -> [[f64; 3]; 3] {
    let mut grad = [[0.0f64; 3]; 3];

    for i in 0..2 {
        for j in 0..2 {
            for k in 0..2 {
                let sign: f64 = (-1.0f64).powi(i + j + k + 1);
                let edge: [f64; 3] = [
                    size[0] / 2.0 * (-1.0f64).powi(i) + center[0],
                    size[1] / 2.0 * (-1.0f64).powi(j) + center[1],
                    size[2] / 2.0 * (-1.0f64).powi(k) + center[2],
                ];

                let dx: f64 = pt[0] - edge[0];
                let dy: f64 = pt[1] - edge[1];
                let dz: f64 = pt[2] - edge[2];

                let dx2: f64 = dx * dx;
                let dy2: f64 = dy * dy;
                let dz2: f64 = dz * dz;

                let dxy: f64 = dx * dy;
                let dxz: f64 = dx * dz;
                let dyz: f64 = dy * dz;

                let dr2: f64 = dx2 + dy2 + dz2;
                let dr: f64 = dr2.sqrt();

                let f_xr_yz: f64 = dx2 * dr2 + dyz * dyz;
                let f_yr_xz: f64 = dy2 * dr2 + dxz * dxz;
                let f_zr_xy: f64 = dz2 * dr2 + dxy * dxy;

                let g_rxr: f64 = dr * dx + dr2;
                let g_ryr: f64 = dr * dy + dr2;
                let g_rzr: f64 = dr * dz + dr2;

                let mut component: [[f64; 3]; 3] = [[0.0f64; 3]; 3];

                // Diagonal Jacobian terms.
                component[0][0] = -polarization[0] * dyz * (dx2 + dr2) / (dr * f_xr_yz)
                    - polarization[1] * dx / g_rzr
                    - polarization[2] * dx / g_ryr;

                component[1][1] = -polarization[0] * dy / g_rzr
                    - polarization[1] * dxz * (dy2 + dr2) / (dr * f_yr_xz)
                    - polarization[2] * dy / g_rxr;

                component[2][2] = -polarization[0] * dz / g_ryr
                    - polarization[1] * dz / g_rxr
                    - polarization[2] * dxy * (dz2 + dr2) / (dr * f_zr_xy);

                // Off-diagonal terms; symmetry follows from curl(B) = 0 outside.
                component[0][1] = polarization[0] * dxz * (dr2 - dy2) / (dr * f_xr_yz)
                    - polarization[1] * dy / g_rzr
                    - polarization[2] / dr;

                component[0][2] = polarization[0] * dxy * (dr2 - dz2) / (dr * f_xr_yz)
                    - polarization[2] * dz / g_ryr
                    - polarization[1] / dr;

                component[1][2] = polarization[1] * dxy * (dr2 - dz2) / (dr * f_yr_xz)
                    - polarization[2] * dz / g_rxr
                    - polarization[0] / dr;

                // Fill the symmetric Jacobian entries.
                component[1][0] = component[0][1];
                component[2][0] = component[0][2];
                component[2][1] = component[1][2];

                for u in 0..3 {
                    for v in 0..3 {
                        grad[u][v] += sign * component[u][v];
                    }
                }
            }
        }
    }

    grad
}

#[cfg(test)]
mod tests {
    use super::{calc_bfield, calc_dbfield};

    #[test]
    fn field_and_gradient_are_finite_outside_the_cuboid() {
        let size = [2.0, 3.0, 4.0];
        let center = [0.2, -0.3, 0.4];
        let polarization = [0.1, -0.2, 0.3];
        let point = [7.0, 8.0, 9.0];

        let field = calc_bfield(&size, &center, &polarization, &point);
        let gradient = calc_dbfield(&size, &center, &polarization, &point);
        assert!([field.0, field.1, field.2]
            .iter()
            .all(|value| value.is_finite()));
        assert!(gradient.iter().flatten().all(|value| value.is_finite()));
        assert_eq!(gradient[0][1], gradient[1][0]);
        assert_eq!(gradient[0][2], gradient[2][0]);
        assert_eq!(gradient[1][2], gradient[2][1]);
    }

    #[test]
    fn analytical_gradient_matches_central_field_difference() {
        let size = [2.0, 3.0, 4.0];
        let center = [0.2, -0.3, 0.4];
        let polarization = [0.1, -0.2, 0.3];
        let point = [7.0, 8.0, 9.0];
        let gradient = calc_dbfield(&size, &center, &polarization, &point);
        let step = 1e-4;

        for derivative_axis in 0..3 {
            let mut lower = point;
            let mut upper = point;
            lower[derivative_axis] -= step;
            upper[derivative_axis] += step;
            let lower_field = calc_bfield(&size, &center, &polarization, &lower);
            let upper_field = calc_bfield(&size, &center, &polarization, &upper);
            let lower_values = [lower_field.0, lower_field.1, lower_field.2];
            let upper_values = [upper_field.0, upper_field.1, upper_field.2];

            for component in 0..3 {
                let difference = (upper_values[component] - lower_values[component]) / (2.0 * step);
                let scale = gradient[derivative_axis][component].abs().max(1e-12);
                let relative_error =
                    (difference - gradient[derivative_axis][component]).abs() / scale;
                assert!(
                    relative_error < 1e-5,
                    "axis={derivative_axis}, component={component}"
                );
            }
        }
    }
}
