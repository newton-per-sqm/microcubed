//! Python bindings and parallel dispatch for Microcubed's analytical kernels.
//!
//! This layer maps Python arrays with shape `(3, N)` to the closed-form cuboid
//! equations in `equations.rs`. Rayon parallelises independent observation
//! points only; it does not approximate the field or its analytical Jacobian.
use indicatif::{ProgressBar, ProgressStyle};
use ndarray::Array2;
use numpy::{PyArray2, PyArrayMethods, ToPyArray};
use pyo3::prelude::*;
use pyo3::types::PyType;
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use std::f64::consts::PI;

mod equations;

const SUB: usize = 1; // Subtraction constant for cpu count
const MU0: f64 = 4.0 * PI * 1e-7; // Permeability of free space in (T * m / A)
const J0: f64 = MU0 / (4.0 * PI); // Conversion factor for magnetization to polarization

#[pyclass(from_py_object)]
#[derive(Clone)]
/// Represents one axis-aligned, uniformly magnetised cuboid.
///
/// Geometry uses the caller's length unit and magnetisation is in A/m. The
/// exterior field has unit tesla when the public Python API constructs this
/// object through `RustMagnet`.
pub struct RustyMagnet {
    size: [f64; 3],          // Size of the cuboid magnet (length, width, height)
    center: [f64; 3],        // Center of the cuboid magnet (x, y, z)
    magnetization: [f64; 3], // Magnetization vector (mx, my, mz)
}

#[pymethods]
/// Python-visible operations for one analytical cuboid magnet.
impl RustyMagnet {
    #[new]
    pub fn new(size: [f64; 3], center: [f64; 3], magnetization: [f64; 3]) -> Self {
        Self {
            size,
            center,
            magnetization,
        }
    }

    /// Create a RustyMagnet from two points defining the bounding box and a magnetization vector.
    #[classmethod]
    pub fn from_bbox(
        _cls: &Bound<'_, PyType>,
        p1: &Bound<'_, PyArray2<f64>>,
        p2: &Bound<'_, PyArray2<f64>>,
        magnetization: Vec<f64>,
    ) -> PyResult<Self> {
        // Accept p1 and p2 as (3,) or (3,1) arrays
        let p1_owned = unsafe { p1.as_array() }.to_owned();
        let p1 = p1_owned
            .to_shape((3,))
            .or_else(|_| p1_owned.to_shape((3,)))
            .map_err(|_| {
                pyo3::exceptions::PyValueError::new_err("p1 must be convertible to shape (3,)")
            })?;
        let p2_owned = unsafe { p2.as_array() }.to_owned();
        let p2 = p2_owned
            .to_shape((3,))
            .or_else(|_| p2_owned.to_shape((3,)))
            .map_err(|_| {
                pyo3::exceptions::PyValueError::new_err("p2 must be convertible to shape (3,)")
            })?;

        let size = [
            (p1[0] - p2[0]).abs(),
            (p1[1] - p2[1]).abs(),
            (p1[2] - p2[2]).abs(),
        ];
        let center = [
            (p1[0] + p2[0]) / 2.0,
            (p1[1] + p2[1]) / 2.0,
            (p1[2] + p2[2]) / 2.0,
        ];

        let magnetization: [f64; 3] = if magnetization.len() == 3 {
            [magnetization[0], magnetization[1], magnetization[2]]
        } else {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "magnetization must be a list or array of length 3",
            ));
        };

        Ok(Self {
            size,
            center,
            magnetization,
        })
    }

    #[getter]
    pub fn polarization(&self) -> [f64; 3] {
        // Convert magnetization to polarization
        [
            -J0 * self.magnetization[0],
            -J0 * self.magnetization[1],
            -J0 * self.magnetization[2],
        ]
    }

    #[getter]
    pub fn bbox(&self) -> [f64; 6] {
        // Calculate bounding box coordinates
        let half_size: [f64; 3] = [self.size[0] / 2.0, self.size[1] / 2.0, self.size[2] / 2.0];
        [
            self.center[0] - half_size[0],
            self.center[1] - half_size[1],
            self.center[2] - half_size[2],
            self.center[0] + half_size[0],
            self.center[1] + half_size[1],
            self.center[2] + half_size[2],
        ]
    }

    #[getter]
    pub fn volume(&self) -> f64 {
        // Calculate the volume of the cuboid magnet
        self.size[0] * self.size[1] * self.size[2]
    }

    #[getter]
    pub fn corners(&self) -> Vec<[f64; 3]> {
        // Calculate the 8 corners of the cuboid magnet
        let half_size: [f64; 3] = [self.size[0] / 2.0, self.size[1] / 2.0, self.size[2] / 2.0];
        let mut corners: Vec<[f64; 3]> = Vec::with_capacity(8);
        for dx in [-1.0, 1.0].iter() {
            for dy in [-1.0, 1.0].iter() {
                for dz in [-1.0, 1.0].iter() {
                    corners.push([
                        self.center[0] + dx * half_size[0],
                        self.center[1] + dy * half_size[1],
                        self.center[2] + dz * half_size[2],
                    ]);
                }
            }
        }
        corners
    }

    #[pyo3(name = "__repr__")]
    fn repr(&self) -> String {
        format!(
            "RustyMagnet(size={:?}, center={:?}, magnetization={:?})",
            self.size, self.center, self.magnetization
        )
    }

    #[pyo3(name = "__eq__")]
    fn eq(&self, other: &Self) -> bool {
        self.size == other.size
            && self.center == other.center
            && self.magnetization == other.magnetization
    }

    /// Calculate analytical exterior flux density at points shaped `(3, N)`.
    ///
    /// The result has shape `(3, N)`. Each column is evaluated independently
    /// with the eight-corner closed-form kernel and may run in parallel.
    pub fn bfield<'py>(
        &self,
        py: Python<'py>,
        points: &Bound<'py, PyArray2<f64>>,
    ) -> Bound<'py, PyArray2<f64>> {
        let points = unsafe { points.as_array() };
        let shape = points.shape();
        if shape.len() != 2 || shape[0] != 3 {
            panic!("Input points array must have shape (3, N). Got {:?}", shape);
        }
        let n_points = shape[1];

        let flat_result: Vec<f64> = (0..n_points)
            .into_par_iter()
            .flat_map(|p| {
                let pt: [f64; 3] = [points[[0, p]], points[[1, p]], points[[2, p]]];
                let (bx, by, bz) =
                    equations::calc_bfield(&self.size, &self.center, &self.polarization(), &pt);
                vec![bx, by, bz]
            })
            .collect();

        let result = Array2::from_shape_vec((n_points, 3), flat_result).unwrap();
        result.t().to_pyarray(py)
    }

    /// Calculate the analytical Jacobian `[partial_i B_j]` at `(3, N)` points.
    ///
    /// The returned flat array has shape `(9, N)` and is reshaped to
    /// `(3, 3, N)` by the Python wrapper. The leading index is the derivative
    /// axis and the second index is the field component.
    pub fn dbfield<'py>(
        &self,
        py: Python<'py>,
        points: &Bound<'py, PyArray2<f64>>,
    ) -> Bound<'py, PyArray2<f64>> {
        let points = unsafe { points.as_array() };
        let shape = points.shape();
        if shape.len() != 2 || shape[0] != 3 {
            panic!("Input points array must have shape (3, N). Got {:?}", shape);
        }
        let n_points = shape[1];

        let flat_result: Vec<f64> = (0..n_points)
            .into_par_iter()
            .flat_map(|p| {
                let pt: [f64; 3] = [points[[0, p]], points[[1, p]], points[[2, p]]];
                let grad: [[f64; 3]; 3] =
                    equations::calc_dbfield(&self.size, &self.center, &self.polarization(), &pt);
                grad.iter()
                    .flat_map(|r: &[f64; 3]| r.iter())
                    .copied()
                    .collect::<Vec<_>>()
            })
            .collect();

        let result = Array2::from_shape_vec((n_points, 9), flat_result).unwrap();
        result.t().to_pyarray(py)
    }

    /// Create a new RustyMagnet mirrored across the specified plane(s).
    #[pyo3(signature = (x = None, y = None, z = None))]
    pub fn mirrored(&self, x: Option<f64>, y: Option<f64>, z: Option<f64>) -> Self {
        let mut new_center = self.center;
        if let Some(xv) = x {
            new_center[0] = 2.0 * xv - new_center[0];
        }
        if let Some(yv) = y {
            new_center[1] = 2.0 * yv - new_center[1];
        }
        if let Some(zv) = z {
            new_center[2] = 2.0 * zv - new_center[2];
        }
        Self {
            size: self.size,
            center: new_center,
            magnetization: self.magnetization,
        }
    }

    /// Create a new RustyMagnet with the center moved by the given vector.
    pub fn moved_by(&self, diff: Vec<f64>) -> Self {
        assert!(diff.len() == 3, "diff must be a vector of length 3");
        let new_center = [
            self.center[0] + diff[0],
            self.center[1] + diff[1],
            self.center[2] + diff[2],
        ];
        Self {
            size: self.size,
            center: new_center,
            magnetization: self.magnetization,
        }
    }

    /// Create a new RustyMagnet with the center moved to the given position.
    pub fn moved_to(&self, center: Vec<f64>) -> Self {
        assert!(center.len() == 3, "center must be a vector of length 3");
        let diff = [
            center[0] - self.center[0],
            center[1] - self.center[1],
            center[2] - self.center[2],
        ];
        self.moved_by(diff.to_vec())
    }

    /// Check if this magnet overlaps (axis-aligned) with another cuboid magnet.
    pub fn overlapping(&self, other: &RustyMagnet) -> bool {
        let a_bbox: [f64; 6] = self.bbox();
        let b_bbox: [f64; 6] = other.bbox();
        (a_bbox[0] < b_bbox[3])
            && (a_bbox[3] > b_bbox[0])
            && (a_bbox[1] < b_bbox[4])
            && (a_bbox[4] > b_bbox[1])
            && (a_bbox[2] < b_bbox[5])
            && (a_bbox[5] > b_bbox[2])
    }

    /// Check if this magnet is touching (axis-aligned) another cuboid magnet.
    pub fn touching(&self, other: &RustyMagnet) -> bool {
        let a_bbox: [f64; 6] = self.bbox();
        let b_bbox: [f64; 6] = other.bbox();
        (a_bbox[0] <= b_bbox[3])
            && (a_bbox[3] >= b_bbox[0])
            && (a_bbox[1] <= b_bbox[4])
            && (a_bbox[4] >= b_bbox[1])
            && (a_bbox[2] <= b_bbox[5])
            && (a_bbox[5] >= b_bbox[2])
    }
}

#[pyclass(from_py_object)]
#[derive(Clone)]
/// Represents a collection of RustyMagnets arranged together.
pub struct RustyArrangement {
    magnets: Vec<RustyMagnet>, // List of RustyMagnets in the arrangement
}

#[pymethods]
/// Methods for the RustyArrangement class.
impl RustyArrangement {
    #[new]
    pub fn new(magnets: Vec<RustyMagnet>) -> Self {
        Self { magnets }
    }

    /// Add a magnet to the arrangement
    pub fn add_magnet(&mut self, magnet: RustyMagnet) {
        self.magnets.push(magnet);
    }

    #[getter]
    pub fn bbox(&self) -> [f64; 6] {
        // Calculate the bounding box that contains all magnets
        let mut min_x: f64 = f64::INFINITY;
        let mut min_y: f64 = f64::INFINITY;
        let mut min_z: f64 = f64::INFINITY;
        let mut max_x: f64 = f64::NEG_INFINITY;
        let mut max_y: f64 = f64::NEG_INFINITY;
        let mut max_z: f64 = f64::NEG_INFINITY;

        for magnet in &self.magnets {
            let bbox: [f64; 6] = magnet.bbox();
            min_x = min_x.min(bbox[0]);
            min_y = min_y.min(bbox[1]);
            min_z = min_z.min(bbox[2]);
            max_x = max_x.max(bbox[3]);
            max_y = max_y.max(bbox[4]);
            max_z = max_z.max(bbox[5]);
        }

        [min_x, min_y, min_z, max_x, max_y, max_z]
    }

    #[getter]
    pub fn corners(&self) -> Vec<[f64; 3]> {
        // Calculate the corners of the bounding box
        let bbox: [f64; 6] = self.bbox();
        let mut corners: Vec<[f64; 3]> = Vec::with_capacity(8);
        for dx in [-1.0, 1.0].iter() {
            for dy in [-1.0, 1.0].iter() {
                for dz in [-1.0, 1.0].iter() {
                    corners.push([
                        bbox[0] + dx * (bbox[3] - bbox[0]) / 2.0,
                        bbox[1] + dy * (bbox[4] - bbox[1]) / 2.0,
                        bbox[2] + dz * (bbox[5] - bbox[2]) / 2.0,
                    ]);
                }
            }
        }
        corners
    }

    /// Get number of magnets in the arrangement
    #[pyo3(name = "__len__")]
    fn len(&self) -> usize {
        self.magnets.len()
    }

    /// Get a string representation of the arrangement
    #[pyo3(name = "__repr__")]
    fn repr(&self) -> String {
        format!("RustyArrangement({} RustyMagnets)", self.len())
    }

    /// Combine two arrangements by concatenating their magnets.
    #[pyo3(name = "__add__")]
    fn add(&self, other: &Self) -> Self {
        let mut new_magnets = self.magnets.clone();
        new_magnets.extend_from_slice(&other.magnets);
        Self {
            magnets: new_magnets,
        }
    }

    /// Calculate the magnetic field at given points for all magnets in the arrangement.
    pub fn bfield<'py>(
        &self,
        py: Python<'py>,
        points: &Bound<'py, PyArray2<f64>>,
    ) -> Bound<'py, PyArray2<f64>> {
        // Limit to available_cpus() - 1, but at least 1
        let n_threads = std::cmp::max(1, num_cpus::get().saturating_sub(SUB));
        let pool = ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .unwrap();

        let points = unsafe { points.as_array() };
        let shape = points.shape();
        let n_points = shape[1];
        let n_magnets = self.magnets.len();

        // Only show progress bar if there are enough magnets and points
        let show_progress = n_magnets >= 100 && n_points >= 100;

        let bar = if show_progress {
            let bar = ProgressBar::new(n_magnets as u64);
            bar.set_style(
                ProgressStyle::with_template(
                    "[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} magnets ({eta})",
                )
                .unwrap()
                .progress_chars("##-"),
            );
            Some(bar)
        } else {
            None
        };

        let mut result = Array2::<f64>::zeros((3, n_points));
        // Tune as needed for memory/performance tradeoff
        let chunk_size = std::cmp::max(1, num_cpus::get() / 2);

        pool.install(|| {
            self.magnets
                .par_chunks(chunk_size)
                .map_init(
                    || bar.clone(),
                    |bar, chunk| {
                        let mut chunk_sum = Array2::<f64>::zeros((3, n_points));
                        for magnet in chunk {
                            let polarization = magnet.polarization();
                            let mut flat_result = Vec::with_capacity(n_points * 3);
                            for p in 0..n_points {
                                let pt = [points[[0, p]], points[[1, p]], points[[2, p]]];
                                let (bx, by, bz) = equations::calc_bfield(
                                    &magnet.size,
                                    &magnet.center,
                                    &polarization,
                                    &pt,
                                );
                                flat_result.push(bx);
                                flat_result.push(by);
                                flat_result.push(bz);
                            }
                            let arr = Array2::from_shape_vec((n_points, 3), flat_result)
                                .unwrap()
                                .t()
                                .to_owned();
                            chunk_sum += &arr;
                            if let Some(bar) = bar.as_ref() {
                                bar.inc(1);
                            }
                        }
                        chunk_sum
                    },
                )
                .collect::<Vec<_>>()
                .into_iter()
                .for_each(|b| {
                    result += &b;
                });
        });

        if let Some(bar) = bar.as_ref() {
            bar.finish_and_clear();
        }

        result.to_pyarray(py)
    }

    /// Calculate the gradient of the magnetic field at given points for all magnets in the arrangement.
    ///
    /// A progress bar is shown only if both the number of magnets and the number of points are at least 100.
    pub fn dbfield<'py>(
        &self,
        py: Python<'py>,
        points: &Bound<'py, PyArray2<f64>>,
    ) -> Bound<'py, PyArray2<f64>> {
        // Limit to available_cpus() - 1, but at least 1
        let n_threads = std::cmp::max(1, num_cpus::get().saturating_sub(SUB));
        let pool = ThreadPoolBuilder::new()
            .num_threads(n_threads)
            .build()
            .unwrap();

        let points = unsafe { points.as_array() };
        let shape = points.shape();
        let n_points = shape[1];
        let n_magnets = self.magnets.len();

        // Only show progress bar if there are enough magnets and points
        let show_progress = n_magnets >= 100 && n_points >= 100;

        let bar = if show_progress {
            let bar = ProgressBar::new(n_magnets as u64);
            bar.set_style(
                ProgressStyle::with_template(
                    "[{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} magnets ({eta})",
                )
                .unwrap()
                .progress_chars("##-"),
            );
            Some(bar)
        } else {
            None
        };

        // Note: The result array is initialized with shape (9, n_points), matching the transposed output of the partials.
        let mut result = Array2::<f64>::zeros((9, n_points));
        // Tune as needed for memory/performance tradeoff
        let chunk_size = std::cmp::max(1, num_cpus::get() / 2);

        pool.install(|| {
            self.magnets
                .par_chunks(chunk_size)
                .map_init(
                    || bar.clone(),
                    |bar, chunk| {
                        let mut chunk_sum = Array2::<f64>::zeros((9, n_points));
                        for magnet in chunk {
                            let polarization = magnet.polarization();
                            for p in 0..n_points {
                                let pt = [points[[0, p]], points[[1, p]], points[[2, p]]];
                                let grad = equations::calc_dbfield(
                                    &magnet.size,
                                    &magnet.center,
                                    &polarization,
                                    &pt,
                                );
                                for r in 0..3 {
                                    for c in 0..3 {
                                        chunk_sum[[r * 3 + c, p]] += grad[r][c];
                                    }
                                }
                            }
                            if let Some(bar) = bar.as_ref() {
                                bar.inc(1);
                            }
                        }
                        chunk_sum
                    },
                )
                .collect::<Vec<_>>()
                .into_iter()
                .for_each(|db| {
                    result += &db;
                });
        });

        if let Some(bar) = bar.as_ref() {
            bar.finish_and_clear();
        }

        result.to_pyarray(py)
    }

    /// Check if the arrangement is valid (no intersections between magnets).
    pub fn valid(&self) -> bool {
        // Parallel brute-force check for overlapping magnets
        (0..self.magnets.len()).into_par_iter().all(|i| {
            ((i + 1)..self.magnets.len())
                .into_par_iter()
                .all(|j| !self.magnets[i].overlapping(&self.magnets[j]))
        })
    }

    /// Create a new RustyArrangement with all magnets mirrored across the specified plane(s).
    #[pyo3(signature = (x = None, y = None, z = None))]
    pub fn mirrored(&self, x: Option<f64>, y: Option<f64>, z: Option<f64>) -> Self {
        let magnets: Vec<RustyMagnet> = self
            .magnets
            .par_iter()
            .map(|m| m.mirrored(x, y, z))
            .collect();
        Self { magnets }
    }

    /// Create a new RustyArrangement with all magnets moved by the given vector.
    pub fn moved_by(&self, diff: Vec<f64>) -> Self {
        assert!(diff.len() == 3, "diff must be a vector of length 3");
        let magnets: Vec<RustyMagnet> = self
            .magnets
            .par_iter()
            .map(|m| m.moved_by(diff.clone()))
            .collect();
        Self { magnets }
    }

    /// Create a new RustyArrangement with all magnets moved to the given center positions.
    /// `centers` must be a (N, 3) array, where N is the number of magnets.
    pub fn moved_to(&self, centers: Vec<Vec<f64>>) -> Self {
        assert!(
            centers.len() == self.magnets.len(),
            "centers must have the same length as the number of magnets"
        );
        let magnets: Vec<RustyMagnet> = self
            .magnets
            .par_iter()
            .zip(centers.into_par_iter())
            .map(|(m, c)| m.moved_to(c))
            .collect();
        Self { magnets }
    }
}

#[pymodule]
/// Internal Rust kernels for Microcubed fields and gradients.
///
/// The public compatibility API is provided by ``microcubed``; this extension
/// contains only the private calculation kernels used by the ``rust`` backend.
fn _rust(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustyMagnet>()?;
    m.add_class::<RustyArrangement>()?;
    Ok(())
}
