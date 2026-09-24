# References and further reading

## Analytical field of a cuboid

R. Ravaud and G. Lemarquand, “Magnetic Field Produced by a Parallelepipedic
Magnet of Various and Uniform Polarization,” *Progress In Electromagnetics
Research*, vol. 98, pp. 207–219, 2009.
[doi:10.2528/PIER09091704](https://doi.org/10.2528/PIER09091704).

This paper derives the scalar potential and all three field components of a
uniformly and arbitrarily polarized parallelepiped. It is the primary basis for
the Microcubed field equations.

## Demagnetizing tensor for cell models

A. J. Newell, W. Williams, and D. J. Dunlop, “A Generalization of the
Demagnetizing Tensor for Nonuniform Magnetization,” *Journal of Geophysical
Research: Solid Earth*, vol. 98, no. B6, pp. 9551–9555, 1993.
[doi:10.1029/93JB00694](https://doi.org/10.1029/93JB00694).

The paper describes mutual demagnetizing tensors for uniformly magnetized
bodies and gives explicit formulas for a block model. This methodology
underlies rectangular finite-difference demagnetizing kernels.

## OOMMF

M. J. Donahue and D. G. Porter, *OOMMF User's Guide*, National Institute of
Standards and Technology,
[`Oxs_Demag` section](https://math.nist.gov/oommf/doc/userguide21a0/userguidexml/sec_oxsEnergies.html).

The manual documents constant magnetization per cell, cell-averaged
demagnetizing fields, FFT convolution, and the analytical and asymptotic
evaluation of the demagnetizing kernel.

## Ubermag example

Ubermag Developers,
[“Calculating a stray field using an airbox method”](https://ubermag.github.io/examples/notebooks/12-tutorial-stray-field.html).

This external tutorial demonstrates an alternative approach using a micromagnetic solver.

## Reproducible project examples

See {doc}`examples` for the self-contained notebooks executed by GitHub Actions.
