# Changelog

## 1.0.0

Initial release of Microcubed in its new repository.

- Analytical magnetic stray fields and gradients for uniformly magnetized
  cuboids and arrangements, with a shared Python API for NumPy and parallel
  Rust backends.
- Projected union boundaries retaining concavities, holes, and disconnected
  components, alongside convex hulls.
- Polygon decomposition and reproducible 2D and 3D Voronoi grain generation,
  including conversion into compact cuboid arrangements.
- Field sampling and Matplotlib visualization of line, plane, and volume sections.
- Self-contained Jupyter examples with automatically generated README snippets.
- Markdown documentation with the Furo theme, executed code, and visual examples.
- Optional Ubermag/OOMMF validation of exterior fields and gradients, with error
  tables, comparison maps, and line profiles.
- Python 3.12–3.14 support, Maturin packaging, and GitHub Actions for
  cross-platform tests, code quality, documentation, examples, and distribution
  validation.
