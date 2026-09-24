"""Sphinx configuration for the installed Microcubed package."""

import os
from importlib.metadata import version as package_version
from pathlib import Path

project = "microcubed"
copyright = "2024–2026, Pascal Muster"
release = package_version(project)
version = release
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "myst_nb",
]
myst_enable_extensions = ["amsmath", "colon_fence", "deflist", "dollarmath", "linkify", "tasklist"]
source_suffix = {".md": "myst-nb", ".ipynb": "myst-nb"}
master_doc = "index"
exclude_patterns = ["_build", "api", "readme.md"]
html_theme = "furo"
html_title = "Microcubed"
html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#0f766e",
        "color-brand-content": "#0f766e",
    },
    "dark_css_variables": {
        "color-brand-primary": "#5eead4",
        "color-brand-content": "#5eead4",
    },
}
html_static_path = ["_static"]


# Execute the canonical examples when Sphinx reads their generated copies.
nb_execution_mode = "force"
nb_execution_timeout = 180
nb_execution_raise_on_error = True
nb_execution_in_temp = True
nb_execution_excludepatterns = (
    [] if os.environ.get("MICROCUBED_RUN_OOMMF") == "1" else ["_notebooks/compare_ubermag.ipynb"]
)


def prepare_notebooks(app):
    root = Path(__file__).resolve().parents[1]
    destination = root / "docs/_notebooks"
    destination.mkdir(exist_ok=True)
    sources = {path.name: path for path in (root / "examples").rglob("*.ipynb")}
    for stale in destination.glob("*.ipynb"):
        if stale.name not in sources:
            stale.unlink()
    for name, source in sources.items():
        (destination / name).write_bytes(source.read_bytes())
    os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"


def setup(app):
    app.connect("builder-inited", prepare_notebooks)
