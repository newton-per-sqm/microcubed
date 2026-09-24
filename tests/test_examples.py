"""Keep the README synchronized and notebook sources clean and compilable.

Actual kernel execution and numerical assertions run in the examples CI job.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "examples").rglob("*.ipynb"))


def test_readme_matches_notebooks():
    assert NOTEBOOKS
    subprocess.run([sys.executable, str(ROOT / "tools/notebooks.py"), "--check"], check=True)


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda path: path.stem)
def test_notebook_sources(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert not cell["outputs"]
            compile("".join(cell["source"]), f"{path.name}:cell-{index}", "exec")
