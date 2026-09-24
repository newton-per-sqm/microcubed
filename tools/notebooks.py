"""Generate README examples and execute/export the notebook suite."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
START = "<!-- examples:start -->"
END = "<!-- examples:end -->"


def readme_examples(paths: list[Path]) -> str:
    blocks = []
    for path in paths:
        notebook = json.loads(path.read_text(encoding="utf-8"))
        for cell in notebook["cells"]:
            if "readme" not in cell.get("metadata", {}).get("tags", []):
                continue
            source = "".join(cell["source"]).strip()
            if cell["cell_type"] == "markdown":
                blocks.append("##" + source)
                blocks.append(f"[Open notebook](examples/{path.name})")
            else:
                blocks.append(f"```python\n{source}\n```")
    return "\n\n".join(blocks)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if README examples are stale")
    parser.add_argument("--execute", action="store_true", help="Execute notebooks and export HTML")
    parser.add_argument(
        "--include-optional", action="store_true", help="Also execute examples requiring external solvers"
    )
    args = parser.parse_args()
    paths = sorted((ROOT / "examples").glob("*.ipynb"))
    if not paths:
        raise SystemExit("No example notebooks found")
    readme = ROOT / "README.md"
    original = readme.read_text(encoding="utf-8")
    before, remainder = original.split(START, 1)
    _, after = remainder.split(END, 1)
    updated = before + START + "\n\n" + readme_examples(paths) + "\n\n" + END + after
    if args.check:
        if updated != original:
            raise SystemExit("README examples are stale; run: python tools/notebooks.py")
    else:
        readme.write_text(updated, encoding="utf-8")
    if args.execute:
        import nbformat
        from nbconvert import HTMLExporter
        from nbconvert.preprocessors import ExecutePreprocessor

        # Use the same headless rendering locally and on GitHub runners.
        os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
        output = ROOT / "build" / "examples"
        output.mkdir(parents=True, exist_ok=True)
        execution_paths = paths + (
            sorted((ROOT / "examples/optional").glob("*.ipynb")) if args.include_optional else []
        )
        for path in execution_paths:
            print(f"Executing {path.name}", flush=True)
            notebook = nbformat.read(path, as_version=4)
            nbformat.validate(notebook)
            ExecutePreprocessor(
                timeout=600 if path.parent.name == "optional" else 180, kernel_name="python3"
            ).preprocess(notebook, {"metadata": {"path": str(ROOT)}})
            nbformat.write(notebook, output / path.name)
            html, _ = HTMLExporter().from_notebook_node(notebook)
            (output / f"{path.stem}.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
