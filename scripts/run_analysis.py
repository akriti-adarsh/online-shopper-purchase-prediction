"""Execute the analysis notebook and rebuild the published result artifacts."""

from pathlib import Path
import subprocess
import sys

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient


def main():
    root = Path(__file__).resolve().parents[1]
    notebook_path = root / "model" / "Online_Shoppers_Model_Analysis.ipynb"
    notebook = nbformat.read(notebook_path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    # Use the interpreter running this script, regardless of installed kernels.
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv = [
        sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"
    ]

    def announce(cell, cell_index):
        if cell.cell_type == "code":
            print(f"Executing notebook cell {cell_index + 1}", flush=True)

    client = NotebookClient(
        notebook,
        km=manager,
        timeout=600,
        resources={"metadata": {"path": str(root)}},
        record_timing=False,
        on_cell_start=announce,
    )
    client.execute(cleanup_kc=True)
    nbformat.validate(notebook)
    with notebook_path.open("w", encoding="utf-8", newline="\n") as stream:
        nbformat.write(notebook, stream)
    subprocess.run([sys.executable, str(root / "scripts" / "build_report.py")],
                   cwd=root, check=True)
    print("Analysis complete. Notebook, models, tables, and figures are refreshed.")


if __name__ == "__main__":
    main()
