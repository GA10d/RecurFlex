"""Execute the result reader and save its tables and images without a kernel server."""
import json
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULT = HERE.parent / "result"
os.environ.setdefault("MPLCONFIGDIR", str(RESULT / ".plotcache"))
os.environ.setdefault("IPYTHONDIR", str(RESULT / ".ipython"))

import nbformat
from IPython.core.interactiveshell import InteractiveShell
from IPython.utils.capture import capture_output


def run():
    if not (RESULT / "verification.json").exists():
        raise FileNotFoundError("Run score_and_report.py before executing the result notebook.")
    path = HERE / "results.ipynb"
    notebook = nbformat.read(path, as_version=4)
    shell = InteractiveShell.instance()
    previous = Path.cwd()
    count = 0
    try:
        os.chdir(HERE)
        for cell in notebook.cells:
            if cell.cell_type != "code":
                continue
            count += 1
            with capture_output() as captured:
                outcome = shell.run_cell(cell.source, store_history=True)
            failure = outcome.error_before_exec or outcome.error_in_exec
            if failure:
                raise failure
            outputs = []
            if captured.stdout:
                outputs.append(nbformat.v4.new_output("stream", name="stdout", text=captured.stdout))
            if captured.stderr:
                outputs.append(nbformat.v4.new_output("stream", name="stderr", text=captured.stderr))
            outputs.extend(nbformat.v4.new_output("display_data", data=item.data, metadata=item.metadata)
                           for item in captured.outputs)
            cell.outputs = outputs
            cell.execution_count = count
        nbformat.validate(notebook)
        nbformat.write(notebook, path)
    finally:
        os.chdir(previous)
    print(json.dumps({"notebook_executed": True, "code_cells": count, "file": str(path)}))


if __name__ == "__main__":
    run()
