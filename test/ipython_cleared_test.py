# -*- coding: utf-8 -*-
"""Checks that IPython notebooks have been cleared."""
import json
import subprocess

from justice import path_util


def test_ipython_cleared():
    project_root = str(path_util.project_root)
    notebooks = subprocess.check_output([
        'git', 'ls-files', '*.ipynb'
    ], cwd=project_root).decode("utf-8").strip().split("\n")
    assert len(notebooks) >= 2, f"Couldn't find notebooks in {path_util.project_root}"
    for notebook in notebooks:
        with open(notebook) as f:
            contents = json.load(f)
        cells = contents["cells"]
        for cell in cells:
            cell_str = "Bad cell with contents {!r}, file {}".format(
                "".join(cell["source"]), notebook
            )
            assert cell.get("execution_count") is None, cell_str
            if cell.get("outputs"):
                raise ValueError(cell_str)
