# -*- coding: utf-8 -*-
"""Checks that IPython notebooks have been cleared."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import json
import os.path

base_dir = os.path.abspath(os.path.join(os.path.abspath(__file__), "../.."))


def test_ipython_cleared():
    notebooks = glob.glob(os.path.join(base_dir, "*.ipynb"))
    assert len(notebooks) >= 2, "Couldn't find notebooks in {}".format(base_dir)
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
