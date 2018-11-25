# -*- coding: utf-8 -*-
"""Helps resolve paths."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pathlib

project_root: pathlib.Path = pathlib.Path(__file__).parent.parent
data_dir: pathlib.Path = project_root / 'data'
models_dir: pathlib.Path = project_root / 'models'
tf_align_data: pathlib.Path = data_dir / 'tf_align_model'
