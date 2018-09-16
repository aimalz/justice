# -*- coding: utf-8 -*-
"""Defines OGLE datasets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

from justice import mmap_array

ogle_dir = os.path.join(mmap_array.default_array_dir, 'ogle_iii')


def for_subset(name):
    return mmap_array.IndexedArrayDescriptor(base_dir=os.path.join(ogle_dir, name))
