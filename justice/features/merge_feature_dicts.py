# -*- coding: utf-8 -*-
"""Utility function to merge feature dicts."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def merge_feature_dicts(feature_dicts):
    feature_dict = {}
    for dct in feature_dicts:
        for key, value in dct.items():
            if key in feature_dict:
                raise ValueError(f"Got multiple values for key {key!r}.")
            feature_dict[key] = value
    return feature_dict
