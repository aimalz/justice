# -*- coding: utf-8 -*-
"""Prefixes tensors with left and right."""
from justice.align_model import graph_typecheck


def _get_lr_features(features, left_or_right: str):
    result = {}
    prefix = f"{left_or_right}."
    for key, value in features.items():
        if key.startswith(prefix):
            result[key[len(prefix):]] = value
    return result


def lr_per_side_sub_model_fn(sub_model_fn, features, params):
    return [
        sub_model_fn(_get_lr_features(features, lr), params=params)
        for lr in ["left", "right"]
    ]


def prefix_tensors(left: dict, right: dict):
    left = graph_typecheck.assert_tensor_dict(left)
    right = graph_typecheck.assert_tensor_dict(right)
    return prefix_dicts(left, right)


def prefix_dicts(left: dict, right: dict):
    result = {}
    for k, v in left.items():
        result[f"left.{k}"] = v
    for k, v in right.items():
        result[f"right.{k}"] = v
    return result
