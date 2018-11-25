# -*- coding: utf-8 -*-
"""Tests for basic vector similarity model."""
import pytest
import tensorflow as tf

from justice.align_model import basic_vector_similarity_model, lr_prefixing
from justice.datasets import plasticc_data
from justice.features import band_settings_params, raw_value_features


def _prefix_lr(left, right):
    result = lr_prefixing.prefix_tensors(left, right)
    result["labels"] = tf.constant(False)
    return result


def _make_test_dataset(params):
    source = plasticc_data.PlasticcBcolzSource.get_default()
    lcs = plasticc_data.PlasticcDatasetLC.bcolz_get_lcs_by_obj_ids(
        bcolz_source=source, dataset="training_set", obj_ids=[745, 10757]
    )
    band_settings = band_settings_params.BandSettings(bands=params["lc_bands"])
    fex = raw_value_features.RawValueExtractor(
        window_size=params["window_size"], band_settings=band_settings
    )
    first_point_features = fex.extract(lcs[0], lcs[0].all_times_unique()[10])
    second_point_features = fex.extract(lcs[1], lcs[1].all_times_unique()[10])
    dataset1 = tf.data.Dataset.from_tensors(first_point_features)
    dataset2 = tf.data.Dataset.from_tensors(second_point_features)
    dataset = tf.data.Dataset.zip((dataset1, dataset2)).map(_prefix_lr)
    return dataset.batch(1, drop_remainder=True)


@pytest.mark.requires_real_data
def test_basic_similarity_model(tf_sess):

    estimator = tf.estimator.Estimator(
        model_fn=basic_vector_similarity_model.model_fn,
        params={
            "batch_size": 1,
            "lc_bands": plasticc_data.PlasticcDatasetLC.expected_bands,
            "window_size": 10,
        }
    )
    estimator.train(_make_test_dataset, max_steps=1)
    variable_names = [x for x in estimator.get_variable_names() if '/Adam' not in x]
    assert variable_names == [
        'beta1_power', 'beta2_power', 'global_step', 'per_side_model/layer_0_dense/bias',
        'per_side_model/layer_0_dense/kernel', 'per_side_model/output_dense/bias',
        'per_side_model/output_dense/kernel'
    ]
