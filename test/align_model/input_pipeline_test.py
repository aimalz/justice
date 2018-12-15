# -*- coding: utf-8 -*-
"""Tests for input pipeline generation."""

import pytest
import tensorflow as tf

import justice.features.example_pair
from justice import lightcurve
from justice.align_model import input_pipeline, synthetic_positives
from justice.features import metadata_features, example_pair


class FakePositivesBuilder(input_pipeline.PositivesDatasetBuilder):
    def __init__(self):
        super(FakePositivesBuilder, self).__init__(seed=65, lc_lookup_chunk_size=4)
        self.basic_synthetic_positives = synthetic_positives.BasicPositivesGenerator()
        self.positives_fex = example_pair.PairFexFromPointwiseFex(
            fex=metadata_features.MetadataValueExtractor(),
            label=True,
        )

    def generate_synthetic_pair(
        self, lc: lightcurve._LC
    ) -> justice.features.example_pair.FullPositivesPair:
        return self.basic_synthetic_positives.make_positive_pair(lc)

    @property
    def feature_extractor(self) -> example_pair.PairFeatureExtractor:
        return self.positives_fex


class FakeNegativesBuilder(input_pipeline.RandomNegativesDatasetBuilder):
    def __init__(self):
        super(FakeNegativesBuilder, self).__init__(seed=0)
        self.negatives_fex = example_pair.PairFexFromPointwiseFex(
            fex=metadata_features.MetadataValueExtractor(),
            label=False,
        )

    @property
    def feature_extractor(self) -> example_pair.PairFeatureExtractor:
        return self.negatives_fex


@pytest.mark.requires_real_data
def test_positives_training_dataset_obj_ids(tf_sess):
    dataset = FakePositivesBuilder().training_dataset()
    assert isinstance(dataset, tf.data.Dataset)
    next_value = dataset.make_one_shot_iterator().get_next()
    expected_object_ids = [
        65049768,
        65049782,
        65049823,
        65049907,
        65050106,
        65050131,
        65050181,
        # This one comes from training_set.
        29283382,
        65050213,
        65050287
    ]
    for expected_object_id in expected_object_ids:
        value = tf_sess.run(next_value)
        assert value == {
            'left.object_id': expected_object_id,
            'right.object_id': expected_object_id,
            'labels': 1
        }


@pytest.mark.requires_real_data
def test_negatives_training_dataset_obj_ids(tf_sess):
    dataset = FakeNegativesBuilder().training_dataset()
    assert isinstance(dataset, tf.data.Dataset)
    next_value = dataset.make_one_shot_iterator().get_next()
    for _ in range(3):
        value = tf_sess.run(next_value)
        assert value['left.object_id'] != value['right.object_id']
        assert value['labels'] == 0
