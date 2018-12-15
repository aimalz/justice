# -*- coding: utf-8 -*-
"""Pair of negative or positive examples."""
import abc
import random
import typing

import tensorflow as tf

from justice import lightcurve
from justice.align_model import lr_prefixing
from justice.features import pointwise_feature_extractor, tf_dataset_builder


class ExamplePair:
    """Simple struct for a pair of examples."""

    __slots__ = ("lca", "lcb", "time_a", "time_b")

    def __init__(
        self, lca: lightcurve._LC, lcb: lightcurve._LC, time_a: float, time_b: float
    ):
        self.lca = lca
        self.lcb = lcb
        self.time_a = time_a
        self.time_b = time_b


class FullPositivesPair(ExamplePair):
    """Simple struct for a pair of positive examples.

    The meaning is that `time_a` of light curve `lca` should be aligned with `time_b` of
    light curve `lcb`. For the current alignment model, we do not have other gold/intended
    transformation parameters, we merely hope that points having the best alignment will
    lead to reasonable inference of dilation.
    """


class FullNegativesPair(ExamplePair):
    """Simple struct for a pair of positive examples.

    The meaning is that `time_a` of light curve `lca` should not be aligned with `time_b`
    of light curve `lcb`. For 'training_set' data, `lca` and `lcb` can be from different
    classes, bun in general we hope that  picking random points and saying these should
    not be aligned will be okay.
    """


class PairFeatureExtractor(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def apply(self, fpp: ExamplePair) -> typing.Dict[str, typing.Any]:
        raise NotImplementedError()

    def make_dataset(self, fpp_gen: typing.Iterator[ExamplePair]) -> tf.data.Dataset:
        return tf_dataset_builder.dataset_from_generator_auto_dtypes(
            self.apply(fpp) for fpp in fpp_gen
        )


class PairFexFromPointwiseFex(PairFeatureExtractor):
    def __init__(
        self, *, fex: pointwise_feature_extractor.PointwiseFeatureExtractor, label: bool
    ):
        self.fex = fex
        self.label = label

    def apply(self, example_pair: ExamplePair) -> typing.Dict[str, typing.Any]:
        first_features = self.fex.extract(example_pair.lca, example_pair.time_a)
        second_features = self.fex.extract(example_pair.lcb, example_pair.time_b)
        result = lr_prefixing.prefix_dicts(first_features, second_features)
        result["labels"] = self.label
        return result


def random_points_for_negative_pair(
    lca: lightcurve._LC, lcb: lightcurve._LC, rng: random.Random = None
) -> FullNegativesPair:
    if rng is None:
        rng = random.Random()
    time_a = rng.choice(lca.all_times_unique())
    time_b = rng.choice(lcb.all_times_unique())
    return FullNegativesPair(
        lca=lca,
        lcb=lcb,
        time_a=time_a,
        time_b=time_b,
    )
