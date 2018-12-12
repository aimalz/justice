# -*- coding: utf-8 -*-
"""Input pipeline helpers for training and evaluation.

Prediction codes should go elsewhere.

This module is intended to be fairly general, so put model-specific stuff like
RawValueExtractor elsewhere. On the other hand, plasticc-specific stuff should go here.
"""
import abc
import logging
import random
import time
import typing

import numpy as np
import tensorflow as tf

import justice.features.example_pair
from justice import lightcurve
from justice.datasets import plasticc_data, sharded_plasticc
from justice.features import tf_dataset_builder, example_pair, lc_id_features


def timed_generator(iterator, name, print_every=30.0, initial_delay=1.0):
    """Records the time it takes to dequeue elements from an iterator.

    Incrementally prints stats on those times.

    :param iterator: Iterator to time.
    :param name: Name of the iterator being timed.
    :param print_every: How many seconds to print.
    :param initial_delay: Initial delay before printing, in seconds.
    :yields: Elements from `iterator`.
    """
    start = time.time()
    last_printed = start - print_every + initial_delay
    elapsed = []
    for x in iterator:
        elapsed.append(time.time() - start)
        if time.time() - last_printed > print_every:
            ntiles = np.percentile(elapsed, q=np.linspace(0, 100, 5))
            print(f"Elapsed time percentiles for {name!r}: {ntiles} s")
            last_printed = time.time()
            elapsed = []
        yield x
        start = time.time()


class PositivesDatasetBuilder(metaclass=abc.ABCMeta):
    def __init__(
        self,
        *,
        seed: int,
        lc_lookup_chunk_size=100,
        source: plasticc_data.PlasticcBcolzSource = None
    ):
        self.seed = seed
        self.lc_lookup_chunk_size = lc_lookup_chunk_size
        self.source = (
            source
            if source is not None else plasticc_data.PlasticcBcolzSource.get_default()
        )

    @abc.abstractmethod
    def generate_synthetic_pair(
        self, lc: lightcurve._LC
    ) -> justice.features.example_pair.FullPositivesPair:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def feature_extractor(self) -> example_pair.PairFeatureExtractor:
        raise NotImplementedError()

    def lc_generator(self, ids_iterator,
                     dataset: str) -> typing.Iterator[plasticc_data.PlasticcDatasetLC]:
        id_chunks = sharded_plasticc.chunk_indefinite(
            ids_iterator, chunk_size=self.lc_lookup_chunk_size
        )
        for id_chunk in id_chunks:
            yield from plasticc_data.PlasticcDatasetLC.bcolz_get_lcs_by_obj_ids(
                bcolz_source=self.source, dataset=dataset, obj_ids=id_chunk
            )

    def _get_dataset(self, dataset: str, split: sharded_plasticc.TrainTune):
        """Gets a dataset for training or test sets.

        :param dataset: String "training_set" or "test_set".
        :param split: Which split of the data should be selected.
        """
        ids_iterator = sharded_plasticc.obj_id_iterator(
            self.seed, dataset=dataset, source=self.source
        )
        split_ids_iterator = sharded_plasticc.filter_by_split(ids_iterator, split)
        lcs_iterator = self.lc_generator(split_ids_iterator, dataset=dataset)
        positive_pairs = (self.generate_synthetic_pair(lc) for lc in lcs_iterator)
        extractor = self.feature_extractor
        features = (extractor.apply(fpp) for fpp in positive_pairs)
        features = timed_generator(features, f"positive example generator ({dataset})")
        return tf_dataset_builder.dataset_from_generator_auto_dtypes(features)

    def dataset_length(self, dataset: str) -> int:
        meta_table = self.source.get_table(f"{dataset}_metadata")
        return len(meta_table['object_id'])

    def dataset_for_split(self, split: sharded_plasticc.TrainTune) -> tf.data.Dataset:
        weights = np.array([
            self.dataset_length("test_set"),
            self.dataset_length("training_set")
        ],
            dtype=np.float64)
        weights /= np.sum(weights)
        datasets = [
            self._get_dataset("test_set", split=split),
            self._get_dataset("training_set", split=split),
        ]
        return tf.data.experimental.sample_from_datasets(
            datasets=datasets, weights=weights, seed=hash(663263 * self.seed + 548099)
        )

    def training_dataset(self) -> tf.data.Dataset:
        return self.dataset_for_split(sharded_plasticc.TrainTune.TRAIN)


class NegativesDatasetBuilder(metaclass=abc.ABCMeta):
    def __init__(
        self,
        *,
        seed: int,
        lc_lookup_chunk_size=100,
        source: plasticc_data.PlasticcBcolzSource = None
    ):
        self.seed = seed
        self.lc_lookup_chunk_size = max(1, lc_lookup_chunk_size // 2)
        self.source = (
            source
            if source is not None else plasticc_data.PlasticcBcolzSource.get_default()
        )

    @abc.abstractmethod
    def generate_negative_pair(
        self,
        lca: lightcurve._LC,
        lcb: lightcurve._LC,
    ) -> example_pair.FullNegativesPair:
        raise NotImplementedError()

    @property
    @abc.abstractmethod
    def feature_extractor(self) -> example_pair.PairFeatureExtractor:
        raise NotImplementedError()

    def lc_generator(self,
                     ids_pairs_iterator,
                     dataset: str) -> typing.Iterator[typing.Tuple[plasticc_data.PlasticcDatasetLC,
                                                                   plasticc_data.PlasticcDatasetLC]]:
        id_pairs_chunks = sharded_plasticc.chunk_indefinite(
            ids_pairs_iterator, chunk_size=self.lc_lookup_chunk_size
        )
        for id_pairs_chunk in id_pairs_chunks:
            id_chunk = frozenset(
                lc_id for left, right in id_pairs_chunk for lc_id in [left, right]
            )
            lcs = plasticc_data.PlasticcDatasetLC.bcolz_get_lcs_by_obj_ids(
                bcolz_source=self.source, dataset=dataset, obj_ids=list(id_chunk)
            )
            lcs_by_id = {lc.meta["object_id"]: lc for lc in lcs}
            for left, right in id_pairs_chunk:
                yield lcs_by_id[left], lcs_by_id[right]

    def _get_dataset(self, dataset: str, split: sharded_plasticc.TrainTune):
        """Gets a dataset for training or test sets.

        :param dataset: String "training_set" or "test_set".
        :param split: Which split of the data should be selected.
        """
        id_pairs_iterator = lc_id_features.negative_pairs_generator(
            meta_table=self.source.get_table(f"{dataset}_metadata")
        )
        logging.warning("TODO: filter by split for negatives")
        # split_ids_iterator = sharded_plasticc.filter_by_split(ids_iterator, split)
        lcs_iterator = self.lc_generator(id_pairs_iterator, dataset=dataset)
        negative_pairs = (
            self.generate_negative_pair(lca, lcb) for lca, lcb in lcs_iterator
        )
        extractor = self.feature_extractor
        features = (extractor.apply(fpp) for fpp in negative_pairs)
        features = timed_generator(features, "negative example generator")
        return tf_dataset_builder.dataset_from_generator_auto_dtypes(features)

    def training_dataset(self) -> tf.data.Dataset:
        return self._get_dataset(
            dataset="training_set", split=sharded_plasticc.TrainTune.TRAIN
        )


class RandomNegativesDatasetBuilder(NegativesDatasetBuilder, abc.ABC):
    def __init__(
        self,
        *,
        seed: int,
        lc_lookup_chunk_size=100,
        source: plasticc_data.PlasticcBcolzSource = None
    ):
        super(RandomNegativesDatasetBuilder, self).__init__(
            seed=seed, lc_lookup_chunk_size=lc_lookup_chunk_size, source=source
        )
        self.rng = random.Random(seed)

    def generate_negative_pair(
        self,
        lca: lightcurve._LC,
        lcb: lightcurve._LC,
    ):
        return example_pair.random_points_for_negative_pair(lca, lcb, rng=self.rng)
