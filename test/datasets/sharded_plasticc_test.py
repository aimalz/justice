# -*- coding: utf-8 -*-
"""Sharded plasticc dataset test."""
import collections
import itertools

import pytest
import scipy.stats

from justice.datasets import sharded_plasticc


def test_id_split_distribution():
    for distribution in [(0.1, 0.9), (0.5, 0.5), (0.1, 0.2, 0.3, 0.4)]:
        split = sharded_plasticc.IdSplit(
            splits={i: value
                    for i, value in enumerate(distribution)}, salt=b'test_salt'
        )
        for i in range(100):
            assert split.hash(i) == split.hash(i), "should be stable"

        for num_to_hash in (30, 100, 1000):
            key_counts = collections.Counter()
            for i in range(num_to_hash):
                key_counts[split.hash(i)] += 1
            kl = scipy.stats.entropy(
                pk=[key_counts[i] for i in range(len(distribution))], qk=distribution
            )
            threshold = len(distribution) / num_to_hash
            assert kl < threshold, "Unexpected distribution of key counts"


@pytest.mark.requires_real_data
def test_unlabeled_iterator():
    def take3(seed, split):
        iterator = sharded_plasticc.obj_id_iterator(seed, "test_set")
        split_iterator = sharded_plasticc.filter_by_split(iterator, split)
        return list(itertools.islice(split_iterator, 3))

    assert take3(0, sharded_plasticc.TrainTune.TRAIN) == [60396676, 60396678, 60396759]
    assert take3(1, sharded_plasticc.TrainTune.TRAIN) == [20899559, 20899631, 20899648]
    assert take3(2, sharded_plasticc.TrainTune.TRAIN) == [8658938, 8659009, 8659184]
    assert take3(0, sharded_plasticc.TrainTune.TUNE) == [60396710, 60397123, 60397177]


def test_chunk_indefinite():
    chunks = sharded_plasticc.chunk_indefinite(itertools.count(0), 3)
    assert next(chunks) == [0, 1, 2]
    assert next(chunks) == [3, 4, 5]
    assert next(chunks) == [6, 7, 8]
