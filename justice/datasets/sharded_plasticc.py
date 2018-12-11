# -*- coding: utf-8 -*-
"""Logic for sharding the PLASTICC dataset, for parallelism and splits."""
import enum
import hashlib
import itertools
import random
import typing

import numpy as np

from justice.datasets import plasticc_data


class IdSplit:
    """Hashes IDs into buckets, e.g. for test/train split.

    This uses sha1 to give a fast but stable hashing. It's about twice as slow as the
    built-in hash() function but probably more reliable.

    (Bad hashing causes headaches so let's not over-optimize CPU performance.)
    """
    _digest_chars = 12
    _max_value = int('f' * _digest_chars, 16)

    def __init__(self, splits: dict, salt: bytes):
        self.keys, values = zip(*splits.items())
        values = np.array(values, dtype=np.float64)
        values *= self._max_value / np.sum(values)
        self.values = np.cumsum(values.astype(np.uint64))
        assert 0.99 < float(self.values[-1]) / float(self._max_value) < 1.01
        self.values[-1] = self._max_value
        self._initial = hashlib.sha1(salt)

    def hash(self, id_value):
        hasher = self._initial.copy()
        hasher.update(str(id_value).encode('ascii'))
        x = int(hasher.hexdigest()[:self._digest_chars], 16)
        for key, value in zip(self.keys, self.values):
            if x <= value:
                return key
        raise ValueError(f"Should never be reached. x={x}, values={self.values}")


class TrainTune(enum.Enum):
    TRAIN = 1
    TUNE = 2


default_train_valid_split = IdSplit({
    TrainTune.TRAIN: 0.9,
    TrainTune.TUNE: 0.1
},
    salt=b"train_tune_20181210")


def obj_id_iterator(
    start_seed: int, dataset: str, source: plasticc_data.PlasticcBcolzSource = None
) -> typing.Iterator[int]:
    source = (
        source if source is not None else plasticc_data.PlasticcBcolzSource.get_default()
    )
    meta_table = source.get_table(f"{dataset}_metadata")
    rng = random.Random(start_seed)
    num_ids = len(meta_table['object_id'])
    start_idx = rng.randint(0, num_ids - 1)
    for offset in itertools.count(0):
        idx = (start_idx + offset) % num_ids
        obj_id = int(meta_table['object_id'][idx])
        yield obj_id


def filter_by_split(iterator: typing.Iterator[int],
                    split: TrainTune) -> typing.Iterator[int]:
    for obj_id in iterator:
        if not isinstance(obj_id, int):
            raise TypeError("Expected integer object IDs")
        if default_train_valid_split.hash(obj_id) == split:
            yield obj_id


def chunk_indefinite(iterator: typing.Iterator,
                     chunk_size: int) -> typing.Iterator[typing.List]:
    """Chunks an indefinite iterator.

    (Initializes arrays to a size instead of calling 'append' repeatedly, this should be
    slightly faster.)

    :param iterator: Input iterator.
    :param chunk_size: Chunk size.
    :yields: Lists of size |chunk_size| containing elements from the input iterator.
    """
    i = 0
    buf = [None] * chunk_size
    while True:
        buf[i] = next(iterator)
        i += 1
        if i == chunk_size:
            yield buf
            i = 0
            buf = [None] * chunk_size
