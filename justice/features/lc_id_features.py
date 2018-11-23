import numpy
import tensorflow as tf
from justice.datasets import plasticc_data


def gen_ids_by_target(meta_table, target):
    """Yields LC ids that have the target class"""
    while True:
        meta_map = meta_table.where('target == {}'.format(target), outcols=['object_id'])
        for row in meta_map:
            yield row.object_id


def _get_targets(meta_table):
    return set(meta_table['target'])


def _gen_pairs(generators):
    while True:
        choices = numpy.random.choice(generators, size=2, replace=False).tolist()
        yield (next(choices[0]), next(choices[1]))


def get_negative_pairs_dataset(meta_table=None):
    if meta_table is None:
        bcolz_source = plasticc_data.PlasticcBcolzSource.get_default()
        meta_table = bcolz_source.get_table('training_set_metadata')
    targets = _get_targets(meta_table)
    generators = [gen_ids_by_target(meta_table, t) for t in targets]
    return tf.data.Dataset.from_generator(
        lambda: _gen_pairs(generators), (tf.int64, tf.int64),
        (tf.TensorShape([]), tf.TensorShape([]))
    )
