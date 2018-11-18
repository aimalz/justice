from justice.features import lc_id_features
import itertools
import tensorflow as tf
import pytest
import numpy

def test_gen_pairs():
    generators = [
        itertools.repeat('a'),
        itertools.repeat('b'),
        itertools.repeat('c'),
    ]
    g = lc_id_features._gen_pairs(generators)
    allowed = {
        ('a','b'),
        ('b','a'),
        ('a','c'),
        ('c','a'),
        ('b','c'),
        ('c','b'),
    }
    seen = set()
    for i in range(1000):
        n = next(g)
        assert n in allowed
        seen.add(n)
    assert seen == allowed

@pytest.mark.requires_real_data
def test_get_negative_pairs_dataset():
    d = lc_id_features.get_negative_pairs_dataset()
    inp = d.make_one_shot_iterator().get_next()
    with tf.Session() as sess:
        values = sess.run({'dataset': inp})
        assert isinstance(values['dataset'][0], numpy.int64)
        assert isinstance(values['dataset'][1], numpy.int64)
        assert len(values['dataset']) == 2