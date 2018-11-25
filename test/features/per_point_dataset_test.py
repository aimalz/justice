# -*- coding: utf-8 -*-
"""Per-point dataset tests."""
import tensorflow as tf
from justice import simulate
from justice.align_model import max_model_kernel
from justice.features import per_point_dataset, raw_value_features, band_settings_params


def model_fn(features, labels, mode, params):
    predictions = max_model_kernel.feature_model_fn(features, params)
    predictions['time'] = features['time']
    return tf.estimator.EstimatorSpec(
        mode=mode, predictions=predictions, loss=tf.constant(0.0), train_op=tf.no_op()
    )


def test_raw_features_dataset():
    lc = simulate.TestLC.make_super_easy()
    assert len(lc.bands['b'].time) == 2, "TestLC.make_super_easy() changed"
    rve = raw_value_features.RawValueExtractor(
        window_size=4, band_settings=band_settings_params.BandSettings(['b'])
    )
    data_gen = per_point_dataset.PerPointDatasetGenerator(
        extract_fcn=rve.extract,
        batch_size=5,
    )

    with tf.Graph().as_default() as g:
        dataset, num_batches, num_non_padding = data_gen.make_dataset(lc)
        assert num_non_padding == 2, "LC has 2 time points"
        assert num_batches == 1
        iterator = dataset.make_one_shot_iterator().get_next()
        with tf.Session(graph=g) as sess:
            assert sess.run(iterator)['band_b.before_padding'].tolist() == [
                4, 3, 4, 4, 4]
        for key, tensor_shape in dataset.output_shapes.items():
            concrete_shape = list(map(int, tensor_shape)
                                  )  # All tensors should have concrete shapes
            assert concrete_shape[0] == 5, "First dimension should be batch dimension."

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        params={
            'batch_size': 5,
            'window_size': 4,
            'flux_scale_epsilon': 0.5,
            'lc_bands': ['b'],
        }
    )
    predictions = list(data_gen.predict_single_lc(estimator, lc))
    assert predictions == [{
        'is_max_soft': [0.11920291930437088],
        'time': 2.0
    }, {
        'is_max_soft': [0.8807970285415649],
        'time': 3.0
    }]
