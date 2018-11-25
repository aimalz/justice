# -*- coding: utf-8 -*-
"""Tests extracted featuers"""
import pytest
import tensorflow as tf

from justice import simulate
from justice.datasets import plasticc_data
from justice.features import raw_value_features, band_settings_params, dense_extracted_features, per_point_dataset


def test_extraction():
    """Tests that extracting dflux_dt from light curves works as expected.

    This uses the default "make_super_easy" class, which has time values [2, 3] and
    flux values [5, 6]. We get features for the first and second points, and then construct
    a dataset by concatenating these two features together. The first batch element
    (corresponding to the first point) should have one non-masked feature with dflux/dt=1,
    in the first "after" position; similarly the second batch element should have one
    non-masked feature with dflux/dt=1, in the first "before" position.
    """
    lc = simulate.TestLC.make_super_easy()
    band_settings = band_settings_params.BandSettings(bands=['b'])
    fex = raw_value_features.RawValueExtractor(
        window_size=4, band_settings=band_settings)
    first_point_features = fex.extract(lc, 1.9)
    second_point_features = fex.extract(lc, 3.0)

    with tf.Graph().as_default() as g:
        dataset1 = tf.data.Dataset.from_tensors(first_point_features)
        dataset2 = tf.data.Dataset.from_tensors(second_point_features)
        dataset = dataset1.concatenate(dataset2).batch(2, drop_remainder=True)
        inp = dataset.make_one_shot_iterator().get_next()
        window_features = dense_extracted_features.WindowFeatures(
            band_features=band_settings.get_band_features(inp, band_name='b'),
            batch_size=2,
            window_size=4,
            band_time_diff=0.2,
        )
        before_flux = inp['band_b.before_flux']
        dflux_dt = window_features.dflux_dt(clip_magnitude=7.0)
        dflux_dt_masked = window_features.masked(dflux_dt, 0, [])

        # Variation where each band has to be sampled within a very strict tolerance.
        window_features_strict = dense_extracted_features.WindowFeatures(
            band_features=band_settings.get_band_features(inp, band_name='b'),
            batch_size=2,
            window_size=4,
            band_time_diff=0.01,
        )
        dflux_dt_masked_strict = window_features_strict.masked(dflux_dt, 0, [])

        with tf.Session(graph=g) as sess:
            values = sess.run({
                'before_flux': before_flux,
                'dflux_dt': dflux_dt,
                'dflux_dt_masked': dflux_dt_masked,
                'dflux_dt_masked_strict': dflux_dt_masked_strict
            })
            values = {k: v.tolist() for k, v in values.items()}

    assert values == {
        'before_flux': [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 5.0]],
        'dflux_dt': [[2.5, 2.5, 2.5, 2.5, 1.0, 2.5, 2.5, 2.5],
                     [2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0]],
        'dflux_dt_masked': [[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]],
        'dflux_dt_masked_strict': [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                   [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]],
    }


@pytest.mark.requires_real_data
def test_dense_feature_extraction():
    source = plasticc_data.PlasticcBcolzSource.get_default()
    lc, = plasticc_data.PlasticcDatasetLC.bcolz_get_lcs_by_obj_ids(
        bcolz_source=source, dataset="training_set", obj_ids=[1598]
    )

    def model_fn(features, labels, mode, params):
        del labels  # unused
        band_settings = band_settings_params.BandSettings.from_params(params)
        results = dense_extracted_features.feature_model_fn(features, params)
        by_band = tf.unstack(results, axis=4)
        predictions = {
            band: tensor for band,
            tensor in zip(
                band_settings.bands,
                by_band)}
        predictions["time"] = features["time"]
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            loss=tf.constant(0.0),
            train_op=tf.no_op()
        )

    window_size = 10
    rve = raw_value_features.RawValueExtractor(
        window_size=window_size,
        band_settings=band_settings_params.BandSettings(lc.expected_bands)
    )
    data_gen = per_point_dataset.PerPointDatasetGenerator(
        extract_fcn=rve.extract,
        batch_size=5,
    )

    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        params={
            'batch_size': 5,
            'window_size': window_size,
            'flux_scale_epsilon': 0.5,
            'lc_bands': lc.expected_bands,
        }
    )
    predictions = list(data_gen.predict_single_lc(estimator, lc, arrays_to_list=False))
    array = predictions[100]['y']
    assert array.shape == (
        20, 3, 32
    )  # 2 * window_size, channels (dflux/dt, dflux, dtime), nbands
    time_array = array[:, 2, :]
    # Should be monotonically increasing as the window shifts, since WindowFeatures
    # computes (point in window time) - (selected time). Should be monotonically
    # decreasing along bins, since each bin fuzzily represents whether the actual value is
    # greater than the bin's center value. As the bin centers increase, these fuzzy
    # greater than values should decrease.
    assert (time_array[1:, :] >= time_array[:-1, :]).all()
    assert (time_array[:, 1:] <= time_array[:, :-1]).all()
