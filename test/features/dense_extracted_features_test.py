# -*- coding: utf-8 -*-
"""Tests extracted featuers"""
import tensorflow as tf

from justice import simulate
from justice.features import raw_value_features, band_settings_params, dense_extracted_features


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
    first_point_features = fex.extract(lc, 2)
    second_point_features = fex.extract(lc, 3)

    with tf.Graph().as_default() as g:
        dataset1 = tf.data.Dataset.from_tensors(first_point_features)
        dataset2 = tf.data.Dataset.from_tensors(second_point_features)
        dataset = dataset1.concatenate(dataset2).batch(2, drop_remainder=True)
        inp = dataset.make_one_shot_iterator().get_next()
        window_features = dense_extracted_features.WindowFeatures(
            band_features=band_settings.get_band_features(inp, band_name='b'),
            batch_size=2,
            window_size=4,
        )
        before_flux = inp['band_b.before_flux']
        dflux_dt = window_features.dflux_dt(clip_magnitude=7.0)
        dflux_dt_masked = window_features.masked(dflux_dt, 0, [])
        with tf.Session(graph=g) as sess:
            values = sess.run({
                'before_flux': before_flux,
                'dflux_dt': dflux_dt,
                'dflux_dt_masked': dflux_dt_masked
            })
            values = {k: v.tolist() for k, v in values.items()}

    assert values == {
        'before_flux': [[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 5.0]],
        'dflux_dt': [[2.5, 2.5, 2.5, 2.5, 1.0, 2.5, 2.5, 2.5],
                     [2.0, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 2.0]],
        'dflux_dt_masked': [[0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]]
    }
