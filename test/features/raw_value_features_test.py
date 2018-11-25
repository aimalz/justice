# -*- coding: utf-8 -*-
"""Tests for raw value window features."""
from justice import simulate
from justice.features import raw_value_features, band_settings_params


def test_basic_extraction():
    lc = simulate.TestLC.make_super_easy()
    fex = raw_value_features.RawValueExtractor(
        window_size=4, band_settings=band_settings_params.BandSettings(bands=['b'])
    )
    first_point_features = fex.extract(lc, 2)
    assert first_point_features['band_b.before_padding'] == 4
    assert first_point_features['band_b.after_padding'] == 3
    assert first_point_features['band_b.closest_time_diff'] == 0
    assert first_point_features['band_b.after_flux'].tolist() == [6, 0, 0, 0]

    second_point_features = fex.extract(lc, 3)
    assert second_point_features['band_b.before_padding'] == 3
    assert second_point_features['band_b.after_padding'] == 4
    assert second_point_features['band_b.closest_time_diff'] == 0
    assert second_point_features['band_b.before_flux'].tolist() == [0, 0, 0, 5]

    time_delta = (
        second_point_features['band_b.closest_time_in_band'] -
        second_point_features['band_b.before_time']
    )
    assert time_delta.tolist()[-1] == 1.0
