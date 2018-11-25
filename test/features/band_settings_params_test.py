# -*- coding: utf-8 -*-
"""Tests for putting per-band features into and out of dictionaries."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import typing

import numpy as np
from justice import lightcurve
from justice.features import band_settings_params


class TwoBandTestLc(lightcurve._LC):
    @property
    def expected_bands(self) -> typing.List[str]:
        return ['a', 'b']


def test_generate_and_extract_per_band_features():
    lc = TwoBandTestLc(
        a=lightcurve.BandData(
            time=np.array([1, 2.0]),
            flux=np.array([4, 2.0]),
            flux_err=np.array([5, 2.0]),
        ),
        b=lightcurve.BandData(
            time=np.array([1, 2.0]),
            flux=np.array([2, 2.0]),
            flux_err=np.array([3, 2.0]),
        ),
    )

    def feature_extractor(band_data: lightcurve.BandData):
        return {
            'first_flux': band_data.flux[0],
            'first_flux_err': band_data.flux_err[0],
        }

    band_settings = band_settings_params.BandSettings(['a', 'b'])
    features = band_settings.generate_per_band_features(feature_extractor, lc)
    assert features == {
        'band_a.first_flux': 4.0,
        'band_a.first_flux_err': 5.0,
        'band_b.first_flux': 2.0,
        'band_b.first_flux_err': 3.0,
    }

    band_a_features = band_settings.get_band_features(features, 'a')
    assert band_a_features == {
        'first_flux': 4.0,
        'first_flux_err': 5.0,
    }
