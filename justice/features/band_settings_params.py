# -*- coding: utf-8 -*-
"""Helper class that reconstructs model parameters from TF params dict."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import typing

from justice import lightcurve
from justice.features import merge_feature_dicts


class BandSettings(object):
    bands: typing.List[str]

    def __init__(self, bands):
        self.bands = bands

    @property
    def nbands(self):
        return len(self.bands)

    def generate_per_band_features(self, fcn, lc: lightcurve._LC):
        feature_dicts = []
        for band_name in self.bands:
            band = lc.bands[band_name]
            band_features = fcn(band)
            if not isinstance(band_features, dict):
                raise TypeError(
                    "generate_per_band_features expected argument to "
                    "generate a dict of per-band features."
                )

            feature_dicts.append({
                f"band_{band_name}.{key}": value
                for key, value in band_features.items()
            })
        return merge_feature_dicts.merge_feature_dicts(feature_dicts)

    def get_band_features(self, features, band_name):
        """Extracts features for a specific band.

        :param features: Feature dictionary, with keys from generate_per_band_features.
        :param band_name: Name of the band to extract.
        :return: New dict with per-band features.
        """
        prefix = f"band_{band_name}."
        return {
            key[len(prefix):]: value
            for key, value in features.items()
            if key.startswith(prefix)
        }

    def per_band_sub_model_fn(self, sub_model_fn, features, params):
        return [
            sub_model_fn(self.get_band_features(features, band_name), params=params)
            for band_name in self.bands
        ]

    @classmethod
    def from_params(cls, params):
        return cls(params["lc_bands"])
