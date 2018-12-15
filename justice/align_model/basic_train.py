# -*- coding: utf-8 -*-
"""Basic local training routine (one hyperparameter, etc.)."""
import argparse
import datetime

import numpy as np
import tensorflow as tf

import justice.features
from justice import lightcurve, path_util
from justice.align_model import basic_vector_similarity_model
from justice.align_model import input_pipeline
from justice.align_model import synthetic_positives
from justice.datasets import plasticc_data
from justice.features import band_settings_params
from justice.features import example_pair
from justice.features import raw_value_features


class BasicPositivesBuilder(input_pipeline.PositivesDatasetBuilder):
    def __init__(
        self, *, seed, bands, window_size, subsample_min_rate, subsample_max_rate
    ):
        super(BasicPositivesBuilder, self).__init__(seed=seed)
        self.basic_synthetic_positives = synthetic_positives.BasicPositivesGenerator()
        self.subsampler = synthetic_positives.PositivePairSubsampler(
            min_rate=subsample_min_rate,
            max_rate=subsample_max_rate,
            rng=np.random.RandomState(seed)
        )
        self.positives_fex = synthetic_positives.RawValuesFullPositives(
            bands, window_size
        )

    def generate_synthetic_pair(
        self, lc: lightcurve._LC
    ) -> justice.features.example_pair.FullPositivesPair:
        fpp = self.basic_synthetic_positives.make_positive_pair(lc)
        return self.subsampler.apply(fpp)

    @property
    def feature_extractor(self) -> example_pair.PairFeatureExtractor:
        return self.positives_fex

    @classmethod
    def from_params(cls, params: dict):
        return cls(
            seed=params["seed"],
            bands=params["lc_bands"],
            window_size=params["window_size"],
            subsample_min_rate=params["subsample_min_rate"],
            subsample_max_rate=params["subsample_max_rate"]
        )


class RawValuesNegatives(example_pair.PairFexFromPointwiseFex):
    def __init__(self, bands, window_size):
        self.band_settings = band_settings_params.BandSettings(bands=bands)
        fex = raw_value_features.RawValueExtractor(
            window_size=window_size, band_settings=self.band_settings
        )
        super(RawValuesNegatives, self).__init__(fex=fex, label=False)


class BasicNegativesBuilder(input_pipeline.RandomNegativesDatasetBuilder):
    def __init__(self, *, seed, bands, window_size):
        super(BasicNegativesBuilder, self).__init__(seed=seed)
        self.negatives_fex = RawValuesNegatives(bands, window_size)

    @property
    def feature_extractor(self) -> example_pair.PairFeatureExtractor:
        return self.negatives_fex

    @classmethod
    def from_params(cls, params: dict):
        return cls(
            seed=params["seed"],
            bands=params["lc_bands"],
            window_size=params["window_size"],
        )


def make_basic_input_dataset(mode, params):
    assert mode == tf.estimator.ModeKeys.TRAIN, "Only train is supported now, sorry."
    positives_dataset = BasicPositivesBuilder.from_params(params).training_dataset()
    negatives_dataset = BasicNegativesBuilder.from_params(params).training_dataset()
    datasets = [positives_dataset, negatives_dataset]
    weights = [0.5, 0.5]
    seed = params["seed"]
    batch_size = params["batch_size"]
    return tf.data.experimental.sample_from_datasets(
        datasets=datasets, weights=weights, seed=hash(655211 * seed + 898769)
    ).batch(
        batch_size=batch_size, drop_remainder=True
    )


def main():
    cmd_args = argparse.ArgumentParser()
    cmd_args.add_argument(
        "--window-size",
        default=10,
        type=int,
        help="For window-based features, number of points to the left and"
        "right of the current point to include."
    )
    cmd_args.add_argument(
        "--seed", default=0, type=int, help="Seed value for random number generators."
    )
    cmd_args.add_argument(
        "--subsample-min-rate",
        default=0.7,
        type=float,
        help="Value between 0 and 1 for minimum synthetic positives subsampling. "
        "Subsampling keeps only a fraction of points in a light curve "
        "(like dropout, but maybe more realistic)."
    )
    cmd_args.add_argument(
        "--subsample-max-rate",
        default=1.0,
        type=float,
        help="value between 0 and 1 for maximum subsampling rate (see above)."
    )
    cmd_args.add_argument(
        "--learning-rate",
        default=1e-3,
        type=float,
        help="Learning rate for Nesterov Adam (gradient descent) algorithm."
    )
    cmd_args.add_argument(
        "--train-steps", default=10_000, type=int, help="Number of training steps."
    )
    cmd_args.add_argument(
        "--batch-size", default=32, type=int, help="Outer batch dimension."
    )
    cmd_args.add_argument(
        "--hidden-sizes", default="128", help="Comma-separated list of hidden sizes."
    )
    cmd_args.add_argument(
        "--output-size",
        default=64,
        type=int,
        help="Output vector size (dimension of encoding of each point)."
    )
    args = cmd_args.parse_args()
    params = vars(args)
    params["lc_bands"] = plasticc_data.PlasticcDatasetLC.expected_bands
    if not params["hidden_sizes"]:
        params["hidden_sizes"] = []
    else:
        params["hidden_sizes"] = [int(x) for x in params["hidden_sizes"].split(",")]

    path_util.align_model_dir.mkdir(parents=True, exist_ok=True)
    d = datetime.datetime.now()
    model_dir = (
        path_util.align_model_dir / f"basic_model_{d.year:04d}_"
        f"{d.month:02d}_{d.day:02d}_{d.hour:02d}.{d.minute:02d}.{d.second:02d}"
    )
    assert not model_dir.exists()

    estimator = tf.estimator.Estimator(
        model_fn=basic_vector_similarity_model.model_fn,
        model_dir=str(model_dir),
        params=params,
    )
    estimator.train(make_basic_input_dataset, max_steps=args.train_steps)


if __name__ == '__main__':
    main()
