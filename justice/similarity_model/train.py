# -*- coding: utf-8 -*-
"""Script that trains a sample model.

NOTE: Run

tensorboard --logdir=justice/similarity_model/tf_model

to see training status. This will also let you train multiple models, so long
as `model_name` below differs, and see their training performance.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path

import tensorflow as tf

from justice import path_util
from justice.similarity_model import model, training_data

model_base_dir = path_util.models_dir / 'legacy_similarity'


def main():
    layer_sizes = [64, 32]
    steps = int(1e5)
    symmetric = True
    for dropout in [0.7]:
        model_name = 'sample_{}_drop{:.2f}_{}_{}_v3'.format(
            'sym' if symmetric else 'asym', dropout, *layer_sizes
        )
        estimator = tf.estimator.Estimator(
            model_fn=model.model_fn,
            model_dir=str(model_base_dir / model_name),
            params={
                'batch_size': 32,
                'window_size': 5,
                'dropout_keep_prob': dropout,
                'layer_sizes': layer_sizes,
                'symmetric': symmetric,
                'lr_decay_steps': steps // 10,
            }
        )
        estimator.train(input_fn=training_data.sample_data_input_fn, max_steps=steps)


if __name__ == '__main__':
    main()
