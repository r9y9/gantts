# coding: utf-8

import tensorflow as tf
import numpy as np

from os.path import join, dirname


hparams_vc = tf.contrib.training.HParams(
    # Acoustic
    order=59,
    frame_period=5,
    windows=[
        (0, 0, np.array([1.0])),
        (1, 1, np.array([-0.5, 0.0, 0.5])),
        (1, 1, np.array([1.0, -2.0, 1.0])),
    ],

    # Training
    batch_size=32,
    weight_decay=0,
    nepoch=200,
    lr=0.01,

    # Data loader
    num_workers=1,
)


hparams_tts = tf.contrib.training.HParams(
    # Linguistic
    use_phone_alignment=False,
    question_path=join(dirname(__file__), "questions",
                       "questions-radio_dnn_416.hed"),

    # Acoustic
    order=59,
    frame_period=5,
    windows=[
        (0, 0, np.array([1.0])),
        (1, 1, np.array([-0.5, 0.0, 0.5])),
        (1, 1, np.array([1.0, -2.0, 1.0])),
    ],

    # Trtaining
    batch_size=32,
    weight_decay=0,
    nepoch=200,
    lr=0.01,

    # Data loader
    num_workers=1,
)


def hparams_debug_string(params):
    values = params.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)
