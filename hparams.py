# coding: utf-8

import tensorflow as tf
import numpy as np

from os.path import join, dirname


# Hyper parameters for voice conversion
vc = tf.contrib.training.HParams(
    # Acoustic features
    order=59,
    frame_period=5,
    windows=[
        (0, 0, np.array([1.0])),
        (1, 1, np.array([-0.5, 0.0, 0.5])),
        (1, 1, np.array([1.0, -2.0, 1.0])),
    ],
    static_dim=59,
    stream_sizes=None,
    has_dynamic_features=None,

    # Generator
    generator="In2OutHighwayNet",
    generator_params={
        "in_dim": 59 * 3,
        "out_dim": 59 * 3,
        "num_hidden": 3,
        "hidden_dim": 512,
        "static_dim": 59,
    },
    optimizer_g="Adagrad",
    optimizer_g_params={
        "lr": 0.01,
        "weight_decay": 0,
    },

    # Discriminator
    discriminator="Discriminator",
    discriminator_params={
        "in_dim": 59,
        "num_hidden": 2,
        "hidden_dim": 256,
    },
    optimizer_d="Adagrad",
    optimizer_d_params={
        "lr": 0.01,
        "weight_decay": 0,
    },

    # This should be overrided
    nepoch=200,

    # Datasets and data loader
    batch_size=32,
    num_workers=1,
    pin_memory=True,
    cache_size=1200,
)


# Hyper paramters for TTS duration model
tts_duration = tf.contrib.training.HParams(
    # Linguistic features
    use_phone_alignment=False,
    question_path=join(dirname(__file__), "questions",
                       "questions-radio_dnn_416.hed"),

    # Duration features
    order=59,
    frame_period=5,
    windows=[
        (0, 0, np.array([1.0])),
    ],
    stream_sizes=[5],
    has_dynamic_features=[False],

    # Generator
    generator="MLP",
    generator_params={
        "in_dim": None,
        "out_dim": None,
        "num_hidden": 3,
        "hidden_dim": 256,
    },
    optimizer_g="Adam",
    optimizer_g_params={
        "lr": 0.001,
        "weight_decay": 1e-7,
    },

    # Discriminator
    discriminator="Discriminator",
    discriminator_params={
        "in_dim": None,
        "num_hidden": 2,
        "hidden_dim": 256,
    },
    optimizer_d="Adam",
    optimizer_d_params={
        "lr": 0.001,
        "weight_decay": 1e-7,
    },

    # This should be overrided
    nepoch=200,

    # Datasets and data loader
    batch_size=32,
    num_workers=1,
    pin_memory=True,
    cache_size=1200,
)

# Hyper paramters for TTS acoustic model
tts_acoustic = tf.contrib.training.HParams(
    # Linguistic
    use_phone_alignment=False,
    question_path=join(dirname(__file__), "questions",
                       "questions-radio_dnn_416.hed"),

    # Acoustic features
    order=59,
    frame_period=5,
    windows=[
        (0, 0, np.array([1.0])),
        (1, 1, np.array([-0.5, 0.0, 0.5])),
        (1, 1, np.array([1.0, -2.0, 1.0])),
    ],
    # Stream info
    stream_sizes=[180, 3, 1, 3],
    has_dynamic_features=[True, True, False, True],

    # Generator
    generator="MLP",
    generator_params={
        "in_dim": None,
        "out_dim": None,
        "num_hidden": 3,
        "hidden_dim": 256,
    },
    optimizer_g="Adam",
    optimizer_g_params={
        "lr": 0.001,
        "weight_decay": 1e-7,
    },

    # Discriminator
    discriminator="Discriminator",
    discriminator_params={
        "in_dim": None,
        "num_hidden": 2,
        "hidden_dim": 256,
    },
    optimizer_d="Adam",
    optimizer_d_params={
        "lr": 0.001,
        "weight_decay": 1e-7,
    },

    # This should be overrided
    nepoch=200,

    # Datasets and data loader
    batch_size=32,
    num_workers=1,
    pin_memory=True,
    cache_size=1200,
)


def hparams_debug_string(params):
    values = params.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)
