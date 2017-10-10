# coding: utf-8

import tensorflow as tf
import numpy as np

from os.path import join, dirname


def hparams_debug_string(params):
    values = params.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)


# Hyper parameters for voice conversion
vc = tf.contrib.training.HParams(
    name="vc",

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

    adversarial_streams=None,

    # Generator
    generator="In2OutHighwayNet",
    generator_params={
        "in_dim": None,
        "out_dim": None,
        "num_hidden": 3,
        "hidden_dim": 512,
        "static_dim": 59,
        "dropout": 0.5,
    },
    optimizer_g="Adagrad",
    optimizer_g_params={
        "lr": 0.01,
        "weight_decay": 0,
    },

    # Discriminator
    discriminator="MLP",
    discriminator_params={
        "in_dim": 59,
        "out_dim": 1,
        "num_hidden": 2,
        "hidden_dim": 256,
        "dropout": 0.5,
        "last_sigmoid": True,
    },
    optimizer_d="Adagrad",
    optimizer_d_params={
        "lr": 0.01,
        "weight_decay": 0,
    },

    # This should be overrided
    nepoch=200,

    # LR schedule
    lr_decay_schedule=False,
    lr_decay_epoch=10,

    # Datasets and data loader
    batch_size=32,
    num_workers=1,
    pin_memory=True,
    cache_size=1200,
)


# Hyper paramters for TTS duration model
tts_duration = tf.contrib.training.HParams(
    name="duration",

    # Linguistic features
    use_phone_alignment=True,
    subphone_features=None,
    add_frame_features=False,
    question_path=join(dirname(__file__), "nnmnkwii_gallery", "data",
                       "questions_jp.hed"),

    # Duration features
    windows=[
        (0, 0, np.array([1.0])),
    ],
    stream_sizes=[1],
    has_dynamic_features=[False],

    # Streams used for computing adversarial loss
    adversarial_streams=[True],
    mask_0th_mgc_for_adv_loss=False,

    # Generator
    generator="LSTMRNN",
    generator_params={
        "in_dim": None,
        "out_dim": None,
        "num_hidden": 3,
        "hidden_dim": 512,
        "bidirectional": True,
        "dropout": 0.5,
        "last_sigmoid": False,
    },
    optimizer_g="Adagrad",
    optimizer_g_params={
        "lr": 0.01,
        "weight_decay": 1e-5,
    },

    # Discriminator
    discriminator="MLP",
    discriminator_params={
        "in_dim": None,
        "out_dim": 1,
        "num_hidden": 2,
        "hidden_dim": 256,
        # "bidirectional": True,
        "dropout": 0.5,
        "last_sigmoid": True,
    },
    optimizer_d="Adagrad",
    optimizer_d_params={
        "lr": 0.01,
        "weight_decay": 1e-5,
    },

    # This should be overrided
    nepoch=200,

    # LR schedule
    lr_decay_schedule=False,
    lr_decay_epoch=25,

    # Datasets and data loader
    batch_size=32,
    num_workers=1,
    pin_memory=True,
    cache_size=1200,
)

# Hyper paramters for TTS acoustic model
tts_acoustic = tf.contrib.training.HParams(
    name="acoustic",

    # Linguistic
    use_phone_alignment=True,
    subphone_features="coarse_coding",
    add_frame_features=True,
    question_path=join(dirname(__file__), "nnmnkwii_gallery", "data",
                       "questions_jp.hed"),

    # Acoustic features
    order=59,
    frame_period=5,
    windows=[
        (0, 0, np.array([1.0])),
        (1, 1, np.array([-0.5, 0.0, 0.5])),
        (1, 1, np.array([1.0, -2.0, 1.0])),
    ],
    # Stream info
    # (mgc, lf0, vuv, bap)
    stream_sizes=[180, 3, 1, 15],
    has_dynamic_features=[True, True, False, True],

    # Streams used for computing adversarial loss
    # NOTE: you should probably change discriminator's `in_dim`
    # if you change the adv_streams
    adversarial_streams=[True, False, False, False],
    # Don't switch this on unless you are sure what you are doing
    # If True, you will need to adjast `in_dim` for discriminator.
    # Rationale for this is that power coefficients are less meaningful
    # to distinguish natrual/generated, especially for frame-level models.
    mask_0th_mgc_for_adv_loss=True,

    # Generator
    generator="MLP",
    generator_params={
        "in_dim": None,
        "out_dim": None,
        "num_hidden": 3,
        "hidden_dim": 512,
        #"bidirectional": True,
        "dropout": 0.5,
        "last_sigmoid": False,
    },
    optimizer_g="Adagrad",
    optimizer_g_params={
        "lr": 0.01,
        "weight_decay": 1e-5,
    },

    # Discriminator
    discriminator="MLP",
    discriminator_params={
        "in_dim": 59,
        "out_dim": 1,
        "num_hidden": 2,
        "hidden_dim": 256,
        "dropout": 0.5,
        "last_sigmoid": True,
    },
    optimizer_d="Adagrad",
    optimizer_d_params={
        "lr": 0.01,
        "weight_decay": 1e-5,
    },

    # This should be overrided
    nepoch=200,

    # LR schedule
    lr_decay_schedule=False,
    lr_decay_epoch=25,

    # Datasets and data loader
    batch_size=16,
    num_workers=1,
    pin_memory=True,
    cache_size=1200,
)
