# coding: utf-8

import numpy as np

order = 59
frame_period = 5
windows = [
    (0, 0, np.array([1.0])),
    (1, 1, np.array([-0.5, 0.0, 0.5])),
    (1, 1, np.array([1.0, -2.0, 1.0])),
]

batch_size = 32
weight_decay = 0
nepoch = 350
num_workers = 1
lr = 0.01
