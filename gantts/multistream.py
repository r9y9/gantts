# coding: utf-8

# Utils for multi-stream features

import torch
from torch import nn
from torch.autograd import Variable

import numpy as np

from nnmnkwii.autograd import unit_variance_mlpg
from nnmnkwii import preprocessing as P


def recompute_delta_features(Y, Y_data_mean, Y_data_std,
                             windows,
                             stream_sizes=[180, 3, 1, 3],
                             has_dynamic_features=[True, True, False, True]):
    start_indices = np.hstack(([0], np.cumsum(stream_sizes)[:-1]))
    end_indices = np.cumsum(stream_sizes)
    static_stream_sizes = get_static_stream_sizes(
        stream_sizes, has_dynamic_features, len(windows))

    for start_idx, end_idx, static_size, has_dynamic in zip(
            start_indices, end_indices, static_stream_sizes, has_dynamic_features):
        if has_dynamic:
            y_static = Y[:, start_idx:start_idx + static_size]
            Y[:, start_idx:end_idx] = P.delta_features(y_static, windows)

    return Y


def select_streams(inputs, stream_sizes=[60, 1, 1, 1],
                   streams=[True, True, True, True]):
    ret = []
    start_indices = np.hstack(([0], np.cumsum(stream_sizes)[:-1]))
    for start_idx, size, enabled in zip(
            start_indices, stream_sizes, streams):
        if not enabled:
            continue
        ret.append(inputs[:, :, start_idx:start_idx + size])

    return torch.cat(ret, dim=-1)


def get_static_stream_sizes(stream_sizes, has_dynamic_features, num_windows):
    """Get static dimention for each feature stream.
    """
    static_stream_sizes = np.array(stream_sizes)
    static_stream_sizes[has_dynamic_features] = \
        static_stream_sizes[has_dynamic_features] / num_windows

    return static_stream_sizes


def get_static_features(inputs, num_windows, stream_sizes=[180, 3, 1, 3],
                        has_dynamic_features=[True, True, False, True],
                        streams=[True, True, True, True]):
    """Get static features from static+dynamic features.
    """
    _, _, D = inputs.size()
    if stream_sizes is None or (len(stream_sizes) == 1 and has_dynamic_features[0]):
        return inputs[:, :, :D // num_windows]
    if len(stream_sizes) == 1 and not has_dynamic_features[0]:
        return inputs

    # Multi stream case
    ret = []
    start_indices = np.hstack(([0], np.cumsum(stream_sizes)[:-1]))
    for start_idx, size, v, enabled in zip(
            start_indices, stream_sizes, has_dynamic_features, streams):
        if not enabled:
            continue
        if v:
            static_features = inputs[:, :, start_idx:start_idx + size // num_windows]
        else:
            static_features = inputs[:, :, start_idx:start_idx + size]
        ret.append(static_features)
    return torch.cat(ret, dim=-1)


def multi_stream_mlpg(inputs, R,
                      stream_sizes=[180, 3, 1, 3],
                      has_dynamic_features=[True, True, False, True],
                      streams=[True, True, True, True]):
    """Split streams and do apply MLPG if stream has dynamic features.
    """
    if R is None:
        num_windows = 1
    else:
        num_windows = R.size(1) / R.size(0)
    B, T, D = inputs.size()
    if D != sum(stream_sizes):
        raise RuntimeError("You probably have specified wrong dimention params.")

    # Straem indices for static+delta features
    # [0,   180, 183, 184]
    start_indices = np.hstack(([0], np.cumsum(stream_sizes)[:-1]))
    # [180, 183, 184, 187]
    end_indices = np.cumsum(stream_sizes)

    # Stream sizes for static features
    # [60, 1, 1, 1]
    static_stream_sizes = get_static_stream_sizes(
        stream_sizes, has_dynamic_features, num_windows)

    # [0,  60, 61, 62]
    static_stream_start_indices = np.hstack(
        ([0], np.cumsum(static_stream_sizes)[:-1]))
    # [60, 61, 62, 63]
    static_stream_end_indices = np.cumsum(static_stream_sizes)

    ret = []
    for in_start_idx, in_end_idx, out_start_idx, out_end_idx, v, enabled in zip(
            start_indices, end_indices, static_stream_start_indices,
            static_stream_end_indices, has_dynamic_features, streams):
        if not enabled:
            continue
        x = inputs[:, :, in_start_idx:in_end_idx]
        y = unit_variance_mlpg(R, x) if v else x
        ret.append(y)

    return torch.cat(ret, dim=-1)
