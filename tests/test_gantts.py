# coding: utf-8
import sys
import numpy as np
from torch.autograd import Variable
import torch
from torch import nn
from torch import optim
from nnmnkwii.paramgen import unit_variance_mlpg_matrix
from nnmnkwii.autograd import unit_variance_mlpg

from gantts.models import In2OutHighwayNet, MLP
from gantts.seqloss import MaskedMSELoss, sequence_mask
from gantts.multistream import multi_stream_mlpg, get_static_features
from gantts.multistream import get_static_stream_sizes, select_streams


def test_model():
    windows = [
        (0, 0, np.array([1.0])),
        (1, 1, np.array([-0.5, 0.0, 0.5])),
    ]

    model = In2OutHighwayNet()
    print(model)
    assert model.include_parameter_generation()

    in_dim = 118
    static_dim = in_dim // 2
    T = 100
    x = Variable(torch.rand(1, T, in_dim))
    R = unit_variance_mlpg_matrix(windows, T)
    R = torch.from_numpy(R)
    _, y = model(x, R)

    print(y.size())
    assert y.size(-1) == static_dim

    # Mini batch
    batch_size = 32
    x = Variable(torch.rand(batch_size, T, in_dim))
    _, y_hat = model(x, R)
    y = Variable(torch.rand(batch_size, T, static_dim), requires_grad=False)

    lengths = [np.random.randint(50, T - 1) for _ in range(batch_size - 1)] + [T]
    lengths = Variable(torch.LongTensor(lengths), requires_grad=False)
    print(x.size(), y.size(), lengths.size())
    MaskedMSELoss()(y_hat, y, lengths).backward()
    print(y.size())
    assert y.size(-1) == static_dim
    assert y.size(0) == batch_size

    # cuda
    if torch.cuda.is_available():
        model = model.cuda()
        x = x.cuda()
        R = R.cuda()
        _, y_hat = model(x, R)


def test_select_streams():
    static_stream_sizes = [60, 1, 1, 1]
    x = torch.zeros(32, 100, 63)
    assert select_streams(x, static_stream_sizes, streams=[
        True, True, True, True]).size() == (32, 100, 63)
    assert select_streams(x, static_stream_sizes, streams=[
        True, False, False, False]).size() == (32, 100, 60)
    assert select_streams(x, static_stream_sizes, streams=[
        True, False, False, True]).size() == (32, 100, 61)

    x = torch.arange(0, 63).expand(32, 100, 63)
    assert (select_streams(x, static_stream_sizes, streams=[
        False, False, False, True]) == x[:, :, -1]).all()
    assert (select_streams(x, static_stream_sizes, streams=[
        False, False, True, False]) == x[:, :, -2]).all()
    assert (select_streams(x, static_stream_sizes, streams=[
        False, True, False, False]) == x[:, :, -3]).all()

    # Multiple selects
    y = select_streams(x, static_stream_sizes, streams=[
        True, False, False, True])
    assert (y[:, :, :60] == x[:, :, :60]).all()
    assert (y[:, :, -1] == x[:, :, -1]).all()

    y = select_streams(x, static_stream_sizes, streams=[
        True, True, False, False])
    assert (y[:, :, :60] == x[:, :, :60]).all()
    assert (y[:, :, 60] == x[:, :, 60]).all()


def test_get_static_stream_sizes():
    stream_sizes = [180, 3, 1, 3]
    has_dynamic_features = [True, True, False, True]
    num_windows = 3

    static_stream_sizes = get_static_stream_sizes(stream_sizes, has_dynamic_features, num_windows)
    print(static_stream_sizes)
    assert np.all(static_stream_sizes == [60, 1, 1, 1])


def test_get_static_features():
    windows = [
        (0, 0, np.array([1.0])),
        (1, 1, np.array([-0.5, 0.0, 0.5])),
        (1, 1, np.array([1.0, -2.0, 1.0])),
    ]
    in_dim = 187
    T = 100
    batch_size = 32
    x = Variable(torch.rand(batch_size, T, in_dim))

    stream_sizes = [180, 3, 1, 3]
    has_dynamic_features = [True, True, False, True]
    static_features = get_static_features(
        x, len(windows), stream_sizes, has_dynamic_features)
    assert static_features.size() == (batch_size, T, 60 + 1 + 1 + 1)

    # 1
    assert get_static_features(
        x, len(windows), stream_sizes, has_dynamic_features,
        streams=[True, False, False, False]).size() == (batch_size, T, 60)
    # 2
    assert get_static_features(
        x, len(windows), stream_sizes, has_dynamic_features,
        streams=[False, True, False, False]).size() == (batch_size, T, 1)

    # 3
    assert get_static_features(
        x, len(windows), stream_sizes, has_dynamic_features,
        streams=[True, False, False, True]).size() == (batch_size, T, 60 + 1)


def test_multi_stream_mlpg():
    windows = [
        (0, 0, np.array([1.0])),
        (1, 1, np.array([-0.5, 0.0, 0.5])),
        (1, 1, np.array([1.0, -2.0, 1.0])),
    ]
    in_dim = 187
    T = 100
    R = unit_variance_mlpg_matrix(windows, T)
    R = torch.from_numpy(R)

    batch_size = 32
    x = Variable(torch.rand(batch_size, T, in_dim))

    stream_sizes = [180, 3, 1, 3]
    has_dynamic_features = [True, True, False, True]
    y = multi_stream_mlpg(x, R, stream_sizes, has_dynamic_features)
    assert y.size() == (batch_size, T, 60 + 1 + 1 + 1)

    mgc = y[:, :, : 60]
    lf0 = y[:, :, 60]
    vuv = y[:, :, 61]
    bap = y[:, :, 62]

    assert (unit_variance_mlpg(R, x[:, :, : 180]) == mgc).data.all()
    assert (unit_variance_mlpg(R, x[:, :, 180: 180 + 3]) == lf0).data.all()
    assert (x[:, :, 183] == vuv).data.all()
    assert (unit_variance_mlpg(R, x[:, :, 184: 184 + 3]) == bap).data.all()

    static_features = get_static_features(
        x, len(windows), stream_sizes, has_dynamic_features)
    assert static_features.size() == y.size()
