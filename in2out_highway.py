# coding: utf-8
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

import numpy as np

from nnmnkwii.autograd import unit_variance_mlpg
from nnmnkwii.paramgen import unit_variance_mlpg_matrix


class In2OutHighwayNet(nn.Module):
    """Input-to-Output Highway Networks.

    Trying to replicate the model described in the following paper:
    https://www.jstage.jst.go.jp/article/transinf/E100.D/8/E100.D_2017EDL8034/

    .. note::
        Since model architecture itself includes parameter generation, we cannot
        simply use the model for multi-stream features (e.g., in TTS, acoustic
        features often includes mgc, f0, vuv and bap.)
    """

    def __init__(self, in_dim=118, out_dim=118, static_dim=118 // 2,
                 num_hidden=3, hidden_dim=512):
        super(In2OutHighwayNet, self).__init__()
        self.static_dim = static_dim
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # Transform gate (can be deep?)
        self.T = nn.Linear(static_dim, static_dim)

        # Hidden layers
        in_sizes = [in_dim] + [hidden_dim] * (num_hidden - 1)
        out_sizes = [hidden_dim] * num_hidden
        self.H = nn.ModuleList(
            [nn.Linear(in_size, out_size) for (in_size, out_size)
             in zip(in_sizes, out_sizes)])
        self.last_linear = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, mlpg_matrix):
        # Add batch axis
        x = x.unsqueeze(0) if x.dim() == 2 else x
        x_static = x[:, :, :self.static_dim]

        # T(x)
        Tx = self.sigmoid(self.T(x_static))

        # G(x)
        for layer in self.H:
            x = self.dropout(self.relu(layer(x)))
        x = self.last_linear(x)
        Gx = unit_variance_mlpg(mlpg_matrix, x)

        # y^ = x + T(x) * G(x)
        return x_static + Tx * Gx


class Discriminator(nn.Module):
    def __init__(self, in_dim=118 // 2, num_hidden=2, hidden_dim=256):
        super(Discriminator, self).__init__()
        in_sizes = [in_dim] + [hidden_dim] * (num_hidden - 1)
        out_sizes = [hidden_dim] * num_hidden
        self.layers = nn.ModuleList(
            [nn.Linear(in_size, out_size) for (in_size, out_size)
             in zip(in_sizes, out_sizes)])
        self.last_linear = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        for layer in self.layers:
            x = self.dropout(self.relu(layer(x)))
        return self.sigmoid(self.last_linear(x))


class MLP(nn.Module):
    """Very simple deep neural networks.
    """

    def __init__(self, in_dim=118, out_dim=118, num_hidden=2, hidden_dim=256):
        super(MLP, self).__init__()
        in_sizes = [in_dim] + [hidden_dim] * (num_hidden - 1)
        out_sizes = [hidden_dim] * num_hidden
        self.layers = nn.ModuleList(
            [nn.Linear(in_size, out_size) for (in_size, out_size)
             in zip(in_sizes, out_sizes)])
        self.last_linear = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.)

    def forward(self, x, lengths=None):
        for layer in self.layers:
            x = self.dropout(self.relu(layer(x)))
        return self.last_linear(x)


class LSTMRNN(nn.Module):
    def __init__(self,  in_dim=118, out_dim=118, num_hidden=2, hidden_dim=256):
        super(LSTMRNN, self).__init__()
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_hidden, batch_first=True)
        self.hidden2out = nn.Linear(hidden_dim, out_dim)

    def forward(self, sequence, lengths):
        if isinstance(lengths, Variable):
            lengths = lengths.data.cpu().long().numpy()
        sequence = nn.utils.rnn.pack_padded_sequence(sequence, lengths, batch_first=True)
        output, _ = self.lstm(sequence)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = self.hidden2out(output)
        return output

# Taken from
# https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation


def sequence_mask(sequence_length):
    max_len = sequence_length.data.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = Variable(seq_range_expand)
    if sequence_length.is_cuda:
        seq_range_expand = seq_range_expand.cuda()
    seq_length_expand = sequence_length.unsqueeze(1) \
        .expand_as(seq_range_expand)
    return (seq_range_expand < seq_length_expand).float()


class MaskedCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(MaskedCrossEntropyLoss, self).__init__()

    def forward(self, logits, target, length):
        """
        Args:
            logits: A Variable containing a FloatTensor of size
                (batch, max_len, num_classes) which contains the
                unnormalized probability for each class.
            target: A Variable containing a LongTensor of size
                (batch, max_len) which contains the index of the true
                class for each corresponding step.
            length: A Variable containing a LongTensor of size (batch,)
                which contains the length of each data in a batch.
        Returns:
            loss: An average loss value masked by the length.
        """

        # logits_flat: (batch * max_len, num_classes)
        logits_flat = logits.view(-1, logits.size(-1))
        # log_probs_flat: (batch * max_len, num_classes)
        log_probs_flat = F.log_softmax(logits_flat)
        # target_flat: (batch * max_len, 1)
        target_flat = target.view(-1, 1)
        # losses_flat: (batch * max_len, 1)
        losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
        # losses: (batch, max_len)
        losses = losses_flat.view(*target.size())
        # mask: (batch, max_len)
        mask = sequence_mask(length)
        losses = losses * mask.float()
        loss = losses.sum() / length.float().sum()
        return loss

# Adapted from:
# https://github.com/facebookresearch/loop/blob/master/model.py


class MaskedMSELoss(nn.Module):
    def __init__(self):
        super(MaskedMSELoss, self).__init__()
        self.criterion = nn.MSELoss(size_average=False)

    def forward(self, input, target, lengths):
        # (B, T, 1)
        mask = sequence_mask(lengths).unsqueeze(-1)
        # (B, T, D)
        mask_ = mask.expand_as(input)
        loss = self.criterion(input * mask_, target * mask_)
        return loss / mask.sum()


class MaskedMGELoss(nn.Module):
    def __init__(self):
        super(MaskedMGELoss, self).__init__()
        self.criterion = MaskedMSELoss()

    def forward(self, inputs, targets, lengths, R):
        inputs = unit_variance_mlpg(R, inputs)
        print(inputs.size(), targets.size())
        assert inputs.size(-1) == targets.size(-1)
        return self.criterion(inputs, targets, lengths)


def get_static_stream_sizes(stream_sizes, has_dynamic_features, num_windows):
    static_stream_sizes = np.array(stream_sizes)
    static_stream_sizes[has_dynamic_features] = \
        static_stream_sizes[has_dynamic_features] / num_windows

    return static_stream_sizes


def get_static_features(inputs, num_windows, stream_sizes=[180, 3, 1, 3],
                        has_dynamic_features=[True, True, False, True]):
    _, _, D = inputs.size()
    if stream_sizes is None or (len(stream_sizes) == 1 and has_dynamic_features[0]):
        return inputs[:, :, :D // num_windows]
    if len(stream_sizes) == 1 and not has_dynamic_features[0]:
        return inputs

    # Multi stream case
    ret = []
    start_indices = np.hstack(([0], np.cumsum(stream_sizes)[:-1]))
    for start_idx, size, v in zip(
            start_indices, stream_sizes, has_dynamic_features):
        if v:
            static_features = inputs[:, :, start_idx:start_idx + size // num_windows]
        else:
            static_features = inputs[:, :, start_idx:start_idx + size]
        ret.append(static_features)
    return torch.cat(ret, dim=-1)


def multi_stream_mlpg(inputs, R,
                      stream_sizes=[180, 3, 1, 3],
                      has_dynamic_features=[True, True, False, True]):
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
    for in_start_idx, in_end_idx, out_start_idx, out_end_idx, v in zip(
            start_indices, end_indices, static_stream_start_indices,
            static_stream_end_indices, has_dynamic_features):
        x = inputs[:, :, in_start_idx:in_end_idx]
        y = unit_variance_mlpg(R, x) if v else x
        ret.append(y)

    return torch.cat(ret, dim=-1)


class MaskedMultiStreamMGELoss(nn.Module):
    def __init__(self):
        super(MaskedMultiStreamMGELoss, self).__init__()
        self.masked_mge = MaskedMGELoss()
        self.masked_mse = MaskedMSELoss()

    def forward(self, inputs, targets, lengths, windows,
                stream_sizes=[180, 3, 1, 3],
                has_dynamic_features=[True, True, False, True],
                ):
        B, T, D = inputs.size()
        assert D == sum(stream_sizes)

        # Straem indices for static+delta features
        # [0,   180, 183, 184]
        start_indices = np.hstack(([0], np.cumsum(stream_sizes)[:-1]))
        # [180, 183, 184, 187]
        end_indices = np.cumsum(stream_sizes)

        # Stream sizes for static features
        # [60, 1, 1, 1]
        static_stream_sizes = get_static_stream_sizes(
            stream_sizes, has_dynamic_features, len(windows))

        # [0,  60, 61, 62]
        static_stream_start_indices = np.hstack(
            ([0], np.cumsum(static_stream_sizes)[:-1]))
        # [60, 61, 62, 63]
        static_stream_end_indices = np.cumsum(static_stream_sizes)

        # MLPG matrix
        R = unit_variance_mlpg_matrix(windows, T)
        R = torch.from_numpy(R)
        R = R.cuda() if inputs.data.is_cuda else R

        loss = 0.0
        for in_start_idx, in_end_idx, out_start_idx, out_end_idx, v in zip(
                start_indices, end_indices, static_stream_start_indices,
                static_stream_end_indices, has_dynamic_features):
            x = inputs[:, :, in_start_idx:in_end_idx]
            y = targets[:, :, out_start_idx:out_end_idx]

            # If the stream has dynamic featrues, then apply MGE loss
            if v:
                loss += self.masked_mge(x, y, lengths, R)
            else:
                loss += self.masked_mse(x, y, lengths)

        return loss
