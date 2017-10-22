# coding: utf-8
import torch
from torch import nn
from torch.autograd import Variable

import numpy as np

from nnmnkwii.autograd import unit_variance_mlpg


class AbstractModel(object):
    """Interface for VC and TTS models
    """

    def include_parameter_generation(self):
        """Whether model includes parameter generation or not.
        """
        return False


class In2OutHighwayNet(AbstractModel, nn.Module):
    """Input-to-Output Highway Networks for voice conversion.

    Trying to replicate the model described in the following paper:
    https://www.jstage.jst.go.jp/article/transinf/E100.D/8/E100.D_2017EDL8034/

    .. note::
        Since model architecture itself includes parameter generation, we cannot
        simply use the model for multi-stream features (e.g., in TTS, acoustic
        features often consist multiple features; mgc, f0, vuv and bap.)
    """

    def __init__(self, in_dim=118, out_dim=118, static_dim=118 // 2,
                 num_hidden=3, hidden_dim=512, dropout=0.5):
        super(In2OutHighwayNet, self).__init__()
        self.static_dim = static_dim
        self.relu = nn.LeakyReLU(inplace=True)
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
        self.dropout = nn.Dropout(dropout)

    def include_parameter_generation(self):
        return True

    def forward(self, x, R, lengths=None):
        # Add batch axis
        x = x.unsqueeze(0) if x.dim() == 2 else x
        x_static = x[:, :, :self.static_dim]

        # T(x)
        Tx = self.sigmoid(self.T(x_static))

        # G(x)
        for layer in self.H:
            x = self.dropout(self.relu(layer(x)))
        x = self.last_linear(x)
        Gx = unit_variance_mlpg(R, x)

        # y^ = x + T(x) * G(x)
        return x, x_static + Tx * Gx


class In2OutRNNHighwayNet(AbstractModel, nn.Module):
    def __init__(self, in_dim=118, out_dim=118, static_dim=118 // 2,
                 num_hidden=3, hidden_dim=512, bidirectional=False, dropout=0.5):
        super(In2OutRNNHighwayNet, self).__init__()
        self.static_dim = static_dim
        self.num_direction = 2 if bidirectional else 1
        self.relu = nn.LeakyReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        # Transform gate (can be deep?)
        self.T = nn.Linear(static_dim, static_dim)

        # Recurrent hidden layers
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_hidden, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout)
        self.hidden2out = nn.Linear(hidden_dim * self.num_direction, out_dim)
        self.dropout = nn.Dropout(dropout)

    def include_parameter_generation(self):
        return True

    def forward(self, x, R, lengths=None):
        # Add batch axis
        x = x.unsqueeze(0) if x.dim() == 2 else x
        x_static = x[:, :, :self.static_dim]

        # T(x)
        Tx = self.sigmoid(self.T(x_static))

        # Pack padded sequence for CuDNN
        if isinstance(lengths, Variable):
            lengths = lengths.data.cpu().long().numpy()
        if lengths is not None:
            inputs = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True)
        else:
            inputs = x

        # G(x)
        output, _ = self.lstm(inputs)
        if lengths is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = self.hidden2out(output)

        Gx = unit_variance_mlpg(R, output)

        # y^ = x + T(x) * G(x)
        return x, x_static + Tx * Gx


class MLP(AbstractModel, nn.Module):
    def __init__(self, in_dim=118, out_dim=1, num_hidden=2, hidden_dim=256,
                 dropout=0.5, last_sigmoid=True, bidirectional=None):
        # bidirectional is dummy
        super(MLP, self).__init__()
        in_sizes = [in_dim] + [hidden_dim] * (num_hidden - 1)
        out_sizes = [hidden_dim] * num_hidden
        self.layers = nn.ModuleList(
            [nn.Linear(in_size, out_size) for (in_size, out_size)
             in zip(in_sizes, out_sizes)])
        self.last_linear = nn.Linear(hidden_dim, out_dim)
        self.relu = nn.LeakyReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.last_sigmoid = last_sigmoid

    def forward(self, x, lengths=None):
        for layer in self.layers:
            x = self.dropout(self.relu(layer(x)))
        x = self.last_linear(x)
        return self.sigmoid(x) if self.last_sigmoid else x


# needs https://github.com/taolei87/sru
class SRURNN(AbstractModel, nn.Module):
    def __init__(self, in_dim=118, out_dim=118, num_hidden=2, hidden_dim=256,
                 bidirectional=False, dropout=0, last_sigmoid=False,
                 use_relu=0, rnn_dropout=0.0):
        super(SRURNN, self).__init__()
        from cuda_functional import SRU
        self.num_direction = 2 if bidirectional else 1
        self.gru = SRU(in_dim, hidden_dim, num_hidden,
                       bidirectional=bidirectional, dropout=dropout,
                       use_relu=use_relu, rnn_dropout=rnn_dropout)
        self.hidden2out = nn.Linear(hidden_dim * self.num_direction, out_dim)
        self.sigmoid = nn.Sigmoid()
        self.last_sigmoid = last_sigmoid

    def forward(self, sequence, lengths):
        # Batch first -> Time first
        sequence = sequence.transpose(0, 1)
        output, _ = self.gru(sequence)
        # Time first -> Batch first
        output = output.transpose(0, 1)
        output = self.hidden2out(output)

        return self.sigmoid(output) if self.last_sigmoid else output


class GRURNN(AbstractModel, nn.Module):
    def __init__(self, in_dim=118, out_dim=118, num_hidden=2, hidden_dim=256,
                 bidirectional=False, dropout=0, last_sigmoid=False):
        super(GRURNN, self).__init__()
        self.num_direction = 2 if bidirectional else 1
        self.gru = nn.LSTM(in_dim, hidden_dim, num_hidden, batch_first=True,
                           bidirectional=bidirectional, dropout=dropout)
        self.hidden2out = nn.Linear(hidden_dim * self.num_direction, out_dim)
        self.sigmoid = nn.Sigmoid()
        self.last_sigmoid = last_sigmoid

    def forward(self, sequence, lengths):
        if isinstance(lengths, Variable):
            lengths = lengths.data.cpu().long().numpy()
        sequence = nn.utils.rnn.pack_padded_sequence(
            sequence, lengths, batch_first=True)
        output, _ = self.gru(sequence)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = self.hidden2out(output)

        return self.sigmoid(output) if self.last_sigmoid else output


class LSTMRNN(AbstractModel, nn.Module):
    def __init__(self, in_dim=118, out_dim=118, num_hidden=2, hidden_dim=256,
                 bidirectional=False, dropout=0, last_sigmoid=False):
        super(LSTMRNN, self).__init__()
        self.num_direction = 2 if bidirectional else 1
        self.lstm = nn.LSTM(in_dim, hidden_dim, num_hidden, batch_first=True,
                            bidirectional=bidirectional, dropout=dropout)
        self.hidden2out = nn.Linear(hidden_dim * self.num_direction, out_dim)
        self.sigmoid = nn.Sigmoid()
        self.last_sigmoid = last_sigmoid

    def forward(self, sequence, lengths):
        if isinstance(lengths, Variable):
            lengths = lengths.data.cpu().long().numpy()
        sequence = nn.utils.rnn.pack_padded_sequence(
            sequence, lengths, batch_first=True)
        output, _ = self.lstm(sequence)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = self.hidden2out(output)

        return self.sigmoid(output) if self.last_sigmoid else output
