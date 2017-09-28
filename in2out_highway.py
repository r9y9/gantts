import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

from nnmnkwii.autograd import unit_variance_mlpg


class In2OutHighwayNet(nn.Module):
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

# Taken from
# https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation


def _sequence_mask(sequence_length):
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
        mask = _sequence_mask(length)
        losses = losses * mask.float()
        loss = losses.sum() / length.float().sum()
        return loss

# Adapted from:
# https://github.com/facebookresearch/loop/blob/master/model.py


class MaskedMSE(nn.Module):
    def __init__(self):
        super(MaskedMSE, self).__init__()
        self.criterion = nn.MSELoss(size_average=False)

    def forward(self, input, target, lengths):
        # (B, T, 1)
        mask = _sequence_mask(lengths).unsqueeze(-1)
        # (B, T, D)
        mask_ = mask.expand_as(input)
        loss = self.criterion(input * mask_, target * mask_)
        return loss / mask.sum()
