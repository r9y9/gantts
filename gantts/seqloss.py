# coding: utf-8
import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

import numpy as np

from nnmnkwii.autograd import unit_variance_mlpg
from nnmnkwii.paramgen import unit_variance_mlpg_matrix

from .multistream import get_static_stream_sizes


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


# Unused now, to be removed
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


class MaskedMGELoss(nn.Module):
    def __init__(self):
        super(MaskedMGELoss, self).__init__()
        self.criterion = MaskedMSELoss()

    def forward(self, inputs, targets, lengths, R):
        inputs = unit_variance_mlpg(R, inputs)
        print(inputs.size(), targets.size())
        assert inputs.size(-1) == targets.size(-1)
        return self.criterion(inputs, targets, lengths)


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
