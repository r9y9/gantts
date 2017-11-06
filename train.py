# coding: utf-8
"""Trainining script for GAN-based TTS and VC models.

usage: train.py [options] <inputs_dir> <outputs_dir>

options:
    --hparams_name=<name>       Name of hyper params [default: vc].
    --hparams=<parmas>          Hyper parameters to be overrided [default: ].
    --checkpoint-dir=<dir>      Where to save models [default: checkpoints].
    --checkpoint-g=<name>       Load generator from checkpoint if given.
    --checkpoint-d=<name>       Load discriminator from checkpoint if given.
    --checkpoint-r=<name>       Load reference model to compute spoofing rate.
    --max_files=<N>             Max num files to be collected. [default: -1]
    --discriminator-warmup      Warmup discriminator.
    --w_d=<f>                   Adversarial (ADV) loss weight [default: 1.0].
    --mse_w=<f>                 Mean squared error (MSE) loss weight [default: 0.0].
    --mge_w=<f>                 Minimum generation error (MGE) loss weight [default: 1.0].
    --restart_epoch=<N>         Restart epoch [default: -1].
    --reset_optimizers          Reset optimizers, otherwise restored from checkpoint.
    --log-event-path=<name>     Log event path.
    --disable-slack             Disable slack message.
    -h, --help                  Show this help message and exit
"""
from docopt import docopt

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
from sklearn.model_selection import train_test_split

import sys
import time
import os
import math
from os.path import splitext, join, abspath, exists
from tqdm import tqdm
from warnings import warn

import tensorboard_logger
from tensorboard_logger import log_value

from nnmnkwii import preprocessing as P
from nnmnkwii import metrics
from nnmnkwii.paramgen import unit_variance_mlpg_matrix
from nnmnkwii.datasets import FileSourceDataset, FileDataSource
from nnmnkwii.datasets import MemoryCacheDataset

import gantts
from gantts.multistream import multi_stream_mlpg, get_static_features
from gantts.multistream import get_static_stream_sizes, select_streams
from gantts.multistream import recompute_delta_features
from gantts.seqloss import MaskedMSELoss, sequence_mask

import hparams
from hparams import hparams_debug_string
hp = None  # to be initailized later

global_epoch = 0
test_size = 0.112  # 1000 training data for cmu arctic
random_state = 1234
checkpoint_interval = 10

use_cuda = torch.cuda.is_available()


class NPYDataSource(FileDataSource):
    def __init__(self, dirname, train=True, max_files=None, test=False):
        self.dirname = dirname
        self.train = train
        self.test = test
        self.max_files = max_files

    def collect_files(self):
        npy_files = list(filter(lambda x: splitext(x)[-1] == ".npy",
                                os.listdir(self.dirname)))
        npy_files = sorted(list(map(lambda d: join(self.dirname, d), npy_files)))
        # last 5 is for real testset
        if self.test:
            return npy_files[len(npy_files) - 5:]
        npy_files = npy_files[:len(npy_files) - 5]
        if self.max_files is not None and self.max_files > 0:
            npy_files = npy_files[:self.max_files]
        train_files, test_files = train_test_split(
            npy_files, test_size=test_size, random_state=random_state)
        return train_files if self.train else test_files

    def collect_features(self, path):
        return np.load(path)


class VCDataset(object):
    def __init__(self, X, Y, data_mean, data_std):
        self.X = X
        self.Y = Y
        self.data_mean = data_mean
        self.data_std = data_std

    def __getitem__(self, idx):
        x = P.scale(self.X[idx], self.data_mean, self.data_std)
        y = P.scale(self.Y[idx], self.data_mean, self.data_std)
        return x, y

    def __len__(self):
        return len(self.X)


class TTSDataset(object):
    def __init__(self, X, Y, X_data_min, X_data_max, Y_data_mean, Y_data_std):
        self.X = X
        self.Y = Y
        self.X_data_min, self.X_data_scale = P.minmax_scale_params(
            X_data_min, X_data_max, feature_range=(0.01, 0.99))
        self.Y_data_mean = Y_data_mean
        self.Y_data_std = Y_data_std

    def __getitem__(self, idx):
        x = P.minmax_scale(
            self.X[idx], min_=self.X_data_min, scale_=self.X_data_scale,
            feature_range=(0.01, 0.99))
        y = P.scale(self.Y[idx], self.Y_data_mean, self.Y_data_std)

        # To handle inconsistent static-delta relationship after normalization
        # This is required to use MSE + MGE loss work
        if hp.recompute_delta_features:
            y = recompute_delta_features(y, self.Y_data_mean, self.Y_data_std,
                                         hp.windows, hp.stream_sizes,
                                         hp.has_dynamic_features)
        return x, y

    def __len__(self):
        return len(self.X)


def _pad_2d(x, max_len):
    x = np.pad(x, [(0, max_len - len(x)), (0, 0)],
               mode="constant", constant_values=0)
    return x


def collate_fn(batch):
    """Create batch"""

    input_lengths = np.array([len(x[0]) for x in batch], dtype=np.int)
    max_len = np.max(input_lengths)
    x_batch = np.array([_pad_2d(x[0], max_len) for x in batch],
                       dtype=np.float32)
    y_batch = np.array([_pad_2d(x[1], max_len) for x in batch],
                       dtype=np.float32)

    x_batch = torch.FloatTensor(x_batch)
    y_batch = torch.FloatTensor(y_batch)
    input_lengths = torch.LongTensor(input_lengths)

    return x_batch, y_batch, input_lengths


def save_checkpoint(model, optimizer, epoch, checkpoint_dir, name):
    checkpoint_path = join(
        checkpoint_dir, "checkpoint_epoch{}_{}.pth".format(
            epoch, name))
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)


def get_vc_data_loaders(X, Y, data_mean, data_var):
    X_train, X_test = X["train"], X["test"]
    Y_train, Y_test = Y["train"], Y["test"]

    # Sequence-wise train loader
    X_train_cache_dataset = MemoryCacheDataset(X_train, cache_size=hp.cache_size)
    Y_train_cache_dataset = MemoryCacheDataset(Y_train, cache_size=hp.cache_size)
    train_dataset = VCDataset(
        X_train_cache_dataset, Y_train_cache_dataset, data_mean, data_std)
    train_loader = data_utils.DataLoader(
        train_dataset, batch_size=hp.batch_size,
        num_workers=hp.num_workers, pin_memory=hp.pin_memory,
        shuffle=True, collate_fn=collate_fn)

    # Sequence-wise test loader
    X_test_cache_dataset = MemoryCacheDataset(X_test, cache_size=hp.cache_size)
    Y_test_cache_dataset = MemoryCacheDataset(Y_test, cache_size=hp.cache_size)
    test_dataset = VCDataset(
        X_test_cache_dataset, Y_test_cache_dataset, data_mean, data_std)
    test_loader = data_utils.DataLoader(
        test_dataset, batch_size=hp.batch_size,
        num_workers=hp.num_workers, pin_memory=hp.pin_memory,
        shuffle=False, collate_fn=collate_fn)

    dataset_loaders = {"train": train_loader, "test": test_loader}
    return dataset_loaders


def get_tts_data_loaders(X, Y, X_data_min, X_data_max, Y_data_mean, Y_data_std):
    X_train, X_test = X["train"], X["test"]
    Y_train, Y_test = Y["train"], Y["test"]

    # Sequence-wise train loader
    X_train_cache_dataset = MemoryCacheDataset(X_train, cache_size=hp.cache_size)
    Y_train_cache_dataset = MemoryCacheDataset(Y_train, cache_size=hp.cache_size)
    train_dataset = TTSDataset(
        X_train_cache_dataset, Y_train_cache_dataset,
        X_data_min, X_data_max, Y_data_mean, Y_data_std)
    train_loader = data_utils.DataLoader(
        train_dataset, batch_size=hp.batch_size,
        num_workers=hp.num_workers, pin_memory=hp.pin_memory,
        shuffle=True, collate_fn=collate_fn)

    # Sequence-wise test loader
    X_test_cache_dataset = MemoryCacheDataset(X_test, cache_size=hp.cache_size)
    Y_test_cache_dataset = MemoryCacheDataset(Y_test, cache_size=hp.cache_size)
    test_dataset = TTSDataset(
        X_test_cache_dataset, Y_test_cache_dataset,
        X_data_min, X_data_max, Y_data_mean, Y_data_std)
    test_loader = data_utils.DataLoader(
        test_dataset, batch_size=hp.batch_size,
        num_workers=hp.num_workers, pin_memory=hp.pin_memory,
        shuffle=False, collate_fn=collate_fn)

    dataset_loaders = {"train": train_loader, "test": test_loader}
    return dataset_loaders


def get_selected_static_stream(y_hat_static):
    static_stream_sizes = get_static_stream_sizes(
        hp.stream_sizes, hp.has_dynamic_features, len(hp.windows))
    y_hat_selected = select_streams(y_hat_static, static_stream_sizes,
                                    streams=hp.adversarial_streams)
    # 0-th mgc with adversarial trainging affects speech quality
    # ref: saito17asja_gan.pdf
    if hp.mask_nth_mgc_for_adv_loss > 0:
        assert hp == hparams.tts_acoustic
        y_hat_selected = y_hat_selected[:, :, hp.mask_nth_mgc_for_adv_loss:]
    return y_hat_selected


def update_discriminator(model_d, optimizer_d, x, y_static, y_hat_static, lengths,
                         mask, phase, eps=1e-20):
    # Select streams
    if hp.adversarial_streams is not None:
        y_static_adv = get_selected_static_stream(y_static)
        y_hat_static_adv = get_selected_static_stream(y_hat_static)
    else:
        y_static_adv, y_hat_static_adv = y_static, y_hat_static

    if hp.discriminator_linguistic_condition:
        y_static_adv = torch.cat((x, y_static_adv), -1)
        y_hat_static_adv = torch.cat((x, y_hat_static_adv), -1)

    T = mask.sum().data[0]

    # Real
    D_real = model_d(y_static_adv, lengths=lengths)
    real_correct_count = ((D_real > 0.5).float() * mask).sum().data[0]

    # Fake
    D_fake = model_d(y_hat_static_adv, lengths=lengths)
    fake_correct_count = ((D_fake < 0.5).float() * mask).sum().data[0]

    # Loss
    loss_real_d = -(torch.log(D_real + eps) * mask).sum() / T
    loss_fake_d = -(torch.log(1 - D_fake + eps) * mask).sum() / T
    loss_d = loss_real_d + loss_fake_d

    if phase == "train":
        loss_d.backward(retain_graph=True)
        torch.nn.utils.clip_grad_norm(model_d.parameters(), 1.0)
        optimizer_d.step()

    return loss_d.data[0], loss_fake_d.data[0], loss_real_d.data[0],\
        real_correct_count, fake_correct_count


def update_generator(model_g, model_d, optimizer_g,
                     x, y, y_hat, y_static, y_hat_static,
                     adv_w, lengths, mask, phase,
                     mse_w=None, mge_w=None, eps=1e-20):
    T = mask.sum().data[0]

    criterion = MaskedMSELoss()

    # MSELoss in static feature domain
    loss_mge = criterion(y_hat_static, y_static, mask=mask)

    # MSELoss in static + delta features domain
    loss_mse = criterion(y_hat, y, mask=mask)

    # Adversarial loss
    if adv_w > 0:
        # Select streams
        if hp.adversarial_streams is not None:
            y_hat_static_adv = get_selected_static_stream(y_hat_static)
        else:
            y_hat_static_adv = y_hat_static

        if hp.discriminator_linguistic_condition:
            y_hat_static_adv = torch.cat((x, y_hat_static_adv), -1)

        loss_adv = -(torch.log(model_d(
            y_hat_static_adv, lengths=lengths) + eps) * mask).sum() / T
    else:
        loss_adv = Variable(y.data.new(1).zero_())

    # MSE + MGE + ADV loss
    # try to decieve discriminator
    loss_g = (mse_w * loss_mse + mge_w * loss_mge) + adv_w * loss_adv
    if phase == "train":
        loss_g.backward()
        torch.nn.utils.clip_grad_norm(model_g.parameters(), 1.0)
        optimizer_g.step()

    return loss_mse.data[0], loss_mge.data[0], loss_adv.data[0], loss_g.data[0]


def exp_lr_scheduler(optimizer, epoch, nepoch, init_lr=0.0001, lr_decay_epoch=100):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {} at epoch {}'.format(lr, epoch))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


def apply_generator(model_g, x, R, lengths):
    if model_g.include_parameter_generation():
        # Case: models include parameter generation in itself
        # Mulistream features cannot be used in this case
        y_hat, y_hat_static = model_g(x, R, lengths=lengths)
    else:
        # Case: generic models (can be sequence model)
        assert hp.has_dynamic_features is not None
        y_hat = model_g(x, lengths=lengths)

        # Handle dimention mismatch
        # This happens when we use pad_packed_sequence
        if y_hat.size(1) != x.size(1):
            y_hat = F.pad(y_hat.unsqueeze(
                0), (0, 0, x.size(1) - y_hat.size(-2), 0)).squeeze(0)

        y_hat_static = multi_stream_mlpg(
            y_hat, R, hp.stream_sizes, hp.has_dynamic_features)

    return y_hat, y_hat_static


def inv_scale(mgc, lf0, vuv, bap, Y_mean, Y_std, binalize_vuv=True):
    # static + dynamic domain
    mgc_dim, lf0_dim, vuv_dim, bap_dim = hp.stream_sizes
    windows = hp.windows

    mgc_start_idx = 0
    lf0_start_idx = mgc_dim
    vuv_start_idx = lf0_start_idx + lf0_dim
    bap_start_idx = vuv_start_idx + vuv_dim

    mgc = P.inv_scale(mgc, Y_mean[:mgc_dim // len(windows)],
                      Y_std[:mgc_dim // len(windows)])
    lf0 = P.inv_scale(lf0, Y_mean[lf0_start_idx:lf0_start_idx + lf0_dim // len(windows)],
                      Y_std[lf0_start_idx:lf0_start_idx + lf0_dim // len(windows)])
    bap = P.inv_scale(bap, Y_mean[bap_start_idx:bap_start_idx + bap_dim // len(windows)],
                      Y_std[bap_start_idx:bap_start_idx + bap_dim // len(windows)])
    vuv = P.inv_scale(vuv, Y_mean[vuv_start_idx], Y_std[vuv_start_idx])
    if binalize_vuv:
        vuv[vuv > 0.5] = 1.0
        vuv[vuv <= 0.5] = 0
        vuv = vuv.long()

    return mgc, lf0, vuv, bap


def split_streams(y_static, Y_data_mean, Y_data_std):
    # static domain
    mgc_dim, lf0_dim, vuv_dim, bap_dim = get_static_stream_sizes(
        hp.stream_sizes, hp.has_dynamic_features, len(hp.windows))
    mgc_start_idx = 0
    lf0_start_idx = mgc_dim
    vuv_start_idx = lf0_start_idx + lf0_dim
    bap_start_idx = vuv_start_idx + vuv_dim
    mgc = y_static[:, :, :lf0_start_idx]
    lf0 = y_static[:, :, lf0_start_idx:vuv_start_idx]
    vuv = y_static[:, :, vuv_start_idx]
    bap = y_static[:, :, bap_start_idx:]

    return inv_scale(mgc, lf0, vuv, bap, Y_data_mean, Y_data_std)


def compute_distortions(y_static, y_hat_static, Y_data_mean, Y_data_std, lengths=None):
    if hp.name == "acoustic":
        mgc, lf0, vuv, bap = split_streams(y_static, Y_data_mean, Y_data_std)
        mgc_hat, lf0_hat, vuv_hat, bap_hat = split_streams(
            y_hat_static, Y_data_mean, Y_data_std)
        try:
            f0_mse = metrics.lf0_mean_squared_error(
                lf0, vuv, lf0_hat, vuv_hat,
                lengths=lengths, linear_domain=True)
        except ZeroDivisionError:
            f0_mse = np.nan

        distortions = {
            "mcd": metrics.melcd(mgc[:, :, 1:], mgc_hat[:, :, 1:], lengths=lengths),
            "bap_mcd": metrics.melcd(bap, bap_hat, lengths=lengths) / 10.0,
            "f0_rmse": np.sqrt(f0_mse),
            "vuv_err": metrics.vuv_error(vuv, vuv_hat, lengths=lengths),
        }
    elif hp.name == "duration":
        y_static_invscale = P.inv_scale(y_static, Y_data_mean, Y_data_std)
        y_hat_static_invscale = P.inv_scale(y_hat_static, Y_data_mean, Y_data_std)
        distortions = {"dur_rmse": math.sqrt(metrics.mean_squared_error(
            y_static_invscale, y_hat_static_invscale, lengths=lengths))}
    elif hp.name == "vc":
        static_dim = hp.order
        y_static_invscale = P.inv_scale(y_static, Y_data_mean[:static_dim], Y_data_std[:static_dim])
        y_hat_static_invscale = P.inv_scale(
            y_hat_static, Y_data_mean[:static_dim], Y_data_std[:static_dim])
        distortions = {"mcd": metrics.melcd(
            y_static_invscale, y_hat_static_invscale, lengths=lengths)}
    else:
        assert False

    return distortions


def train_loop(models, optimizers, dataset_loaders,
               w_d=0.0, mse_w=0.0, mge_w=1.0,
               update_d=True, update_g=True,
               reference_discriminator=None):
    model_g, model_d = models
    optimizer_g, optimizer_d = optimizers
    if use_cuda:
        model_g, model_d = model_g.cuda(), model_d.cuda()
        if reference_discriminator is not None:
            reference_discriminator = reference_discriminator.cuda()
            reference_discriminator.eval()

    if hp == hparams.vc:
        Y_data_mean = dataset_loaders["train"].dataset.data_mean
        Y_data_std = dataset_loaders["train"].dataset.data_std
    else:
        Y_data_mean = dataset_loaders["train"].dataset.Y_data_mean
        Y_data_std = dataset_loaders["train"].dataset.Y_data_std
    Y_data_mean = torch.from_numpy(Y_data_mean)
    Y_data_std = torch.from_numpy(Y_data_std)
    if use_cuda:
        Y_data_mean = Y_data_mean.cuda()
        Y_data_std = Y_data_std.cuda()

    E_loss_mge = 1
    E_loss_adv = 1
    has_dynamic = np.any(hp.has_dynamic_features)
    global global_epoch

    for global_epoch in tqdm(range(global_epoch + 1, hp.nepoch + 1)):
        # LR schedule
        if hp.lr_decay_schedule and update_g:
            optimizer_g = exp_lr_scheduler(optimizer_g, global_epoch - 1, hp.nepoch,
                                           init_lr=hp.optimizer_g_params["lr"],
                                           lr_decay_epoch=hp.lr_decay_epoch)
        if hp.lr_decay_schedule and update_d:
            optimizer_d = exp_lr_scheduler(optimizer_d, global_epoch - 1, hp.nepoch,
                                           init_lr=hp.optimizer_d_params["lr"],
                                           lr_decay_epoch=hp.lr_decay_epoch)

        for phase in ["train", "test"]:
            running_loss = {"generator": 0.0, "mse": 0.0, "mge": 0.0,
                            "loss_real_d": 0.0,
                            "loss_fake_d": 0.0,
                            "loss_adv": 0.0,
                            "discriminator": 0.0}
            if phase == "train":
                model_g.train()
                model_d.train()
            else:
                model_g.eval()
                model_d.eval()
            running_metrics = {}
            real_correct_count, fake_correct_count = 0, 0
            regard_fake_as_natural = 0
            N = len(dataset_loaders[phase])
            total_num_frames = 0
            for x, y, lengths in dataset_loaders[phase]:
                # Sort by lengths. This is needed for pytorch's PackedSequence
                sorted_lengths, indices = torch.sort(
                    lengths.view(-1), dim=0, descending=True)
                sorted_lengths = sorted_lengths.long()
                cpu_sorted_lengths = list(sorted_lengths)
                max_len = sorted_lengths[0]

                # Get sorted batch
                x, y = x[indices], y[indices]

                # Generator noise
                if hp.generator_add_noise:
                    z = torch.rand(x.size(0), max_len, hp.generator_noise_dim)
                else:
                    z = None

                # Construct MLPG paramgen matrix for every batch
                if has_dynamic:
                    R = unit_variance_mlpg_matrix(hp.windows, max_len)
                    R = torch.from_numpy(R)
                    R = R.cuda() if use_cuda else R
                else:
                    R = None

                if use_cuda:
                    x, y = x.cuda(), y.cuda()
                    sorted_lengths = sorted_lengths.cuda()
                    z = z.cuda() if z is not None else None

                # Pack into variables
                x, y = Variable(x), Variable(y)
                z = Variable(z) if z is not None else None
                sorted_lengths = Variable(sorted_lengths)

                # Static features
                y_static = get_static_features(
                    y, len(hp.windows), hp.stream_sizes, hp.has_dynamic_features)

                # Num frames in batch
                total_num_frames += sorted_lengths.float().sum().data[0]

                # Mask
                mask = sequence_mask(sorted_lengths).unsqueeze(-1)

                # Reset optimizers state
                optimizer_g.zero_grad()
                optimizer_d.zero_grad()

                # Apply model (generator)
                generator_input = torch.cat((x, z), -1) if z is not None else x
                y_hat, y_hat_static = apply_generator(
                    model_g, generator_input, R, cpu_sorted_lengths)
                # Should have same time length
                assert x.size(1) == y_hat.size(1)

                # Compute spoofing rate
                if reference_discriminator is not None:
                    if hp.adversarial_streams is not None:
                        y_hat_static_ref = get_selected_static_stream(y_hat_static)
                    else:
                        y_hat_static_ref = y_hat_static
                    target = reference_discriminator(
                        y_hat_static_ref, lengths=cpu_sorted_lengths)
                    # Count samples classified as natural, while inputs are
                    # actually generated.
                    regard_fake_as_natural += ((target > 0.5).float() * mask).sum().data[0]

                ### Update discriminator ###
                # Natural: 1, Genrated: 0
                if update_d:
                    loss_d, loss_fake_d, loss_real_d, _real_correct_count,\
                        _fake_correct_count = update_discriminator(
                            model_d, optimizer_d, x, y_static, y_hat_static,
                            cpu_sorted_lengths, mask, phase)
                    running_loss["discriminator"] += loss_d
                    running_loss["loss_fake_d"] += loss_fake_d
                    running_loss["loss_real_d"] += loss_real_d
                    real_correct_count += _real_correct_count
                    fake_correct_count += _fake_correct_count

                ### Update generator ###
                if update_g:
                    adv_w = w_d * float(np.clip(E_loss_mge / E_loss_adv, 0, 1e+3))
                    loss_mse, loss_mge, loss_adv, loss_g = update_generator(
                        model_g, model_d, optimizer_g, x, y, y_hat,
                        y_static, y_hat_static,
                        adv_w, cpu_sorted_lengths, mask, phase,
                        mse_w=mse_w, mge_w=mge_w)

                    running_loss["mse"] += loss_mse
                    running_loss["mge"] += loss_mge
                    running_loss["loss_adv"] += loss_adv
                    running_loss["generator"] += loss_g

                    # Distotions
                    distortions = compute_distortions(
                        y_static.data, y_hat_static.data,
                        Y_data_mean, Y_data_std, sorted_lengths.data)
                    for k, v in distortions.items():
                        try:
                            running_metrics[k] += float(v)
                        except KeyError:
                            running_metrics[k] = float(v)

            # Update expectation
            # NOTE: E_loss_mge is not exactly same as E[L_mge(y,y_hat)]
            # in thier papers, since we add MSE term in the loss.
            # It will be same if mse_w = 0 and mge_w = 1.
            if update_d and update_g and phase == "train":
                E_loss_mge = (mse_w * running_loss["mse"] +
                              mge_w * running_loss["mge"]) / N
                E_loss_adv = running_loss["loss_adv"] / N
                log_value("E(mge)", E_loss_mge, global_epoch)
                log_value("E(adv)", E_loss_adv, global_epoch)
                log_value("MGE/ADV loss weight",  E_loss_mge / E_loss_adv, global_epoch)

            # Log loss
            for ty, enabled in [("mse", update_g),
                                ("mge", update_g),
                                ("discriminator", update_d),
                                ("loss_real_d", update_d),
                                ("loss_fake_d", update_d),
                                ("loss_adv", update_g and update_d),
                                ("generator", update_g)]:
                if enabled:
                    ave_loss = running_loss[ty] / N
                    log_value(
                        "{} {} loss".format(phase, ty), ave_loss, global_epoch)

            # Log eval metrics
            for k, v in running_metrics.items():
                log_value(
                    "{} {} metric".format(phase, k), v / N, global_epoch)

            # Log discriminator classification accuracy
            if update_d:
                log_value("Real {} acc".format(phase),
                          real_correct_count / total_num_frames, global_epoch)
                log_value("Fake {} acc".format(phase),
                          fake_correct_count / total_num_frames, global_epoch)

            # Log spoofing rate for generated features by reference model
            if reference_discriminator is not None:
                log_value("{} spoofing rate".format(phase),
                          regard_fake_as_natural / total_num_frames, global_epoch)

        # Save checkpoints
        if global_epoch % checkpoint_interval == 0:
            for model, optimizer, enabled, name in [
                    (model_g, optimizer_g, update_g, "Generator"),
                    (model_d, optimizer_d, update_d, "Discriminator")]:
                if enabled:
                    save_checkpoint(
                        model, optimizer, global_epoch, checkpoint_dir, name)

    return 0


def load_checkpoint(model, optimizer, checkpoint_path):
    global global_epoch
    print("Load checkpoint from: {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    global_epoch = checkpoint["global_epoch"]


if __name__ == "__main__":
    since = time.time()
    args = docopt(__doc__)
    print("Command line args:\n", args)
    hp = getattr(hparams, args["--hparams_name"])

    # Override hyper parameters
    hp.parse(args["--hparams"])
    print(hparams_debug_string(hp))

    inputs_dir = args["<inputs_dir>"]
    outputs_dir = args["<outputs_dir>"]

    # Assuming inputs and outputs are in same parent directoy
    # This can be relaxed, but for now it's fine.
    data_dir = abspath(join(inputs_dir, os.pardir))
    assert data_dir == abspath(join(outputs_dir, os.pardir))

    checkpoint_dir = args["--checkpoint-dir"]
    checkpoint_path_d = args["--checkpoint-d"]
    checkpoint_path_g = args["--checkpoint-g"]
    checkpoint_path_r = args["--checkpoint-r"]
    max_files = int(args["--max_files"])
    w_d = float(args["--w_d"])
    mse_w = float(args["--mse_w"])
    mge_w = float(args["--mge_w"])
    discriminator_warmup = args["--discriminator-warmup"]
    restart_epoch = int(args["--restart_epoch"])

    reset_optimizers = args["--reset_optimizers"]
    log_event_path = args["--log-event-path"]
    disable_slack = args["--disable-slack"]

    # Flags to update discriminator/generator or not
    update_d = w_d > 0
    update_g = False if discriminator_warmup else True

    if not exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    X = {"train": {}, "test": {}}
    Y = {"train": {}, "test": {}}
    utt_lengths = {"train": {}, "test": {}}

    for phase in ["train", "test"]:
        train = True if phase == "train" else False
        X[phase] = FileSourceDataset(
            NPYDataSource(inputs_dir, train=train, max_files=max_files))
        Y[phase] = FileSourceDataset(
            NPYDataSource(outputs_dir, train=train, max_files=max_files))
        # Assuming X and Y are time aligned.
        x_lengths = np.array([len(x) for x in X[phase]])
        y_lengths = np.array([len(y) for y in Y[phase]])
        assert np.allclose(x_lengths, y_lengths)
        utt_lengths[phase] = x_lengths
        print("Size of dataset for {}: {}".format(phase, len(X[phase])))

    # Collect stats for noramlization (from training data)
    # if this becomes performance heavy (not now), this can be done in a separte
    # script
    phase = "train"
    # TODO: ugly?
    if hp == hparams.vc:
        # Collect mean/var from source and target features
        data_mean, data_var, last_sample_count = P.meanvar(
            X[phase], utt_lengths[phase], return_last_sample_count=True)
        data_mean, data_var = P.meanvar(
            Y[phase], utt_lengths[phase], mean_=data_mean, var_=data_var,
            last_sample_count=last_sample_count)
        data_std = np.sqrt(data_var)

        np.save(join(data_dir, "data_mean"), data_mean)
        np.save(join(data_dir, "data_var"), data_var)

        if hp.generator_params["in_dim"] is None:
            hp.generator_params["in_dim"] = data_mean.shape[-1]
        if hp.generator_params["out_dim"] is None:
            hp.generator_params["out_dim"] = data_mean.shape[-1]

        # Dataset loaders
        dataset_loaders = get_vc_data_loaders(X, Y, data_mean, data_std)
    else:
        ty = "acoustic" if hp == hparams.tts_acoustic else "duration"
        X_data_min, X_data_max = P.minmax(X[phase])
        Y_data_mean, Y_data_var = P.meanvar(Y[phase])
        Y_data_std = np.sqrt(Y_data_var)

        np.save(join(data_dir, "X_{}_data_min".format(ty)), X_data_min)
        np.save(join(data_dir, "X_{}_data_max".format(ty)), X_data_max)
        np.save(join(data_dir, "Y_{}_data_mean".format(ty)), Y_data_mean)
        np.save(join(data_dir, "Y_{}_data_var".format(ty)), Y_data_var)

        if hp.generator_params["in_dim"] is None:
            D = X_data_min.shape[-1]
            if hp.generator_add_noise:
                D = D + hp.generator_noise_dim
            hp.generator_params["in_dim"] = D
        if hp.generator_params["out_dim"] is None:
            hp.generator_params["out_dim"] = Y_data_mean.shape[-1]
        if hp.discriminator_params["in_dim"] is None:
            sizes = get_static_stream_sizes(
                hp.stream_sizes, hp.has_dynamic_features, len(hp.windows))
            D = int(np.array(sizes[hp.adversarial_streams]).sum())
            if hp.adversarial_streams[0]:
                D -= hp.mask_nth_mgc_for_adv_loss
            if hp.discriminator_linguistic_condition:
                D = D + X_data_min.shape[-1]
            hp.discriminator_params["in_dim"] = D
        dataset_loaders = get_tts_data_loaders(
            X, Y, X_data_min, X_data_max, Y_data_mean, Y_data_std)

    # Models
    model_g = getattr(gantts.models, hp.generator)(**hp.generator_params)
    model_d = getattr(gantts.models, hp.discriminator)(**hp.discriminator_params)
    print("Generator:", model_g)
    print("Discriminator:", model_d)

    # Reference discriminator model to compute spoofing rate
    if checkpoint_path_r is not None:
        reference_discriminator = getattr(
            gantts.models, hp.discriminator)(**hp.discriminator_params)
        try:
            load_checkpoint(reference_discriminator, None, checkpoint_path_r)
        except:
            warn("Invalid cehckpoint for reference discriminator")
            reference_discriminator = None
    else:
        reference_discriminator = None

    if use_cuda:
        model_g, model_d = model_g.cuda(), model_d.cuda()
        if reference_discriminator is not None:
            reference_discriminator = reference_discriminator.cuda()

    # Optimizers
    optimizer_g = getattr(optim, hp.optimizer_g)(model_g.parameters(),
                                                 **hp.optimizer_g_params)
    optimizer_d = getattr(optim, hp.optimizer_d)(model_d.parameters(),
                                                 **hp.optimizer_d_params)

    # Load checkpoint
    if checkpoint_path_d:
        if reset_optimizers:
            load_checkpoint(model_d, None, checkpoint_path_d)
        else:
            load_checkpoint(model_d, optimizer_d, checkpoint_path_d)
    if checkpoint_path_g:
        if reset_optimizers:
            load_checkpoint(model_g, None, checkpoint_path_g)
        else:
            load_checkpoint(model_g, optimizer_g, checkpoint_path_g)

    # Restart iteration at restart_epoch
    if restart_epoch >= 0:
        global_epoch = restart_epoch

    # Setup tensorboard logger
    if log_event_path is None:
        log_event_path = "log/run-test" + str(np.random.randint(100000))
    print("Los event path: {}".format(log_event_path))
    tensorboard_logger.configure(log_event_path)

    # Train
    print("Start training from epoch {}".format(global_epoch))
    train_loop((model_g, model_d), (optimizer_g, optimizer_d),
               dataset_loaders, w_d=w_d, update_d=update_d, update_g=update_g,
               reference_discriminator=reference_discriminator,
               mse_w=mse_w, mge_w=mge_w)

    # Save models
    for model, optimizer, enabled, name in [
            (model_g, optimizer_g, update_g, "Generator"),
            (model_d, optimizer_d, update_d, "Discriminator")]:
        if enabled:
            save_checkpoint(
                model, optimizer, global_epoch, checkpoint_dir, name)

    if not disable_slack and "SLACK_API_TOKEN" in os.environ:
        from slackclient import SlackClient

        print("Posting to slack...")
        slack_token = os.environ["SLACK_API_TOKEN"]
        sc = SlackClient(slack_token)

        try:
            sc.api_call(
                "chat.postMessage",
                channel="#research",
                text="""
Finally train.py finished! :tada:. Elapsed time: {}mins.
Command line args:
{}

{}
""".format((time.time() - since) // 60, args, hparams_debug_string(hp)))
        except Exception as e:
            print(str(e))

    print("Finished!")
    sys.exit(0)
