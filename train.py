# coding: utf-8
"""Trainining script

usage: train.py [options] <inputs_dir> <outputs_dir>

options:
    --type=<ty>                 vc or tts [default: vc].
    --checkpoint-dir=<dir>      Directory where to save model checkpoints [default: checkpoints].
    --hparams=<parmas>          Hyper parameters [default: ].
    --checkpoint-path-g=<name>  Restore generator from checkpoint if given.
    --checkpoint-path-d=<name>  Restore discriminator from checkpoint if given.
    --max_files=<N>             Max num files to be collected. [default: -1]
    --discriminator-warmup      Warmup discriminator.
    --w_d=<f>                   Weight for loss weighting [default: 1.0].
    --reference-discriminator=<name>    Reference discriminator.
    --restart_epoch=<N>         Restart epoch [default: -1].
    --reset_optimizers          Reset optimizers.
    -h, --help                  Show this help message and exit
"""
from docopt import docopt

import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
from torch import optim
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
from sklearn.model_selection import train_test_split

import sys
import os
from os.path import splitext, join
from tqdm import tqdm

import tensorboard_logger
from tensorboard_logger import log_value

from nnmnkwii import preprocessing as P
from nnmnkwii.paramgen import unit_variance_mlpg_matrix
from nnmnkwii.datasets import FileSourceDataset, FileDataSource
from nnmnkwii.datasets import MemoryCacheDataset

from in2out_highway import In2OutHighwayNet, Discriminator
from in2out_highway import MaskedMSE, _sequence_mask

from hparams import hparams_debug_string, hparams_vc, hparams_tts
hp = None  # to be initailized later

global_epoch = 0
test_size = 0.05
random_state = 1234

use_cuda = torch.cuda.is_available()


class NPYDataSource(FileDataSource):
    def __init__(self, dirname, train=True, max_files=None):
        self.dirname = dirname
        self.train = train
        self.max_files = max_files

    def collect_files(self):
        npy_files = list(filter(lambda x: splitext(x)[-1] == ".npy",
                                os.listdir(self.dirname)))
        npy_files = list(map(lambda d: join(self.dirname, d), npy_files))
        if self.max_files is not None:
            npy_files = npy_files[:self.max_files]
        train_files, test_files = train_test_split(
            npy_files, test_size=test_size, random_state=random_state)
        return train_files if self.train else test_files

    def collect_features(self, path):
        return np.load(path)


class PyTorchDataset(object):
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


def save_checkpoint(model, optimizer, epoch, checkpoint_dir):
    checkpoint_path = join(
        checkpoint_dir, "checkpoint_epoch{}_{}.pth".format(
            epoch, type(model).__name__))
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)


def get_data_loaders(X, Y, data_mean, data_var):
    X_train, X_test = X["train"], X["test"]
    Y_train, Y_test = Y["train"], Y["test"]

    # Sequence-wise train loader
    X_train_cache_dataset = MemoryCacheDataset(X_train)
    Y_train_cache_dataset = MemoryCacheDataset(Y_train)
    train_dataset = PyTorchDataset(
        X_train_cache_dataset, Y_train_cache_dataset, data_mean, data_std)
    train_loader = data_utils.DataLoader(
        train_dataset, batch_size=hp.batch_size,
        num_workers=hp.num_workers,
        shuffle=True, collate_fn=collate_fn)

    # Sequence-wise test loader
    X_test_cache_dataset = MemoryCacheDataset(X_test)
    Y_test_cache_dataset = MemoryCacheDataset(Y_test)
    test_dataset = PyTorchDataset(
        X_test_cache_dataset, Y_test_cache_dataset, data_mean, data_std)
    test_loader = data_utils.DataLoader(
        test_dataset, batch_size=hp.batch_size,
        num_workers=hp.num_workers,
        shuffle=False, collate_fn=collate_fn)

    dataset_loaders = {"train": train_loader, "test": test_loader}
    return dataset_loaders


def update_discriminator(model_d, optimizer_d, y_static, y_hat_static, mask,
                         phase, eps=1e-20):
    T = mask.sum().data[0]

    # Real
    D_real = model_d(y_static)
    real_correct_count = ((D_real > 0.5).float() * mask).sum().data[0]

    # Fake
    D_fake = model_d(y_hat_static)
    fake_correct_count = ((D_fake < 0.5).float() * mask).sum().data[0]

    # Loss
    loss_real_d = -(torch.log(D_real + eps) * mask).sum() / T
    loss_fake_d = -(torch.log(1 - D_fake + eps) * mask).sum() / T
    loss_d = loss_real_d + loss_fake_d

    if phase == "train":
        loss_d.backward(retain_graph=True)
        optimizer_d.step()

    return loss_d.data[0], loss_fake_d.data[0], loss_real_d.data[0],\
        real_correct_count, fake_correct_count


def update_generator(model_g, model_d, optimizer_g, y_static, y_hat_static,
                     weight, lengths, mask, phase, eps=1e-20):
    T = mask.sum().data[0]

    # MGE loss
    loss_mge = MaskedMSE()(y_hat_static, y_static, lengths)

    # Adversarial loss
    if weight > 0:
        loss_adv = -(torch.log(model_d(y_hat_static) + eps) * mask).sum() / T
    else:
        loss_adv = Variable(y_static.data.new(1).zero_())

    # MGE + ADV loss
    # try to decieve discriminator
    loss_g = loss_mge + weight * loss_adv
    if phase == "train":
        loss_g.backward()
        optimizer_g.step()

    return loss_mge.data[0], loss_adv.data[0], loss_g.data[0]


def train_loop(models, optimizers, dataset_loaders, w_d=0.0,
               update_d=True, update_g=True,
               reference_discriminator=None):
    model_g, model_d = models
    optimizer_g, optimizer_d = optimizers
    if use_cuda:
        model_g, model_d = model_g.cuda(), model_d.cuda()
        if reference_discriminator is not None:
            reference_discriminator = reference_discriminator.cuda()
            reference_discriminator.eval()
    model_g.train()
    model_d.train()

    static_dim = model_g.static_dim

    E_loss_mge = 1
    E_loss_adv = 1
    global global_epoch
    for global_epoch in tqdm(range(global_epoch + 1, hp.nepoch + 1)):
        for phase in ["train", "test"]:
            running_loss = {"generator": 0.0, "mge": 0.0,
                            "loss_real_d": 0.0,
                            "loss_fake_d": 0.0,
                            "loss_adv": 0.0,
                            "discriminator": 0.0}
            real_correct_count, fake_correct_count = 0, 0
            regard_fake_as_natural = 0
            N = len(dataset_loaders[phase])
            total_num_frames = 0
            for x, y, lengths in dataset_loaders[phase]:
                # Sort by lengths. This is needed for pytorch's PackedSequence
                sorted_lengths, indices = torch.sort(
                    lengths.view(-1), dim=0, descending=True)
                sorted_lengths = sorted_lengths.long()
                max_len = sorted_lengths[0]

                # Get sorted batch
                x, y = x[indices], y[indices]
                y_static = y[:, :, :static_dim]

                # MLPG paramgen matrix
                R = unit_variance_mlpg_matrix(hp.windows, max_len)
                R = torch.from_numpy(R)

                if use_cuda:
                    x, y_static, R = x.cuda(), y_static.cuda(), R.cuda()
                    sorted_lengths = sorted_lengths.cuda()

                # Pack into variables
                x, y_static = Variable(x), Variable(y_static)
                sorted_lengths = Variable(sorted_lengths)

                # Num frames in batch
                total_num_frames += sorted_lengths.float().sum().data[0]

                # Mask
                mask = _sequence_mask(sorted_lengths).unsqueeze(-1)

                # Reset optimizers state
                optimizer_g.zero_grad()
                optimizer_d.zero_grad()

                # Apply model (generator)
                y_hat_static = model_g(x, R)

                # Compute spoofing rate
                if reference_discriminator is not None:
                    target = reference_discriminator(y_hat_static)
                    # Count samples classified as natural, while inputs are
                    # actually generated.
                    regard_fake_as_natural += ((target > 0.5).float() * mask).sum().data[0]

                ### Update discriminator ###
                # Natural: 1, Genrated: 0
                if update_d:
                    loss_d, loss_fake_d, loss_real_d, _real_correct_count,\
                        _fake_correct_count = update_discriminator(
                            model_d, optimizer_d, y_static, y_hat_static,
                            mask, phase)
                    running_loss["discriminator"] += loss_d
                    running_loss["loss_fake_d"] += loss_fake_d
                    running_loss["loss_real_d"] += loss_real_d
                    real_correct_count += _real_correct_count
                    fake_correct_count += _fake_correct_count

                ### Update generator ###
                if update_g:
                    weight = float(w_d * E_loss_mge / E_loss_adv)
                    loss_mge, loss_adv, loss_g, = update_generator(
                        model_g, model_d, optimizer_g, y_static, y_hat_static,
                        weight, sorted_lengths, mask, phase)

                    running_loss["mge"] += loss_mge
                    running_loss["loss_adv"] += loss_adv
                    running_loss["generator"] += loss_g

            # Update expectation
            if update_d and update_g and phase == "train":
                E_loss_mge = running_loss["mge"] / N
                E_loss_adv = running_loss["loss_adv"] / N
                log_value("E(mge)", E_loss_mge, global_epoch)
                log_value("E(adv)", E_loss_adv, global_epoch)
                log_value("MGE/ADV loss weight",  E_loss_mge / E_loss_adv, global_epoch)

            # Log loss
            for ty, enabled in [("mge", update_g),
                                ("discriminator", update_d),
                                ("loss_real_d", update_d),
                                ("loss_fake_d", update_d),
                                ("loss_adv", update_g and w_d > 0),
                                ("generator", update_g)]:
                if enabled:
                    ave_loss = running_loss[ty] / N
                    log_value(
                        "{} {} loss".format(phase, ty), ave_loss, global_epoch)

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
    args = docopt(__doc__)
    print("Command line args:\n", args)
    ty = args["--type"]
    hp = hparams_vc if ty == "vc" else hparams_tts

    # Override hyper parameters
    hp.parse(args["--hparams"])
    print(hparams_debug_string(hp))

    inputs_dir = args["<inputs_dir>"]
    outputs_dir = args["<outputs_dir>"]
    checkpoint_dir = args["--checkpoint-dir"]
    checkpoint_path_d = args["--checkpoint-path-d"]
    checkpoint_path_g = args["--checkpoint-path-g"]
    max_files = int(args["--max_files"])
    w_d = float(args["--w_d"])
    discriminator_warmup = args["--discriminator-warmup"]
    restart_epoch = int(args["--restart_epoch"])

    reference_discriminator_path = args["--reference-discriminator"]
    reset_optimizers = args["--reset_optimizers"]

    update_d = w_d > 0
    update_g = False if discriminator_warmup else True

    os.makedirs(checkpoint_dir, exist_ok=True)

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

    print("Eval files:\n", X["test"].collected_files)

    # Collect stats for noramlization (from training data)
    ty = "train"
    data_mean, data_var, last_sample_count = P.meanvar(
        X[ty], utt_lengths[ty], return_last_sample_count=True)
    data_mean, data_var = P.meanvar(
        Y[ty], utt_lengths[ty], mean_=data_mean, var_=data_var,
        last_sample_count=last_sample_count)
    data_std = np.sqrt(data_var)

    np.save("data_mean", data_mean)
    np.save("data_var", data_var)

    # Dataset loaders
    dataset_loaders = get_data_loaders(X, Y, data_mean, data_std)

    # Models
    static_dim = hp.order
    in_dim = static_dim * len(hp.windows)
    out_dim = in_dim
    model_g = In2OutHighwayNet(
        in_dim=in_dim, out_dim=out_dim, static_dim=static_dim)
    model_d = Discriminator(in_dim=static_dim)
    print(model_g)
    print(model_d)

    # Reference discriminator model to compute spoofing rate
    if reference_discriminator_path is not None:
        reference_discriminator = Discriminator(in_dim=static_dim)
        load_checkpoint(reference_discriminator, None, reference_discriminator_path)
    else:
        reference_discriminator = None

    if use_cuda:
        model_g, model_d = model_g.cuda(), model_d.cuda()
        if reference_discriminator is not None:
            reference_discriminator = reference_discriminator.cuda()

    # Optimizers
    optimizer_g = optim.Adagrad(model_g.parameters(),
                                lr=hp.lr,
                                weight_decay=hp.weight_decay)
    optimizer_d = optim.Adagrad(model_d.parameters(),
                                lr=hp.lr,
                                weight_decay=hp.weight_decay)

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

    print("Start training from epoch {}".format(global_epoch))

    # Setup tensorboard logger
    ext = "discriminator_warmup" if discriminator_warmup else ""
    log_path = "log/run-test" + ext + str(np.random.randint(100000))
    print("Los path: {}".format(log_path))
    tensorboard_logger.configure(log_path)

    # Train
    train_loop((model_g, model_d), (optimizer_g, optimizer_d),
               dataset_loaders, w_d=w_d, update_d=update_d, update_g=update_g,
               reference_discriminator=reference_discriminator)

    # Save models
    for model, optimizer, enabled in [(model_g, optimizer_g, update_g),
                                      (model_d, optimizer_d, update_d)]:
        save_checkpoint(
            model, optimizer, global_epoch, checkpoint_dir)

    print("Finished!")
    sys.exit(0)
