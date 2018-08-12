# coding: utf-8
"""Evaluation script for GAN-based VC models.

usage: evaluation_vc.py [options] <checkpoint> <data_dir> <wav_dir> <outputs_dir>

options:
    -h, --help                  Show this help message and exit
    --diffvc                    Enable DIFF VC.
"""
from docopt import docopt
import numpy as np

import torch
from torch import nn
from torch.autograd import Variable

from scipy.io import wavfile
import pysptk
from pysptk.synthesis import Synthesizer, MLSADF
import pyworld

import sys
import os
from os.path import splitext, join, abspath, basename, exists

from nnmnkwii import preprocessing as P
from nnmnkwii.paramgen import unit_variance_mlpg_matrix
from nnmnkwii.datasets import FileSourceDataset, FileDataSource

import gantts
from gantts.multistream import multi_stream_mlpg, get_static_features
from gantts.multistream import get_static_stream_sizes, select_streams
from gantts.seqloss import MaskedMSELoss, sequence_mask

from hparams import vc as hp

from train import NPYDataSource

fs = 16000
hop_length = int(fs * (hp.frame_period * 0.001))


def test_vc_from_path(model, path, data_mean, data_std, diffvc=True):
    model.eval()

    fs, x = wavfile.read(path)
    x = x.astype(np.float64)
    f0, timeaxis = pyworld.dio(x, fs, frame_period=hp.frame_period)
    f0 = pyworld.stonemask(x, f0, timeaxis, fs)
    spectrogram = pyworld.cheaptrick(x, f0, timeaxis, fs)
    aperiodicity = pyworld.d4c(x, f0, timeaxis, fs)
    alpha = pysptk.util.mcepalpha(fs)
    mc = pysptk.sp2mc(spectrogram, order=hp.order, alpha=alpha)
    c0, mc = mc[:, 0], mc[:, 1:]
    static_dim = mc.shape[-1]
    mc = P.modspec_smoothing(mc, fs / hop_length, cutoff=50)
    mc = P.delta_features(mc, hp.windows).astype(np.float32)

    T = mc.shape[0]

    inputs = mc[:, :static_dim].copy()

    # Normalization
    mc_scaled = P.scale(mc, data_mean, data_std)

    mc_scaled = Variable(torch.from_numpy(mc_scaled))
    lengths = [len(mc_scaled)]

    # Add batch axis
    mc_scaled = mc_scaled.view(1, -1, mc_scaled.size(-1))

    # For MLPG
    R = unit_variance_mlpg_matrix(hp.windows, T)
    R = torch.from_numpy(R)

    # Apply model
    if model.include_parameter_generation():
        # Case: models include parameter generation in itself
        # Mulistream features cannot be used in this case
        y_hat, y_hat_static = model(mc_scaled, R, lengths=lengths)
    else:
        # Case: generic models (can be sequence model)
        assert hp.has_dynamic_features is not None
        y_hat = model(mc_scaled, lengths=lengths)
        y_hat_static = multi_stream_mlpg(
            y_hat, R, hp.stream_sizes, hp.has_dynamic_features)

    mc_static_pred = y_hat_static.data.cpu().numpy().reshape(-1, static_dim)

    # Denormalize
    mc_static_pred = P.inv_scale(
        mc_static_pred, data_mean[:static_dim], data_std[:static_dim])

    outputs = mc_static_pred.copy()

    if diffvc:
        mc_static_pred = mc_static_pred - mc[:, :static_dim]

    mc = np.hstack((c0[:, None], mc_static_pred))
    if diffvc:
        mc[:, 0] = 0  # remove power coefficients
        engine = Synthesizer(MLSADF(order=hp.order, alpha=alpha),
                             hopsize=hop_length)
        b = pysptk.mc2b(mc.astype(np.float64), alpha=alpha)
        waveform = engine.synthesis(x, b)
    else:
        fftlen = pyworld.get_cheaptrick_fft_size(fs)
        spectrogram = pysptk.mc2sp(
            mc.astype(np.float64), alpha=alpha, fftlen=fftlen)
        waveform = pyworld.synthesize(
            f0, spectrogram, aperiodicity, fs, hp.frame_period)

    return waveform, inputs, outputs


def load_checkpoint(model, optimizer, checkpoint_path):
    print("Load checkpoint from: {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])


def get_wav_files(data_dir, wav_dir, test=False):
    if test:
        files = NPYDataSource(join(data_dir, "X"), test=True).collect_files()
    else:
        files = NPYDataSource(join(data_dir, "X"), train=False).collect_files()

    wav_files = list(map(
        lambda f: join(wav_dir, splitext(basename(f))[0] + ".wav"), files))
    return wav_files


if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    checkpoint_path = args["<checkpoint>"]
    data_dir = args["<data_dir>"]
    wav_dir = args["<wav_dir>"]
    outputs_dir = args["<outputs_dir>"]
    diffvc = args["--diffvc"]

    # Collect stats
    data_mean = np.load(join(data_dir, "data_mean.npy"))
    data_var = np.load(join(data_dir, "data_var.npy"))
    data_std = np.sqrt(data_var)

    if hp.generator_params["in_dim"] is None:
        hp.generator_params["in_dim"] = data_mean.shape[-1]
    if hp.generator_params["out_dim"] is None:
        hp.generator_params["out_dim"] = data_mean.shape[-1]

    # Model
    model = getattr(gantts.models, hp.generator)(**hp.generator_params)
    load_checkpoint(model, None, checkpoint_path)
    print(model)

    # Generate samples for
    # 1. Evaluation set
    # 2. Test set
    eval_dir = join(outputs_dir, "eval")
    test_dir = join(outputs_dir, "test")
    if not exists(eval_dir):
        os.makedirs(eval_dir)
    if not exists(test_dir):
        os.makedirs(test_dir)
    eval_files = get_wav_files(data_dir, wav_dir, test=False)
    test_files = get_wav_files(data_dir, wav_dir, test=True)
    for dst_dir, files in [(eval_dir, eval_files), (test_dir, test_files)]:
        for path in files:
            print(dst_dir, path)
            name = splitext(basename(path))[0]
            dst_path = join(dst_dir, name + ".wav")
            waveform, _, _ = test_vc_from_path(
                model, path, data_mean, data_std, diffvc=diffvc)
            wavfile.write(dst_path, fs, waveform.astype(np.int16))

    sys.exit(0)
