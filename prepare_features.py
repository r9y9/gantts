"""Prepare acoustic features to be used for DNN trainingself.

usage:
    prepare_features.py [options]

options:
    --max_files=<N>      Max num files to be collected. [default: 100]
    -h, --help           show this help message and exit
"""
from __future__ import division, print_function, absolute_import

from docopt import docopt
import numpy as np

from nnmnkwii.datasets import FileSourceDataset
from nnmnkwii import preprocessing as P
from nnmnkwii.preprocessing.alignment import DTWAligner
from nnmnkwii.datasets import cmu_arctic

import pysptk
import pyworld
from scipy.io import wavfile
from tqdm import tqdm
from os.path import basename, splitext, exists, expanduser, join
import os
import sys

import hparams


class MGCSource(cmu_arctic.WavFileDataSource):
    def __init__(self, data_root, speakers, max_files=None):
        super(MGCSource, self).__init__(data_root, speakers, max_files=max_files)
        self.alpha = None

    def collect_features(self, wav_path):
        fs, x = wavfile.read(wav_path)
        x = x.astype(np.float64)
        f0, timeaxis = pyworld.dio(x, fs, frame_period=hparams.frame_period)
        f0 = pyworld.stonemask(x, f0, timeaxis, fs)
        spectrogram = pyworld.cheaptrick(x, f0, timeaxis, fs)
        spectrogram = P.trim_zeros_frames(spectrogram)
        if self.alpha is None:
            self.alpha = pysptk.util.mcepalpha(fs)
        mgc = pysptk.sp2mc(spectrogram, order=hparams.order, alpha=self.alpha)
        # Drop 0-th coefficient
        mgc = mgc[:, 1:]
        # Smoothing
        hop_length = int(fs * (hparams.frame_period * 0.001))
        modfs = fs / hop_length
        mgc = P.modspec_smoothing(mgc, modfs, cutoff=50)
        # Add delta
        mgc = P.delta_features(mgc, hparams.windows)
        return mgc.astype(np.float32)


if __name__ == "__main__":
    args = docopt(__doc__)
    max_files = int(args["--max_files"])

    DATA_ROOT = join(expanduser("~"), "data", "cmu_arctic")

    X_dataset = FileSourceDataset(MGCSource(DATA_ROOT, ["clb"],
                                            max_files=max_files))
    Y_dataset = FileSourceDataset(MGCSource(DATA_ROOT, ["slt"],
                                            max_files=max_files))

    # Convert to arrays
    print("Convert datasets to arrays")
    X, Y = X_dataset.asarray(verbose=1), Y_dataset.asarray(verbose=1)

    # Alignment
    print("Perform alignment")
    X, Y = DTWAligner().transform((X, Y))

    # Save
    for speaker in ["clb", "slt"]:
        os.makedirs(join("data", speaker), exist_ok=True)
    print("Save features to disk")
    for idx, (x, y) in tqdm(enumerate(zip(X, Y))):
        # paths
        src_name = splitext(basename(X_dataset.collected_files[idx][0]))[0]
        tgt_name = splitext(basename(Y_dataset.collected_files[idx][0]))[0]
        src_path = join("data", "clb", src_name)
        tgt_path = join("data", "slt", tgt_name)

        # Trim and ajast frames
        x = P.trim_zeros_frames(x)
        y = P.trim_zeros_frames(y)
        x, y = P.adjast_frame_lengths(x, y, pad=True, divisible_by=2)

        # Save
        np.save(src_path, x)
        np.save(tgt_path, y)

    sys.exit(0)
