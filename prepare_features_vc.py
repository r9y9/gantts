"""Prepare acoustic features for one-to-one voice conversion.

usage:
    prepare_features_vc.py [options] <DATA_ROOT> <source_speaker> <target_speaker>

options:
    --max_files=<N>      Max num files to be collected. [default: 100]
    --dst_dir=<d>        Destination directory [default: data/cmu_arctic_vc].
    --overwrite          Overwrite files.
    -h, --help           show this help message and exit
"""
from __future__ import division, print_function, absolute_import

from docopt import docopt
import numpy as np

from nnmnkwii.datasets import FileSourceDataset
from nnmnkwii import preprocessing as P
from nnmnkwii.preprocessing.alignment import DTWAligner
from nnmnkwii.datasets import cmu_arctic, voice_statistics, vcc2016

import pysptk
import pyworld
from scipy.io import wavfile
from tqdm import tqdm
from os.path import basename, splitext, exists, expanduser, join, dirname
import os
import sys

from hparams import vc as hp
from hparams import hparams_debug_string


# vcc2016.WavFileDataSource and voice_statistics.WavFileDataSource can be
# drop-in replacement. See below for details:
# https://r9y9.github.io/nnmnkwii/latest/references/datasets.html#builtin-data-sources
class MGCSource(cmu_arctic.WavFileDataSource):
    def __init__(self, data_root, speakers, max_files=None):
        super(MGCSource, self).__init__(data_root, speakers,
                                        max_files=max_files)
        self.alpha = None

    def collect_features(self, wav_path):
        fs, x = wavfile.read(wav_path)
        x = x.astype(np.float64)
        f0, timeaxis = pyworld.dio(x, fs, frame_period=hp.frame_period)
        f0 = pyworld.stonemask(x, f0, timeaxis, fs)
        spectrogram = pyworld.cheaptrick(x, f0, timeaxis, fs)
        spectrogram = P.trim_zeros_frames(spectrogram)
        if self.alpha is None:
            self.alpha = pysptk.util.mcepalpha(fs)
        mgc = pysptk.sp2mc(spectrogram, order=hp.order, alpha=self.alpha)
        # Drop 0-th coefficient
        mgc = mgc[:, 1:]
        # 50Hz cut-off MS smoothing
        hop_length = int(fs * (hp.frame_period * 0.001))
        modfs = fs / hop_length
        mgc = P.modspec_smoothing(mgc, modfs, cutoff=50)
        # Add delta
        mgc = P.delta_features(mgc, hp.windows)
        return mgc.astype(np.float32)


if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    DATA_ROOT = args["<DATA_ROOT>"]
    source_speaker = args["<source_speaker>"]
    target_speaker = args["<target_speaker>"]
    max_files = int(args["--max_files"])
    dst_dir = args["--dst_dir"]
    overwrite = args["--overwrite"]

    print(hparams_debug_string(hp))

    X_dataset = FileSourceDataset(MGCSource(DATA_ROOT, [source_speaker],
                                            max_files=max_files))
    Y_dataset = FileSourceDataset(MGCSource(DATA_ROOT, [target_speaker],
                                            max_files=max_files))

    skip_feature_extraction = exists(join(dst_dir, "X")) \
        and exists(join(dst_dir, "Y"))
    if overwrite:
        skip_feature_extraction = False
    if skip_feature_extraction:
        print("Features seems to be prepared, skipping feature extraction.")
        sys.exit(0)

    # Create dirs
    for speaker, name in [(source_speaker, "X"), (target_speaker, "Y")]:
        d = join(dst_dir, name)
        print("Destination dir for {}: {}".format(speaker, d))
        if not exists(d):
            os.makedirs(d)

    # Convert to arrays
    print("Convert datasets to arrays")
    X, Y = X_dataset.asarray(verbose=1), Y_dataset.asarray(verbose=1)

    # Alignment
    print("Perform alignment")
    X, Y = DTWAligner().transform((X, Y))

    print("Save features to disk")
    for idx, (x, y) in tqdm(enumerate(zip(X, Y))):
        # paths
        src_name = splitext(basename(X_dataset.collected_files[idx][0]))[0]
        tgt_name = splitext(basename(Y_dataset.collected_files[idx][0]))[0]
        src_path = join(dst_dir, "X", src_name)
        tgt_path = join(dst_dir, "Y", tgt_name)

        # Trim and ajast frames
        x = P.trim_zeros_frames(x)
        y = P.trim_zeros_frames(y)
        x, y = P.adjast_frame_lengths(x, y, pad=True, divisible_by=2)

        # Save
        np.save(src_path, x)
        np.save(tgt_path, y)

    sys.exit(0)
