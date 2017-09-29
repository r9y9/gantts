"""Prepare acoustic features to be used for DNN training.

usage:
    prepare_features_vc.py [options] <DATA_ROOT>

options:
    --max_files=<N>      Max num files to be collected. [default: -1]
    --dst_dir=<d>        Destination directory [default: data/cmu_arcic_tts].
    --overwrite          Overwrite files
    -h, --help           show this help message and exit
"""
from __future__ import division, print_function, absolute_import

from docopt import docopt
import numpy as np

from nnmnkwii.datasets import FileSourceDataset, FileDataSource
from nnmnkwii import preprocessing as P
from nnmnkwii.io import hts
from nnmnkwii.frontend import merlin as fe

import pysptk
import pyworld
from scipy.io import wavfile
from tqdm import tqdm
from os.path import basename, splitext, exists, expanduser, join
import os
import sys
from glob import glob

from hparams import hparams_tts as hp
from hparams import hparams_debug_string


class LinguisticSource(FileDataSource):
    def __init__(self, data_root, max_files=None, add_frame_features=False,
                 subphone_features=None):
        self.data_root = data_root
        self.max_files = max_files
        self.add_frame_features = add_frame_features
        self.subphone_features = subphone_features
        self.test_paths = None
        self.binary_dict, self.continuous_dict = hts.load_question_set(
            hp.question_path)

    def collect_files(self):
        label_dir_name = "label_phone_align" if hp.use_phone_alignment \
            else "label_state_align"
        files = sorted(glob(join(self.data_root, label_dir_name, "*.lab")))
        if self.max_files is not None and self.max_files > 0:
            return files[:self.max_files]
        else:
            return files

    def collect_features(self, path):
        labels = hts.load(path)
        features = fe.linguistic_features(
            labels, self.binary_dict, self.continuous_dict,
            add_frame_features=self.add_frame_features,
            subphone_features=self.subphone_features)
        if self.add_frame_features:
            indices = labels.silence_frame_indices().astype(np.int)
        else:
            indices = labels.silence_phone_indices()
        features = np.delete(features, indices, axis=0)

        return features.astype(np.float32)


class DurationSource(FileDataSource):
    def __init__(self, data_root, max_files=None):
        self.data_root = data_root
        self.max_files = max_files

    def collect_files(self):
        label_dir_name = "label_phone_align" if hp.use_phone_alignment \
            else "label_state_align"
        files = sorted(glob(join(self.data_root, label_dir_name, "*.lab")))
        if self.max_files is not None and self.max_files > 0:
            return files[:self.max_files]
        else:
            return files

    def collect_features(self, path):
        labels = hts.load(path)
        features = fe.duration_features(labels)
        indices = labels.silence_phone_indices()
        features = np.delete(features, indices, axis=0)
        return features.astype(np.float32)


class AcousticSource(FileDataSource):
    def __init__(self, data_root, max_files=None):
        self.data_root = data_root
        self.max_files = max_files
        self.alpha = None

    def collect_files(self):
        wav_paths = sorted(glob(join(self.data_root, "wav", "*.wav")))
        label_dir_name = "label_phone_align" if hp.use_phone_alignment \
            else "label_state_align"
        label_paths = sorted(glob(join(self.data_root, label_dir_name, "*.lab")))
        if self.max_files is not None and self.max_files > 0:
            return wav_paths[:self.max_files], label_paths[:self.max_files]
        else:
            return wav_paths, label_paths

    def collect_features(self, wav_path, label_path):
        fs, x = wavfile.read(wav_path)
        x = x.astype(np.float64)
        f0, timeaxis = pyworld.dio(x, fs, frame_period=hp.frame_period)
        f0 = pyworld.stonemask(x, f0, timeaxis, fs)
        spectrogram = pyworld.cheaptrick(x, f0, timeaxis, fs)
        aperiodicity = pyworld.d4c(x, f0, timeaxis, fs)

        bap = pyworld.code_aperiodicity(aperiodicity, fs)
        if self.alpha is None:
            self.alpha = pysptk.util.mcepalpha(fs)
        mgc = pysptk.sp2mc(spectrogram, order=hp.order, alpha=self.alpha)
        f0 = f0[:, None]
        lf0 = f0.copy()
        nonzero_indices = np.nonzero(f0)
        lf0[nonzero_indices] = np.log(f0[nonzero_indices])
        vuv = (lf0 != 0).astype(np.float32)
        lf0 = P.interp1d(lf0, kind="slinear")

        mgc = P.delta_features(mgc, hp.windows)
        lf0 = P.delta_features(lf0, hp.windows)
        bap = P.delta_features(bap, hp.windows)

        features = np.hstack((mgc, lf0, vuv, bap))

        # Cut silence frames by HTS alignment
        labels = hts.load(label_path)
        features = features[:labels.num_frames()]
        indices = labels.silence_frame_indices()
        features = np.delete(features, indices, axis=0)

        return features.astype(np.float32)


if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    DATA_ROOT = args["<DATA_ROOT>"]
    max_files = int(args["--max_files"])
    dst_dir = args["--dst_dir"]
    overwrite = args["--overwrite"]

    print(hparams_debug_string(hp))

    # Features required to train duration model
    # X -> Y
    # X: linguistic
    # Y: duration
    X_duration_source = LinguisticSource(DATA_ROOT, max_files,
                                         add_frame_features=False, subphone_features=None)
    Y_duration_source = DurationSource(DATA_ROOT, max_files)

    X_duration = FileSourceDataset(X_duration_source)
    Y_duration = FileSourceDataset(Y_duration_source)

    # Features required to train acoustic model
    # X -> Y
    # X: linguistic
    # Y: acoustic
    subphone_features = "full" if not hp.use_phone_alignment else "coarse_coding"
    X_acoustic_source = LinguisticSource(DATA_ROOT, max_files,
                                         add_frame_features=True, subphone_features=subphone_features)
    Y_acoustic_source = AcousticSource(DATA_ROOT, max_files)
    X_acoustic = FileSourceDataset(X_acoustic_source)
    Y_acoustic = FileSourceDataset(Y_acoustic_source)

    # Save as files
    X_duration_root = join(dst_dir, "X_duration")
    Y_duration_root = join(dst_dir, "Y_duration")
    X_acoustic_root = join(dst_dir, "X_acoustic")
    Y_acoustic_root = join(dst_dir, "Y_acoustic")

    skip_duration_feature_extraction = exists(
        X_duration_root) and exists(Y_duration_root)
    skip_acoustic_feature_extraction = exists(
        X_acoustic_root) and exists(Y_acoustic_root)

    if overwrite:
        skip_acoustic_feature_extraction = False
        skip_duration_feature_extraction = False

    for d in [X_duration_root, Y_duration_root, X_acoustic_root, Y_acoustic_root]:
        if not os.path.exists(d):
            print("mkdirs: {}".format(d))
            os.makedirs(d)

    # Save features for duration model
    if not skip_duration_feature_extraction:
        print("Duration linguistic feature dim", X_duration[0].shape[-1])
        print("Duration feature dim", Y_duration[0].shape[-1])
        for idx in tqdm(range(len(X_duration))):
            x, y = X_duration[idx], Y_duration[idx]
            name = splitext(basename(X_duration.collected_files[idx][0]))[0]
            xpath = join(X_duration_root, name)
            ypath = join(Y_duration_root, name)
            np.save(xpath, x)
            np.save(ypath, y)
    else:
        print("Features for duration model training found, skipping feature extraction.")

    # Save features for acoustic model
    if not skip_acoustic_feature_extraction:
        print("Acoustic linguistic feature dim", X_acoustic[0].shape[-1])
        print("Acoustic feature dim", Y_acoustic[0].shape[-1])
        for idx in tqdm(range(len(X_acoustic))):
            x, y = X_acoustic[idx], Y_acoustic[idx]
            name = splitext(basename(X_acoustic.collected_files[idx][0]))[0]
            xpath = join(X_acoustic_root, name)
            ypath = join(Y_acoustic_root, name)
            np.save(xpath, x)
            np.save(ypath, y)
    else:
        print("Features for acousic model training found, skipping feature extraction.")

    print("Finished!")
    sys.exit(0)
