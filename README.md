# GAN TTS

[![Build Status](https://travis-ci.org/r9y9/gantts.svg?branch=master)](https://travis-ci.org/r9y9/gantts)

PyTorch implementation of Generative adversarial Networks (GAN) based text-to-speech (TTS) and voice conversion (VC).

1. [Saito, Yuki, Shinnosuke Takamichi, and Hiroshi Saruwatari. "Statistical Parametric Speech Synthesis Incorporating Generative Adversarial Networks." IEEE/ACM Transactions on Audio, Speech, and Language Processing (2017).](http://ieeexplore.ieee.org/abstract/document/8063435/)
2. [Shan Yang, Lei Xie, Xiao Chen, Xiaoyan Lou, Xuan Zhu, Dongyan Huang, Haizhou Li, "
Statistical Parametric Speech Synthesis Using Generative Adversarial Networks Under A Multi-task Learning Framework", 	arXiv:1707.01670, Jul 2017.](https://arxiv.org/abs/1707.01670)

## Generated audio samples

Audio samples are available in the Jupyter notebooks at the link below:

- [Voice conversion (en, MLP)](http://nbviewer.jupyter.org/github/r9y9/gantts/blob/master/notebooks/Test%20VC.ipynb)
- [Voice conversion (en, RNN)](http://nbviewer.jupyter.org/github/r9y9/gantts/blob/master/notebooks/Test%20RNN%20VC.ipynb)
- [Text-to-speech synthesis (en, MLP)](http://nbviewer.jupyter.org/github/r9y9/gantts/blob/master/notebooks/Test%20TTS.ipynb)
- [Text-to-speech synthesis (ja, MLP)](http://nbviewer.jupyter.org/gist/r9y9/185a56417cee27d9f785b8caf1c9f5ec)


## Notes on hyper parameters

- `adversarial_streams`, which represents streams (mgc, lf0, vuv, bap) to be used to compute adversarial loss, is a very speech quality sensitive parameter. Computing adversarial loss on mgc features (except for first few dimensions) seems to be working good.
- If `mask_nth_mgc_for_adv_loss` > 0, first `mask_nth_mgc_for_adv_loss` dimension for mgc will be ignored for computing adversarial loss. As described in [saito2017asja](http://sython.org/papers/ASJ/saito2017asja.pdf), I confirmed that using 0-th (and 1-th) mgc for computing adversarial loss affects speech quality. From my experience, `mask_nth_mgc_for_adv_loss` = 1 for mgc order 25, `mask_nth_mgc_for_adv_loss` = 2 for mgc order 59 are working to me.
- F0 extracted by WORLD will be spline interpolated. Set `f0_interpolation_kind` to "slinear" if you want frist-order spline interpolation, which is same as Merlin's default.
- Set `use_harvest` to True if you want to use Harvest F0 estimation algorithm. If False, Dio and StoneMask are used to estimate/refine F0.
- If you see `cuda runtime error (2) : out of memory`, try smaller batch size. https://github.com/r9y9/gantts/issues/3

### Notes on [2]

Though I haven't got improvements over Saito's approach [1] yet, but the GAN-based models described in [2] should be achieved by the following configurations:

- Set `generator_add_noise` to True. This will enable generator to use Gaussian noise as input. Linguistic features are concatenated with the noise vector.
- Set `discriminator_linguistic_condition` to True. The discriminator uses linguistic features as condition.

## Requirements

- [PyTorch](http://pytorch.org/) >= v0.2.0
- [TensorFlow](https://www.tensorflow.org/) (just for `tf.contrib.training.HParams`)
- [nnmnkwii](https://github.com/r9y9/nnmnkwii)
- https://github.com/taolei87/sru (if you want to try SRU-based models)
- Python

## Installation

Please install PyTorch, TensorFlow and SRU (if needed) first. Once you have those, then

```
git clone --recursive https://github.com/r9y9/gantts & cd gantts
pip install -e ".[train]"
```

should install all other dependencies.

## Repository structure

- **gantts/**: Network definitions, utilities for working on sequence-loss optimization.
- **prepare_features_vc.py**: Acoustic feature extraction script for voice conversion.
- **prepare_features_tts.py**: Linguistic/duration/acoustic feature extraction script for TTS.
- **train.py**: GAN-based training script. This is written to be generic so that can be used for training voice conversion models as well as text-to-speech models (duration/acoustic).
- **train_gan.sh**: Adversarial training wrapper script for `train.py`.
- **hparams.py**: Hyper parameters for VC and TTS experiments.
- **evaluation_vc.py**: Evaluation script for VC.
- **evaluation_tts.py**: Evaluation script for TTS.

Feature extraction scripts are written for CMU ARCTIC dataset, but can be easily adapted for other datasets.

## Run demos

### Voice conversion (en)

`vc_demo.sh` is a `clb` to `clt` voice conversion demo script. Before running the script, please download wav files for `clb` and `slt` from [CMU ARCTIC](http://festvox.org/cmu_arctic/) and check that you have all data in a directory as follows:

```
> tree ~/data/cmu_arctic/ -d -L 1
/home/ryuichi/data/cmu_arctic/
├── cmu_us_awb_arctic
├── cmu_us_bdl_arctic
├── cmu_us_clb_arctic
├── cmu_us_jmk_arctic
├── cmu_us_ksp_arctic
├── cmu_us_rms_arctic
└── cmu_us_slt_arctic
```

Once you have downloaded datasets, then:

```
./vc_demo.sh ${experimental_id} ${your_cmu_arctic_data_root}
```

e.g.,

```
 ./vc_demo.sh vc_gan_test ~/data/cmu_arctic/
```

Model checkpoints will be saved at `./checkpoints/${experimental_id}` and audio samples
are saved at `./generated/${experimental_id}`.

### Text-to-speech synthesis (en)

`tts_demo.sh` is a self-contained TTS demo script. The usage is:

```
./tts_demo.sh ${experimental_id}
```

This will download `slt_arctic_full_data` used in Merlin's demo, perform feature extraction, train models and synthesize audio samples for eval/test set. `${experimenta_id}` can be arbitrary string, for example,

```
./tts_demo.sh tts_test
```


Model checkpoints will be saved at `./checkpoints/${experimental_id}` and audio samples
are saved at `./generated/${experimental_id}`.

## Hyper paramters

See ``hparams.py``.

## Monitoring training progress

```
tensorboard --logdir=log
```

## References

- [Yuki Saito, Shinnosuke Takamichi, Hiroshi Saruwatari, "Statistical Parametric Speech Synthesis Incorporating Generative Adversarial Networks", arXiv:1709.08041 [cs.SD], Sep. 2017](https://arxiv.org/abs/1709.08041)
- [Yuki Saito, Shinnosuke Takamichi, and Hiroshi Saruwatari, "Training algorithm to deceive anti-spoofing verification for DNN-based text-to-speech synthesis," IPSJ SIG Technical Report, 2017-SLP-115, no. 1, pp. 1-6, Feb., 2017. (in Japanese)](http://sython.org/papers/SIG-SLP/saito201702slp.pdf)
- [Yuki Saito, Shinnosuke Takamichi, and Hiroshi Saruwatari, "Voice conversion using input-to-output highway networks," IEICE Transactions on Information and Systems, Vol.E100-D, No.8, pp.1925--1928, Aug. 2017](https://www.jstage.jst.go.jp/article/transinf/E100.D/8/E100.D_2017EDL8034/_article)
- https://www.slideshare.net/ShinnosukeTakamichi/dnnantispoofing
- https://www.slideshare.net/YukiSaito8/Saito2017icassp

## Notice

The repository doesn't try to reproduce same results reported in their papers because 1) data is not publically available and 2). hyper parameters are highly depends on data. Instead, I tried same ideas on different data with different hyper parameters.
