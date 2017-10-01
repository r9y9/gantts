# GAN TTS

PyTorch implementation of Generative adversarial Networks (GAN) based text-to-speech (TTS) and voice conversion (VC). Models, training algorithms and demos for TTS and VC using [CMU ARCTIC](http://festvox.org/cmu_arctic/) are available.

## Requirements

- [PyTorch](http://pytorch.org/)
- [TensorFlow](https://www.tensorflow.org/) (just for `tf.contrib.training.HParams`)
- [nnmnkwii](https://github.com/r9y9/nnmnkwii)

## Installation

```
git clone --recursive https://github.com/r9y9/gantts & cd gantts
pip install -e . # or python setup.py develop
```

If you want to run the training script, then you need to install additional dependencies.

```
pip install -e ".[train]"
```

## Repository structure

- **gantts/**: Network definitions, utilities for working on sequence-loss optimization.
- **prepare_features_vc.py**: Acoustic feature extraction script for voice conversion.
- **prepare_features_tts.py**: Linguistic/duration/acoustic feature extraction script for TTS.
- **train.py**: GAN-based training script. This is written to be generic so that can be used for training voice conversion models as well as text-to-speech models (duration/acoustic).
- **hparams.py**: Hyper parameters VC and TTS experiments.

Feature extraction scripts are written for CMU ARCTIC dataset, but can be easily adapted for other datasets.

## Run demos

### Text-to-speech synthesis (en)

```
./tts_demo.sh
```
This will download slt_arctic_full_data used in Merlin's demo, perform feature extraction and training models.

### Voice conversion (en)

Please download [CMU ARCTIC](http://festvox.org/cmu_arctic/) datasets, at least for two speakers (e.g., `clb`, `slt`) and check that you have all data in a directory as follows:

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
./vc_demo.sh ${your_cmu_arctic_data_root} # in my case, data root is `~/data/cmu_arctic`
```


## References

- [Yuki Saito, Shinnosuke Takamichi, Hiroshi Saruwatari, "Statistical Parametric Speech Synthesis Incorporating Generative Adversarial Networks", arXiv:1709.08041 [cs.SD], Sep. 2017](https://arxiv.org/abs/1709.08041)
- [Yuki Saito, Shinnosuke Takamichi, and Hiroshi Saruwatari, "Training algorithm to deceive anti-spoofing verification for DNN-based text-to-speech synthesis," IPSJ SIG Technical Report, 2017-SLP-115, no. 1, pp. 1-6, Feb., 2017. (in Japanese)](http://sython.org/papers/SIG-SLP/saito201702slp.pdf)
- [Yuki Saito, Shinnosuke Takamichi, and Hiroshi Saruwatari, "Voice conversion using input-to-output highway networks," IEICE Transactions on Information and Systems, Vol.E100-D, No.8, pp.1925--1928, Aug. 2017](https://www.jstage.jst.go.jp/article/transinf/E100.D/8/E100.D_2017EDL8034/_article)
- https://www.slideshare.net/ShinnosukeTakamichi/dnnantispoofing
- https://www.slideshare.net/YukiSaito8/Saito2017icassp

## Notice

The repository doesn't try to reproduce same results reported in their papers because 1) data is not publically available and 2). hyper parameters are highly depends on data. Instead, I tried same ideas on different data with different hyper parameters.
