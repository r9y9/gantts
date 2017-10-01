#!/bin/bash

# GAN-based voice cnoversion demo

set -e

data_dir=$1

# Feature extraction
python prepare_features_vc.py --max_files=500 ${data_dir} \
    clb slt --dst_dir=./data/cmu_arctic_vc

# train_gan.sh args:
# 1. Hyper param name
# 2. X: Network inputs
# 3. Y: Network outputs
# 4. Where to save checkpoints
# 5. Generator wamup epoch
# 6.discriminator_warmup_epoch
# 7. Total epoch for spoofing model training
# 8. Total epoch for GAN

# Traing models
./train_gan.sh vc data/cmu_arctic_vc/X data/cmu_arctic_vc/Y \
    checkpoints/vc \
    50 10 50 200
