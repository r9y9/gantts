#!/bin/bash

# GAN-based TTS demo

set -e

# Flags
run_duration_training=1
run_acoustic_training=1

# Download merin's demo data
./nnmnkwii_gallery/scripts/download_data.sh slt_arctic_full_data


# Linguistic/duration/acoustic feature extraction
# for all data.
dst_dir=./data/cmu_arctic_tts
python prepare_features_tts.py --max_files=-1 \
    ./nnmnkwii_gallery/data/slt_arctic_full_data/ \
    --dst_dir=${dst_dir}


# train_gan.sh args:
# 1. Hyper param name
# 2. X: Network inputs
# 3. Y: Network outputs
# 4. Where to save checkpoints
# 5. Generator wamup epoch
# 6.discriminator_warmup_epoch
# 7. Total epoch for spoofing model training
# 8. Total epoch for GAN

# Train duration model
if [ "$run_duration_training" == 1 ]; then
    ./train_gan.sh tts_duration \
        data/cmu_arctic_tts/X_duration/ \
        data/cmu_arctic_tts/Y_duration/ \
        checkpoints/tts_duration \
        50 5 50 100
fi

# Train acoustic model
if [ "$run_acoustic_training" == 1 ]; then
    ./train_gan.sh tts_acoustic \
        data/cmu_arctic_tts/X_acoustic/ \
        data/cmu_arctic_tts/Y_acoustic/ \
        checkpoints/tts_acoustic \
        50 5 20 100
fi
