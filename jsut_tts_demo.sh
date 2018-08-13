#!/bin/bash

# GAN-based Japanese TTS demo

set -e

experiment_id=$1
fs=48000

# Corpus path
DATA_ROOT=~/data/jsut_ver1.1/basic5000

# Flags
run_duration_training=1
run_acoustic_training=1
generated_audio_dir=./generated/${experiment_id}
checkpoints_dir=./checkpoints/${experiment_id}

# Linguistic/duration/acoustic feature extraction
# for all data.
data_dir=./data/jsut_basic5000_order59
python prepare_features_tts.py --max_files=-1 \
    ${DATA_ROOT} --dst_dir=${data_dir}

# train_gan.sh args:
# 1. Hyper param name
# 2. X: Network inputs
# 3. Y: Network outputs
# 4. Where to save checkpoints
#
# 5. Generator wamup epoch
# 6. discriminator_warmup_epoch
# 7. Total epoch for spoofing model training
# 8. Total epoch for GAN

# Train duration model
if [ "$run_duration_training" == 1 ]; then
    ./train_gan.sh tts_duration \
        ${data_dir}/X_duration/ \
        ${data_dir}/Y_duration/ \
        ${checkpoints_dir}/tts_duration \
        15 5 10 30 $experiment_id
fi

# Train acoustic model
if [ "$run_acoustic_training" == 1 ]; then
    ./train_gan.sh tts_acoustic \
        ${data_dir}/X_acoustic/ \
        ${data_dir}/Y_acoustic/ \
        ${checkpoints_dir}/tts_acoustic \
        15 5 10 30 $experiment_id
fi

# Generate audio samples for eval and test set
for ty in baseline gan
do
    python evaluation_tts.py \
        ${checkpoints_dir}/tts_acoustic/$ty/checkpoint_epoch30_Generator.pth \
        ${checkpoints_dir}/tts_duration/$ty/checkpoint_epoch30_Generator.pth \
        ${data_dir} \
        ${DATA_ROOT}/lab \
        ${generated_audio_dir}/duration_acousic/$ty \
        --fs=$fs
    python evaluation_tts.py \
        ${checkpoints_dir}/tts_acoustic/$ty/checkpoint_epoch30_Generator.pth \
        ${checkpoints_dir}/tts_duration/$ty/checkpoint_epoch30_Generator.pth \
        ${data_dir} \
        ${DATA_ROOT}/lab \
        ${generated_audio_dir}/acoustic_only/$ty --disable-duraton-gen \
        --fs=$fs
done
