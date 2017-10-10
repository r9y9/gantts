#!/bin/bash

# GAN-based Japanese TTS demo

set -e

experiment_id=$1
fs=48000

# Needs adjastment
HTS_DEMO_ROOT=~/local/HTS-demo_NIT-ATR503-M001

# Flags
run_duration_training=1
run_acoustic_training=1
generated_audio_dir=./generated/${experiment_id}
checkpoints_dir=./checkpoints/${experiment_id}

# Download HTS's demo data
./nnmnkwii_gallery/scripts/copy_from_htsdemo.sh $HTS_DEMO_ROOT

# Linguistic/duration/acoustic feature extraction
# for all data.
data_dir=./data/nit_atr503_tts_order59
python prepare_features_tts.py --max_files=-1 \
    ./nnmnkwii_gallery/data/NIT-ATR503 \
    --dst_dir=${data_dir}


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
        50 5 10 100 $experiment_id
fi

# Train acoustic model
if [ "$run_acoustic_training" == 1 ]; then
    ./train_gan.sh tts_acoustic \
        ${data_dir}/X_acoustic/ \
        ${data_dir}/Y_acoustic/ \
        ${checkpoints_dir}/tts_acoustic \
        25 5 10 50 $experiment_id
fi

# Generate audio samples for eval and test set
for ty in baseline gan
do
    python evaluation_tts.py \
        ${checkpoints_dir}/tts_acoustic/$ty/checkpoint_epoch50_Generator.pth \
        ${checkpoints_dir}/tts_duration/$ty/checkpoint_epoch100_Generator.pth \
        ${data_dir} \
        ./nnmnkwii_gallery/data/NIT-ATR503/label_phone_align/ \
        ${generated_audio_dir}/duration_acousic/$ty \
        --fs=$fs
    python evaluation_tts.py \
        ${checkpoints_dir}/tts_acoustic/$ty/checkpoint_epoch50_Generator.pth \
        ${checkpoints_dir}/tts_duration/$ty/checkpoint_epoch100_Generator.pth \
        ${data_dir} \
        ./nnmnkwii_gallery/data/NIT-ATR503/label_phone_align/ \
        ${generated_audio_dir}/acoustic_only/$ty --disable-duraton-gen \
        --fs=$fs
done
