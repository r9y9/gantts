#!/bin/bash

# GAN-based TTS demo

set -e

experiment_id=$1

# Flags
run_feature_extraction=1
run_duration_training=1
run_acoustic_training=1
run_evaluations=1

generated_audio_dir=./generated/${experiment_id}
checkpoints_dir=./checkpoints/${experiment_id}

# Download merin's demo data
./nnmnkwii_gallery/scripts/download_data.sh slt_arctic_full_data


# Linguistic/duration/acoustic feature extraction
# for all data.
# data_dir=./data/cmu_arctic_tts_order24
data_dir=./data/cmu_arctic_tts_order59

if [ "$2" != "" ]; then
    data_dir=$2
fi

echo "Experimental id:" $experiment_id
echo "Data dir:" $data_dir

if [ "${run_feature_extraction}" == 1 ]; then
    python prepare_features_tts.py --max_files=-1 \
        ./nnmnkwii_gallery/data/slt_arctic_full_data/ \
        --dst_dir=${data_dir}
fi

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
if [ "$run_evaluations" == 1 ]; then
    for ty in baseline gan
    do
        python evaluation_tts.py \
            ${checkpoints_dir}/tts_acoustic/$ty/checkpoint_epoch50_Generator.pth \
            ${checkpoints_dir}/tts_duration/$ty/checkpoint_epoch100_Generator.pth \
            ${data_dir} \
            ./nnmnkwii_gallery/data/slt_arctic_full_data/label_state_align/ \
            ${generated_audio_dir}/duration_acousic/$ty
    done
fi
