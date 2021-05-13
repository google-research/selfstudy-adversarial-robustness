#!/bin/bash
#
# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
# This script trains all models for the self-study in parallel on available GPUs
#
# Usage:
#
#   train_all.sh [--gpus=GPU_IDS] [--keep_all_ckpt] [--cleanup_out_dir=YES|NO]
#
# Where:
#
#   GPU_IDS
#       Comma separated list of IDs of GPUs to use for training, ex: 0,1,2
#       If not provided then all available GPUs will be used for training.
#
#   --keep_all_ckpt
#       If this flag is provided then all intermediate checkpoints will be kept.
#       Otherwise only final checkpoint will be kept. Note that intermediate
#       checkpoints are not needed in the self-study.
#
#   --cleanup_out_dir=YES|NO
#       Specifies whether to cleanup output directory. Default is YES.
#

################################################################################
# Preparation - parse arguments, setup variables, etc...
################################################################################

# Fail after any error
set -e

# Parse command line arguments
GPU_IDS=""
KEEP_ALL_CKPT="NO"
CLEANUP_OUT_DIR="YES"

for i in "$@"
do
case $i in
    --gpus=*)
    GPU_IDS="${i#*=}"
    shift
    ;;
    --cleanup_out_dir=*)
    CLEANUP_OUT_DIR="${i#*=}"
    shift
    ;;
    --keep_all_ckpt)
    KEEP_ALL_CKPT="YES"
    shift
    ;;
    *)
    # unknown option
    ;;
esac
done

# If no GPU IDS provided then use all available GPUs
if [[ -z ${GPU_IDS} ]]; then
    echo "Testing availability of GPUs."
    GPU_COUNT=`nvidia-smi --query-gpu=count --format=csv,noheader | tail -n 1`
    echo "Found ${GPU_COUNT} GPUs."
    GPU_IDS=`seq --separator=, 0 $((${GPU_COUNT}-1))`
fi

if [[ -z ${GPU_IDS} ]]; then
  echo "No GPUs found for training."
  exit 1
fi

echo "GPUs to be used: ${GPU_IDS}"

# Convert GPU IDs into array
IFS=',' read -ra GPU_IDS <<< "${GPU_IDS}"

# Change directory to repository root (in case this script is run from somewhere else)
cd "$( dirname "${BASH_SOURCE[0]}" )/.."

################################################################################
# Helper code which schedules training on different GPUs
################################################################################

NEXT_GPU_IDX=0
NUM_GPUS=${#GPU_IDS[@]}
MODEL_DIR="$(pwd)/checkpoints"

function wait_until_done() {
    NEXT_GPU_IDX=0
    echo ""
    echo "Waiting until training finishes."
    echo ""
    wait
}


function run() {
    local training_script=$1
    local log_file="/tmp/${training_script}.log"
    local gpu_idx=${NEXT_GPU_IDX}

    echo "Starting ${training_script} on GPU ${gpu_idx}. Log saved to ${log_file}"

    CUDA_VISIBLE_DEVICES=${gpu_idx} python3 "training/${training_script}" --model_dir="${MODEL_DIR}" &> ${log_file} &

    NEXT_GPU_IDX=$((${NEXT_GPU_IDX}+1))

    if [[ "${NEXT_GPU_IDX}" == "${NUM_GPUS}" ]]; then
        wait_until_done
    fi
}

################################################################################
# Prepare checkpoint directory
################################################################################

echo ""
echo "Checkpoint directory: ${MODEL_DIR}"
echo ""
mkdir -p ${MODEL_DIR}

if [[ "${CLEANUP_OUT_DIR}" == "YES" ]]; then
    echo "All existing data will be deleted from the checkpoint directory,"
    echo "new models will be trained and saved there."
    read -p "Continue? (y|N) " -r

    if [[ ("${REPLY}" != "y") && ("${REPLY}" != "yes") ]]; then
        exit
    fi

    echo ""

    rm -rf "${MODEL_DIR}/*"
fi

################################################################################
# Run training scripts
################################################################################

run train_baseline.py
run train_blur.py
run train_ls.py
run train_jump.py
run train_injection.py
run train_diverse.py
run train_multiple_binary.py
run train_discretize.py
run train_transform.py
run train_contrastive.py
run train_advtrain.py

wait_until_done

################################################################################
# Cleanup all aux checkpoints
################################################################################

if [[ "${KEEP_ALL_CKPT}" -eq "NO" ]]; then
    echo "Cleaning up auxiliary checkpoints"
    find "${MODEL_DIR}" -name "checkpoint*" -type f -delete
fi
