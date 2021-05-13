#!/bin/bash

# Fail after any error
set -e

# Change directory to repository root (in case this script is run from somewhere else)
cd "$( dirname "${BASH_SOURCE[0]}" )/.."

# Setup python path
export PYTHONPATH="$PYTHONPATH:$(pwd):$(pwd)/public"

# Tests rely on pretrained model checkpoints, make sure they exist
function check_checkpoint_exist {
  local ckpt_prefix=$1
  local some_ckpt_file="${ckpt_prefix}.index"

  if [[ ! -f "${some_ckpt_file}" ]]; then
    echo "Checkpoint does not exist: ${ckpt_prefix}"
    echo "Make sure that all checkpoints are prepared before running tests."
    exit 1
  fi
}

check_checkpoint_exist "checkpoints/baseline/final_checkpoint-1"

# Run tests
python -m pytest tests/*.py
