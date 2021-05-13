#!/bin/bash

# Fail after any error
set -e

# Change directory to repository root (in case this script is run from somewhere else)
cd "$( dirname "${BASH_SOURCE[0]}" )/.."

# Setup python path
export PYTHONPATH="$PYTHONPATH:$(pwd):$(pwd)/public"

flake8 --max-line-length=120 .
