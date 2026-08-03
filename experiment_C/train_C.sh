#!/bin/bash

set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "${REPO_ROOT}/utils/train_split_member.sh"

EXPERIMENT_DIR="experiment_C"
CONFIG_FILE="experiment_C/config_C.yml"
DATASET_PREFIX="dataset/ani1x_energy_split"
TRAIN_RELATIVE_PATH="dft_train.xyz"
VALID_RELATIVE_PATH="dft_val.xyz"
TEST_RELATIVE_PATH="dft_test_id.xyz"
FOUNDATION_EXPERIMENT_DIR=""

run_split_member "$@"
