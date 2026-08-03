#!/bin/bash

set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "${REPO_ROOT}/utils/train_split_member.sh"

EXPERIMENT_DIR="experiment_A"
CONFIG_FILE="experiment_A/config_A.yml"
DATASET_PREFIX="dataset/ani1x_system_split"
TRAIN_RELATIVE_PATH="dft_train.xyz"
VALID_RELATIVE_PATH="dft_val.xyz"
TEST_RELATIVE_PATH="dft_test_id.xyz"
FOUNDATION_EXPERIMENT_DIR=""

run_split_member "$@"
