#!/bin/bash

set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "${REPO_ROOT}/utils/train_split_member.sh"

EXPERIMENT_DIR="experiment_wC"
CONFIG_FILE="experiment_wC/config_wC.yml"
DATASET_PREFIX="dataset/water"
TRAIN_RELATIVE_PATH="ccsdt/train.xyz"
VALID_RELATIVE_PATH="ccsdt/val.xyz"
TEST_RELATIVE_PATH="ccsdt/test.xyz"
FOUNDATION_EXPERIMENT_DIR=""

run_split_member "$@"
