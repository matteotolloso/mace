#!/bin/bash

set -euo pipefail

REPO_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
source "${REPO_ROOT}/utils/train_split_member.sh"

EXPERIMENT_DIR="experiment_wA"
CONFIG_FILE="experiment_wA/config_wA.yml"
DATASET_PREFIX="dataset/water"
TRAIN_RELATIVE_PATH="blyp/train.xyz"
VALID_RELATIVE_PATH="blyp/val.xyz"
TEST_RELATIVE_PATH="blyp/test.xyz"
FOUNDATION_EXPERIMENT_DIR=""

run_split_member "$@"
