

import warnings
warnings.filterwarnings("ignore")
from mace.cli.run_train import main as mace_run_train_main
import sys
import logging
import shutil
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"


shutil.rmtree("MACE_models", ignore_errors=True)
shutil.rmtree("MACE_models_logs", ignore_errors=True)
shutil.rmtree("MACE_models_checkpoints", ignore_errors=True)
shutil.rmtree("MACE_models_results", ignore_errors=True)

def train_mace(config_file_path):
    logging.getLogger().handlers.clear()
    sys.argv = ["program", "--config", config_file_path]
    mace_run_train_main()

train_mace("config/config-01.yml")
