#!/bin/bash
# Create the `mace` conda env on Leonardo from environment.yml, install this repository in
# editable mode and run the CPU checks. Budget-free serial partition (has internet; login nodes
# kill processes after 10 min of CPU). Submit from $WORK/uq-mf/slurm:
#   sbatch $WORK/uq-mf/mace/papers/iclr27/rebuttal/leonardo/setup_env.sbatch
# Requires the Anaconda ToS for the `defaults` channels to be accepted for our Miniconda
# (user decision; see LEONARDO.md).
#SBATCH --account=IscrC_UQ-MF
#SBATCH --partition=lrd_all_serial
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --job-name=uq-setup-env
#SBATCH --output=%x_%j.out
set -euo pipefail

UQ=/leonardo_work/IscrC_UQ-MF/uq-mf
CONDA=$UQ/miniconda3/bin/conda
ENV=$UQ/miniconda3/envs/mace
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL PYTHONPATH   # never the other project's conda
export PATH=$UQ/miniconda3/bin:/usr/bin:/bin
export TMPDIR=${TMPDIR:-/tmp} PIP_CACHE_DIR=$TMPDIR/pip-cache

cd $UQ/mace
echo "commit $(git rev-parse --short HEAD), $(date)"
if [ ! -x $ENV/bin/python ]; then
  $CONDA env create -y -p $ENV -f environment.yml
fi
$ENV/bin/pip install --no-deps --no-build-isolation -e .
$ENV/bin/python -c "import torch, e3nn, mace, uq; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'e3nn', e3nn.__version__)"
$ENV/bin/python -m uq check
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=4 $ENV/bin/python -m unittest discover -s uq/tests -t .
echo "setup done $(date)"
