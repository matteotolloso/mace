"""Uncertainty quantification for machine-learned interatomic potentials.

Research code of this repository (the forked MACE package lives in ``mace/``):

    uq.paths            directory layout (the only place that hard-codes it)
    uq.registry         experiment and dataset specs (YAML)
    uq.cli              the ``uq`` command: list, status, train, evaluate, aggregate, ...
    uq.datasets         dataset builders (ANI-1x splits, water)
    uq.training         member launcher, <=2-GPU scheduler, completeness
    uq.evaluation       per-split inference, metrics and five-split aggregation
    uq.active_learning  active-learning workflow
    uq.legacy           old experiment letters and pre-2026-09 paths

Start with ``uq --help`` and README.md.
"""
