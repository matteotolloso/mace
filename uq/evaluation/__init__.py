"""Evaluation of trained ensembles: per-split inference, metrics and five-split aggregation.

One module per evaluation step (run as ``python -m uq.evaluation.<module>``);
``pipeline`` turns an experiment spec into the ordered list of steps, and
``uq evaluate`` / ``uq aggregate`` run them. See docs/evaluation.md.
"""
