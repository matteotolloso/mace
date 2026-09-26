"""Single-round active learning on ANI-1x Energy-OOD (proof of concept).

    workflow         one split: data partition, acquisition (random/AU/EU/TU), warm-start
                     fine-tuning of the 10 members, held-out evaluation, report
    five_splits      runs/aggregates the five splits with 95% intervals
    support_filter   re-aggregates with the geometry-only training-support filter
    plot_results     figures of an aggregate summary
    train_member     continues one member from its checkpoint (called by workflow)
    runtime          model loading, loaders, ensemble prediction
    common           constants, provenance (sha256 inventories), data helpers
    entry            ``uq al run|run-all``

Runs are written to ``runs/active-learning/ani1x-energy/<name>/``; the parents are
the final members of ``ani1x-energy/mace-mve/{hf-only,lf-to-hf}``. See
docs/active-learning.md.
"""
