"""Training completeness: ``uq status [experiment ...] [-v]``.

A member is complete when both its final model
``runs/<id>/split_<s>/checkpoints/<name>_run-<m>.model`` and its MACE result log
``runs/<id>/split_<s>/results/<name>_run-<m>_train.txt`` exist; a member with only
epoch checkpoints is partial. Also shows which splits have evaluation outputs.
Exit code 1 when anything is incomplete. Replaces ``check_experiments.sh``.
"""

from __future__ import annotations

from uq import paths, registry


def member_state(exp, split, member):
    if exp.member_complete(split, member):
        return "complete"
    checkpoints = paths.checkpoints_dir(exp.id, split)
    if exp.final_model(split, member).exists() or exp.train_log(split, member).exists() \
            or any(checkpoints.glob(f"{exp.run_name}_run-{member}_epoch-*.pt")):
        return "partial"
    return "missing"


def main(names=(), verbose=False) -> int:
    experiments = registry.resolve(names)
    width = max(len(exp.label) for exp in experiments)
    total = complete = 0
    incomplete = []
    print(f"{'experiment':<{width}}  trained members per split          evaluated splits")
    for exp in experiments:
        cells, notes = [], []
        for split in exp.splits:
            states = [member_state(exp, split, m) for m in exp.members]
            done = states.count("complete")
            total += len(states)
            complete += done
            cells.append(f"{done:>2}/{len(states)}")
            if done < len(states):
                incomplete.append(f"{exp.label} split {split}")
                if verbose:
                    for kind in ("partial", "missing"):
                        members = [m for m, st in zip(exp.members, states) if st == kind]
                        if members:
                            notes.append(f"    split {split} {kind}: {members}")
        evaluated = [s for s in exp.splits if any(paths.split_evaluation_dir(exp.id, s).glob("reliability_*_raw.csv"))]
        aggregated = any(paths.evaluation_dir(exp.id).glob("reliability_*_summary.csv"))
        print(f"{exp.label:<{width}}  {' '.join(cells)}   {len(evaluated)}/{len(exp.splits)}"
              f"{' + aggregate' if aggregated else ''}")
        for note in notes:
            print(note)
    print(f"\n{complete}/{total} members complete" + (f"; incomplete: {', '.join(incomplete)}" if incomplete else ""))
    return 1 if incomplete else 0
