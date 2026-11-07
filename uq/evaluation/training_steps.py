#!/usr/bin/env python3
"""Gradient steps per protocol, up to each member's selected checkpoint (CPU, logs only; rebuttal WP4.1).

For every ANI-1x MVE experiment with all members trained (A-F), per split and member:

  train_size        the "training dataset size" MACE logs at start (``logs/<name>_run-<m>.log``)
  steps_per_epoch   floor(train_size / batch_size): MACE's training loader drops the last batch
  selected_epoch    the epoch the evaluation pipeline selects (lowest validation loss,
                    ``reliability.select_best_checkpoints``); its checkpoint is saved after epoch + 1 epochs
  steps_selected    (selected_epoch + 1) x steps_per_epoch, plus the parent's ``steps_selected`` for LF->HF
                    (the fine-tuning starts from the parent's selected checkpoint)
  epochs_run, steps_run   the same for the whole run (last validation epoch + 1)

Output (``--output``, default ``results/tables/training_steps.csv``): one row per experiment with the
median, minimum and maximum over its 50 members of each quantity.

    python -m uq.evaluation.training_steps
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from uq import paths, registry
from uq.evaluation.reliability import select_best_checkpoints
from uq.evaluation.replicate_statistics import write_csv

EXPERIMENTS = [f"ani1x-{d}/mace-mve/{p}" for d in ("system", "energy") for p in ("lf-only", "lf-to-hf", "hf-only")]
SIZE = re.compile(r"training dataset size: (\d+)")
EPOCH = re.compile(r"_epoch-(\d+)\.pt$")


def member_rows(exp):
    batch = int(registry._config(exp.config_path)["batch_size"])  # noqa: SLF001 - the spec's own config
    rows = {}
    for split in exp.splits:
        selected = select_best_checkpoints(paths.checkpoints_dir(exp.id, split), paths.results_dir(exp.id, split),
                                           exp.run_name, "loss", "min", None)
        for path in selected:
            member = int(re.search(r"_run-(\d+)_epoch-", path.name).group(1))
            log = paths.logs_dir(exp.id, split) / f"{exp.run_name}_run-{member}.log"
            size = int(SIZE.search(log.read_text()).group(1))
            epochs = [json.loads(line).get("epoch") for line in exp.train_log(split, member).read_text().splitlines()
                      if '"mode": "eval"' in line]
            per_epoch = size // batch
            sel = int(EPOCH.search(path.name).group(1))
            rows[(split, member)] = {"train_size": size, "steps_per_epoch": per_epoch, "selected_epoch": sel,
                                     "steps_selected": (sel + 1) * per_epoch,
                                     "epochs_run": max(e for e in epochs if e is not None) + 1}
            rows[(split, member)]["steps_run"] = rows[(split, member)]["epochs_run"] * per_epoch
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, default=paths.RESULTS / "tables" / "training_steps.csv")
    args = parser.parse_args(argv)
    members = {name: member_rows(registry.experiment(name)) for name in EXPERIMENTS}
    out = []
    for name in EXPERIMENTS:
        exp = registry.experiment(name)
        rows = members[name]
        if len(rows) != len(exp.splits) * len(exp.members):
            raise SystemExit(f"{name}: {len(rows)} members selected, expected {len(exp.splits) * len(exp.members)}")
        cumulative = {k: r["steps_selected"] + (members[exp.parent.id][k]["steps_selected"] if exp.parent else 0)
                      for k, r in rows.items()}
        record = {"experiment": name, "legacy_id": exp.legacy_id or "", "protocol": exp.protocol,
                  "members": len(rows)}
        for key in ("train_size", "steps_per_epoch", "selected_epoch", "steps_selected", "epochs_run", "steps_run"):
            values = np.array([r[key] for r in rows.values()])
            record.update({f"{key}_median": float(np.median(values)), f"{key}_min": int(values.min()),
                           f"{key}_max": int(values.max())})
        values = np.array(list(cumulative.values()))
        record.update({"steps_selected_incl_parent_median": float(np.median(values)),
                       "steps_selected_incl_parent_min": int(values.min()),
                       "steps_selected_incl_parent_max": int(values.max())})
        out.append(record)
    write_csv(args.output, out)
    print(f"wrote {paths.rel(args.output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
