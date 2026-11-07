"""Snapshot of the training and evaluation commands of every recorded experiment.

The commands of the ICLR27 experiments were verified equal to the pre-restructure
scripts; this snapshot keeps any experiment's commands from changing by accident when
the pipeline grows. Experiments that are not in the snapshot yet are not checked.
After an intended change, rewrite it and review the diff before committing:

    python -m uq.tests.test_commands --update
"""

import json
import re
import sys
import unittest
from pathlib import Path

from uq import paths, registry
from uq.evaluation import pipeline
from uq.training import launch

SNAPSHOT = Path(__file__).with_name("snapshots") / "commands.json"


def _normalize(text: str) -> str:
    return text.replace(f"{paths.RUNS}/", "runs/").replace(f"{paths.ROOT}/", "")


def train_command(exp_id: str, split: int, member: int) -> str:
    return _normalize(" ".join(["python", *launch.command(exp_id, split, member)[1:]]))


def evaluation_commands(exp_id: str, split: int) -> list:
    return [_normalize(f"{step.name}: {step}") for step in pipeline.steps(exp_id, split)]


def snapshot_of(exp_id: str) -> dict:
    """Commands of split 0 and member 0; the other splits and members differ only by their numbers."""
    return {"train": train_command(exp_id, 0, 0), "evaluate": evaluation_commands(exp_id, 0),
            "aggregate": _normalize(str(pipeline.aggregate_step(exp_id)))}


def _as_split_0(text: str, split: int) -> str:
    text = re.sub(rf"--split {split}(?!\d)", "--split 0", text)   # lf_on_hf takes the split number
    return re.sub(rf"split_{split}(?!\d)", "split_0", text)


class CommandSnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recorded = json.loads(SNAPSHOT.read_text())

    def test_recorded_experiments_still_exist(self):
        missing = sorted(set(self.recorded) - set(registry.experiments()))
        self.assertEqual(missing, [], "experiments in the snapshot were removed or renamed")

    def test_commands_match_the_snapshot(self):
        for exp_id, recorded in self.recorded.items():
            with self.subTest(experiment=exp_id):
                self.assertEqual(snapshot_of(exp_id), recorded,
                                 "commands changed; if intended: python -m uq.tests.test_commands --update")

    def test_splits_differ_only_by_their_number(self):
        for exp in registry.experiments().values():
            base = evaluation_commands(exp.id, 0)
            for split in exp.splits:
                with self.subTest(experiment=exp.id, split=split):
                    self.assertEqual([_as_split_0(c, split) for c in evaluation_commands(exp.id, split)], base)

    def test_members_differ_only_by_seed(self):
        for exp in registry.experiments().values():
            base = train_command(exp.id, 0, 0)
            for split in exp.splits:
                for member in exp.members:
                    text = _as_split_0(train_command(exp.id, split, member), split)
                    text = (text.replace(f"--seed {member} ", "--seed 0 ").replace(f"_seed_{member} ", "_seed_0 ")
                            .replace(f"_run-{member}.model", "_run-0.model"))
                    self.assertEqual(text, base, f"{exp.id} split {split} member {member}")


def update() -> None:
    SNAPSHOT.parent.mkdir(exist_ok=True)
    table = {exp_id: snapshot_of(exp_id) for exp_id in registry.experiments()}
    SNAPSHOT.write_text(json.dumps(table, indent=1) + "\n")
    print(f"wrote {paths.rel(SNAPSHOT)} ({len(table)} experiments); review `git diff` before committing")


if __name__ == "__main__":
    if "--update" in sys.argv:
        update()
    else:
        unittest.main()
