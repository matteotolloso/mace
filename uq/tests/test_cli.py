"""Smoke tests of the `uq` command: help texts, read-only commands and spec validity."""

import contextlib
import dataclasses
import io
import unittest
from pathlib import Path

from uq import cli, legacy, paths, registry


def run(*argv):
    out = io.StringIO()
    with contextlib.redirect_stdout(out):
        code = cli.main(list(argv))
    return code, out.getvalue()


class CliTests(unittest.TestCase):
    def test_every_help_renders(self):
        parser = cli.build_parser()
        parser.format_help()
        for action in parser._subparsers._group_actions:
            for sub in action.choices.values():
                sub.format_help()

    def test_specs_are_valid(self):
        self.assertEqual(registry.validate(), [])

    def test_read_only_commands(self):
        self.assertEqual(run("list")[0], 0)
        self.assertEqual(run("check")[0], 0)
        code, text = run("show", "D")
        self.assertEqual(code, 0)
        self.assertIn("ani1x-energy/mace-mve/lf-to-hf", text)

    def test_unknown_backend_fails_fast(self):
        exp = dataclasses.replace(registry.experiment("A"), model="gaussian-process")
        with self.assertRaisesRegex(registry.SpecError, "no training/evaluation backend"):
            registry.require_backend(exp)

    def test_experiment_index_is_current(self):
        """experiments/README.md embeds `uq list --markdown`; regenerate it after adding an experiment."""
        code, table = run("list", "--markdown")
        self.assertEqual(code, 0)
        self.assertIn(table.strip(), (paths.EXPERIMENTS / "README.md").read_text())

    def test_legacy_letters_resolve(self):
        for letter, exp_id in legacy.LETTERS.items():
            self.assertEqual(registry.experiment(letter).id, exp_id)

    def test_absolute_old_paths_from_any_checkout(self):
        old = "/some/old/checkout/experiment_D/checkpoints_3/mace_run-2.model"
        self.assertEqual(legacy.translate(old),
                         paths.RUNS / "ani1x-energy/mace-mve/lf-to-hf/split_3/checkpoints/mace_run-2.model")
        self.assertEqual(legacy.translate("/old/mace/mace/modules/models.py"), paths.ROOT / "mace/modules/models.py")
        self.assertEqual(legacy.translate(paths.ROOT / "uq/cli.py"), paths.ROOT / "uq/cli.py")
        self.assertEqual(legacy.translate("/not/in/any/checkout.bin"), Path("/not/in/any/checkout.bin"))

    def test_where_translates_old_paths(self):
        code, text = run("where", "experiment_D/checkpoints_3/mace_run-2.model")
        self.assertEqual(code, 0)
        self.assertIn("runs/ani1x-energy/mace-mve/lf-to-hf/split_3/checkpoints/mace_run-2.model", text)


if __name__ == "__main__":
    unittest.main()
