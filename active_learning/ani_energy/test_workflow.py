"""CPU tests; no reads/writes to existing experiments or datasets."""

from __future__ import annotations

import os
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from common import (
    BUDGET, CASES, CONTROL, ENERGY_KEY, HERE, MEMBERS, REGIMES,
    augment_training, comparison, config_id, geometry_only,
    inventory, load_cached, load_json, partition_ood, prepare_data, save_cached,
    save_json, select_ids,
)

os.environ["MPLCONFIGDIR"] = str(HERE / "runs" / "_test_cache" / "matplotlib")
os.environ["XDG_CACHE_HOME"] = str(HERE / "runs" / "_test_cache")

import numpy as np
import torch
import yaml
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read, write
from e3nn import o3
from mace import modules
from runtime import load_ensemble, make_loader, predict, reliability
from train_member import continue_member
import workflow


def molecule(index, system="H2", distance=0.75):
    atoms = Atoms("H2", positions=[[0, 0, 0], [distance, 0, 0]])
    atoms.info = {"system": system, "conf_idx": index, ENERGY_KEY: -1.0 - index / 10000,
                  "config_type": "cc_train", "relative_energy_eV": index / 10,
                  "energy_quantile": 0.9, "wb97x_tz.energy": -7.0}
    return atoms


class ConstantMVE(torch.nn.Module):
    def __init__(self, mean, variance):
        super().__init__()
        self.mean = torch.nn.Parameter(torch.tensor(float(mean)))
        self.variance = float(variance)
        self.register_buffer("atomic_numbers", torch.tensor([1]))
        self.register_buffer("r_max", torch.tensor(3.0))
        self.heads = ["Default"]
        self.predict_mve = True

    def forward(self, graph, **kwargs):
        size = len(graph["ptr"]) - 1
        return {"energy_mean": self.mean.expand(size),
                "energy_var": torch.full((size,), self.variance, device=self.mean.device)}


class DataTests(unittest.TestCase):
    def test_balanced_partition_and_reproducibility(self):
        atoms = [molecule(i, f"system_{i // 20:03d}") for i in range(1200)]
        first = partition_ood(atoms, 7)
        self.assertEqual(first, partition_ood(atoms, 7))
        self.assertNotEqual(first[0], partition_ood(atoms, 8)[0])
        pool, test, systems = first
        self.assertEqual(len(pool), 600)
        self.assertFalse(set(pool) & set(test))
        self.assertEqual(set(pool) | set(test), set(range(1200)))
        self.assertTrue(all(counts == {"pool": 10, "heldout": 10} for counts in systems.values()))

    def test_label_metadata_and_calculator_removed(self):
        atoms = molecule(1)
        atoms.calc = SinglePointCalculator(atoms, energy=-90, forces=np.ones((2, 3)))
        atoms.arrays["label_array"] = np.ones(2)
        atoms.set_cell([9, 9, 9])
        atoms.pbc = True
        public = geometry_only(atoms)
        self.assertEqual(set(public.info), {"system", "conf_idx", "al_id"})
        self.assertEqual(set(public.arrays), {"numbers", "positions"})
        self.assertIsNone(public.calc)
        np.testing.assert_array_equal(public.cell, atoms.cell)
        np.testing.assert_array_equal(public.pbc, atoms.pbc)

    def test_selection_is_exact_reproducible_and_finite(self):
        ids = [f"molecule:{i:04d}" for i in range(800)]
        scores = {key: float(i) for i, key in enumerate(ids)}
        chosen = select_ids(ids, "tu", 0, scores)
        self.assertEqual(chosen, list(reversed(ids))[:BUDGET])
        random = select_ids(ids, "random", 3)
        self.assertEqual(random, select_ids(ids, "random", 3))
        self.assertEqual(len(set(random)), BUDGET)
        self.assertNotEqual(random, select_ids(ids, "random", 4))
        tied = dict.fromkeys(ids, 1.0)
        self.assertEqual(select_ids(ids[::-1], "tu", 0, tied), ids[:BUDGET])
        scores[ids[0]] = float("nan")
        with self.assertRaises(ValueError):
            select_ids(ids, "tu", 0, scores)

    def test_prepare_reveal_and_no_original_mutations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            write(source / "cc_test_ood.xyz", [molecule(i, f"s_{i // 20}") for i in range(1200)])
            for theory in ("cc", "dft"):
                for i, split in enumerate(("train", "val", "test_id")):
                    write(source / f"{theory}_{split}.xyz", [molecule(2000 + i)])
            originals = inventory(source.glob("*"))
            output = root / "al"
            split = prepare_data(source, output, 0)
            self.assertEqual(set(load_json(output / "oracle.json")), set(split["pool_ids"]))
            self.assertTrue(all(ENERGY_KEY not in atoms.info for atoms in read(output / "pool.xyz", ":")))
            ids = select_ids(split["pool_ids"], "random", 0)
            augmented = output / "train.xyz"
            augment_training(source / "cc_train.xyz", output / "pool.xyz", output / "oracle.json",
                             ids, split["heldout_ids"], augmented)
            result = read(augmented, index=":")
            self.assertEqual(len(result), BUDGET + 1)
            self.assertEqual({config_id(atoms) for atoms in result[1:]}, set(ids))
            self.assertTrue(augmented.read_bytes().startswith((source / "cc_train.xyz").read_bytes()))
            self.assertEqual(originals, inventory(source.glob("*")))
            with self.assertRaises(ValueError):
                augment_training(source / "cc_train.xyz", output / "pool.xyz", output / "oracle.json",
                                 [split["heldout_ids"][0], *ids[1:]], split["heldout_ids"], augmented)
            write(source / "dft_train.xyz", [molecule(0, "s_0")])
            with self.assertRaises(ValueError):
                prepare_data(source, root / "leaky", 0)

    def test_cache_integrity_and_reporting_formulas(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "cache.json"
            artifact = Path(temporary) / "artifact.json"
            save_json(artifact, [1, 2])
            save_cached(path, {"seed": 0}, {"value": 1}, [artifact])
            self.assertEqual(load_cached(path, {"seed": 0}), {"value": 1})
            with self.assertRaises(RuntimeError):
                load_cached(path, {"seed": 1})
            save_json(artifact, [3, 4])
            with self.assertRaises(RuntimeError):
                load_cached(path, {"seed": 0})
        stats = comparison(10, 8, 6)
        self.assertEqual(stats["relative_improvement_random_percent"], 20)
        self.assertEqual(stats["relative_improvement_tu_percent"], 40)
        self.assertEqual(stats["tu_gain_over_random_meV_per_atom"], 2)
        self.assertEqual(stats["tu_gain_over_random_percentage_points"], 20)
        self.assertIsNone(comparison(0, 0, 0)["relative_improvement_tu_percent"])


class RuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.set_default_dtype(torch.float32)

    def test_actual_shared_tu_definition_and_hidden_labels(self):
        models = [ConstantMVE(i, 4) for i in range(10)]
        atoms = [molecule(0), molecule(1)]
        public_loader = make_loader(atoms, models[0], 2, labeled=False)
        batch = next(iter(public_loader))
        self.assertTrue(torch.all(batch.energy == 0))
        self.assertTrue(torch.all(batch.energy_weight == 0))
        rows = reliability.evaluate_split(models, public_loader, torch.device("cpu"), True, None, "test", 0)
        for row in rows:
            self.assertAlmostEqual(row["aleatoric_var"], 1.0)
            self.assertAlmostEqual(row["epistemic_var"], 2.0625)
            self.assertAlmostEqual(row["total_var"], 3.0625)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pool.xyz"
            write(path, atoms)
            with patch("runtime.load_ensemble", return_value=models):
                first = predict([], path, "cpu", 2, labeled=False)
                for atom in atoms:
                    atom.info[ENERGY_KEY] = 123456789
                write(path, atoms)
                second = predict([], path, "cpu", 2, labeled=False)
            self.assertEqual(first, second)
            self.assertNotIn("sq_error", first[0])
            self.assertNotIn("ref_energy", first[0])

    def test_real_mve_checkpoint_continuation_one_epoch(self):
        torch.manual_seed(0)
        model = modules.ScaleShiftMACE(
            r_max=3, num_bessel=4, num_polynomial_cutoff=5, max_ell=0,
            interaction_cls=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
            interaction_cls_first=modules.interaction_classes["RealAgnosticResidualInteractionBlock"],
            num_interactions=2, num_elements=1, hidden_irreps=o3.Irreps("4x0e"),
            MLP_irreps=o3.Irreps("4x0e"), gate=torch.nn.functional.silu,
            atomic_energies=np.array([0.0]), avg_num_neighbors=1, atomic_numbers=[1],
            correlation=1, radial_type="bessel", atomic_inter_scale=1.0,
            atomic_inter_shift=0.0, predict_mve=True,
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial = root / "initial.model"
            torch.save(model, initial)
            loaded = load_ensemble([initial], "cpu")[0]
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, loaded.state_dict()[name]), name)
            self.assertTrue(all(not param.requires_grad for param in loaded.parameters()))
            write(root / "train.xyz", [molecule(i, distance=0.7 + i / 20) for i in range(4)])
            write(root / "val.xyz", [molecule(10), molecule(11)])
            config = {
                "name": "mace", "seed": 0, "device": "cpu", "default_dtype": "float32",
                "train_file": str(root / "train.xyz"), "valid_file": str(root / "val.xyz"),
                "energy_key": ENERGY_KEY, "batch_size": 2, "valid_batch_size": 2,
                "max_num_epochs": 1, "lr": 0.001, "compute_forces": False,
                "forces_weight": 0, "predict_mve": True, "loss": "gaussian_nll", "swa": False,
                "num_channels": 4, "max_L": 0,
            }
            for key in ("work_dir", "model_dir", "log_dir", "results_dir", "checkpoints_dir", "downloads_dir"):
                config[key] = str(root / key)
            config_path = root / "config.yml"
            config_path.write_text(yaml.safe_dump(config))
            continue_member(config_path, initial)
            finished = load_json(root / "finished.json")
            trained = load_ensemble([finished["model"]], "cpu")[0]
            self.assertTrue(any(not torch.equal(value, trained.state_dict()[name])
                                for name, value in model.state_dict().items()))
            self.assertTrue(torch.equal(model.atomic_energies_fn.atomic_energies,
                                        trained.atomic_energies_fn.atomic_energies))
            rows = predict([finished["model"]], root / "val.xyz", "cpu", 2, labeled=True)
            self.assertEqual(len(rows), 2)
            self.assertTrue(np.isfinite(reliability.compute_energy_rmse(rows)))


class OrchestrationTests(unittest.TestCase):
    def test_four_cases_optional_control_and_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, run = root / "source", root / "run"
            source.mkdir()
            write(source / "cc_test_ood.xyz", [molecule(i, f"s_{i // 20}") for i in range(1200)])
            for theory in ("cc", "dft"):
                for j, split in enumerate(("train", "val", "test_id")):
                    write(source / f"{theory}_{split}.xyz", [molecule(2000 + j)])
            prepare_data(source, run / "data", 0)
            initial = {}
            for regime in REGIMES:
                folder = run / "initial" / regime
                folder.mkdir(parents=True)
                (folder / "source_config.yml").write_text(yaml.safe_dump({
                    "lr": 0.01 if regime == "hf_only" else 0.001,
                    "max_num_epochs": 300 if regime == "hf_only" else 100,
                }))
                initial[regime] = []
                for member in MEMBERS:
                    path = folder / f"{member}.model"
                    save_json(path, {"regime": regime, "member": member})
                    initial[regime].append({"model": str(path)})
            manifest = {"initial": initial, "source_dataset": str(source), "settings": {"budget": BUDGET}}
            save_json(run / "manifest.json", manifest)
            args = SimpleNamespace(seed=0, case=None, members=None, common_evaluator=False,
                                   device="cpu", batch_size=64, epochs=None)

            def fake_inference(run, name, models, xyz, args, *, labeled):
                self.assertEqual(len(models), 10)
                rows = []
                for i, atoms in enumerate(read(xyz, index=":")):
                    row = {"al_id": config_id(atoms), "total_var": float(i if "hf_only" in name else 1200 - i)}
                    if labeled:
                        error = 1.0 if name.startswith("before") else (0.8 if "random" in name else 0.6)
                        row["sq_error"] = error ** 2
                    else:
                        self.assertNotIn(ENERGY_KEY, atoms.info)
                    rows.append(row)
                path = run / "inference" / f"{name}.json"
                save_cached(path, {}, rows)
                return rows, path

            def fake_training(command, *, cwd, **kwargs):
                config = yaml.safe_load((cwd / "config.yml").read_text())
                case, member = cwd.parent.parent.name, config["seed"]
                regime, acquisition = ("lf_hf", "hf_only_tu") if case == CONTROL else CASES[case]
                self.assertEqual(command[-1], initial[regime][member]["model"])
                self.assertEqual(config["train_file"], str(run / "acquisition" / acquisition / "train.xyz"))
                self.assertEqual(config["valid_file"], str(source / "cc_val.xyz"))
                self.assertNotIn("test_file", config)
                self.assertFalse(config["wandb"])
                self.assertEqual(config["lr"], 0.001)
                self.assertEqual(config["max_num_epochs"], args.epochs if args.epochs is not None else 100)
                path = cwd / "models" / "best.model"
                save_json(path, {"case": case, "member": member})
                save_json(cwd / "finished.json", {"model": str(path), "artifacts": inventory([path])})

            with patch("workflow.inference", side_effect=fake_inference), \
                    patch("workflow.subprocess.run", side_effect=fake_training) as trainer, \
                    redirect_stdout(io.StringIO()):
                workflow.acquire(run, manifest, args)
                workflow.train(run, manifest, args)
                self.assertEqual(trainer.call_count, 40)
                workflow.train(run, manifest, args)
                self.assertEqual(trainer.call_count, 40)
                workflow.evaluate(run, manifest, args)
                workflow.report(run, args)
                summary = load_json(run / "report" / "summary.json")
                self.assertEqual(len(summary["rows"]), 4)
                self.assertAlmostEqual(summary["rows"][0]["relative_improvement_tu_percent"], 40)
                self.assertEqual(summary["common_evaluator"], [])
                args.common_evaluator = True
                workflow.train(run, manifest, args)
                self.assertEqual(trainer.call_count, 50)
                workflow.evaluate(run, manifest, args)
                workflow.report(run, args)
                self.assertEqual(len(load_json(run / "report" / "summary.json")["common_evaluator"]), 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
