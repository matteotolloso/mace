#!/usr/bin/env python3
"""Single-round ANI Energy-OOD active learning; all writes stay in this directory."""

from __future__ import annotations

import argparse
import csv
import fcntl
import importlib.metadata
import logging
import math
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from common import (
    ACQUISITION_METRICS, RUN_TAG, AL_EPOCHS, AL_LEARNING_RATE, BUDGET, CASES, CONTROL, ENERGY_KEY, HERE, MEMBERS, REGIMES, ROOT,
    augment_training, comparison, inventory, load_cached, load_json, prepare_data,
    save_cached, save_json, select_ids, sha256, verify_inventory,
)

import yaml

LOGGER = logging.getLogger("ani_al")


def settings(args):
    return {"schema": 1, "split_seed": args.split_seed, "seed": args.seed,
            "budget": BUDGET, "members": list(MEMBERS), "pool_fraction": 0.5,
            "epochs_override": args.epochs, "selection_key": "loss", "selection_mode": "min",
            "learning_rate": AL_LEARNING_RATE,
            "acquisition_metrics": list(ACQUISITION_METRICS),
            "additional_epochs": args.epochs if args.epochs is not None else AL_EPOCHS,
            "uncertainty": "raw AU + population EU, per-atom variance (eV/atom)^2",
            "warm_start": "full model, fresh optimizer/scheduler, original validation"}


def source_files(split_seed):
    dataset = ROOT / "dataset" / f"ani1x_energy_split_{split_seed}"
    files = [dataset / f"{theory}_{split}.xyz" for theory in ("cc", "dft")
             for split in ("train", "val", "test_id")]
    files.append(dataset / "cc_test_ood.xyz")
    files += [ROOT / f"experiment_{letter}" / f"config_{letter}.yml" for letter in REGIMES.values()]
    files += [HERE / name for name in ("common.py", "runtime.py", "train_member.py", "workflow.py")]
    files += [ROOT / name for name in (
        "eval/reliability.py", "dataset/ani1x_energy_splitter.py", "mace/tools/train.py",
        "mace/tools/scripts_utils.py", "mace/tools/arg_parser.py", "mace/tools/arg_parser_tools.py",
        "mace/tools/checkpoint.py", "mace/modules/models.py", "mace/modules/loss.py",
        "mace/data/utils.py", "mace/data/atomic_data.py",
    )]
    return files


def prepare(run, args):
    manifest_path = run / "manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        if manifest["settings"] != settings(args):
            raise RuntimeError("Run settings changed. Choose a new --name.")
        verify_inventory(manifest["sources"])
        verify_inventory(manifest["artifacts"])
        return manifest

    from runtime import reliability

    inputs = inventory(source_files(args.split_seed))
    source = ROOT / "dataset" / f"ani1x_energy_split_{args.split_seed}"
    LOGGER.info("Preparing pool / held-out OOD partition and checking low-energy ID disjointness")
    split = prepare_data(source, run / "data", args.seed)
    initial = {}
    artifacts = list((run / "data").glob("*"))
    for regime, letter in REGIMES.items():
        experiment = ROOT / f"experiment_{letter}"
        checkpoints = experiment / f"checkpoints_{args.split_seed}"
        results = experiment / f"results_{args.split_seed}"
        chosen = reliability.select_best_checkpoints(checkpoints, results, "mace", "loss", "min", None)
        seeds = [int(reliability.CHECKPOINT_PATTERN.match(path.name).group("seed")) for path in chosen]
        if seeds != list(MEMBERS):
            raise RuntimeError(f"{regime}: expected completed members 0..9, found {seeds}.")
        target = run / "initial" / regime
        target.mkdir(parents=True, exist_ok=True)
        source_config = experiment / f"config_{letter}.yml"
        config = yaml.safe_load(source_config.read_text())
        if config.get("energy_key") != ENERGY_KEY or not config.get("predict_mve"):
            raise RuntimeError(f"{source_config} is not the expected CC MVE protocol.")
        shutil.copyfile(source_config, target / "source_config.yml")
        initial[regime] = []
        for member, path in zip(MEMBERS, chosen):
            destination = target / path.name
            companion = checkpoints / f"mace_run-{member}.model"
            frozen_companion = target / companion.name
            logfile = results / f"mace_run-{member}_train.txt"
            copied_log = target / logfile.name
            provenance = inventory([path, companion, logfile])
            # CUDA TorchScript blocks in some e3nn companions cannot be loaded
            # without a GPU. Preparation only copies bytes; scoring loads later.
            shutil.copyfile(path, destination)
            shutil.copyfile(companion, frozen_companion)
            shutil.copyfile(logfile, copied_log)
            verify_inventory(provenance)
            if [sha256(destination), sha256(frozen_companion), sha256(copied_log)] != list(provenance.values()):
                raise RuntimeError("Checkpoint snapshot does not match its source.")
            initial[regime].append({
                "member": member, "experiment": f"experiment_{letter}", "model": str(destination),
                "checkpoint": str(path), "companion": str(frozen_companion),
                "epoch": int(reliability.CHECKPOINT_PATTERN.match(path.name).group("epoch")),
                "source_provenance": provenance,
            })
            artifacts += [destination, frozen_companion, copied_log]
        artifacts.append(target / "source_config.yml")
    versions = {}
    for name in ("numpy", "ase", "torch", "e3nn", "mace-torch", "PyYAML"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed as distribution"
    # Detect changes during preparation before accepting a reproducible snapshot.
    verify_inventory(inputs)
    manifest = {"settings": settings(args), "sources": inputs, "artifacts": inventory(artifacts),
                "source_dataset": str(source), "initial": initial, "versions": versions,
                "python": sys.version, "pool_count": len(split["pool_ids"]),
                "heldout_count": len(split["heldout_ids"])}
    save_json(manifest_path, manifest)
    LOGGER.info("Prepared %d pool and %d held-out OOD configurations", manifest["pool_count"], manifest["heldout_count"])
    return manifest


def inference(run, name, models, xyz, args, *, labeled):
    from runtime import predict

    path = run / "inference" / f"{name}.json"
    inputs = {"files": inventory([*models, xyz]), "device": args.device,
              "batch_size": args.batch_size, "per_atom": True, "labeled": labeled,
              "manifest_sha256": sha256(run / "manifest.json")}
    rows = load_cached(path, inputs)
    if rows is None:
        rows = predict(models, xyz, args.device, args.batch_size, labeled=labeled)
        for row in rows:
            if any(isinstance(value, (float, int)) and not math.isfinite(value) for value in row.values()):
                raise RuntimeError(f"Non-finite prediction in {name}; refusing to discard configurations.")
        save_cached(path, inputs, rows)
    else:
        LOGGER.info("Using cached predictions: %s", name)
    return rows, path


def acquire(run, manifest, args):
    split = load_json(run / "data" / "split.json")
    pool_ids = split["pool_ids"]
    selections = {"random": ("random", None, None)}
    for regime in REGIMES:
        models = [entry["model"] for entry in manifest["initial"][regime]]
        rows, score_path = inference(run, f"pool_{regime}", models, run / "data" / "pool.xyz", args, labeled=False)
        if [row["al_id"] for row in rows] != pool_ids:
            raise RuntimeError("Cached prediction IDs do not match the acquisition pool.")
        for method, field in ACQUISITION_METRICS.items():
            selections[f"{regime}_{method}"] = (method, {row["al_id"]: row[field] for row in rows}, score_path)
    for name, (method, scores, score_path) in selections.items():
        directory = run / "acquisition" / name
        selection_path = directory / "selection.json"
        inputs = {"manifest_sha256": sha256(run / "manifest.json"), "method": method,
                  "seed": args.seed, "budget": BUDGET,
                  "scores": inventory([score_path]) if score_path else {}}
        selected = load_cached(selection_path, inputs)
        if selected is None:
            selected = {"ids": select_ids(pool_ids, method, args.seed, scores),
                        "method": method, "seed": args.seed, "budget": BUDGET,
                        "acquirer": name.removesuffix(f"_{method}") if method != "random" else "shared_random"}
            save_cached(selection_path, inputs, selected)
        augmented_path = directory / "train.xyz"
        augmented_inputs = inventory([selection_path, run / "data" / "oracle.json",
                                      Path(manifest["source_dataset"]) / "cc_train.xyz"])
        stamp = directory / "augmented.json"
        if load_cached(stamp, augmented_inputs) is None:
            counts = augment_training(
                Path(manifest["source_dataset"]) / "cc_train.xyz", run / "data" / "pool.xyz",
                run / "data" / "oracle.json", selected["ids"], split["heldout_ids"], augmented_path,
            )
            save_cached(stamp, augmented_inputs, counts, [augmented_path])
        LOGGER.info("Ready: %s, %d acquired labels", name, BUDGET)


def configured_cases(args):
    cases = dict(CASES)
    if args.common_evaluator:
        cases[CONTROL] = ("lf_hf", "hf_only_tu")
    if args.case:
        if args.case not in cases:
            raise ValueError("The control case requires --common-evaluator.")
        return {args.case: cases[args.case]}
    return cases


def member_inputs(run, manifest, case, member, args):
    regime, acquisition = ("lf_hf", "hf_only_tu") if case == CONTROL else CASES[case]
    data_stamp = run / "acquisition" / acquisition / "augmented.json"
    if not data_stamp.exists():
        raise RuntimeError("Acquisition is not ready. Run --stage acquire first.")
    stamp = load_json(data_stamp)
    verify_inventory(stamp["inputs"])
    verify_inventory(stamp["artifacts"])
    model = manifest["initial"][regime][member]["model"]
    return {"initial": inventory([model]), "acquisition": inventory([data_stamp]),
            "manifest_sha256": sha256(run / "manifest.json"), "member": member,
            "device": args.device}, regime, acquisition, model


def train(run, manifest, args):
    for case in configured_cases(args):
        for member in args.members or MEMBERS:
            inputs, regime, acquisition, model = member_inputs(run, manifest, case, member, args)
            directory = run / "cases" / case / f"member_{member}"
            done = directory / "complete.json"
            if load_cached(done, inputs) is not None:
                LOGGER.info("Already complete: %s member %d", case, member)
                continue
            directory.mkdir(parents=True, exist_ok=True)
            attempts = sorted(directory.glob("attempt_*"))
            attempt = directory / f"attempt_{len(attempts):03d}"
            attempt.mkdir()
            config = yaml.safe_load((run / "initial" / regime / "source_config.yml").read_text())
            for key in ("foundation_model", "test_file", "test_dir", "wandb_entity", "wandb_project"):
                config.pop(key, None)
            config.update({
                "name": "mace", "seed": member, "device": args.device,
                "lr": AL_LEARNING_RATE,
                "max_num_epochs": AL_EPOCHS,
                "train_file": str(run / "acquisition" / acquisition / "train.xyz"),
                "valid_file": str(Path(manifest["source_dataset"]) / "cc_val.xyz"),
                "wandb": False, "plot": False, "log_epoch_outputs": False,
                "multiheads_finetuning": False,
                "restart_latest": False, "save_all_checkpoints": True, "keep_checkpoints": True,
                "work_dir": str(attempt), "wandb_dir": str(attempt / "wandb"),
                "downloads_dir": str(attempt / "downloads"),
            })
            for key, folder in (("model_dir", "models"), ("log_dir", "logs"),
                                ("results_dir", "results"), ("checkpoints_dir", "checkpoints")):
                config[key] = str(attempt / folder)
            if args.epochs is not None:
                config["max_num_epochs"] = args.epochs
            config_path = attempt / "config.yml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=False))
            LOGGER.info("Training %s member %d; log: %s", case, member, attempt / "console.log")
            command = [sys.executable, "-B", str(HERE / "train_member.py"),
                       "--config", str(config_path), "--initial", model]
            save_json(attempt / "invocation.json", {"command": command, "inputs": inputs,
                      "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                      "warm_start": "all model weights/buffers; optimizer and scheduler reset"})
            with (attempt / "console.log").open("w") as log:
                try:
                    subprocess.run(command, cwd=attempt, stdout=log, stderr=subprocess.STDOUT, check=True)
                except subprocess.CalledProcessError as exc:
                    raise RuntimeError(f"Training failed; inspect {attempt / 'console.log'}") from exc
            finished = load_json(attempt / "finished.json")
            verify_inventory(finished["artifacts"])
            save_cached(done, inputs, finished, [*finished["artifacts"], attempt / "finished.json"])


def evaluation_models(run, manifest, case, args):
    paths = []
    for member in MEMBERS:
        stamp_path = run / "cases" / case / f"member_{member}" / "complete.json"
        if not stamp_path.exists():
            raise RuntimeError(f"{case} member {member} is incomplete. Run --stage train first.")
        inputs, _, _, _ = member_inputs(run, manifest, case, member, args)
        completed = load_cached(stamp_path, inputs)
        paths.append(completed["model"])
    return paths


def evaluate(run, manifest, args):
    from runtime import reliability

    models = {f"before_{regime}": [entry["model"] for entry in entries]
              for regime, entries in manifest["initial"].items()}
    models.update({case: evaluation_models(run, manifest, case, args) for case in configured_cases(args)})
    tests = {"energy_ood": run / "data" / "heldout_ood.xyz",
             "energy_id": Path(manifest["source_dataset"]) / "cc_test_id.xyz"}
    for name, ensemble in models.items():
        for test, xyz in tests.items():
            rows, predictions = inference(run, f"{name}_{test}", ensemble, xyz, args, labeled=True)
            rmse = reliability.compute_energy_rmse(rows)
            if not math.isfinite(rmse):
                raise RuntimeError("Non-finite RMSE; refusing to publish partial metrics.")
            save_json(run / "metrics" / f"{name}_{test}.json", {
                "condition": name, "test": test, "count": len(rows), "test_file": str(xyz),
                "test_sha256": sha256(xyz), "rmse_eV_per_atom": rmse,
                "rmse_meV_per_atom": 1000 * rmse, "prediction_file": str(predictions),
                "prediction_sha256": sha256(predictions), "models": inventory(ensemble),
                "manifest_sha256": sha256(run / "manifest.json"),
            })


def report(run, args):
    rows = []
    gain_comparison = {}
    metrics_used = []

    def metric(condition, test):
        path = run / "metrics" / f"{condition}_{test}.json"
        if not path.exists():
            raise RuntimeError(f"Missing metrics: {path}. Complete training/evaluation first.")
        item = load_json(path)
        verify_inventory({item["test_file"]: item["test_sha256"],
                          item["prediction_file"]: item["prediction_sha256"],
                          str(run / "manifest.json"): item["manifest_sha256"], **item["models"]})
        metrics_used.append(path)
        return item

    def matched(values):
        if len({(item["test_sha256"], item["count"]) for item in values}) != 1:
            raise RuntimeError("Conditions were not evaluated on exactly the same test set.")

    for test in ("energy_ood", "energy_id"):
        current = {}
        test_metrics = []
        for regime in REGIMES:
            values = [metric(name, test) for name in
                      (f"before_{regime}", f"{regime}_random", *(f"{regime}_{m}" for m in ACQUISITION_METRICS))]
            test_metrics.extend(values)
            stats = {}
            for method, item in zip(ACQUISITION_METRICS, values[2:]):
                stats.update(comparison(values[0]["rmse_meV_per_atom"], values[1]["rmse_meV_per_atom"],
                                        item["rmse_meV_per_atom"], method))
            current[regime] = stats
            rows.append({"regime": regime, "test": test, "count": values[0]["count"], **stats})
        matched(test_metrics)
        hf, lf = current["hf_only"], current["lf_hf"]
        gain_comparison[test] = {}
        for method in ACQUISITION_METRICS:
            for unit in ("meV_per_atom", "percentage_points"):
                key = f"{method}_gain_over_random_{unit}"
                gain_comparison[test][f"lf_hf_minus_hf_only_{method}_gain_{unit}"] = (
                    lf[key] - hf[key] if lf[key] is not None and hf[key] is not None else None
                )
    control = []
    if args.common_evaluator:
        for test in ("energy_ood", "energy_id"):
            hf, lf = metric(CONTROL, test), metric("lf_hf_tu", test)
            matched([hf, lf])
            control.append({"test": test, "downstream": "lf_hf",
                            "hf_only_acquisition_rmse_meV_per_atom": hf["rmse_meV_per_atom"],
                            "lf_hf_acquisition_rmse_meV_per_atom": lf["rmse_meV_per_atom"],
                            "lf_hf_acquisition_gain_meV_per_atom": hf["rmse_meV_per_atom"] - lf["rmse_meV_per_atom"]})
    destination = run / "report"
    destination.mkdir(exist_ok=True)
    exclusions = load_json(run / "manifest.json").get("evaluation_exclusions", [])
    save_json(destination / "summary.json", {
        "evaluation_exclusions": exclusions,
        "settings": load_json(run / "manifest.json")["settings"], "rows": rows,
        "cross_regime_gain": gain_comparison, "common_evaluator": control,
        "inputs": inventory(metrics_used), "positive_gain_favors": "uncertainty acquisition / LF->HF",
        "scope": "One dataset split, one acquisition seed; no significance claim or CI.",
    })
    with (destination / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# ANI Energy-OOD Active Learning", "", "RMSE in meV/atom; positive gain favors uncertainty acquisition.", "",
             "| Regime | Test | Method | Before | Random-500 | Acquired-500 | Random improvement % | Method improvement % | Gain over random |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        for method in ACQUISITION_METRICS:
            values = [row[key] for key in ("rmse_before_meV_per_atom", "rmse_random500_meV_per_atom",
                      f"rmse_{method}500_meV_per_atom", "relative_improvement_random_percent",
                      f"relative_improvement_{method}_percent", f"{method}_gain_over_random_meV_per_atom")]
            lines.append("| " + " | ".join([row["regime"], row["test"], method.upper(),
                         *("n/a" if value is None else f"{value:.4f}" for value in values)]) + " |")
    lines += ["", "LF->HF minus HF-only acquisition gain (positive supports the POC hypothesis):"]
    for test, values in gain_comparison.items():
        for method in ACQUISITION_METRICS:
            lines.append(f"- {test}, {method.upper()}: {values[f'lf_hf_minus_hf_only_{method}_gain_meV_per_atom']:.4f} meV/atom")
    if control:
        lines += ["", "Common LF->HF evaluator; positive gain favors the LF->HF-selected batch:"]
        for item in control:
            lines.append(f"- {item['test']}: HF-only selection {item['hf_only_acquisition_rmse_meV_per_atom']:.4f}, "
                         f"LF->HF selection {item['lf_hf_acquisition_rmse_meV_per_atom']:.4f}, "
                         f"gain {item['lf_hf_acquisition_gain_meV_per_atom']:.4f} meV/atom")
    if exclusions:
        lines += ["", "Post-hoc held-out exclusions (results are conditional on this modified test set):"]
        lines += [f"- {item['al_id']}: {item['reason']}" for item in exclusions]
    lines += ["", "Single-round, single-split POC. No confidence interval or significance claim.", ""]
    (destination / "summary.md").write_text("\n".join(lines))
    print("\n".join(lines))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "prepare", "acquire", "train", "evaluate", "report"), default="all")
    parser.add_argument("--split-seed", type=int, choices=range(5), default=0)
    parser.add_argument("--seed", type=int, default=0,
                        help="Internal acquisition RNG seed (default 0); recorded in metadata, not a repetition layer")
    parser.add_argument("--name", help="Isolated run directory name (no paths)")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=64, help="Inference batch size only")
    parser.add_argument("--epochs", type=int, help="Override BOTH regimes' additional epochs; default: 100 for both")
    parser.add_argument("--common-evaluator", action="store_true")
    parser.add_argument("--case", choices=(*CASES, CONTROL), help="Select one case for train/evaluate stages")
    parser.add_argument("--members", nargs="+", type=int, choices=MEMBERS, help="Train these member seeds only")
    args = parser.parse_args()
    if args.seed < 0 or args.batch_size <= 0 or (args.epochs is not None and args.epochs <= 0):
        parser.error("Seeds must be nonnegative; batch size and epochs must be positive.")
    if args.case and args.stage not in ("train", "evaluate"):
        parser.error("--case is supported with --stage train/evaluate only.")
    if args.members and args.stage != "train":
        parser.error("--members is supported with --stage train only.")
    if args.name is None:
        args.name = f"split_{args.split_seed}" + (f"_{RUN_TAG}" if RUN_TAG else "") + (f"_epochs_{args.epochs}" if args.epochs else "")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.name):
        parser.error("--name must be a simple directory name, not a path.")
    return args


def main():
    args = parse_args()
    run = (HERE / "runs" / args.name).resolve()
    if not run.is_relative_to(HERE) or run == HERE:
        raise ValueError("Output directory must stay inside active_learning/ani_energy/.")
    run.mkdir(parents=True, exist_ok=True)
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    os.environ["MPLCONFIGDIR"] = str(run / "runtime_cache" / "matplotlib")
    os.environ["XDG_CACHE_HOME"] = str(run / "runtime_cache")
    os.environ["WANDB_MODE"] = "disabled"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with (run / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process is using {run}; run stages sequentially.") from exc
        manifest = prepare(run, args)
        if args.stage not in ("prepare", "report") and args.device == "cuda":
            import torch
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable. Preparation is saved; rerun on a GPU-enabled machine/session.")
        for stage, operation in (("acquire", acquire), ("train", train), ("evaluate", evaluate)):
            if args.stage in ("all", stage):
                operation(run, manifest, args)
        if args.stage in ("all", "report"):
            report(run, args)
    print(f"AL artifacts: {run}")


if __name__ == "__main__":
    main()
