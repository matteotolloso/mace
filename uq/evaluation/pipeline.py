"""Evaluation pipeline: experiment spec -> ordered evaluation steps.

    uq evaluate B --gpu 6 --dry-run      # list the commands
    uq evaluate B --gpu 6                # every split, then the five-split aggregate
    uq aggregate                         # all aggregates from per-split outputs (CPU only)

Per split, in this order (``experiment.yaml`` -> ``evaluation`` selects sets/extras):

  train_curves             ensemble train/val loss and RMSE per epoch
  reliability_<set>_<cal>  final ensemble (validation-NLL checkpoints): per-configuration
                           rows + reliability bins, uncalibrated and isotonic
  epoch_raw_train          mean AU/EU and RMSE on the train set every 10 epochs
  epoch_quality_<set>      Spearman/AUSE/ENCE/magnitudes/RMSE/GNLL every 5 epochs
  epoch_quality_finetune_mixed_<set>   (LF->HF) parent + fine-tuning trajectory
  finetune                 (``finetune_comparison``) parent vs fine-tuned ensemble
  distribution_<set>       per-configuration AU, EU and error distributions
  energy_ood               (``energy_quantiles``) AU/EU by within-system energy quantile
  member_predictions_<set> (``member_predictions``) each member's prediction and variance (npz)
  ensemble_size_<set>      (``member_predictions``, CPU) metrics vs ensemble size M = 1..10

Outputs go to ``experiments/<id>/evaluation/split_<s>/``; the aggregate writes the
five-split CSV/SVG files to ``experiments/<id>/evaluation/``. The commands are the
ones the old ``experiment_X/eval_X.sh`` scripts ran (verified against them).
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from typing import List

from uq import paths, registry

SET_TITLES = {"id": "ID", "ood": "OOD", "train": "Train", "test": "Test", "val": "Val"}


@dataclass
class Step:
    name: str
    module: str
    args: List[str]

    def command(self) -> List[str]:
        return [sys.executable, "-m", f"uq.evaluation.{self.module}", *self.args]

    def __str__(self) -> str:
        return f"python -m uq.evaluation.{self.module} " + " ".join(shlex.quote(a) for a in self.args)


def _r(path) -> str:
    return str(paths.rel(path))


def steps(experiment, split: int) -> List[Step]:
    """The evaluation steps of one split, in execution order."""
    exp = registry.experiment(experiment)
    registry.require_backend(exp)
    ev = exp.evaluation
    key = exp.fidelity.energy_key
    name = exp.run_name
    ck, rs = _r(paths.checkpoints_dir(exp.id, split)), _r(paths.results_dir(exp.id, split))
    out = paths.split_evaluation_dir(exp.id, split)
    o = lambda file: _r(out / file)  # noqa: E731
    f = lambda role: _r(exp.data_file(role, split))  # noqa: E731
    bs, trim = str(ev.get("batch_size", 256)), str(ev.get("trim", 0.0))
    sets = list(ev.get("sets", exp.dataset.tests))
    log_axes = ev.get("reliability_log_axes")
    result = []

    args = ["--checkpoints-dir", ck, "--experiment-name", name, "--train-split", f("train"),
            "--validation-split", f("val"), "--energy-key-train", key, "--energy-key-val", key,
            "--every-n-epochs", "5", "--device", "cuda", "--output-csv", o("train_curves.csv"),
            "--output-plot", o("train_curves.svg")]
    if ev.get("train_curves_batch_size"):
        args += ["--batch-size", str(ev["train_curves_batch_size"])]
    result.append(Step("train_curves", "train_curves", args))

    for role in ev.get("reliability", exp.dataset.tests):
        for calibrated in (True, False):
            tag = f"reliability_{role}_{'cal' if calibrated else 'nocal'}"
            args = ["--checkpoints-dir", ck, "--results-dir", rs, "--experiment-name", name,
                    "--validation-split", f("val"), "--test-split", f(role),
                    "--energy-key-val", key, "--energy-key-test", key, "--selection-key", "loss",
                    "--selection-mode", "min", "--num-bins", "15", "--trim", trim,
                    "--isotonic-calibration", str(calibrated)]
            if log_axes:
                args += ["--log-log-scale", "--axis-min", str(log_axes[0]), "--axis-max", str(log_axes[1])]
            args += ["--output-csv-raw", o(f"{tag}_raw.csv"), "--output-csv-bins", o(f"{tag}_bins.csv"),
                     "--output-plot", o(f"{tag}.svg"), "--log-level", "INFO", "--log-every-batches", "1",
                     "--device", "cuda", "--batch-size", bs]
            result.append(Step(tag, "reliability", args))

    result.append(Step("epoch_raw_train", "epoch_raw", [
        "--checkpoints-dir", ck, "--experiment-name", name, "--split-path", f("train"), "--energy-key", key,
        "--device", "cuda", "--batch-size", bs, "--every-n-epochs", "10",
        "--output-csv", o("epoch_raw_train.csv"), "--output-plot", o("epoch_raw_train.svg")]))

    for role in sets:
        result.append(Step(f"epoch_quality_{role}", "epoch_quality", [
            "--checkpoints-dir", ck, "--experiment-name", name, "--split-path", f(role), "--energy-key", key,
            "--device", "cuda", "--batch-size", bs, "--every-n-epochs", "5", "--free-scale",
            "--output-csv", o(f"epoch_quality_{role}.csv"), "--output-plot", o(f"epoch_quality_{role}.svg")]))

    parent = exp.parent
    if parent:
        pck, prs = _r(paths.checkpoints_dir(parent.id, split)), _r(paths.results_dir(parent.id, split))
        for role in sets:
            tag = f"epoch_quality_finetune_mixed_{role}"
            title = (f"Epoch Quality {SET_TITLES.get(role, role)}: {parent.fidelity.name} Pretrain + "
                     f"{exp.fidelity.name} Finetune")
            result.append(Step(tag, "epoch_quality_mixed", [
                "--pretrain-checkpoints-dir", pck, "--pretrain-experiment-name", parent.run_name,
                "--pretrain-split-path", _r(parent.data_file(role, split)),
                "--pretrain-energy-key", parent.fidelity.energy_key,
                "--finetune-checkpoints-dir", ck, "--finetune-experiment-name", name,
                "--finetune-split-path", f(role), "--finetune-energy-key", key,
                "--device", "cuda", "--batch-size", bs, "--every-n-epochs-pretrain", "5",
                "--every-n-epochs-finetune", "5", "--free-scale",
                "--output-csv", o(f"{tag}.csv"), "--output-plot", o(f"{tag}.svg"), "--title", title]))
        if ev.get("finetune_comparison"):
            result.append(Step("finetune", "finetune_comparison", [
                "--pretrain-checkpoints-dir", pck, "--pretrain-results-dir", prs,
                "--pretrain-experiment-name", parent.run_name,
                "--finetune-checkpoints-dir", ck, "--finetune-results-dir", rs, "--finetune-experiment-name", name,
                "--test-id-split", f("id"), "--test-ood-split", f("ood"), "--energy-key-test", key,
                "--selection-key-pretrain", "loss", "--selection-key-finetune", "loss",
                "--selection-mode-pretrain", "min", "--selection-mode-finetune", "min", "--trim", trim,
                "--device", "cuda", "--batch-size", bs, "--output-csv", o("finetune.csv"),
                "--output-plot", o("finetune.svg")]))

    for role in sets:
        result.append(Step(f"distribution_{role}", "distribution", [
            "--checkpoints-dir", ck, "--results-dir", rs, "--experiment-name", name, "--split-path", f(role),
            "--energy-key", key, "--selection-key", "loss", "--selection-mode", "min", "--trim", trim,
            "--device", "cuda", "--batch-size", bs, "--output-csv", o(f"distribution_{role}.csv"),
            "--output-plot", o(f"distribution_{role}.svg")]))

    if ev.get("energy_quantiles"):
        result.append(Step("energy_ood", "energy_ood", [
            "--checkpoints-dir", ck, "--results-dir", rs, "--experiment-name", name,
            "--test-id-split", f("id"), "--test-ood-split", f("ood"), "--energy-key-test", key,
            "--selection-key", "loss", "--selection-mode", "min", "--device", "cuda", "--batch-size", bs,
            "--output-csv-raw", o("energy_ood_raw.csv"), "--output-csv-bins", o("energy_ood_bins.csv"),
            "--output-plot", o("energy_ood.svg")]))

    for role in ev.get("member_predictions", []):
        result.append(Step(f"member_predictions_{role}", "member_predictions", [
            "--checkpoints-dir", ck, "--results-dir", rs, "--experiment-name", name, "--split-path", f(role),
            "--energy-key", key, "--selection-key", "loss", "--selection-mode", "min", "--device", "cuda",
            "--batch-size", bs, "--output", o(f"members_{role}.npz")]))
        result.append(Step(f"ensemble_size_{role}", "ensemble_size", [
            "--members-npz", o(f"members_{role}.npz"), "--output-csv", o(f"ensemble_size_{role}.csv")]))
    return result


def aggregate_step(experiment) -> Step:
    exp = registry.experiment(experiment)
    ev_dir = _r(paths.evaluation_dir(exp.id))
    args = ["--cache-dir", ev_dir, "--output-dir", ev_dir, "--split-seeds", *map(str, exp.splits)]
    log_axes = exp.evaluation.get("reliability_log_axes")
    if log_axes:
        args += ["--log-log-reliability", "--reliability-axis-min", str(log_axes[0]),
                 "--reliability-axis-max", str(log_axes[1])]
    return Step("aggregate", "aggregate", args)


def _run(step: Step, gpu=None) -> int:
    env = dict(os.environ)
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    print(f"[{step.name}] {step}", flush=True)
    return subprocess.call(step.command(), cwd=paths.ROOT, env=env)


def evaluate(experiment, gpu, splits=None, only_steps=None, aggregate=True, dry_run=False) -> int:
    """Run the per-split steps on one GPU (stop at the first failure), then aggregate."""
    exp = registry.experiment(experiment)
    only = set(only_steps or [])
    for split in splits or exp.splits:
        for step in steps(exp.id, split):
            if only and step.name not in only:
                continue
            if dry_run:
                print(f"CUDA_VISIBLE_DEVICES={gpu} {step}")
                continue
            status = _run(step, gpu)
            if status:
                print(f"{exp.label} split {split}: step {step.name} failed (exit {status})", file=sys.stderr)
                return status
    if aggregate:
        step = aggregate_step(exp.id)
        if dry_run:
            print(step)
            return 0
        return _run(step)
    return 0


def aggregate_all(names=(), dry_run=False, only=None) -> int:
    """Rebuild five-split aggregates from per-split outputs (CPU only); with no names,
    every experiment with outputs plus the active-learning plots. ``only``: file-name
    prefixes to rebuild (SVGs embed a date and random ids, so a full rebuild rewrites
    every tracked SVG even when nothing changed)."""
    status = 0
    for exp in registry.resolve(names):
        if not paths.split_evaluation_dir(exp.id, exp.splits[0]).is_dir():
            print(f"skipping {exp.label}: no per-split evaluation outputs yet")
            continue
        step = aggregate_step(exp.id)
        if only:
            step.args += ["--only", *only]
        if dry_run:
            print(step)
        else:
            status |= _run(step)
    if not names and not only:
        from uq.active_learning import entry

        status |= entry.replot_all(dry_run=dry_run)
    return status
