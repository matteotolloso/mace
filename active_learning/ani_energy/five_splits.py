#!/usr/bin/env python3
"""Run the five existing AL splits and aggregate paired statistics with 95% CIs."""

from __future__ import annotations

import argparse
import csv
import fcntl
import io
import math
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

from common import AL_EPOCHS, AL_LEARNING_RATE, HERE, comparison, inventory, load_json, save_json, verify_inventory

T95_DF4 = 2.7764451051977987
TESTS = ("energy_ood", "energy_id")
REGIMES = ("hf_only", "lf_hf")


def confidence_interval(values):
    if len(values) != 5:
        raise ValueError("Exactly five split-level values are required.")
    if any(value is not None and not math.isfinite(value) for value in values):
        raise ValueError("Non-finite split-level statistic.")
    if any(value is None for value in values):
        return {"n": sum(value is not None for value in values), "mean": None,
                "ci95_lower": None, "ci95_upper": None, "status": "undefined_statistic"}
    mean = statistics.mean(values)
    margin = T95_DF4 * statistics.stdev(values) / math.sqrt(5)
    return {"n": 5, "mean": mean, "ci95_lower": mean - margin,
            "ci95_upper": mean + margin, "status": "ok"}


def read_report(run):
    """Do not read a summary while the existing single-split runner is writing it."""
    path = run / "report" / "summary.json"
    if not path.exists():
        return None
    with (run / ".lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return None
        report = load_json(path)
        verify_inventory(report["inputs"])
        manifest = load_json(run / "manifest.json")
        if report["settings"] != manifest["settings"]:
            raise ValueError(f"Report/manifest settings mismatch: {run}")
        report["_verified_inputs"] = {
            **inventory([path, run / "manifest.json"]), **report["inputs"]
        }
        return report


def aggregate(reports, settings, common_evaluator=False):
    if len(reports) != 5:
        raise ValueError("All five split reports are required.")
    groups = defaultdict(list)
    for split, report in enumerate(reports):
        actual = dict(report["settings"])
        if actual.pop("split_seed") != split or actual != settings:
            raise ValueError(f"Split {split}: incompatible experimental settings.")
        rows = {(row["regime"], row["test"]): row for row in report["rows"]}
        if len(report["rows"]) != 4 or set(rows) != {(r, t) for r in REGIMES for t in TESTS}:
            raise ValueError(f"Split {split}: missing or duplicate condition/test rows.")
        for test in TESTS:
            stats = {}
            if rows[("hf_only", test)]["count"] != rows[("lf_hf", test)]["count"]:
                raise ValueError("Regimes have different test-set counts.")
            for regime in REGIMES:
                row = rows[(regime, test)]
                stats[regime] = comparison(*(row[key] for key in (
                    "rmse_before_meV_per_atom", "rmse_random500_meV_per_atom", "rmse_tu500_meV_per_atom"
                )))
                for metric, value in stats[regime].items():
                    groups[(regime, test, metric)].append(value)
            # Form paired differences within each split, not differences of CI bounds.
            for unit in ("meV_per_atom", "percentage_points"):
                key = f"tu_gain_over_random_{unit}"
                hf, lf = stats["hf_only"][key], stats["lf_hf"][key]
                groups[("cross_regime", test, f"lf_hf_minus_hf_only_tu_gain_{unit}")].append(
                    None if hf is None or lf is None else lf - hf
                )
        if common_evaluator:
            control = report.get("common_evaluator", [])
            if len(control) != 2 or {row["test"] for row in control} != set(TESTS):
                raise ValueError(f"Split {split}: common-evaluator results are incomplete.")
            for row in control:
                if row["downstream"] != "lf_hf":
                    raise ValueError("Unexpected common-evaluator regime.")
                hf = row["hf_only_acquisition_rmse_meV_per_atom"]
                lf = row["lf_hf_acquisition_rmse_meV_per_atom"]
                for metric, value in (("hf_only_acquisition_rmse_meV_per_atom", hf),
                                      ("lf_hf_acquisition_rmse_meV_per_atom", lf),
                                      ("lf_hf_acquisition_gain_meV_per_atom", hf - lf)):
                    groups[("common_evaluator", row["test"], metric)].append(value)
    return [{"regime": regime, "test": test, "metric": metric,
             **confidence_interval(values), "split_values": values}
            for (regime, test, metric), values in sorted(groups.items())]


def write_report(destination, rows, inputs, settings):
    destination.mkdir(parents=True, exist_ok=True)
    with (destination / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        summary_path = destination / "summary_ci95.json"
        if summary_path.exists():
            cached = load_json(summary_path)
            if cached["inputs"] == inputs:
                verify_inventory(cached["artifacts"])
                print(f"Using cached aggregate: {summary_path}", flush=True)
                return
        fields = [key for key in rows[0] if key != "split_values"]
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=[*fields, *(f"split_{s}" for s in range(5))])
        writer.writeheader()
        for row in rows:
            writer.writerow({**{key: row[key] for key in fields},
                             **{f"split_{s}": value for s, value in enumerate(row["split_values"])}})
        lines = ["# Five-Split ANI Active Learning", "",
                 "Mean and approximate 95% Student-t CI across five split-level results (df=4).",
                 "The 10 ensemble members are not treated as independent repetitions.",
                 "Intervals assume independent, approximately normal split-level estimates;",
                 "overlapping datasets and only five repetitions limit significance claims.", "",
                 "| Regime | Test | Metric | Mean | 95% CI |", "|---|---|---|---:|---|"]
        for row in rows:
            mean = "undefined" if row["mean"] is None else f"{row['mean']:.5g}"
            ci = "undefined" if row["mean"] is None else f"[{row['ci95_lower']:.5g}, {row['ci95_upper']:.5g}]"
            lines.append(f"| {row['regime']} | {row['test']} | {row['metric']} | {mean} | {ci} |")
        artifacts = []
        for suffix, content in (("csv", buffer.getvalue()), ("md", "\n".join(lines) + "\n")):
            path = destination / f"summary_ci95.{suffix}"
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_text(content)
            temporary.replace(path)
            artifacts.append(path)
        save_json(summary_path, {"inputs": inputs, "artifacts": inventory(artifacts),
                  "settings": settings, "rows": rows, "confidence_level": 0.95,
                  "method": "mean +/- t(df=4, p=.975) * sample_sd / sqrt(5)",
                  "paired": "Gains and relative improvements computed within each split first.",
                  "caveat": "Approximate CI: overlapping splits are not independent datasets."})
        print(f"Aggregate reports: {destination}/summary_ci95.{{csv,json,md}}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--common-evaluator", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true", help="Never launch training/evaluation")
    parser.add_argument("--wait", action="store_true", help="Wait for five reports; requires --aggregate-only")
    parser.add_argument("--poll-seconds", type=float, default=60)
    args = parser.parse_args()
    if args.gpu < 0 or args.seed < 0 or args.poll_seconds <= 0 or (args.epochs is not None and args.epochs <= 0):
        parser.error("GPU/seed must be nonnegative; epochs/poll interval must be positive.")
    if args.wait and not args.aggregate_only:
        parser.error("--wait requires --aggregate-only.")
    suffix = f"_epochs_{args.epochs}" if args.epochs is not None else ""
    runs = [HERE / "runs" / f"split_{split}{suffix}" for split in range(5)]
    if not args.aggregate_only:
        for split in range(5):
            existing = read_report(runs[split])
            if existing is not None:
                saved = existing["settings"]
                if saved.get("learning_rate") != AL_LEARNING_RATE:
                    raise ValueError(
                        f"Split {split}: saved run predates the shared AL learning rate. "
                        "Existing results are unchanged; use --aggregate-only to summarize them."
                    )
                expected_epochs = args.epochs if args.epochs is not None else AL_EPOCHS
                if saved.get("additional_epochs") != expected_epochs:
                    raise ValueError(
                        f"Split {split}: saved run does not match the shared AL epoch limit. "
                        "Existing results are unchanged; use --aggregate-only to summarize them."
                    )
                if saved["split_seed"] != split or saved["seed"] != args.seed or saved["epochs_override"] != args.epochs:
                    raise ValueError(f"Split {split}: saved settings differ from this invocation.")
                if not args.common_evaluator or len(existing.get("common_evaluator", [])) == 2:
                    print(f"Split {split}/4 already evaluated; reusing its report.", flush=True)
                    continue
            command = ["bash", str(HERE / "run.sh"), str(args.gpu),
                       "--split-seed", str(split), "--seed", str(args.seed)]
            if args.epochs is not None:
                command += ["--epochs", str(args.epochs)]
            if args.common_evaluator:
                command += ["--common-evaluator"]
            print(f"Running/resuming split {split}/4 on GPU {args.gpu}", flush=True)
            subprocess.run(command, check=True)
    while True:
        reports = [read_report(run) for run in runs]
        pending = [s for s, report in enumerate(reports)
                   if report is None or (args.common_evaluator and len(report.get("common_evaluator", [])) != 2)]
        if not pending:
            break
        if not args.wait:
            raise RuntimeError(f"Missing or still-running splits: {pending}. Use --aggregate-only --wait to wait.")
        print(f"Waiting for splits {pending}; no training will be launched.", flush=True)
        time.sleep(args.poll_seconds)
    settings = {key: value for key, value in reports[0]["settings"].items() if key != "split_seed"}
    if settings["seed"] != args.seed or settings["epochs_override"] != args.epochs:
        raise ValueError("Requested settings do not match the saved reports.")
    rows = aggregate(reports, settings, args.common_evaluator)
    files = inventory([Path(__file__), HERE / "common.py"])
    for report in reports:
        files.update(report["_verified_inputs"])
    verify_inventory(files)
    inputs = {"files": files,
              "common_evaluator": args.common_evaluator}
    output = HERE / "runs" / f"aggregate{suffix}"
    if args.common_evaluator:
        output = output / "common_evaluator"
    write_report(output, rows, inputs, settings)
    subprocess.run([sys.executable, "-B", str(HERE / "plot_results.py"),
                    "--summary", str(output / "summary_ci95.json")], check=True)


if __name__ == "__main__":
    main()
