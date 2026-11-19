"""The ``uq`` command (also ``python -m uq``). Run from anywhere; paths are resolved
from the repository layout in ``uq.paths``. Experiments are named by id
(``ani1x-system/mace-mve/lf-to-hf``) or legacy letter (``B``).

Common tasks:
    uq list                                  what experiments exist
    uq status                                training completeness (450 members)
    uq summary B --test ood                  headline metrics from the tracked aggregates
    uq train B --gpus 6 7 --dry-run          train missing members (<= 3 GPUs)
    uq evaluate B --gpu 6 --dry-run          per-split evaluation + five-split aggregate
    uq aggregate                             rebuild all five-split aggregates (CPU only)
    uq where experiment_D/checkpoints_3      old (pre-2026-09) path -> new path
"""

from __future__ import annotations

import argparse
import sys

MAX_GPUS = 3  # project rule (user, 2026-11-03): never more than three GPUs at the same time


def _ints(text: str) -> list:
    """Parse ``0-4`` or ``0,2,3`` into a list of ints."""
    out = []
    for part in text.split(","):
        if "-" in part:
            low, high = part.split("-")
            out.extend(range(int(low), int(high) + 1))
        elif part:
            out.append(int(part))
    return out


# -- handlers ---------------------------------------------------------------------------

def cmd_list(args) -> int:
    from uq import registry

    rows = [(exp.id, exp.legacy_id or "", exp.dataset_id, exp.protocol, exp.parent_id or "")
            for exp in registry.experiments().values()]
    if args.markdown:
        print("| Experiment | Legacy | Dataset | Protocol | Parent |\n|---|---|---|---|---|")
        for row in rows:
            print("| " + " | ".join(f"`{c}`" if i in (0, 4) and c else c for i, c in enumerate(row)) + " |")
    else:
        width = max(len(r[0]) for r in rows)
        for row in rows:
            print(f"{row[0]:<{width}}  {row[1]:<3} {row[3]:<9} {'<- ' + row[4] if row[4] else ''}")
    return 0


def cmd_show(args) -> int:
    from uq import paths, registry

    exp = registry.experiment(args.experiment)
    data = exp.dataset
    print(f"id:        {exp.id}  (legacy {exp.legacy_id})")
    print(f"dataset:   {data.id} [{exp.target} = {exp.fidelity.method}, key {exp.fidelity.energy_key}]")
    print(f"protocol:  {exp.protocol}" + (f", parent {exp.parent.label}" if exp.parent else ""))
    print(f"members:   {len(exp.members)} x splits {list(exp.splits)}; model {exp.model}, uq {exp.uq}")
    print(f"spec:      {paths.rel(exp.dir / 'experiment.yaml')}")
    print(f"config:    {paths.rel(exp.config_path)}")
    print(f"data:      {paths.rel(data.split_dir(0)).as_posix().replace('split_0', 'split_<s>')}/"
          f"{{{', '.join(f'{r}: {f}' for r, f in exp.fidelity.files.items())}}}")
    print(f"runs:      {paths.rel(paths.run_dir(exp.id, 0)).as_posix().replace('split_0', 'split_<s>')}"
          "/{checkpoints,models,logs,results}")
    print(f"eval:      {paths.rel(paths.evaluation_dir(exp.id))}/ (aggregates), split_<s>/ (per split)")
    return 0


def cmd_check(args) -> int:
    from uq import registry

    problems = registry.validate()
    for problem in problems:
        print(problem)
    print(f"{len(registry.experiments())} experiments, {len(registry.datasets())} datasets: "
          + ("OK" if not problems else f"{len(problems)} problem(s)"))
    return 1 if problems else 0


def cmd_status(args) -> int:
    from uq.training import status

    return status.main(args.experiments, verbose=args.verbose)


def cmd_train(args) -> int:
    from uq.training import scheduler

    if len(set(args.gpus)) > MAX_GPUS:
        raise SystemExit(f"at most {MAX_GPUS} GPUs may be used at the same time (got {args.gpus})")
    return scheduler.run(args.experiments, gpus=args.gpus, splits=args.splits, members=args.members,
                         jobs_per_gpu=args.jobs_per_gpu, dry_run=args.dry_run,
                         by_experiment=args.by_experiment, threads_per_job=args.threads_per_job,
                         resume=args.resume)


def cmd_train_slurm(args) -> int:
    from uq.training import slurm

    return slurm.run(args.experiments, splits=args.splits, members=args.members, pack=args.pack,
                     by_experiment=args.by_experiment, resume=args.resume, qos=args.qos, walltime=args.time,
                     account=args.account, partition=args.partition, out=args.out, submit=args.submit,
                     dry_run=args.dry_run, after=args.after, after_ok=args.after_ok)


def cmd_prune(args) -> int:
    from uq.training import prune

    return prune.main(args.experiments, stride=args.stride, apply=args.apply)


def cmd_train_member(args) -> int:
    from uq.training import launch

    return launch.train_member(args.experiment, args.split, args.member, args.gpu, dry_run=args.dry_run,
                               resume=args.resume)


def cmd_evaluate(args) -> int:
    from uq.evaluation import pipeline

    return pipeline.evaluate(args.experiment, gpu=args.gpu, splits=args.splits, only_steps=args.steps,
                             aggregate=not args.no_aggregate, dry_run=args.dry_run)


def cmd_evaluate_slurm(args) -> int:
    from uq.evaluation import slurm

    return slurm.run(args.experiments, splits=args.splits, after=args.after, prune=args.prune, qos=args.qos,
                     walltime=args.time, account=args.account, partition=args.partition, out=args.out,
                     submit=args.submit, dry_run=args.dry_run)


def cmd_sync_back(args) -> int:
    from uq.training import sync

    return sync.run(args.experiments, args.host, splits=args.splits, remote_root=args.remote_root,
                    remote_runs=args.remote_runs, checkpoints=args.checkpoints, apply=args.apply)


def cmd_aggregate(args) -> int:
    from uq.evaluation import pipeline

    return pipeline.aggregate_all(args.experiments, dry_run=args.dry_run, only=args.only)


def cmd_summary(args) -> int:
    from uq.evaluation import summary

    return summary.main(args.experiment, test=args.test, calibrated=args.cal, members=args.members)


def cmd_where(args) -> int:
    from uq import legacy, paths

    for old in args.paths:
        print(f"{old} -> {paths.rel(legacy.translate(old))}")
    return 0


def cmd_data(args) -> int:
    if args.data_command == "build":
        from uq.datasets import build

        return build.build(args.dataset, splits=args.splits, dry_run=args.dry_run)
    from uq.evaluation import support_filter

    return support_filter.main(["--build"] if args.build else ["--report"])


def cmd_al(args) -> int:
    from uq.active_learning import entry

    return entry.main(args.al_command, args.gpu, args.rest)


def cmd_figures(args) -> int:
    import subprocess

    from uq import paths

    script = paths.PAPERS / args.paper / "figures" / "make_all.sh"
    if not script.is_file():
        raise SystemExit(f"no {paths.rel(script)}")
    return subprocess.call(["bash", str(script)])


# -- parser -----------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="uq", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    p = sub.add_parser("list", help="list experiments (id, legacy letter, protocol, parent)")
    p.add_argument("--markdown", action="store_true", help="print a Markdown table")
    p.set_defaults(func=cmd_list)

    p = sub.add_parser("show", help="spec and resolved paths of one experiment")
    p.add_argument("experiment")
    p.set_defaults(func=cmd_show)

    p = sub.add_parser("check", help="validate all experiment and dataset specs")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("status", help="training completeness per experiment and split")
    p.add_argument("experiments", nargs="*")
    p.add_argument("-v", "--verbose", action="store_true", help="list incomplete members")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("train", help="train all missing members on <= 3 GPUs (skips complete ones)")
    p.add_argument("experiments", nargs="+")
    p.add_argument("--gpus", type=int, nargs="+", required=True)
    p.add_argument("--splits", type=_ints, help="e.g. 0-4 or 1,3 (default: all of the spec)")
    p.add_argument("--members", type=_ints, help="e.g. 0-9 (default: all of the spec)")
    p.add_argument("--jobs-per-gpu", type=int, default=1, help="concurrent members per GPU")
    p.add_argument("--threads-per-job", type=int, default=8, help="CPU threads per member (default 8)")
    p.add_argument("--by-experiment", action="store_true",
                   help="finish experiments in the order given instead of interleaving members")
    p.add_argument("--resume", action="store_true",
                   help="continue interrupted members after their latest epoch checkpoint instead of "
                        "refusing them (their results log is backed up and cut to that epoch)")
    p.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("train-slurm", help="write (and submit) 1-GPU SLURM jobs for the missing members")
    p.add_argument("experiments", nargs="+")
    p.add_argument("--splits", type=_ints)
    p.add_argument("--members", type=_ints)
    p.add_argument("--pack", type=int, default=1, help="member chains run in parallel on each GPU (default 1)")
    p.add_argument("--by-experiment", action="store_true")
    p.add_argument("--resume", action="store_true", help="continue interrupted members")
    p.add_argument("--qos", default="normal", help="e.g. normal, boost_qos_dbg, boost_qos_lprod")
    p.add_argument("--time", default="24:00:00", help="walltime per job")
    p.add_argument("--account", help="default $UQ_SLURM_ACCOUNT")
    p.add_argument("--partition", help="default $UQ_SLURM_PARTITION or boost_usr_prod")
    p.add_argument("--out", help="directory for scripts and SLURM output (default runs/_slurm/<time>)")
    p.add_argument("--submit", action="store_true", help="sbatch the scripts (otherwise only write them)")
    p.add_argument("--dry-run", action="store_true", help="print the scripts, write nothing")
    p.add_argument("--after", type=lambda t: [x for x in t.split(",") if x],
                   help="comma-separated job ids to wait for (afterany); keeps children whose parent they train")
    p.add_argument("--after-ok", type=lambda t: [x for x in t.split(",") if x],
                   help="comma-separated job ids that must succeed first (afterok; e.g. prune jobs that free space)")
    p.set_defaults(func=cmd_train_slurm)

    p = sub.add_parser("train-member", help="train one ensemble member (split, member seed)")
    p.add_argument("experiment")
    p.add_argument("split", type=int)
    p.add_argument("member", type=int)
    p.add_argument("--gpu", type=int, default=None,
                   help="GPU index (default: keep the inherited CUDA_VISIBLE_DEVICES, e.g. under SLURM)")
    p.add_argument("--resume", action="store_true", help="continue an interrupted member (see uq train --resume)")
    p.add_argument("--dry-run", action="store_true", help="print the command, run nothing")
    p.set_defaults(func=cmd_train_member)

    p = sub.add_parser("prune-checkpoints", help="delete epoch checkpoints no analysis reads (dry run unless --apply)")
    p.add_argument("experiments", nargs="*", help="default: all")
    p.add_argument("--stride", type=int, default=5, help="keep epochs divisible by this (must divide 5)")
    p.add_argument("--apply", action="store_true", help="really delete (irreversible)")
    p.set_defaults(func=cmd_prune)

    p = sub.add_parser("evaluate", help="GPU evaluation of every split, then the five-split aggregate")
    p.add_argument("experiment")
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--splits", type=_ints, help="default: all splits of the spec")
    p.add_argument("--steps", nargs="+", help="only these step names (see --dry-run output)")
    p.add_argument("--no-aggregate", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="print the commands, run nothing")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("evaluate-slurm", help="write (and submit) one 1-GPU SLURM evaluation job per split")
    p.add_argument("experiments", nargs="+")
    p.add_argument("--splits", type=_ints, help="default: all splits of each spec")
    p.add_argument("--after", type=lambda t: [x for x in t.split(",") if x],
                   help="comma-separated SLURM job ids to wait for (afterok)")
    p.add_argument("--prune", action="store_true",
                   help="add a CPU job that prunes the epoch checkpoints after all evaluations succeeded")
    p.add_argument("--qos", default="normal")
    p.add_argument("--time", default="24:00:00", help="walltime per evaluation job")
    p.add_argument("--account", help="default $UQ_SLURM_ACCOUNT")
    p.add_argument("--partition", help="default $UQ_SLURM_PARTITION or boost_usr_prod")
    p.add_argument("--out", help="directory for scripts and SLURM output (default runs/_slurm/eval-<time>)")
    p.add_argument("--submit", action="store_true", help="sbatch the scripts (otherwise only write them)")
    p.add_argument("--dry-run", action="store_true", help="print the scripts, write nothing")
    p.set_defaults(func=cmd_evaluate_slurm)

    p = sub.add_parser("sync-back", help="copy run and evaluation outputs back from a remote cluster (no overwrite)")
    p.add_argument("experiments", nargs="+")
    p.add_argument("--host", required=True, help="ssh host, e.g. leonardo")
    p.add_argument("--splits", type=_ints)
    p.add_argument("--remote-root", help="repository clone on the remote (default $UQ_REMOTE_ROOT)")
    p.add_argument("--remote-runs", help="UQ_RUNS on the remote (default $UQ_REMOTE_RUNS)")
    p.add_argument("--checkpoints", action="store_true", help="also copy pruned checkpoint folders")
    p.add_argument("--apply", action="store_true", help="really copy (default: dry run)")
    p.set_defaults(func=cmd_sync_back)

    p = sub.add_parser("aggregate", help="rebuild five-split aggregates from per-split outputs (CPU)")
    p.add_argument("experiments", nargs="*", help="default: all")
    p.add_argument("--only", nargs="+", metavar="PREFIX",
                   help="rebuild only aggregates of per-split CSVs with these name prefixes (e.g. ensemble_size_)")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_aggregate)

    p = sub.add_parser("summary", help="headline metrics (mean [95%% CI]) from the tracked aggregates")
    p.add_argument("experiment")
    p.add_argument("--test", help="test role, e.g. id, ood, test, train (default: all)")
    p.add_argument("--cal", action="store_true", help="isotonic-recalibrated variances")
    p.add_argument("--members", type=int, default=None,
                   help="ensemble of M members from ensemble_size_<test>.csv (1 = single model)")
    p.set_defaults(func=cmd_summary)

    p = sub.add_parser("where", help="translate old (pre-2026-09) paths to the current layout")
    p.add_argument("paths", nargs="+")
    p.set_defaults(func=cmd_where)

    p = sub.add_parser("data", help="dataset builders and the training-support geometry cache")
    data_sub = p.add_subparsers(dest="data_command", required=True)
    q = data_sub.add_parser("build", help="generate missing splits (never overwrites)")
    q.add_argument("dataset")
    q.add_argument("--splits", type=_ints)
    q.add_argument("--dry-run", action="store_true")
    q = data_sub.add_parser("support", help="training-support filter bounds / geometry cache")
    q.add_argument("--build", action="store_true", help="(re)build missing geometry caches")
    p.set_defaults(func=cmd_data)

    p = sub.add_parser("al", help="active learning (uq/active_learning); args after GPU pass through")
    p.add_argument("al_command", choices=["run", "run-all"])
    p.add_argument("gpu", type=int)
    p.add_argument("rest", nargs=argparse.REMAINDER)
    p.set_defaults(func=cmd_al)

    p = sub.add_parser("figures", help="rebuild a paper's figures from caches (CPU)")
    p.add_argument("paper", help="e.g. iclr27")
    p.set_defaults(func=cmd_figures)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        return 130
    except BrokenPipeError:  # output piped into `head` & co.
        import os

        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0
    except Exception as error:  # show spec errors briefly, everything else with a traceback
        from uq.registry import SpecError

        if isinstance(error, SpecError):
            print(f"uq: {error}", file=sys.stderr)
            return 2
        raise


if __name__ == "__main__":
    raise SystemExit(main())
