#!/usr/bin/env python3
"""One-off migration of the repository layout (2026-09). Kept for provenance.

    python docs/restructure/migrate_layout.py plan     # print every step, change nothing
    python docs/restructure/migrate_layout.py apply    # run it; writes runs/_migration/manifest.json
    python docs/restructure/migrate_layout.py verify   # re-count the moved directories
    python docs/restructure/migrate_layout.py revert   # move everything back (from the manifest)

Moves are renames on one filesystem: nothing is copied or rewritten, except the
seven water n2p2 source files that are *copied* to data/raw/water-n2p2/. Tracked
paths move with `git mv` (history kept, untracked files inside move along);
untracked heavy directories (checkpoints, run logs, datasets, W&B, active-learning
runs) move with os.rename and their file counts are recorded and re-checked.
`apply` refuses to start if any destination exists or any source is missing.
Obsolete launchers are removed with `git rm` (they are in the tag iclr27-submission).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from uq.legacy import LETTERS  # noqa: E402

MANIFEST = ROOT / "runs" / "_migration" / "manifest.json"
SPLITS = range(5)
RUN_KINDS = ("checkpoints", "models", "logs", "results")
GEOMETRY_FILES = ("cc_train", "cc_val", "dft_train", "dft_val", "cc_test_id", "cc_test_ood", "dft_test_id", "dft_test_ood")
WATER_THEORIES = ("afqmc", "blyp", "ccsd", "ccsdt", "hf", "revpbe-d3")


def steps():
    """Ordered (kind, src, dst): kind in git (git mv), fs (os.rename), copy, rm (git rm), rmdir."""
    out = []
    # datasets (tracked metadata + ignored xyz move together)
    for s in SPLITS:
        out.append(("git", f"dataset/ani1x_system_split_{s}", f"data/ani1x-system/split_{s}"))
        out.append(("git", f"dataset/ani1x_energy_split_{s}", f"data/ani1x-energy/split_{s}"))
        out.append(("git", f"dataset/water_{s}", f"data/water-random/split_{s}"))
    out.append(("fs", "dataset/ani1x-release.h5", "data/raw/ani1x-release.h5"))
    out.append(("fs", "dataset/ani1x-processed", "data/legacy/ani1x-processed"))
    for theory in WATER_THEORIES:
        out.append(("copy", f"data/water-random/split_0/{theory}/input.data", f"data/raw/water-n2p2/{theory}/input.data"))
    out.append(("copy", "data/water-random/split_0/afqmc/AFQMC-energies-stderr.txt",
                "data/raw/water-n2p2/afqmc/AFQMC-energies-stderr.txt"))
    # support-filter geometry cache -> next to the split files
    for kind in ("system", "energy"):
        for s in SPLITS:
            for name in GEOMETRY_FILES:
                out.append(("git", f"eval/cache/geometry/{kind}_{s}_{name}.npz",
                            f"data/ani1x-{kind}/split_{s}/geometry/{name}.npz"))
    # evaluation outputs (tracked)
    for letter, exp in LETTERS.items():
        out.append(("git", f"experiment_{letter}/evaluation", f"experiments/{exp}/evaluation"))
        for s in SPLITS:
            out.append(("git", f"experiments/{exp}/evaluation/cache/split_{s}", f"experiments/{exp}/evaluation/split_{s}"))
        out.append(("rmdir", f"experiments/{exp}/evaluation/cache", ""))
    # per-epoch per-configuration predictions (paper Figs. 3, 5) -> their experiment
    for letter in ("C", "D", "F"):
        for s in SPLITS:
            base = f"experiments/{LETTERS[letter]}/evaluation/split_{s}/epoch_predictions_cc_test_ood"
            out.append(("git", f"papers/iclr27/figures/_cache/epochs/{letter}_{s}.npz", f"{base}.npz"))
            out.append(("fs", f"papers/iclr27/figures/_cache/epochs/{letter}_{s}.log", f"{base}.log"))
    out += [("rmdir", "papers/iclr27/figures/_cache/epochs", ""), ("rmdir", "papers/iclr27/figures/_cache", "")]
    # heavy run artifacts (untracked)
    for letter, exp in LETTERS.items():
        for s in SPLITS:
            for kind in RUN_KINDS:
                out.append(("fs", f"experiment_{letter}/{kind}_{s}", f"runs/{exp}/split_{s}/{kind}"))
    out.append(("fs", "active_learning/ani_energy/runs", "runs/active-learning/ani1x-energy"))
    out.append(("fs", "wandb", "runs/wandb"))
    out.append(("fs", "experiments/runs", "runs/_logs"))
    # obsolete launchers (replaced by the uq CLI; kept in the tag)
    for letter in LETTERS:
        out.append(("rm", f"experiment_{letter}/train_{letter}.sh", ""))
        out.append(("rm", f"experiment_{letter}/eval_{letter}.sh", ""))
    for path in ("eval.sh", "eval_water.sh", "check_experiments.sh", "utils/train_split_member.sh",
                 "eval/split_cache.sh", "experiments/scripts/runner",
                 "experiments/scripts/eval_water_after_training.sh", "dataset/.gitignore"):
        out.append(("rm", path, ""))
    # emptied legacy directories (bytecode caches inside are deleted)
    for path in (*(f"experiment_{x}" for x in LETTERS), "eval/cache/geometry", "eval/cache", "eval", "utils",
                 "dataset", "active_learning/ani_energy", "active_learning", "experiments/scripts"):
        out.append(("rmdir", path, ""))
    return out


def count_files(path: Path) -> int:
    if path.is_file() or path.is_symlink():
        return 1
    return sum(len(files) + sum(1 for d in dirs if (Path(root) / d).is_symlink()) for root, dirs, files in os.walk(path))


def git(*args):
    subprocess.run(["git", *args], cwd=ROOT, check=True)


def preflight(plan, start=0):
    """Problems with the steps from ``start`` on (earlier steps are already applied)."""
    problems = []
    moved_into = {dst for kind, _, dst in plan[:start] if kind in ("git", "fs", "copy")}
    for kind, src, dst in plan[start:]:
        if kind in ("git", "fs", "copy"):
            source_exists = (ROOT / src).exists() or any(src.startswith(d + "/") for d in moved_into)
            if not source_exists:
                problems.append(f"missing source {src}")
            if (ROOT / dst).exists():
                problems.append(f"destination exists {dst}")
            moved_into.add(dst)
        elif kind == "rm" and not (ROOT / src).exists():
            problems.append(f"missing file to remove {src}")
    return problems


def apply(plan):
    """Run the plan; resumes after the last step recorded in the manifest."""
    manifest = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"steps": []}
    start = len(manifest["steps"])
    if [(r["kind"], r["src"], r["dst"]) for r in manifest["steps"]] != [tuple(step) for step in plan[:start]]:
        raise SystemExit("manifest does not match the plan; refusing to resume")
    problems = preflight(plan, start)
    if problems:
        raise SystemExit("refusing to start:\n  " + "\n  ".join(problems))
    if start:
        print(f"resuming after step {start}")
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    for kind, src, dst in plan[start:]:
        s, d = ROOT / src, ROOT / dst
        record = {"kind": kind, "src": src, "dst": dst}
        if kind in ("git", "fs", "copy"):
            record["files"] = count_files(s)
            d.parent.mkdir(parents=True, exist_ok=True)
        if kind == "git":
            git("mv", src, dst)
        elif kind == "fs":
            os.rename(s, d)
        elif kind == "copy":
            shutil.copy2(s, d)
        elif kind == "rm":
            git("rm", "-q", src)
        elif kind == "rmdir":
            for cache in s.glob("**/__pycache__"):
                shutil.rmtree(cache)
            if s.is_dir():
                if any(s.iterdir()):
                    raise SystemExit(f"{src} is not empty after the moves: {sorted(p.name for p in s.iterdir())}")
                s.rmdir()
        if kind in ("git", "fs", "copy"):
            record["files_after"] = count_files(d)
            if record["files_after"] != record["files"]:
                raise SystemExit(f"file count changed for {src} -> {dst}: {record['files']} -> {record['files_after']}")
        manifest["steps"].append(record)
        MANIFEST.write_text(json.dumps(manifest, indent=1))
    copies = [dst for kind, _, dst in plan if kind == "copy" and dst.endswith("input.data")]
    git("add", *copies)
    print(f"applied {len(plan)} steps; manifest {MANIFEST.relative_to(ROOT)}")


def verify():
    """Re-count every destination; files that later steps moved in or out are accounted for."""
    manifest = json.loads(MANIFEST.read_text())
    records = manifest["steps"]
    inside = lambda path, root: path == root or path.startswith(root + "/")  # noqa: E731
    bad = 0
    for i, record in enumerate(records):
        if record["kind"] not in ("git", "fs", "copy"):
            continue
        expected = record["files"]
        for later in records[i + 1:]:
            if later["kind"] not in ("git", "fs", "copy"):
                continue
            if inside(later["dst"], record["dst"]) and not inside(later["src"], record["dst"]):
                expected += later["files"]
            if inside(later["src"], record["dst"]) and not inside(later["dst"], record["dst"]) \
                    and later["kind"] != "copy":
                expected -= later["files"]
        now = count_files(ROOT / record["dst"])
        if now != expected:
            bad += 1
            print(f"MISMATCH {record['dst']}: {now} files, expected {expected}")
    moved = sum(r["files"] for r in records if r["kind"] in ("git", "fs"))
    print(f"{len(records)} steps, {moved} files moved; " + ("all counts match" if not bad else f"{bad} mismatches"))
    return 1 if bad else 0


def revert():
    manifest = json.loads(MANIFEST.read_text())
    for record in reversed(manifest["steps"]):
        kind, src, dst = record["kind"], record["src"], record["dst"]
        if kind == "git":
            (ROOT / src).parent.mkdir(parents=True, exist_ok=True)
            git("mv", dst, src)
        elif kind == "fs":
            (ROOT / src).parent.mkdir(parents=True, exist_ok=True)
            os.rename(ROOT / dst, ROOT / src)
        elif kind == "copy":
            git("rm", "-q", "--cached", "--ignore-unmatch", dst)
            (ROOT / dst).unlink()
        elif kind == "rm":
            git("checkout", "HEAD", "--", src)
    print("reverted; review `git status`")


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "plan"
    plan = steps()
    if command == "plan":
        for kind, src, dst in plan:
            print(f"{kind:5s} {src}" + (f" -> {dst}" if dst else ""))
        problems = preflight(plan)
        print(f"{len(plan)} steps; preflight: " + ("OK" if not problems else "\n  " + "\n  ".join(problems)))
    elif command == "apply":
        apply(plan)
    elif command == "verify":
        sys.exit(verify())
    elif command == "revert":
        revert()
    else:
        raise SystemExit(__doc__)
