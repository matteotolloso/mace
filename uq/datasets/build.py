"""Generate missing dataset splits from ``data/<dataset>/dataset.yaml``.

``uq data build <dataset> [--splits 0-4] [--dry-run]`` runs the dataset's builder
module (``build.module``) with ``build.args``, where ``{source}``, ``{split_dir}``
and ``{split}`` are filled in per split. A split that already has any of its
files is **never** rebuilt: completed experiments depend on the exact split files.
For builders that read per-split copies of the raw data (water), the raw
``<theory>/input.data`` files are copied into the split directory first.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

from uq import paths, registry


def _existing_files(data, split):
    return [f for fidelity in data.fidelities.values() for f in fidelity.files.values()
            if (data.split_dir(split) / f).exists()]


def _copy_raw(source, split_dir, dry_run):
    for raw in sorted(source.glob("*/*")):
        target = split_dir / raw.relative_to(source)
        if target.exists():
            continue
        print(f"copy {paths.rel(raw)} -> {paths.rel(target)}")
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(raw, target)


def command(dataset, split):
    data = registry.dataset(dataset)
    spec = data.build
    source = paths.DATA / spec["source"]
    fill = {"source": str(source), "split_dir": str(data.split_dir(split)), "split": str(split)}
    return [sys.executable, "-m", spec["module"], *(arg.format(**fill) for arg in spec["args"])]


def build(dataset, splits=None, dry_run=False):
    data = registry.dataset(dataset)
    if not data.build:
        raise SystemExit(f"{dataset}: no build section in data/{dataset}/dataset.yaml")
    status = 0
    for split in splits or data.splits:
        existing = _existing_files(data, split)
        if existing:
            print(f"{dataset} split {split}: exists ({len(existing)} files), never overwritten - skipped")
            continue
        source = paths.DATA / data.build["source"]
        if source.is_dir():
            _copy_raw(source, data.split_dir(split), dry_run)
        cmd = command(dataset, split)
        print(" ".join(cmd))
        if not dry_run:
            status |= subprocess.call(cmd, cwd=paths.ROOT)
    return status
