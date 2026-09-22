#!/usr/bin/env python3
"""Back up and migrate AL run names plus their path/hash dependency graph."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import re
import tarfile
from contextlib import ExitStack
from pathlib import Path

import yaml

from common import HERE, inventory, load_json, save_json, sha256

HASH = re.compile(r"[0-9a-f]{64}\Z")
SOURCE_NAMES = ("workflow.py", "five_splits.py")


def remap(text, mappings):
    for old, new in mappings.items():
        text = text.replace(old + "/", new + "/")
        if text == old:
            text = new
    return text


def transform_graph(contents, mappings, source_hashes, content_updates=None):
    """Resolve dependent digests before rewriting their parents; never touch model binaries."""
    by_hash = {}
    for path, raw in contents.items():
        by_hash.setdefault(hashlib.sha256(raw).hexdigest(), path)
    resolved, visiting, hashes = {}, set(), dict(source_hashes)

    def rewrite(value):
        if isinstance(value, dict):
            return {remap(key, mappings): rewrite(item) for key, item in value.items()}
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, str):
            if value in hashes:
                return hashes[value]
            if value in by_hash:
                resolve(by_hash[value])
                return hashes[value]
            return remap(value, mappings)
        return value

    def resolve(path):
        if path in resolved:
            return
        if path in visiting:
            raise ValueError(f"Cyclic metadata hashes: {path}")
        visiting.add(path)
        raw = contents[path]
        updated = (content_updates or {}).get(path, raw)
        data = json.loads(updated) if path.endswith(".json") else yaml.safe_load(updated)
        changed = rewrite(data)
        if changed == data:
            result = updated
        elif path.endswith(".json"):
            result = (json.dumps(changed, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
        else:
            result = yaml.safe_dump(changed, sort_keys=False).encode()
        resolved[path] = result
        hashes[hashlib.sha256(raw).hexdigest()] = hashlib.sha256(result).hexdigest()
        visiting.remove(path)

    for path in contents:
        resolve(path)
    return resolved


def hash_references(data):
    if isinstance(data, dict):
        for key, value in data.items():
            if key.startswith("/") and isinstance(value, str) and HASH.fullmatch(value):
                yield key, value
            else:
                yield from hash_references(value)
    elif isinstance(data, list):
        for value in data:
            yield from hash_references(value)


def prepare(root, backup):
    if backup.exists():
        raise ValueError(f"Backup already exists: {backup}")
    mappings = {}
    for path in sorted(root.iterdir()):
        match = re.fullmatch(r"(split_[0-4]|aggregate)_seed_0(_epochs_[0-9]+)?", path.name)
        if match and path.is_dir():
            target = root / (match[1] + (match[2] or ""))
            if target.exists():
                raise ValueError(f"Destination already exists: {target}")
            mappings[str(path)] = str(target)
    if not mappings:
        raise ValueError("No legacy seed-0 run names found.")
    with ExitStack() as stack:
        paths = []
        for name in mappings:
            run = Path(name)
            lock = stack.enter_context((run / ".lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            paths.extend(p for p in run.rglob("*") if p.suffix in (".json", ".yml", ".yaml") and "runtime_cache" not in p.parts)
        print(f"Checking {len(paths)} metadata files before backup...", flush=True)
        expected = {}
        for path in paths:
            if path.suffix == ".json":
                for name, digest in hash_references(load_json(path)):
                    if name in expected and expected[name] != digest:
                        raise ValueError(f"Conflicting cached hashes for {name}")
                    expected[name] = digest
        for name, digest in expected.items():
            if sha256(name) != digest:
                raise ValueError(f"Already stale before migration: {name}")
        backup.mkdir(parents=True)
        sources = [HERE / name for name in SOURCE_NAMES]
        with tarfile.open(backup / "original_metadata.tar.gz", "w:gz") as archive:
            for path in [*paths, *sources]:
                archive.add(path, arcname=str(path.relative_to(HERE)), recursive=False)
        save_json(backup / "plan.json", {"mappings": mappings, "metadata": inventory(paths),
                  "sources": inventory(sources), "validated_references": expected})
        print(f"Backup and migration plan: {backup}", flush=True)


def apply(backup):
    plan = load_json(backup / "plan.json")
    mappings = plan["mappings"]
    if (backup / "complete.json").exists():
        raise ValueError("This migration has already completed.")
    with ExitStack() as stack:
        for old, new in mappings.items():
            existing = Path(old) if Path(old).exists() else Path(new)
            if Path(old).exists() and Path(new).exists():
                raise ValueError(f"Both source and target exist: {old}, {new}")
            lock = stack.enter_context((existing / ".lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        contents = {}
        with tarfile.open(backup / "original_metadata.tar.gz", "r:gz") as archive:
            for name, expected in plan["metadata"].items():
                raw = archive.extractfile(str(Path(name).relative_to(HERE))).read()
                if hashlib.sha256(raw).hexdigest() != expected:
                    raise ValueError(f"Backup corruption: {name}")
                contents[name] = raw
        source_hashes = {old: sha256(name) for name, old in plan["sources"].items()}
        transformed = transform_graph(contents, mappings, source_hashes)
        if not (backup / "started.json").exists():
            for name, expected in plan["metadata"].items():
                if sha256(name) != expected:
                    raise ValueError(f"Metadata changed after backup: {name}")
            save_json(backup / "started.json", {"sources_after": inventory(plan["sources"])})
        elif load_json(backup / "started.json")["sources_after"] != inventory(plan["sources"]):
            raise ValueError("Migration code dependencies changed during a partial migration.")
        for old, new in mappings.items():
            if Path(old).exists():
                Path(old).rename(new)
        changed = 0
        for old_path, raw in transformed.items():
            path = Path(remap(old_path, mappings))
            if path.read_bytes() != raw:
                temporary = path.with_suffix(path.suffix + ".migration_tmp")
                temporary.write_bytes(raw)
                temporary.replace(path)
                changed += 1
        expected = {}
        for name, raw in transformed.items():
            if name.endswith(".json"):
                for path, digest in hash_references(json.loads(raw)):
                    if path in expected and expected[path] != digest:
                        raise ValueError(f"Conflicting migrated hashes for {path}")
                    expected[path] = digest
        print(f"Verifying {len(expected)} migrated references...", flush=True)
        for name, digest in expected.items():
            if sha256(name) != digest:
                raise ValueError(f"Invalid migrated reference: {name}")
        save_json(backup / "complete.json", {"mappings": mappings, "rewritten_metadata_files": changed,
                  "verified_references": len(expected), "sources_after": inventory(plan["sources"]),
                  "note": "Only metadata paths/digests changed; no model, prediction, selection or metric values changed."})
        print(f"Migrated {len(mappings)} folders; rewrote {changed} metadata files.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Apply the prepared plan after code edits")
    options = parser.parse_args()
    backup = HERE / "runs" / "migration_backup"
    if options.apply:
        apply(backup)
    else:
        prepare(HERE / "runs", backup)
