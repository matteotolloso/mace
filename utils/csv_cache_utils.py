from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple


def parse_float(value: Any) -> float:
    if value is None or value == "":
        return float("nan")
    return float(value)


def parse_int(value: Any) -> int:
    if value is None or value == "":
        raise ValueError("Expected an integer value, found an empty string.")
    return int(value)


def parse_str(value: Any) -> str:
    return "" if value is None else str(value)


def read_csv_rows(path: Path) -> Tuple[List[str], List[Dict[str, Any]]]:
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV file {path} has no header.")
        return list(reader.fieldnames), list(reader)


def _parse_row(
    row: Dict[str, Any],
    *,
    path: Path,
    row_number: int,
    field_parsers: Dict[str, Callable[[Any], Any]],
) -> Dict[str, Any]:
    parsed: Dict[str, Any] = {}
    for key, value in row.items():
        parser = field_parsers.get(key, parse_str)
        try:
            parsed[key] = parser(value)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to parse column '{key}' on line {row_number} of {path}: {exc}"
            ) from exc
    return parsed


def parse_csv_rows(
    path: Path,
    *,
    fieldnames: Sequence[str],
    rows: List[Dict[str, Any]],
    required_fields: Iterable[str],
    field_parsers: Dict[str, Callable[[Any], Any]],
) -> List[Dict[str, Any]]:
    missing = [field for field in required_fields if field not in fieldnames]
    if missing:
        raise RuntimeError(f"CSV file {path} is missing required columns: {missing}")

    return [
        _parse_row(row, path=path, row_number=row_number, field_parsers=field_parsers)
        for row_number, row in enumerate(rows, start=2)
    ]


def write_csv_rows(path: Path, fieldnames: Sequence[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def read_typed_csv(
    path: Path,
    *,
    required_fields: Iterable[str],
    field_parsers: Dict[str, Callable[[str], Any]],
) -> List[Dict[str, Any]]:
    fieldnames, rows = read_csv_rows(path)
    return parse_csv_rows(
        path,
        fieldnames=fieldnames,
        rows=rows,
        required_fields=required_fields,
        field_parsers=field_parsers,
    )


def load_cached_csv_rows(
    path: Path,
    *,
    required_fields: Iterable[str],
    field_parsers: Dict[str, Callable[[str], Any]],
    logger: Optional[logging.Logger] = None,
    label: str = "plot",
    key_fields: Optional[Sequence[str]] = None,
    compute_missing_rows: Optional[Callable[[], List[Dict[str, Any]]]] = None,
) -> Optional[List[Dict[str, Any]]]:
    if not path.exists():
        return None

    try:
        fieldnames, raw_rows = read_csv_rows(path)
        missing = [field for field in required_fields if field not in fieldnames]
        if missing:
            if compute_missing_rows is None or key_fields is None:
                raise RuntimeError(f"CSV file {path} is missing required columns: {missing}")
            if any(field not in fieldnames for field in key_fields):
                raise RuntimeError(
                    f"CSV file {path} is missing one or more key columns needed for cache upgrade: {list(key_fields)}"
                )

            if logger is not None:
                logger.info(
                    "Cached %s CSV %s is missing columns %s; computing only the missing columns.",
                    label,
                    path,
                    missing,
                )

            computed_rows = compute_missing_rows()
            computed_map = {
                tuple(parse_str(row.get(field)) for field in key_fields): row
                for row in computed_rows
            }
            for row in raw_rows:
                key = tuple(parse_str(row.get(field)) for field in key_fields)
                computed = computed_map.get(key)
                if computed is None:
                    raise RuntimeError(
                        f"Could not find matching computed row for key {key} while upgrading cached CSV {path}."
                    )
                for field in missing:
                    row[field] = computed.get(field, "")

            updated_fieldnames = list(fieldnames) + [field for field in missing if field not in fieldnames]
            write_csv_rows(path, updated_fieldnames, raw_rows)
            fieldnames = updated_fieldnames

        rows = parse_csv_rows(
            path,
            fieldnames=fieldnames,
            rows=raw_rows,
            required_fields=required_fields,
            field_parsers=field_parsers,
        )
    except Exception as exc:
        if logger is not None:
            logger.warning(
                "Cached %s CSV %s could not be used (%s). Recomputing.",
                label,
                path,
                exc,
            )
        return None

    if logger is not None:
        logger.info("Using cached %s CSV from %s", label, path)
    return rows
