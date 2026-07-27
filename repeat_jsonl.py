#!/usr/bin/env python3
"""Repeat JSONL rows cyclically until the requested row count is reached."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repeat input JSONL rows cyclically to a target row count."
    )
    parser.add_argument("input", type=Path, help="Source JSONL file")
    parser.add_argument("output", type=Path, help="Generated JSONL file")
    parser.add_argument("target_rows", type=int, help="Number of rows to generate")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if args.target_rows <= 0:
        raise ValueError("target_rows must be positive")
    if input_path == output_path:
        raise ValueError("input and output must use different paths")
    if not input_path.is_file():
        raise FileNotFoundError(f"input file not found: {input_path}")

    rows = []
    with input_path.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON at {input_path}:{line_number}: {error}"
                ) from error
            rows.append(line.rstrip(b"\r\n"))

    if not rows:
        raise ValueError(f"input file has no JSONL rows: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_path.parent,
            prefix=f".{output_path.name}.",
            delete=False,
        ) as stream:
            temp_path = Path(stream.name)
            for index in range(args.target_rows):
                stream.write(rows[index % len(rows)])
                stream.write(b"\n")
        os.replace(temp_path, output_path)
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()

    print(
        f"Generated {output_path}: "
        f"{args.target_rows} rows from {len(rows)} source rows"
    )


if __name__ == "__main__":
    main()
