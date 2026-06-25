#!/usr/bin/env python3
"""Build a manifest of source zip sets for online-teacher distillation."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def natural_key(value: str) -> list[int | str]:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a source-zip manifest.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    zip_paths = sorted(args.input_root.rglob("*.zip"), key=lambda path: natural_key(str(path.relative_to(args.input_root))))
    if args.limit is not None:
        zip_paths = zip_paths[: args.limit]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        handle.write("# Source SLC-PFM zip sets, relative to input root, with .pt suffix for trainer compatibility.\n")
        handle.write(f"# input_root={args.input_root}\n")
        for zip_path in zip_paths:
            handle.write(f"{zip_path.relative_to(args.input_root).with_suffix('.pt').as_posix()}\n")
    print(f"Wrote {len(zip_paths)} source zip set(s) to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
