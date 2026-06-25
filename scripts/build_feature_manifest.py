#!/usr/bin/env python3
"""Build a manifest of complete feature sets for C-RADIO distillation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator, Sequence


def natural_key(value: str) -> list[int | str]:
    import re

    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def parse_encoder_list(value: str) -> list[str]:
    encoders = [item.strip() for item in value.split(",") if item.strip()]
    if not encoders:
        raise argparse.ArgumentTypeError("at least one encoder is required")
    return encoders


def iter_feature_rel_paths(first_root: Path) -> Iterator[Path]:
    for child in sorted(first_root.iterdir(), key=lambda path: natural_key(path.name)):
        if child.is_file() and child.suffix == ".pt":
            yield child.relative_to(first_root)
        elif child.is_dir():
            paths = sorted(
                child.rglob("*.pt"),
                key=lambda path: natural_key(str(path.relative_to(first_root))),
            )
            for path in paths:
                yield path.relative_to(first_root)


def discover_complete_feature_sets(
    feature_root: Path,
    encoders: Sequence[str],
    token_feature_root: Path | None,
    limit: int | None,
) -> list[Path]:
    first_root = feature_root / encoders[0]
    if not first_root.exists():
        raise FileNotFoundError(f"missing feature directory: {first_root}")

    complete: list[Path] = []
    for rel_path in iter_feature_rel_paths(first_root):
        has_summary = all((feature_root / encoder / rel_path).exists() for encoder in encoders)
        has_tokens = token_feature_root is None or all(
            (token_feature_root / encoder / rel_path).exists() for encoder in encoders
        )
        if has_summary and has_tokens:
            complete.append(rel_path)
            if limit is not None and len(complete) >= limit:
                break
    return complete


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a complete feature-set manifest.")
    parser.add_argument("--feature-root", type=Path, required=True)
    parser.add_argument("--encoders", type=parse_encoder_list, default=["virchow2", "hoptimus1", "uni_v2"])
    parser.add_argument("--token-feature-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rel_paths = discover_complete_feature_sets(
        feature_root=args.feature_root,
        encoders=args.encoders,
        token_feature_root=args.token_feature_root,
        limit=args.limit,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        handle.write("# Complete SLC-PFM feature sets, relative to each encoder feature root.\n")
        handle.write(f"# feature_root={args.feature_root}\n")
        handle.write(f"# encoders={','.join(args.encoders)}\n")
        if args.token_feature_root is not None:
            handle.write(f"# token_feature_root={args.token_feature_root}\n")
        for rel_path in rel_paths:
            handle.write(f"{rel_path.as_posix()}\n")
    print(f"Wrote {len(rel_paths)} complete feature set(s) to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
