#!/usr/bin/env python3
"""Build a tile-level manifest for deterministic sampled distillation."""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


IMAGE_SUFFIXES = (".webp", ".png", ".jpg", ".jpeg")


@dataclass(frozen=True)
class TileSampler:
    denominator: int = 1
    offset: int = 0

    def keep(self, rel_path: Path, tile_name: str) -> bool:
        if self.denominator <= 1:
            return True
        key = f"{rel_path.as_posix()}\0{tile_name}".encode("utf-8")
        digest = hashlib.blake2b(key, digest_size=8).digest()
        value = int.from_bytes(digest, byteorder="big", signed=False)
        return value % self.denominator == self.offset


def natural_key(text: str) -> list[object]:
    import re

    return [int(chunk) if chunk.isdigit() else chunk.lower() for chunk in re.split(r"(\d+)", text)]


def image_members(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        return sorted(
            [
                info.filename
                for info in archive.infolist()
                if not info.is_dir() and info.filename.lower().endswith(IMAGE_SUFFIXES)
            ],
            key=natural_key,
        )


def build_manifest(
    *,
    input_root: Path,
    output: Path,
    sampler: TileSampler,
    limit_zips: int | None,
) -> tuple[int, int, int]:
    zip_paths = sorted(input_root.rglob("*.zip"), key=lambda path: natural_key(str(path.relative_to(input_root))))
    output.parent.mkdir(parents=True, exist_ok=True)
    zips_seen = 0
    tiles_seen = 0
    tiles_kept = 0
    with output.open("w", encoding="utf-8") as handle:
        for zip_path in zip_paths:
            if limit_zips is not None and zips_seen >= limit_zips:
                break
            zips_seen += 1
            rel_path = zip_path.relative_to(input_root).with_suffix(".pt")
            try:
                tile_names = image_members(zip_path)
            except zipfile.BadZipFile as exc:
                print(f"[warn] bad zip {zip_path}: {exc}", flush=True)
                continue
            tiles_seen += len(tile_names)
            kept_here = 0
            for tile_name in tile_names:
                if not sampler.keep(rel_path, tile_name):
                    continue
                handle.write(f"{rel_path.as_posix()}\t{tile_name}\n")
                kept_here += 1
            tiles_kept += kept_here
            if zips_seen % 100 == 0:
                print(
                    f"[progress] zips={zips_seen} tiles_seen={tiles_seen} tiles_kept={tiles_kept}",
                    flush=True,
                )
    return zips_seen, tiles_seen, tiles_kept


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True, help="Root containing chunk_* zip folders.")
    parser.add_argument("--output", type=Path, required=True, help="Output text manifest path.")
    parser.add_argument("--sample-rate-denominator", type=int, default=1)
    parser.add_argument("--sample-rate-offset", type=int, default=0)
    parser.add_argument("--limit-zips", type=int, help="Optional smoke-test limit.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.sample_rate_denominator < 1:
        raise ValueError("--sample-rate-denominator must be >= 1")
    if args.sample_rate_offset < 0 or args.sample_rate_offset >= args.sample_rate_denominator:
        raise ValueError("--sample-rate-offset must be in [0, --sample-rate-denominator)")
    sampler = TileSampler(
        denominator=args.sample_rate_denominator,
        offset=args.sample_rate_offset,
    )
    zips_seen, tiles_seen, tiles_kept = build_manifest(
        input_root=args.input_root,
        output=args.output,
        sampler=sampler,
        limit_zips=args.limit_zips,
    )
    print(
        f"[done] wrote {args.output} zips={zips_seen} tiles_seen={tiles_seen} tiles_kept={tiles_kept}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
