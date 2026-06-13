#!/usr/bin/env python3
"""Extract tile features from zip archives of WebP images.

The expected input layout is:

    input_root/
      chunk_8/
        chunk8_id000_....zip
          1.webp
          2.webp
          ...

Outputs are written one `.pt` file per input zip and encoder.
"""

from __future__ import annotations

import argparse
import io
import math
import os
import re
import sys
import time
import zipfile
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


SUPPORTED_ENCODERS = ("virchow2", "hoptimus1", "uni_v2")


@dataclass(frozen=True)
class EncoderSpec:
    name: str
    hf_model: str
    prefix_tokens: int


ENCODER_SPECS = {
    "virchow2": EncoderSpec(
        name="virchow2",
        hf_model="hf-hub:paige-ai/Virchow2",
        prefix_tokens=5,  # cls token + 4 register tokens
    ),
    "hoptimus1": EncoderSpec(
        name="hoptimus1",
        hf_model="hf-hub:bioptimus/H-optimus-1",
        prefix_tokens=1,  # cls token
    ),
    "uni_v2": EncoderSpec(
        name="uni_v2",
        hf_model="hf-hub:MahmoodLab/UNI2-h",
        prefix_tokens=9,  # cls token + 8 register tokens
    ),
}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def natural_key(value: str) -> list[int | str]:
    parts = re.split(r"(\d+)", value)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def parse_encoder_list(value: str) -> list[str]:
    if value == "all":
        return list(SUPPORTED_ENCODERS)
    encoders = [item.strip() for item in value.split(",") if item.strip()]
    unknown = sorted(set(encoders).difference(SUPPORTED_ENCODERS))
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown encoder(s): {', '.join(unknown)}; choose from {', '.join(SUPPORTED_ENCODERS)}"
        )
    return encoders


def find_zip_files(input_root: Path, chunks: Sequence[str] | None) -> list[Path]:
    if chunks:
        roots = [input_root / chunk for chunk in chunks]
    else:
        roots = [input_root]

    zip_files: list[Path] = []
    for root in roots:
        if not root.exists():
            print(f"[warn] input path does not exist: {root}", file=sys.stderr)
            continue
        zip_files.extend(path for path in root.rglob("*.zip") if path.is_file())

    return sorted(zip_files, key=lambda path: natural_key(str(path.relative_to(input_root))))


def zip_image_members(zip_path: Path) -> list[str]:
    with zipfile.ZipFile(zip_path) as archive:
        members = [
            info.filename
            for info in archive.infolist()
            if not info.is_dir() and info.filename.lower().endswith((".webp", ".png", ".jpg", ".jpeg"))
        ]
    return sorted(members, key=natural_key)


def load_image_from_zip(archive: zipfile.ZipFile, member: str) -> Any:
    from PIL import Image

    with archive.open(member) as handle:
        image_bytes = handle.read()
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def load_encoder(spec: EncoderSpec, device: str, tile_size: int | None) -> tuple[Any, Any]:
    import timm
    import torch
    from timm.data import resolve_data_config
    from timm.data.transforms_factory import create_transform
    from timm.layers import SwiGLUPacked
    from torchvision import transforms

    if spec.name == "virchow2":
        model = timm.create_model(
            spec.hf_model,
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    elif spec.name == "hoptimus1":
        model = timm.create_model(
            spec.hf_model,
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=False,
        )
        transform_steps: list[Any] = []
        if tile_size:
            transform_steps.append(transforms.Resize((tile_size, tile_size)))
        transform_steps.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=(0.707223, 0.578729, 0.703617),
                    std=(0.211883, 0.230117, 0.177517),
                ),
            ]
        )
        transform = transforms.Compose(transform_steps)
    elif spec.name == "uni_v2":
        timm_kwargs = {
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": timm.layers.SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        }
        model = timm.create_model(spec.hf_model, pretrained=True, **timm_kwargs)
        transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    else:
        raise ValueError(f"unsupported encoder: {spec.name}")

    model.to(device)
    model.eval()
    return model, transform


def select_device(requested: str) -> str:
    if requested != "auto":
        return requested

    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def autocast_context(device: str, amp_dtype: str):
    if amp_dtype == "off" or device == "cpu":
        return nullcontext()

    import torch

    if device.startswith("cuda"):
        dtype = torch.float16 if amp_dtype == "fp16" else torch.bfloat16
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


def convert_save_dtype(tensor: Any, save_dtype: str) -> Any:
    if save_dtype == "float16":
        return tensor.half()
    if save_dtype == "bfloat16":
        return tensor.bfloat16()
    if save_dtype == "float32":
        return tensor.float()
    return tensor


def tensor_from_output(output: Any) -> Any:
    if isinstance(output, dict):
        for key in ("x", "features", "last_hidden_state"):
            if key in output:
                return output[key]
        raise ValueError(f"model returned a dict without a recognized feature key: {sorted(output)}")
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


def maybe_patch_map(tokens: Any, prefix_tokens: int) -> tuple[Any | None, tuple[int, int] | None]:
    if tokens.ndim != 3 or tokens.shape[1] <= prefix_tokens:
        return None, None

    patch_tokens = tokens[:, prefix_tokens:, :]
    side = int(math.sqrt(patch_tokens.shape[1]))
    if side * side != patch_tokens.shape[1]:
        return patch_tokens, None
    return patch_tokens.reshape(patch_tokens.shape[0], side, side, patch_tokens.shape[2]), (side, side)


def run_batch(
    model: Any,
    batch: Any,
    spec: EncoderSpec,
    device: str,
    amp_dtype: str,
    include_token_maps: bool,
) -> tuple[Any, Any | None, tuple[int, int] | None]:
    import torch

    with torch.inference_mode(), autocast_context(device, amp_dtype):
        output = tensor_from_output(model(batch))

        if output.ndim == 2:
            embeddings = output
            token_source = None
            if include_token_maps and hasattr(model, "forward_features"):
                token_source = tensor_from_output(model.forward_features(batch))
        elif output.ndim == 3:
            embeddings = output[:, 0, :]
            token_source = output
        else:
            raise ValueError(f"unexpected model output shape: {tuple(output.shape)}")

    token_maps = None
    grid_size = None
    if include_token_maps and token_source is not None:
        token_maps, grid_size = maybe_patch_map(token_source, spec.prefix_tokens)

    return embeddings, token_maps, grid_size


def output_path_for(zip_path: Path, input_root: Path, output_root: Path, encoder: str) -> Path:
    relative = zip_path.relative_to(input_root).with_suffix(".pt")
    return output_root / encoder / relative


def save_payload(payload: dict[str, Any], output_path: Path) -> None:
    import torch

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    log(f"[save] writing {output_path}")
    torch.save(payload, tmp_path)
    tmp_path.replace(output_path)
    log(f"[save] finished {output_path}")


def extract_zip_features(
    zip_path: Path,
    input_root: Path,
    output_root: Path,
    spec: EncoderSpec,
    model: Any,
    transform: Any,
    device: str,
    batch_size: int,
    limit_tiles: int | None,
    amp_dtype: str,
    save_dtype: str,
    include_token_maps: bool,
    overwrite: bool,
    fail_fast: bool,
    log_every_batches: int,
) -> bool:
    import torch

    out_path = output_path_for(zip_path, input_root, output_root, spec.name)
    if out_path.exists() and not overwrite:
        log(f"[skip] {spec.name}: {zip_path} -> {out_path}")
        return False

    started_at = time.monotonic()
    log(f"[zip-start] {spec.name}: {zip_path}")
    members = zip_image_members(zip_path)
    if limit_tiles is not None:
        members = members[:limit_tiles]
    log(f"[zip-members] {spec.name}: {zip_path} has {len(members)} image tile(s)")

    tile_names: list[str] = []
    embeddings_out: list[Any] = []
    token_maps_out: list[Any] = []
    errors: list[dict[str, str]] = []
    grid_size: tuple[int, int] | None = None
    pending_images: list[Any] = []
    pending_names: list[str] = []
    batches_done = 0

    def flush_batch() -> None:
        nonlocal batches_done, grid_size
        if not pending_images:
            return
        batch_tile_count = len(pending_images)
        batch = torch.stack(pending_images, dim=0).to(device, non_blocking=True)
        embeddings, token_maps, batch_grid_size = run_batch(
            model=model,
            batch=batch,
            spec=spec,
            device=device,
            amp_dtype=amp_dtype,
            include_token_maps=include_token_maps,
        )
        embeddings_out.append(convert_save_dtype(embeddings.detach().cpu(), save_dtype))
        if token_maps is not None:
            token_maps_out.append(convert_save_dtype(token_maps.detach().cpu(), save_dtype))
            grid_size = batch_grid_size
        tile_names.extend(pending_names)
        batches_done += 1
        if log_every_batches > 0 and batches_done % log_every_batches == 0:
            log(
                f"[batch] {spec.name}: {zip_path.name} "
                f"batch={batches_done} tiles_done={len(tile_names)} last_batch={batch_tile_count}"
            )
        pending_images.clear()
        pending_names.clear()

    with zipfile.ZipFile(zip_path) as archive:
        for member in members:
            try:
                image = load_image_from_zip(archive, member)
                pending_images.append(transform(image))
                pending_names.append(member)
                if len(pending_images) >= batch_size:
                    flush_batch()
            except Exception as exc:  # noqa: BLE001 - continue past bad tiles unless requested.
                if fail_fast:
                    raise
                errors.append({"tile": member, "error": repr(exc)})
                print(f"[warn] failed tile {zip_path}!{member}: {exc}", file=sys.stderr, flush=True)

    flush_batch()

    if embeddings_out:
        features = torch.cat(embeddings_out, dim=0)
    else:
        features = torch.empty((0, 0), dtype=torch.float32)

    payload: dict[str, Any] = {
        "encoder": spec.name,
        "source_zip": str(zip_path),
        "tile_names": tile_names,
        "features": features,
        "feature_kind": "cls_embedding",
        "errors": errors,
        "created_unix_time": time.time(),
    }
    if token_maps_out:
        payload["token_maps"] = torch.cat(token_maps_out, dim=0)
        payload["token_map_kind"] = "patch_tokens"
        payload["token_grid_size"] = grid_size

    save_payload(payload, out_path)
    elapsed = time.monotonic() - started_at
    log(f"[done] {spec.name}: {zip_path} -> {out_path} ({len(tile_names)} tiles, {elapsed:.1f}s)")
    return True


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract feature tensors from zipped WebP tile archives."
    )
    parser.add_argument("--input-root", type=Path, required=True, help="Root containing chunk_* folders.")
    parser.add_argument("--output-root", type=Path, required=True, help="Folder for extracted features.")
    parser.add_argument(
        "--encoders",
        type=parse_encoder_list,
        default=list(SUPPORTED_ENCODERS),
        help="Comma-separated encoders or 'all'. Choices: virchow2,hoptimus1,uni_v2.",
    )
    parser.add_argument("--chunks", nargs="*", help="Optional chunk folder names, e.g. chunk_8 chunk_9.")
    parser.add_argument("--batch-size", type=positive_int, default=64)
    parser.add_argument("--limit-zips", type=positive_int, help="Process only the first N zip files.")
    parser.add_argument("--limit-tiles", type=positive_int, help="Process only the first N tiles per zip.")
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:0, mps, or cpu.")
    parser.add_argument("--amp-dtype", choices=("fp16", "bf16", "off"), default="fp16")
    parser.add_argument("--save-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument(
        "--tile-size",
        type=non_negative_int,
        default=224,
        help="Resize H-optimus-1 tiles to this size. Use 0 to disable resizing.",
    )
    parser.add_argument(
        "--include-token-maps",
        action="store_true",
        help="Also save patch-token feature maps when available. This can greatly increase output size.",
    )
    parser.add_argument("--hf-cache-dir", type=Path, help="Set HF_HOME before loading gated model weights.")
    parser.add_argument(
        "--log-every-batches",
        type=non_negative_int,
        default=4,
        help="Print progress every N inference batches within a zip. Use 0 to disable batch progress.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Regenerate outputs that already exist.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first corrupt/unreadable tile.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    args = build_parser().parse_args(argv)

    if args.hf_cache_dir:
        os.environ["HF_HOME"] = str(args.hf_cache_dir)
        args.hf_cache_dir.mkdir(parents=True, exist_ok=True)

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    tile_size = None if args.tile_size == 0 else args.tile_size

    zip_files = find_zip_files(input_root, args.chunks)
    if args.limit_zips is not None:
        zip_files = zip_files[: args.limit_zips]
    if not zip_files:
        print(f"No zip files found under {input_root}", file=sys.stderr)
        return 2

    device = select_device(args.device)
    log(f"Using device: {device}")
    log(f"Input root: {input_root}")
    log(f"Output root: {output_root}")
    log(f"HF_HOME: {os.environ.get('HF_HOME', '<default>')}")
    log(f"Found {len(zip_files)} zip file(s).")

    for encoder_name in args.encoders:
        spec = ENCODER_SPECS[encoder_name]
        encoder_started_at = time.monotonic()
        log(f"[encoder-load-start] {encoder_name} ({spec.hf_model})")
        model, transform = load_encoder(spec, device=device, tile_size=tile_size)
        log(f"[encoder-load-done] {encoder_name} loaded in {time.monotonic() - encoder_started_at:.1f}s")
        for zip_path in zip_files:
            extract_zip_features(
                zip_path=zip_path,
                input_root=input_root,
                output_root=output_root,
                spec=spec,
                model=model,
                transform=transform,
                device=device,
                batch_size=args.batch_size,
                limit_tiles=args.limit_tiles,
                amp_dtype=args.amp_dtype,
                save_dtype=args.save_dtype,
                include_token_maps=args.include_token_maps,
                overwrite=args.overwrite,
                fail_fast=args.fail_fast,
                log_every_batches=args.log_every_batches,
            )
        del model
        try:
            import torch

            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
