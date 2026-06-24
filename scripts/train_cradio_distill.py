#!/usr/bin/env python3
"""Continue C-RADIO distillation from extracted Virchow2, H-optimus-1, and UNI2 features.

This uses the actual NVlabs/RADIO C-RADIO checkpoint as the student backbone:

* one checkpoint-initialized C-RADIO student backbone
* one summary projection head per teacher embedding space
* one dense token projection head per teacher token space, when dense loss is enabled
* balanced per-teacher summary losses plus cached or on-the-fly dense losses

The extracted summary feature files are used as teacher targets, while input
images are streamed from the original zipped tile archives. Dense patch-token
targets can either be loaded from cached token maps or computed on the fly by
frozen teacher encoders.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
import shutil
import sys
import time
import zipfile
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

from torch.utils.data import IterableDataset


DEFAULT_ENCODERS = ("virchow2", "hoptimus1", "uni_v2")


@dataclass(frozen=True)
class TileSampler:
    denominator: int = 1
    offset: int = 0

    @property
    def enabled(self) -> bool:
        return self.denominator > 1

    def keep(self, rel_path: Path, tile_name: str) -> bool:
        if not self.enabled:
            return True
        key = f"{rel_path.as_posix()}\0{tile_name}".encode("utf-8")
        digest = hashlib.blake2b(key, digest_size=8).digest()
        value = int.from_bytes(digest, byteorder="big", signed=False)
        return value % self.denominator == self.offset

    def describe(self) -> str:
        if not self.enabled:
            return "full dataset"
        return f"deterministic tile sample offset={self.offset}/{self.denominator}"


@dataclass(frozen=True)
class OnlineTeacherSpec:
    name: str
    hf_model: str
    prefix_tokens: int


ONLINE_TEACHER_SPECS = {
    "virchow2": OnlineTeacherSpec(
        name="virchow2",
        hf_model="hf-hub:paige-ai/Virchow2",
        prefix_tokens=5,
    ),
    "hoptimus1": OnlineTeacherSpec(
        name="hoptimus1",
        hf_model="hf-hub:bioptimus/H-optimus-1",
        prefix_tokens=1,
    ),
    "uni_v2": OnlineTeacherSpec(
        name="uni_v2",
        hf_model="hf-hub:MahmoodLab/UNI2-h",
        prefix_tokens=9,
    ),
}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def parse_tag_list(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def format_duration(seconds: float) -> str:
    seconds = max(float(seconds), 0.0)
    minutes, sec = divmod(int(round(seconds)), 60)
    hours, minute = divmod(minutes, 60)
    days, hour = divmod(hours, 24)
    if days:
        return f"{days}d{hour:02d}h{minute:02d}m{sec:02d}s"
    if hours:
        return f"{hour}h{minute:02d}m{sec:02d}s"
    if minutes:
        return f"{minute}m{sec:02d}s"
    return f"{sec}s"


def natural_key(value: str) -> list[int | str]:
    import re

    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def parse_encoder_list(value: str) -> list[str]:
    encoders = [item.strip() for item in value.split(",") if item.strip()]
    if not encoders:
        raise argparse.ArgumentTypeError("at least one encoder is required")
    return encoders


def load_payload(path: Path) -> dict[str, Any]:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def load_feature_tensors(
    path: Path,
    *,
    require_features: bool,
    require_token_maps: bool,
) -> tuple[list[str], Any | None, Any | None]:
    payload = load_payload(path)
    if require_features and "features" not in payload:
        raise KeyError(f"{path} does not contain a 'features' tensor")
    features = payload.get("features")
    token_maps = payload.get("token_maps")
    if require_token_maps and token_maps is None:
        raise KeyError(f"{path} does not contain 'token_maps'; re-run extraction with --include-token-maps")
    return (
        list(payload["tile_names"]),
        features.float() if features is not None else None,
        token_maps.float() if token_maps is not None else None,
    )


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


def discover_feature_sets(
    feature_root: Path,
    encoders: Sequence[str],
    token_feature_root: Path | None = None,
    limit: int | None = None,
) -> list[Path]:
    first_root = feature_root / encoders[0]
    if not first_root.exists():
        raise FileNotFoundError(f"missing feature directory: {first_root}")

    complete = []
    for rel_path in iter_feature_rel_paths(first_root):
        has_summary = all((feature_root / encoder / rel_path).exists() for encoder in encoders)
        has_tokens = token_feature_root is None or all((token_feature_root / encoder / rel_path).exists() for encoder in encoders)
        if has_summary and has_tokens:
            complete.append(rel_path)
            if limit is not None and len(complete) >= limit:
                break
    return complete


def sampled_tile_indices(tile_sampler: TileSampler, rel_path: Path, tile_names: Sequence[str]) -> list[int]:
    if not tile_sampler.enabled:
        return list(range(len(tile_names)))
    return [index for index, tile_name in enumerate(tile_names) if tile_sampler.keep(rel_path, tile_name)]


def image_from_zip(archive: zipfile.ZipFile, member: str) -> Any:
    from PIL import Image

    with archive.open(member) as handle:
        image_bytes = handle.read()
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def build_transform(tile_size: int) -> Any:
    from torchvision import transforms

    return transforms.Compose(
        [
            transforms.Resize((tile_size, tile_size)),
            transforms.ToTensor(),
        ]
    )


def tensor_from_model_output(output: Any) -> Any:
    if isinstance(output, dict):
        for key in ("x", "features", "last_hidden_state"):
            if key in output:
                return output[key]
        raise ValueError(f"model returned a dict without a recognized feature key: {sorted(output)}")
    if isinstance(output, (tuple, list)):
        return output[0]
    return output


@dataclass
class OnlineTeacher:
    name: str
    model: Any
    mean: Any
    std: Any
    prefix_tokens: int
    token_dim: int
    token_grid_size: tuple[int, int]


def normalize_for_teacher(images: Any, teacher: OnlineTeacher) -> Any:
    return (images - teacher.mean) / teacher.std


def split_square_patch_tokens(output: Any, preferred_prefix_tokens: int, teacher_name: str) -> tuple[Any, int, tuple[int, int]]:
    if output.ndim != 3:
        raise ValueError(f"{teacher_name} did not return token features; got shape={tuple(output.shape)}")

    total_tokens = int(output.shape[1])
    candidates: list[tuple[int, int, int, int]] = []
    for prefix_tokens in range(total_tokens):
        patch_tokens = total_tokens - prefix_tokens
        side = math.isqrt(patch_tokens)
        if side > 0 and side * side == patch_tokens:
            candidates.append((abs(prefix_tokens - preferred_prefix_tokens), -patch_tokens, prefix_tokens, side))

    if not candidates:
        raise ValueError(f"{teacher_name} returned no square patch-token suffix; got shape={tuple(output.shape)}")

    _, _, prefix_tokens, side = min(candidates)
    return output[:, prefix_tokens:, :].float(), prefix_tokens, (side, side)


def extract_online_patch_tokens(
    teacher: OnlineTeacher,
    images: Any,
    *,
    device: str,
    amp_dtype: str,
) -> Any:
    with inference_context(device, amp_dtype):
        teacher_images = normalize_for_teacher(images, teacher)
        if hasattr(teacher.model, "forward_features"):
            output = tensor_from_model_output(teacher.model.forward_features(teacher_images))
        else:
            output = tensor_from_model_output(teacher.model(teacher_images))

    patch_tokens, _, _ = split_square_patch_tokens(output, teacher.prefix_tokens, teacher.name)
    return patch_tokens


def infer_online_token_dim(
    *,
    model: Any,
    mean: Any,
    std: Any,
    prefix_tokens: int,
    device: str,
    tile_size: int,
    amp_dtype: str,
) -> tuple[int, int, tuple[int, int]]:
    import torch

    probe = torch.zeros(1, 3, tile_size, tile_size, device=device)
    with inference_context(device, amp_dtype):
        probe_input = (probe - mean) / std
        if hasattr(model, "forward_features"):
            output = tensor_from_model_output(model.forward_features(probe_input))
        else:
            output = tensor_from_model_output(model(probe_input))
    patch_tokens, actual_prefix_tokens, token_grid_size = split_square_patch_tokens(output, prefix_tokens, "online teacher probe")
    return int(patch_tokens.shape[-1]), actual_prefix_tokens, token_grid_size


def load_online_teacher(
    encoder: str,
    *,
    device: str,
    tile_size: int,
    amp_dtype: str,
) -> OnlineTeacher:
    import timm
    import torch
    from timm.layers import SwiGLUPacked

    if encoder not in ONLINE_TEACHER_SPECS:
        raise ValueError(f"unsupported online teacher: {encoder}")
    spec = ONLINE_TEACHER_SPECS[encoder]
    log(f"Loading online patch-token teacher {encoder} from {spec.hf_model}")

    if encoder == "virchow2":
        model = timm.create_model(
            spec.hf_model,
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )
        cfg = getattr(model, "pretrained_cfg", {}) or {}
        mean = tuple(cfg.get("mean", (0.485, 0.456, 0.406)))
        std = tuple(cfg.get("std", (0.229, 0.224, 0.225)))
    elif encoder == "hoptimus1":
        model = timm.create_model(
            spec.hf_model,
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=False,
        )
        mean = (0.707223, 0.578729, 0.703617)
        std = (0.211883, 0.230117, 0.177517)
    elif encoder == "uni_v2":
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
        cfg = getattr(model, "pretrained_cfg", {}) or {}
        mean = tuple(cfg.get("mean", (0.485, 0.456, 0.406)))
        std = tuple(cfg.get("std", (0.229, 0.224, 0.225)))
    else:
        raise ValueError(f"unsupported online teacher: {encoder}")

    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    mean_tensor = torch.tensor(mean, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    std_tensor = torch.tensor(std, device=device, dtype=torch.float32).view(1, 3, 1, 1)
    token_dim, actual_prefix_tokens, token_grid_size = infer_online_token_dim(
        model=model,
        mean=mean_tensor,
        std=std_tensor,
        prefix_tokens=spec.prefix_tokens,
        device=device,
        tile_size=tile_size,
        amp_dtype=amp_dtype,
    )
    log(
        f"Loaded online patch-token teacher {encoder}: dim={token_dim} "
        f"prefix_tokens={actual_prefix_tokens} configured_prefix_tokens={spec.prefix_tokens} "
        f"token_grid_size={token_grid_size[0]}x{token_grid_size[1]}"
    )
    return OnlineTeacher(
        name=encoder,
        model=model,
        mean=mean_tensor,
        std=std_tensor,
        prefix_tokens=actual_prefix_tokens,
        token_dim=token_dim,
        token_grid_size=token_grid_size,
    )


class ZipFeatureDistillDataset(IterableDataset):
    def __init__(
        self,
        *,
        input_root: Path,
        feature_root: Path,
        token_feature_root: Path | None,
        rel_paths: Sequence[Path],
        encoders: Sequence[str],
        tile_size: int,
        require_token_maps: bool,
        seed: int,
        shuffle_zips: bool,
        tile_sampler: TileSampler,
        epoch: int = 0,
    ) -> None:
        self.input_root = input_root
        self.feature_root = feature_root
        self.token_feature_root = token_feature_root
        self.rel_paths = list(rel_paths)
        self.encoders = list(encoders)
        self.transform = build_transform(tile_size)
        self.require_token_maps = require_token_maps
        self.seed = seed
        self.shuffle_zips = shuffle_zips
        self.tile_sampler = tile_sampler
        self.epoch = epoch

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __iter__(self) -> Iterator[dict[str, Any]]:
        import torch
        from torch.utils.data import get_worker_info

        worker_info = get_worker_info()
        rel_paths = list(self.rel_paths)
        epoch_seed = self.seed + self.epoch
        if self.shuffle_zips:
            rng = random.Random(epoch_seed)
            rng.shuffle(rel_paths)

        if worker_info is not None:
            rel_paths = rel_paths[worker_info.id :: worker_info.num_workers]

        for rel_path in rel_paths:
            zip_path = (self.input_root / rel_path).with_suffix(".zip")
            if not zip_path.exists():
                print(f"[warn] missing source zip: {zip_path}", file=sys.stderr, flush=True)
                continue

            teacher_names: dict[str, list[str]] = {}
            summary_name_to_index: dict[str, dict[str, int]] = {}
            token_name_to_index: dict[str, dict[str, int]] = {}
            teacher_features: dict[str, Any] = {}
            teacher_token_maps: dict[str, Any] = {}
            for encoder in self.encoders:
                feature_path = self.feature_root / encoder / rel_path
                names, features, token_maps = load_feature_tensors(
                    feature_path,
                    require_features=True,
                    require_token_maps=self.require_token_maps and self.token_feature_root is None,
                )
                teacher_names[encoder] = names
                summary_name_to_index[encoder] = {name: index for index, name in enumerate(names)}
                assert features is not None
                teacher_features[encoder] = features
                if self.require_token_maps and self.token_feature_root is not None:
                    token_path = self.token_feature_root / encoder / rel_path
                    token_names, _, token_maps = load_feature_tensors(
                        token_path,
                        require_features=False,
                        require_token_maps=True,
                    )
                    if token_names != names:
                        print(
                            f"[warn] tile order differs for {feature_path} and {token_path}; aligning by tile name",
                            file=sys.stderr,
                            flush=True,
                        )
                    teacher_names[encoder] = sorted(set(names) & set(token_names), key=natural_key)
                    token_name_to_index[encoder] = {name: index for index, name in enumerate(token_names)}
                else:
                    token_name_to_index[encoder] = summary_name_to_index[encoder]
                if token_maps is not None:
                    teacher_token_maps[encoder] = token_maps

            common_tiles = set(teacher_names[self.encoders[0]])
            for encoder in self.encoders[1:]:
                common_tiles &= set(teacher_names[encoder])
            tile_names = sorted(common_tiles, key=natural_key)
            if self.tile_sampler.enabled:
                tile_names = [
                    tile_name
                    for tile_name in tile_names
                    if self.tile_sampler.keep(rel_path, tile_name)
                ]
            if not tile_names:
                continue

            try:
                with zipfile.ZipFile(zip_path) as archive:
                    for tile_name in tile_names:
                        try:
                            image = self.transform(image_from_zip(archive, tile_name))
                        except Exception as exc:  # noqa: BLE001
                            print(f"[warn] failed image {zip_path}!{tile_name}: {exc}", file=sys.stderr, flush=True)
                            continue

                        targets = {
                            encoder: teacher_features[encoder][summary_name_to_index[encoder][tile_name]]
                            for encoder in self.encoders
                        }
                        token_targets = {
                            encoder: teacher_token_maps[encoder][token_name_to_index[encoder][tile_name]]
                            for encoder in self.encoders
                            if encoder in teacher_token_maps
                        }
                        yield {"image": image, "teachers": targets, "teacher_tokens": token_targets}
            except zipfile.BadZipFile as exc:
                print(f"[warn] bad zip {zip_path}: {exc}", file=sys.stderr, flush=True)


def build_student(
    *,
    radio_version: str,
    radio_repo: str,
    radio_source: str,
    radio_force_reload: bool,
    vitdet_window_size: int | None,
    summary_dims: dict[str, int],
    token_dims: dict[str, int],
) -> Any:
    import torch
    from torch import nn

    def trust_torch_hub_repo(repo: str) -> None:
        if radio_source != "github":
            return
        repo_without_ref = repo.split(":", 1)[0]
        if "/" not in repo_without_ref:
            return
        owner, name = repo_without_ref.split("/", 1)
        trusted_name = f"{owner}_{name}"
        hub_dir = Path(torch.hub.get_dir())
        hub_dir.mkdir(parents=True, exist_ok=True)
        trusted_list = hub_dir / "trusted_list"
        existing = set()
        if trusted_list.exists():
            existing = {line.strip() for line in trusted_list.read_text(encoding="utf-8").splitlines()}
        if trusted_name not in existing:
            with trusted_list.open("a", encoding="utf-8") as handle:
                handle.write(f"{trusted_name}\n")
            log(f"Added TorchHub trusted repo entry {trusted_name} in {trusted_list}")

    class Student(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            trust_torch_hub_repo(radio_repo)
            hub_kwargs: dict[str, Any] = {
                "source": radio_source,
                "version": radio_version,
                "progress": True,
                "skip_validation": True,
                "force_reload": radio_force_reload,
                "trust_repo": True,
            }
            if vitdet_window_size is not None:
                hub_kwargs["vitdet_window_size"] = vitdet_window_size
            self.radio = torch.hub.load(radio_repo, "radio_model", **hub_kwargs)
            self.radio.float()
            input_conditioner = getattr(self.radio, "input_conditioner", None)
            if input_conditioner is not None and hasattr(input_conditioner, "dtype"):
                input_conditioner.dtype = torch.float32
            student_dim = int(self.radio.summary_dim)
            self.summary_heads = nn.ModuleDict(
                {
                    encoder: nn.Sequential(
                        nn.LayerNorm(student_dim),
                        nn.Linear(student_dim, student_dim * 2),
                        nn.GELU(),
                        nn.Linear(student_dim * 2, dim),
                    )
                    for encoder, dim in summary_dims.items()
                }
            )
            self.spatial_heads = nn.ModuleDict(
                {
                    encoder: nn.Sequential(
                        nn.LayerNorm(int(self.radio.embed_dim)),
                        nn.Linear(int(self.radio.embed_dim), int(self.radio.embed_dim) * 2),
                        nn.GELU(),
                        nn.Linear(int(self.radio.embed_dim) * 2, dim),
                    )
                    for encoder, dim in token_dims.items()
                }
            )

        def forward(self, images: Any) -> dict[str, dict[str, Any]]:
            radio_output = self.radio(images)
            if isinstance(radio_output, dict):
                backbone_output = radio_output["backbone"]
            else:
                backbone_output = radio_output
            summary = backbone_output.summary
            spatial = backbone_output.features
            summary = summary.float()
            spatial = spatial.float()
            summary_outputs = {encoder: head(summary) for encoder, head in self.summary_heads.items()}
            spatial_outputs = {encoder: head(spatial) for encoder, head in self.spatial_heads.items()}
            return {"summary": summary_outputs, "spatial": spatial_outputs}

    return Student()


@dataclass
class TeacherStats:
    count: int
    dim: int
    mean_direction: Any
    angular_dispersion: float


@dataclass
class DistillStats:
    summary: dict[str, TeacherStats]
    spatial: dict[str, TeacherStats]


def compute_teacher_stats(
    *,
    feature_root: Path,
    token_feature_root: Path | None,
    rel_paths: Sequence[Path],
    encoders: Sequence[str],
    max_files: int | None,
    include_token_maps: bool,
    tile_sampler: TileSampler,
) -> DistillStats:
    import torch
    import torch.nn.functional as F

    paths = list(rel_paths[:max_files]) if max_files is not None else list(rel_paths)
    if not paths:
        raise RuntimeError("no feature files available for teacher statistics")

    summary_sums: dict[str, Any] = {}
    summary_counts: dict[str, int] = {encoder: 0 for encoder in encoders}
    summary_dims: dict[str, int] = {}
    spatial_sums: dict[str, Any] = {}
    spatial_counts: dict[str, int] = {encoder: 0 for encoder in encoders}
    spatial_dims: dict[str, int] = {}

    log(f"Computing teacher mean directions from {len(paths)} zip feature set(s); sample={tile_sampler.describe()}")

    def load_token_maps_for_stats(
        encoder: str,
        rel_path: Path,
        fallback_names: Sequence[str],
        fallback_token_maps: Any | None = None,
    ) -> tuple[list[str], Any | None]:
        if not include_token_maps:
            return list(fallback_names), None
        if token_feature_root is not None:
            token_names, _, maps = load_feature_tensors(
                token_feature_root / encoder / rel_path,
                require_features=False,
                require_token_maps=True,
            )
            return token_names, maps
        return list(fallback_names), fallback_token_maps

    for rel_path in paths:
        for encoder in encoders:
            names, features, token_maps = load_feature_tensors(
                feature_root / encoder / rel_path,
                require_features=True,
                require_token_maps=include_token_maps and token_feature_root is None,
            )
            assert features is not None
            indices = sampled_tile_indices(tile_sampler, rel_path, names)
            if not indices:
                continue
            features = features[indices]
            unit = F.normalize(features.float(), dim=-1)
            summary_sums[encoder] = unit.sum(dim=0) if encoder not in summary_sums else summary_sums[encoder] + unit.sum(dim=0)
            summary_counts[encoder] += int(unit.shape[0])
            summary_dims[encoder] = int(unit.shape[-1])
            token_names, token_maps = load_token_maps_for_stats(encoder, rel_path, names, token_maps)
            if token_maps is not None:
                token_indices = sampled_tile_indices(tile_sampler, rel_path, token_names)
                if not token_indices:
                    continue
                token_maps = token_maps[token_indices]
                tokens = token_maps.reshape(-1, token_maps.shape[-1])
                token_unit = F.normalize(tokens.float(), dim=-1)
                spatial_sums[encoder] = (
                    token_unit.sum(dim=0) if encoder not in spatial_sums else spatial_sums[encoder] + token_unit.sum(dim=0)
                )
                spatial_counts[encoder] += int(token_unit.shape[0])
                spatial_dims[encoder] = int(token_unit.shape[-1])

    missing_summary = [encoder for encoder in encoders if summary_counts[encoder] == 0]
    if missing_summary:
        raise RuntimeError(
            f"tile sampling selected no summary features for {missing_summary}; "
            "increase --stats-max-files, lower --sample-rate-denominator, or use a different --sample-rate-offset"
        )

    mean_dirs = {
        encoder: F.normalize(summary_sums[encoder] / max(summary_counts[encoder], 1), dim=0)
        for encoder in encoders
    }
    spatial_mean_dirs = {
        encoder: F.normalize(spatial_sums[encoder] / max(spatial_counts[encoder], 1), dim=0)
        for encoder in spatial_sums
    }

    sq_sums = {encoder: 0.0 for encoder in encoders}
    spatial_sq_sums = {encoder: 0.0 for encoder in spatial_sums}
    for rel_path in paths:
        for encoder in encoders:
            names, features, token_maps = load_feature_tensors(
                feature_root / encoder / rel_path,
                require_features=True,
                require_token_maps=include_token_maps and token_feature_root is None,
            )
            assert features is not None
            indices = sampled_tile_indices(tile_sampler, rel_path, names)
            if not indices:
                continue
            features = features[indices]
            unit = F.normalize(features.float(), dim=-1)
            sq_sums[encoder] += float(((unit - mean_dirs[encoder]) ** 2).sum(dim=-1).sum().item())
            token_names, token_maps = load_token_maps_for_stats(encoder, rel_path, names, token_maps)
            if token_maps is not None:
                token_indices = sampled_tile_indices(tile_sampler, rel_path, token_names)
                if not token_indices:
                    continue
                token_maps = token_maps[token_indices]
                tokens = token_maps.reshape(-1, token_maps.shape[-1])
                token_unit = F.normalize(tokens.float(), dim=-1)
                spatial_sq_sums[encoder] += float(
                    ((token_unit - spatial_mean_dirs[encoder]) ** 2).sum(dim=-1).sum().item()
                )

    summary_stats = {}
    for encoder in encoders:
        dispersion = math.sqrt(sq_sums[encoder] / max(summary_counts[encoder], 1))
        summary_stats[encoder] = TeacherStats(
            count=summary_counts[encoder],
            dim=summary_dims[encoder],
            mean_direction=mean_dirs[encoder],
            angular_dispersion=max(float(dispersion), 1e-4),
        )
        log(
            f"Teacher summary stats {encoder}: count={summary_stats[encoder].count} "
            f"dim={summary_stats[encoder].dim} angular_dispersion={summary_stats[encoder].angular_dispersion:.6f}"
        )
    spatial_stats = {}
    for encoder in spatial_sums:
        dispersion = math.sqrt(spatial_sq_sums[encoder] / max(spatial_counts[encoder], 1))
        spatial_stats[encoder] = TeacherStats(
            count=spatial_counts[encoder],
            dim=spatial_dims[encoder],
            mean_direction=spatial_mean_dirs[encoder],
            angular_dispersion=max(float(dispersion), 1e-4),
        )
        log(
            f"Teacher spatial stats {encoder}: count={spatial_stats[encoder].count} "
            f"dim={spatial_stats[encoder].dim} angular_dispersion={spatial_stats[encoder].angular_dispersion:.6f}"
        )
    return DistillStats(summary=summary_stats, spatial=spatial_stats)


def teacher_stats_to_device(stats: dict[str, TeacherStats], device: str) -> dict[str, TeacherStats]:
    return {
        encoder: TeacherStats(value.count, value.dim, value.mean_direction.to(device), value.angular_dispersion)
        for encoder, value in stats.items()
    }


def stats_to_device(stats: DistillStats, device: str) -> DistillStats:
    return DistillStats(
        summary=teacher_stats_to_device(stats.summary, device),
        spatial=teacher_stats_to_device(stats.spatial, device),
    )


def tile_sampler_metadata(tile_sampler: TileSampler) -> dict[str, int]:
    return {
        "sample_rate_denominator": tile_sampler.denominator,
        "sample_rate_offset": tile_sampler.offset,
    }


def load_stats_sample_metadata(path: Path) -> dict[str, int] | None:
    raw = load_payload(path)
    return raw.get("tile_sampler")


def save_stats(stats: DistillStats, path: Path, tile_sampler: TileSampler) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "summary": {encoder: asdict(value) for encoder, value in stats.summary.items()},
            "spatial": {encoder: asdict(value) for encoder, value in stats.spatial.items()},
            "tile_sampler": tile_sampler_metadata(tile_sampler),
        },
        path,
    )


def load_stats(path: Path) -> DistillStats:
    raw = load_payload(path)
    if "summary" not in raw:
        raw = {"summary": raw, "spatial": {}}
    return DistillStats(
        summary={encoder: TeacherStats(**value) for encoder, value in raw["summary"].items()},
        spatial={encoder: TeacherStats(**value) for encoder, value in raw.get("spatial", {}).items()},
    )


def init_wandb(args: argparse.Namespace, config: dict[str, Any], output_dir: Path) -> Any | None:
    wandb_project = args.wandb_project
    if not wandb_project and os.environ.get("WANDB_API_KEY") and args.wandb_mode != "disabled":
        wandb_project = "slc-pfm"
    if not wandb_project:
        return None
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "WandB logging was requested but the 'wandb' package is not installed in this environment."
        ) from exc

    run_id = args.wandb_id or output_dir.name
    run_name = args.wandb_name or output_dir.name
    run = wandb.init(
        project=wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        id=run_id,
        group=args.wandb_group,
        tags=parse_tag_list(args.wandb_tags),
        resume="allow",
        mode=args.wandb_mode,
        config=config,
        dir=str(output_dir),
    )
    log(f"WandB logging enabled: project={wandb_project} run_id={run_id} name={run_name}")
    return run


def latest_checkpoint_path(output_dir: Path) -> Path | None:
    latest = output_dir / "checkpoint_latest.pt"
    if latest.exists():
        return latest

    checkpoints = sorted(
        output_dir.glob("checkpoint_step*.pt"),
        key=lambda path: natural_key(path.name),
    )
    return checkpoints[-1] if checkpoints else None


def save_checkpoint(
    *,
    output_dir: Path,
    step: int,
    epoch: int,
    epoch_step: int,
    completed_epochs: int,
    model: Any,
    optimizer: Any,
    scaler: Any,
    args: argparse.Namespace,
    teacher_dims: dict[str, int],
    token_dims: dict[str, int],
    stats: DistillStats,
    config: dict[str, Any],
) -> Path:
    import torch

    checkpoint_path = output_dir / f"checkpoint_step{step:07d}.pt"
    payload = {
        "step": step,
        "epoch": epoch,
        "epoch_step": epoch_step,
        "completed_epochs": completed_epochs,
        "radio_version": args.radio_version,
        "radio_repo": args.radio_repo,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "teacher_dims": teacher_dims,
        "token_dims": token_dims,
        "teacher_stats": {
            "summary": {encoder: asdict(value) for encoder, value in stats.summary.items()},
            "spatial": {encoder: asdict(value) for encoder, value in stats.spatial.items()},
        },
        "config": config,
    }
    torch.save(payload, checkpoint_path)
    latest_path = output_dir / "checkpoint_latest.pt"
    shutil.copyfile(checkpoint_path, latest_path)
    return checkpoint_path


def balanced_summary_loss(prediction: Any, target: Any, stats: TeacherStats) -> Any:
    import torch.nn.functional as F

    pred_unit = F.normalize(prediction.float(), dim=-1)
    target_unit = F.normalize(target.float(), dim=-1)
    mean_dir = stats.mean_direction.view(1, -1)
    pred_balanced = (pred_unit - mean_dir) / stats.angular_dispersion
    target_balanced = (target_unit - mean_dir) / stats.angular_dispersion
    return F.mse_loss(pred_balanced, target_balanced)


def batch_balanced_loss(prediction: Any, target: Any) -> Any:
    import torch.nn.functional as F

    pred_unit = F.normalize(prediction.float(), dim=-1)
    target_unit = F.normalize(target.float(), dim=-1)
    mean_dir = F.normalize(target_unit.mean(dim=0, keepdim=True), dim=-1)
    dispersion = (((target_unit - mean_dir) ** 2).sum(dim=-1).mean()).sqrt().clamp_min(1e-4)
    pred_balanced = (pred_unit - mean_dir) / dispersion
    target_balanced = (target_unit - mean_dir) / dispersion
    return F.mse_loss(pred_balanced, target_balanced)


def resize_spatial_prediction(prediction: Any, target: Any) -> Any:
    if prediction.ndim != 3:
        raise ValueError(f"expected spatial prediction [B, L, C], got {tuple(prediction.shape)}")
    if target.ndim == 4:
        target_tokens = target.reshape(target.shape[0], -1, target.shape[-1])
    elif target.ndim == 3:
        target_tokens = target
    else:
        raise ValueError(f"expected target token map [B, H, W, C] or [B, L, C], got {tuple(target.shape)}")
    if prediction.shape[1] == target_tokens.shape[1]:
        return prediction

    import torch.nn.functional as F

    pred_side = int(math.sqrt(prediction.shape[1]))
    target_side = int(math.sqrt(target_tokens.shape[1]))
    if pred_side * pred_side != prediction.shape[1] or target_side * target_side != target_tokens.shape[1]:
        raise ValueError(
            f"cannot resize non-square token grids: pred={prediction.shape[1]} target={target_tokens.shape[1]}"
        )
    pred = prediction.transpose(1, 2).reshape(prediction.shape[0], prediction.shape[2], pred_side, pred_side)
    pred = F.interpolate(pred, size=(target_side, target_side), mode="bilinear", align_corners=False)
    return pred.flatten(2).transpose(1, 2)


def select_device(requested: str) -> str:
    if requested != "auto":
        return requested

    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def autocast_context(device: str, amp_dtype: str):
    if not device.startswith("cuda") or amp_dtype == "off":
        return nullcontext()

    import torch

    dtype = torch.float16 if amp_dtype == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


@contextmanager
def inference_context(device: str, amp_dtype: str):
    import torch

    with torch.inference_mode(), autocast_context(device, amp_dtype):
        yield


def collate_batch(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    import torch

    images = torch.stack([sample["image"] for sample in samples], dim=0)
    encoders = samples[0]["teachers"].keys()
    teachers = {
        encoder: torch.stack([sample["teachers"][encoder] for sample in samples], dim=0)
        for encoder in encoders
    }
    token_encoders = samples[0]["teacher_tokens"].keys()
    teacher_tokens = {
        encoder: torch.stack([sample["teacher_tokens"][encoder] for sample in samples], dim=0)
        for encoder in token_encoders
    }
    return {"image": images, "teachers": teachers, "teacher_tokens": teacher_tokens}


def train(args: argparse.Namespace) -> None:
    import torch
    from torch.utils.data import DataLoader

    run_started_at = time.monotonic()
    if args.sample_rate_denominator < 1:
        raise ValueError("--sample-rate-denominator must be >= 1")
    if args.sample_rate_offset < 0 or args.sample_rate_offset >= args.sample_rate_denominator:
        raise ValueError("--sample-rate-offset must be in [0, --sample-rate-denominator)")
    tile_sampler = TileSampler(
        denominator=args.sample_rate_denominator,
        offset=args.sample_rate_offset,
    )
    device = select_device(args.device)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    encoders = args.encoders
    token_feature_root = args.token_feature_root or args.feature_root
    cached_token_root = token_feature_root if args.include_token_maps and not args.online_token_teachers else None
    need_cached_spatial_stats = args.include_token_maps and not args.online_token_teachers
    rel_paths = discover_feature_sets(
        args.feature_root,
        encoders,
        cached_token_root,
        limit=args.limit_zips,
    )
    if not rel_paths:
        raise RuntimeError(f"no complete teacher feature sets under {args.feature_root}")
    log(f"Found {len(rel_paths)} complete zip feature set(s)")
    log(f"Dataset sampling: {tile_sampler.describe()}")

    stats_path = args.stats_path or (output_dir / "teacher_stats.pt")
    if stats_path.exists() and not args.recompute_stats:
        metadata = load_stats_sample_metadata(stats_path)
        if metadata is None:
            metadata = {"sample_rate_denominator": 1, "sample_rate_offset": 0}
        expected_metadata = tile_sampler_metadata(tile_sampler)
        if metadata != expected_metadata:
            raise RuntimeError(
                f"{stats_path} was computed with tile_sampler={metadata}, "
                f"but this run requested tile_sampler={expected_metadata}. "
                "Use --recompute-stats or a fresh OUTPUT_DIR."
            )
        log(f"Loading teacher stats from {stats_path}")
        stats = load_stats(stats_path)
        if need_cached_spatial_stats and not all(encoder in stats.spatial for encoder in encoders):
            missing = [encoder for encoder in encoders if encoder not in stats.spatial]
            raise RuntimeError(
                f"{stats_path} does not contain spatial stats for {missing}. "
                "Use --recompute-stats or a fresh OUTPUT_DIR for dense token-map training."
            )
    else:
        stats = compute_teacher_stats(
            feature_root=args.feature_root,
            token_feature_root=cached_token_root,
            rel_paths=rel_paths,
            encoders=encoders,
            max_files=args.stats_max_files,
            include_token_maps=need_cached_spatial_stats,
            tile_sampler=tile_sampler,
        )
        save_stats(stats, stats_path, tile_sampler)
        log(f"Saved teacher stats to {stats_path}")

    teacher_dims = {encoder: stats.summary[encoder].dim for encoder in encoders}
    online_teachers = {}
    if args.online_token_teachers:
        log("Loading frozen online patch-token teachers")
        online_teachers = {
            encoder: load_online_teacher(
                encoder,
                device=device,
                tile_size=args.tile_size,
                amp_dtype=args.amp_dtype,
            )
            for encoder in encoders
        }
    token_dims = (
        {encoder: teacher.token_dim for encoder, teacher in online_teachers.items()}
        if args.online_token_teachers
        else {encoder: stats.spatial[encoder].dim for encoder in encoders if encoder in stats.spatial}
    )
    stats = stats_to_device(stats, device)
    log(f"Loading C-RADIO student {args.radio_version} from {args.radio_repo}")
    model = build_student(
        radio_version=args.radio_version,
        radio_repo=args.radio_repo,
        radio_source=args.radio_source,
        radio_force_reload=args.radio_force_reload,
        vitdet_window_size=args.vitdet_window_size,
        summary_dims=teacher_dims,
        token_dims=token_dims if args.include_token_maps else {},
    ).to(device)

    dataset = ZipFeatureDistillDataset(
        input_root=args.input_root,
        feature_root=args.feature_root,
        token_feature_root=cached_token_root,
        rel_paths=rel_paths,
        encoders=encoders,
        tile_size=args.tile_size,
        require_token_maps=need_cached_spatial_stats,
        seed=args.seed,
        shuffle_zips=not args.no_shuffle_zips,
        tile_sampler=tile_sampler,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.startswith("cuda"),
        collate_fn=collate_batch,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=device.startswith("cuda") and args.amp_dtype == "fp16")

    config_path = output_dir / "train_config.json"
    config = vars(args).copy()
    config["encoders"] = encoders
    config["input_root"] = str(args.input_root)
    config["feature_root"] = str(args.feature_root)
    config["token_feature_root"] = str(token_feature_root) if need_cached_spatial_stats else None
    config["output_dir"] = str(output_dir)
    config["stats_path"] = str(stats_path)
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")
    wandb_run = init_wandb(args, config, output_dir)

    resume_path = args.resume_checkpoint
    if args.auto_resume and resume_path is None:
        resume_path = latest_checkpoint_path(output_dir)

    step = 0
    start_epoch = 0
    start_epoch_step = 0
    if resume_path is not None:
        if not resume_path.exists():
            raise FileNotFoundError(f"resume checkpoint does not exist: {resume_path}")
        checkpoint = load_payload(resume_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        step = int(checkpoint.get("step", 0))
        start_epoch = int(checkpoint.get("completed_epochs", checkpoint.get("epoch", 0)))
        start_epoch_step = int(checkpoint.get("epoch_step", 0))
        log(
            f"Resumed checkpoint {resume_path}: step={step} completed_epochs={start_epoch} "
            f"epoch_step={start_epoch_step}"
        )

    if args.epochs is None and args.max_steps is None and args.max_run_steps is None:
        raise ValueError("set --epochs, --max-steps, --max-run-steps, or a combination")

    target_epochs = args.epochs
    run_stop_step = step + args.max_run_steps if args.max_run_steps is not None else None
    if target_epochs is not None and start_epoch >= target_epochs:
        log(f"Nothing to do: completed_epochs={start_epoch} >= target_epochs={target_epochs}")
        return

    def stop_reason() -> str | None:
        if args.max_steps is not None and step >= args.max_steps:
            return f"max_steps={args.max_steps}"
        if run_stop_step is not None and step >= run_stop_step:
            return f"max_run_steps={args.max_run_steps}"
        return None

    model.train()
    images_seen = 0
    train_started_at = time.monotonic()
    train_step_started = step
    log(
        f"Starting distillation on {device}: "
        f"student={args.radio_version} batch_size={args.batch_size} "
        f"start_step={step} start_epoch={start_epoch} start_epoch_step={start_epoch_step} "
        f"target_epochs={target_epochs} max_steps={args.max_steps} max_run_steps={args.max_run_steps}"
    )
    epoch = start_epoch
    while True:
        if target_epochs is not None and epoch >= target_epochs:
            break
        if stop_reason() is not None:
            break

        dataset.set_epoch(epoch)
        epoch_started_at = time.monotonic()
        resume_epoch_step = start_epoch_step if epoch == start_epoch else 0
        epoch_step_base = step - resume_epoch_step
        log(f"Starting epoch {epoch + 1}" + (f"/{target_epochs}" if target_epochs is not None else ""))
        if resume_epoch_step:
            log(f"Skipping {resume_epoch_step} already-trained batch(es) in epoch {epoch + 1}")
        produced_batch_count = 0
        for batch_index, batch in enumerate(loader):
            if batch_index < resume_epoch_step:
                continue
            if stop_reason() is not None:
                break
            produced_batch_count += 1

            images = batch["image"].to(device, non_blocking=True)
            teachers = {
                encoder: tensor.to(device, non_blocking=True)
                for encoder, tensor in batch["teachers"].items()
            }
            teacher_tokens = {
                encoder: tensor.to(device, non_blocking=True)
                for encoder, tensor in batch["teacher_tokens"].items()
            }

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, args.amp_dtype):
                outputs = model(images)
                losses = {
                    encoder: balanced_summary_loss(outputs["summary"][encoder], teachers[encoder], stats.summary[encoder])
                    for encoder in encoders
                }
                spatial_losses = {}
                if args.include_token_maps:
                    if args.online_token_teachers:
                        for encoder, teacher in online_teachers.items():
                            target_tokens = extract_online_patch_tokens(
                                teacher,
                                images,
                                device=device,
                                amp_dtype=args.amp_dtype,
                            )
                            pred_tokens = resize_spatial_prediction(outputs["spatial"][encoder], target_tokens)
                            spatial_losses[encoder] = batch_balanced_loss(
                                pred_tokens.reshape(-1, pred_tokens.shape[-1]),
                                target_tokens.reshape(-1, target_tokens.shape[-1]),
                            )
                    else:
                        for encoder in token_dims:
                            pred_tokens = resize_spatial_prediction(outputs["spatial"][encoder], teacher_tokens[encoder])
                            spatial_losses[encoder] = balanced_summary_loss(
                                pred_tokens.reshape(-1, pred_tokens.shape[-1]),
                                teacher_tokens[encoder].reshape(-1, teacher_tokens[encoder].shape[-1]),
                                stats.spatial[encoder],
                            )
                loss = sum(losses.values()) / len(losses)
                if spatial_losses:
                    loss = loss + args.spatial_loss_weight * (sum(spatial_losses.values()) / len(spatial_losses))

            scaler.scale(loss).backward()
            if args.grad_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()

            step += 1
            images_seen += int(images.shape[0])
            if step % args.log_every == 0 or step == 1:
                elapsed = max(time.monotonic() - train_started_at, 1e-6)
                total_elapsed = max(time.monotonic() - run_started_at, 1e-6)
                measured_steps = max(step - train_step_started, 1)
                steps_per_sec = measured_steps / elapsed
                images_per_sec = images_seen / elapsed
                if args.max_steps is not None:
                    smoke_total_seconds = total_elapsed + max(args.max_steps - step, 0) / max(steps_per_sec, 1e-12)
                    smoke_total_bits = f"estimate_smoke_total_time={format_duration(smoke_total_seconds)} "
                else:
                    smoke_total_bits = ""
                estimate_steps = args.estimate_total_steps
                full_estimate_bits = ""
                if estimate_steps is not None and estimate_steps > 0:
                    full_total_seconds = total_elapsed + max(estimate_steps - step, 0) / max(steps_per_sec, 1e-12)
                    full_estimate_bits = (
                        f" estimate_total_steps={estimate_steps} "
                        f"estimate_total_time={format_duration(full_total_seconds)} "
                        f"estimate_remaining_time={format_duration(max(full_total_seconds - total_elapsed, 0.0))}"
                    )
                loss_bits = " ".join(f"{encoder}={float(value.detach().cpu()):.5f}" for encoder, value in losses.items())
                spatial_bits = " ".join(
                    f"{encoder}_spatial={float(value.detach().cpu()):.5f}" for encoder, value in spatial_losses.items()
                )
                if wandb_run is not None:
                    metrics = {
                        "train/loss": float(loss.detach().cpu()),
                        "train/epoch": epoch + 1,
                        "train/epoch_step": step - epoch_step_base,
                        "train/images_seen_this_run": images_seen,
                        "train/lr": float(optimizer.param_groups[0]["lr"]),
                        "perf/steps_per_sec": steps_per_sec,
                        "perf/images_per_sec": images_per_sec,
                        "time/elapsed_seconds": elapsed,
                        "time/total_elapsed_seconds": total_elapsed,
                    }
                    metrics.update(
                        {f"loss/summary_{encoder}": float(value.detach().cpu()) for encoder, value in losses.items()}
                    )
                    metrics.update(
                        {
                            f"loss/spatial_{encoder}": float(value.detach().cpu())
                            for encoder, value in spatial_losses.items()
                        }
                    )
                    wandb_run.log(metrics, step=step)
                log(
                    f"step={step} loss={float(loss.detach().cpu()):.5f} "
                    f"epoch={epoch + 1} "
                    f"elapsed={elapsed:.1f}s total_elapsed={total_elapsed:.1f}s "
                    f"steps_per_sec={steps_per_sec:.4f} images_per_sec={images_per_sec:.2f} images_seen={images_seen} "
                    f"{smoke_total_bits}"
                    f"{full_estimate_bits} "
                    f"{loss_bits} {spatial_bits}"
                )

            reason = stop_reason()
            if step % args.save_every == 0 or reason is not None:
                checkpoint_path = save_checkpoint(
                    output_dir=output_dir,
                    step=step,
                    epoch=epoch,
                    epoch_step=step - epoch_step_base,
                    completed_epochs=epoch,
                    model=model,
                    optimizer=optimizer,
                    scaler=scaler,
                    args=args,
                    teacher_dims=teacher_dims,
                    token_dims=token_dims,
                    stats=stats,
                    config=config,
                )
                log(f"saved checkpoint {checkpoint_path}")

        if produced_batch_count == 0 and stop_reason() is None:
            raise RuntimeError("epoch produced no batches; check dataset/feature alignment")

        reason = stop_reason()
        if reason is not None:
            checkpoint_path = save_checkpoint(
                output_dir=output_dir,
                step=step,
                epoch=epoch,
                epoch_step=step - epoch_step_base,
                completed_epochs=epoch,
                model=model,
                optimizer=optimizer,
                scaler=scaler,
                args=args,
                teacher_dims=teacher_dims,
                token_dims=token_dims,
                stats=stats,
                config=config,
            )
            log(f"reached {reason}; saved checkpoint {checkpoint_path}")
            break

        epoch += 1
        start_epoch_step = 0
        epoch_elapsed = time.monotonic() - epoch_started_at
        checkpoint_path = save_checkpoint(
            output_dir=output_dir,
            step=step,
            epoch=epoch - 1,
            epoch_step=0,
            completed_epochs=epoch,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            args=args,
            teacher_dims=teacher_dims,
            token_dims=token_dims,
            stats=stats,
            config=config,
        )
        log(
            f"completed epoch {epoch}"
            + (f"/{target_epochs}" if target_epochs is not None else "")
            + f" steps_this_epoch={step - epoch_step_base} elapsed={format_duration(epoch_elapsed)} "
            + f"checkpoint={checkpoint_path}"
        )

    if wandb_run is not None:
        wandb_run.finish()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="C-RADIO multi-teacher distillation from extracted features.")
    parser.add_argument("--input-root", type=Path, required=True, help="Root containing chunk_* zip folders.")
    parser.add_argument("--feature-root", type=Path, required=True, help="Root containing per-encoder feature folders.")
    parser.add_argument(
        "--token-feature-root",
        type=Path,
        help="Optional separate root containing token-map-only files. Defaults to --feature-root.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for checkpoints and stats.")
    parser.add_argument("--encoders", type=parse_encoder_list, default=list(DEFAULT_ENCODERS))
    parser.add_argument(
        "--radio-version",
        default="c-radio_v4-so400m",
        help="C-RADIO checkpoint version, e.g. c-radio_v4-so400m or c-radio_v4-h.",
    )
    parser.add_argument("--radio-repo", default="NVlabs/RADIO", help="TorchHub repo or local RADIO checkout.")
    parser.add_argument("--radio-source", choices=("github", "local"), default="github")
    parser.add_argument("--radio-force-reload", action="store_true")
    parser.add_argument("--vitdet-window-size", type=int, help="Optional C-RADIO ViTDet window size.")
    parser.add_argument("--tile-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, help="Optional maximum number of optimizer steps.")
    parser.add_argument(
        "--max-run-steps",
        type=int,
        help="Optional number of optimizer steps to run in this submission before checkpointing and exiting.",
    )
    parser.add_argument("--epochs", type=int, help="Number of full dataset passes to train.")
    parser.add_argument(
        "--estimate-total-steps",
        type=int,
        help="Optional full-run step count used only for ETA logging during smoke tests.",
    )
    parser.add_argument("--resume-checkpoint", type=Path, help="Checkpoint path to resume from.")
    parser.add_argument(
        "--auto-resume",
        dest="auto_resume",
        action="store_true",
        default=True,
        help="Resume from OUTPUT_DIR/checkpoint_latest.pt when present. Enabled by default.",
    )
    parser.add_argument(
        "--no-auto-resume",
        dest="auto_resume",
        action="store_false",
        help="Disable automatic resume from OUTPUT_DIR/checkpoint_latest.pt.",
    )
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16", "off"), default="bf16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--limit-zips", type=int, help="Limit zip feature sets for smoke tests.")
    parser.add_argument(
        "--sample-rate-denominator",
        type=int,
        default=1,
        help="Deterministically keep roughly 1/N tiles across the dataset. Use 1000 for a 1/1000 sample.",
    )
    parser.add_argument(
        "--sample-rate-offset",
        type=int,
        default=0,
        help="Hash bucket offset used with --sample-rate-denominator for reproducible non-overlapping folds.",
    )
    parser.add_argument("--stats-path", type=Path, help="Optional path to cached teacher stats.")
    parser.add_argument("--stats-max-files", type=int, help="Limit files used to estimate teacher stats.")
    parser.add_argument("--recompute-stats", action="store_true")
    parser.add_argument("--include-token-maps", action="store_true", help="Train dense spatial losses from token_maps.")
    parser.add_argument(
        "--online-token-teachers",
        action="store_true",
        help="Train dense spatial losses by running frozen teacher encoders on the fly instead of loading token_maps.",
    )
    parser.add_argument("--spatial-loss-weight", type=float, default=1.0)
    parser.add_argument("--no-shuffle-zips", action="store_true")
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--wandb-project", help="Enable WandB logging to this project.")
    parser.add_argument("--wandb-entity", help="Optional WandB entity/team.")
    parser.add_argument("--wandb-name", help="Optional WandB run display name. Defaults to output directory name.")
    parser.add_argument("--wandb-id", help="Optional stable WandB run id. Defaults to output directory name.")
    parser.add_argument("--wandb-group", help="Optional WandB run group.")
    parser.add_argument("--wandb-tags", help="Comma-separated WandB tags.")
    parser.add_argument("--wandb-mode", choices=("online", "offline", "disabled"), default="online")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    args = build_parser().parse_args(argv)
    if args.online_token_teachers:
        args.include_token_maps = True
    random.seed(args.seed)

    train(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
