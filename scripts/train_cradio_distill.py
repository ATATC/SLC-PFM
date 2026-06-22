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
import io
import json
import math
import os
import random
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


def discover_feature_sets(feature_root: Path, encoders: Sequence[str], token_feature_root: Path | None = None) -> list[Path]:
    first_root = feature_root / encoders[0]
    if not first_root.exists():
        raise FileNotFoundError(f"missing feature directory: {first_root}")

    rel_paths = sorted(
        (path.relative_to(first_root) for path in first_root.rglob("*.pt")),
        key=lambda path: natural_key(str(path)),
    )
    complete = []
    for rel_path in rel_paths:
        has_summary = all((feature_root / encoder / rel_path).exists() for encoder in encoders)
        has_tokens = token_feature_root is None or all((token_feature_root / encoder / rel_path).exists() for encoder in encoders)
        if has_summary and has_tokens:
            complete.append(rel_path)
    return complete


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


def normalize_for_teacher(images: Any, teacher: OnlineTeacher) -> Any:
    return (images - teacher.mean) / teacher.std


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

    if output.ndim != 3 or output.shape[1] <= teacher.prefix_tokens:
        raise ValueError(
            f"{teacher.name} did not return patch tokens from forward_features; got shape={tuple(output.shape)}"
        )
    return output[:, teacher.prefix_tokens :, :].float()


def infer_online_token_dim(
    *,
    model: Any,
    mean: Any,
    std: Any,
    prefix_tokens: int,
    device: str,
    tile_size: int,
    amp_dtype: str,
) -> int:
    import torch

    probe = torch.zeros(1, 3, tile_size, tile_size, device=device)
    with inference_context(device, amp_dtype):
        probe_input = (probe - mean) / std
        if hasattr(model, "forward_features"):
            output = tensor_from_model_output(model.forward_features(probe_input))
        else:
            output = tensor_from_model_output(model(probe_input))
    if output.ndim != 3 or output.shape[1] <= prefix_tokens:
        raise ValueError(f"expected patch tokens from online teacher probe, got shape={tuple(output.shape)}")
    return int(output.shape[-1])


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
    token_dim = infer_online_token_dim(
        model=model,
        mean=mean_tensor,
        std=std_tensor,
        prefix_tokens=spec.prefix_tokens,
        device=device,
        tile_size=tile_size,
        amp_dtype=amp_dtype,
    )
    log(f"Loaded online patch-token teacher {encoder}: dim={token_dim} prefix_tokens={spec.prefix_tokens}")
    return OnlineTeacher(
        name=encoder,
        model=model,
        mean=mean_tensor,
        std=std_tensor,
        prefix_tokens=spec.prefix_tokens,
        token_dim=token_dim,
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

    def __iter__(self) -> Iterator[dict[str, Any]]:
        import torch
        from torch.utils.data import get_worker_info

        worker_info = get_worker_info()
        rel_paths = list(self.rel_paths)
        epoch_seed = self.seed + int(time.time() // 3600)
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

    class Student(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            hub_kwargs: dict[str, Any] = {
                "source": radio_source,
                "version": radio_version,
                "progress": True,
                "skip_validation": True,
                "force_reload": radio_force_reload,
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

    log(f"Computing teacher mean directions from {len(paths)} zip feature set(s)")

    def load_token_maps_for_stats(encoder: str, rel_path: Path, fallback_token_maps: Any | None = None) -> Any | None:
        if not include_token_maps:
            return None
        if token_feature_root is not None:
            _, _, maps = load_feature_tensors(
                token_feature_root / encoder / rel_path,
                require_features=False,
                require_token_maps=True,
            )
            return maps
        return fallback_token_maps

    for rel_path in paths:
        for encoder in encoders:
            _, features, token_maps = load_feature_tensors(
                feature_root / encoder / rel_path,
                require_features=True,
                require_token_maps=include_token_maps and token_feature_root is None,
            )
            assert features is not None
            unit = F.normalize(features.float(), dim=-1)
            summary_sums[encoder] = unit.sum(dim=0) if encoder not in summary_sums else summary_sums[encoder] + unit.sum(dim=0)
            summary_counts[encoder] += int(unit.shape[0])
            summary_dims[encoder] = int(unit.shape[-1])
            token_maps = load_token_maps_for_stats(encoder, rel_path, token_maps)
            if token_maps is not None:
                tokens = token_maps.reshape(-1, token_maps.shape[-1])
                token_unit = F.normalize(tokens.float(), dim=-1)
                spatial_sums[encoder] = (
                    token_unit.sum(dim=0) if encoder not in spatial_sums else spatial_sums[encoder] + token_unit.sum(dim=0)
                )
                spatial_counts[encoder] += int(token_unit.shape[0])
                spatial_dims[encoder] = int(token_unit.shape[-1])

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
            _, features, token_maps = load_feature_tensors(
                feature_root / encoder / rel_path,
                require_features=True,
                require_token_maps=include_token_maps and token_feature_root is None,
            )
            assert features is not None
            unit = F.normalize(features.float(), dim=-1)
            sq_sums[encoder] += float(((unit - mean_dirs[encoder]) ** 2).sum(dim=-1).sum().item())
            token_maps = load_token_maps_for_stats(encoder, rel_path, token_maps)
            if token_maps is not None:
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


def save_stats(stats: DistillStats, path: Path) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "summary": {encoder: asdict(value) for encoder, value in stats.summary.items()},
            "spatial": {encoder: asdict(value) for encoder, value in stats.spatial.items()},
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
    )
    if args.limit_zips is not None:
        rel_paths = rel_paths[: args.limit_zips]
    if not rel_paths:
        raise RuntimeError(f"no complete teacher feature sets under {args.feature_root}")
    log(f"Found {len(rel_paths)} complete zip feature set(s)")

    stats_path = args.stats_path or (output_dir / "teacher_stats.pt")
    if stats_path.exists() and not args.recompute_stats:
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
        )
        save_stats(stats, stats_path)
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

    model.train()
    step = 0
    log(
        f"Starting distillation on {device}: "
        f"student={args.radio_version} batch_size={args.batch_size}"
    )
    while step < args.max_steps:
        for batch in loader:
            if step >= args.max_steps:
                break

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
            if step % args.log_every == 0 or step == 1:
                loss_bits = " ".join(f"{encoder}={float(value.detach().cpu()):.5f}" for encoder, value in losses.items())
                spatial_bits = " ".join(
                    f"{encoder}_spatial={float(value.detach().cpu()):.5f}" for encoder, value in spatial_losses.items()
                )
                log(f"step={step} loss={float(loss.detach().cpu()):.5f} {loss_bits} {spatial_bits}")

            if step % args.save_every == 0 or step == args.max_steps:
                checkpoint_path = output_dir / f"checkpoint_step{step:07d}.pt"
                torch.save(
                    {
                        "step": step,
                        "radio_version": args.radio_version,
                        "radio_repo": args.radio_repo,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "teacher_dims": teacher_dims,
                        "token_dims": token_dims,
                        "teacher_stats": {
                            "summary": {encoder: asdict(value) for encoder, value in stats.summary.items()},
                            "spatial": {encoder: asdict(value) for encoder, value in stats.spatial.items()},
                        },
                        "config": config,
                    },
                    checkpoint_path,
                )
                log(f"saved checkpoint {checkpoint_path}")


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
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--amp-dtype", choices=("bf16", "fp16", "off"), default="bf16")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--limit-zips", type=int, help="Limit zip feature sets for smoke tests.")
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
