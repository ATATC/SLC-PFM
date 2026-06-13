#!/usr/bin/env python3
"""Print one-minute GPU utilization peaks using nvidia-smi."""

from __future__ import annotations

import argparse
import csv
import math
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from io import StringIO
from typing import Sequence


RUNNING = True


@dataclass(frozen=True)
class DeviceSample:
    device_id: str
    name: str
    gpu_utilization_percent: float | None
    memory_utilization_percent: float
    memory_used_mib: float
    memory_total_mib: float


def stop(_signum: int, _frame: object) -> None:
    global RUNNING
    RUNNING = False


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def parse_optional_float(value: str) -> float | None:
    cleaned = value.strip()
    if not cleaned or cleaned.upper() == "N/A":
        return None
    try:
        parsed = float(cleaned)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def load_gpu_info(nvidia_smi_bin: str) -> list[DeviceSample]:
    command = [
        nvidia_smi_bin,
        "--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, text=True, capture_output=True)

    samples: list[DeviceSample] = []
    reader = csv.reader(StringIO(result.stdout))
    for row in reader:
        if len(row) < 6:
            continue
        device_id = row[0].strip()
        name = row[1].strip()
        gpu_utilization = parse_optional_float(row[2])
        memory_used = parse_optional_float(row[4])
        memory_total = parse_optional_float(row[5])
        if memory_used is None or memory_total is None or memory_total <= 0:
            continue
        samples.append(
            DeviceSample(
                device_id=device_id,
                name=name,
                gpu_utilization_percent=gpu_utilization,
                memory_utilization_percent=100.0 * memory_used / memory_total,
                memory_used_mib=memory_used,
                memory_total_mib=memory_total,
            )
        )
    return samples


def emit_window_report(
    *,
    label: str,
    window_started_at: float,
    samples: int,
    peak_memory_by_device: dict[str, float],
    peak_gpu_by_device: dict[str, float],
    peak_used_mib_by_device: dict[str, float],
    total_mib_by_device: dict[str, float],
    latest_names: dict[str, str],
) -> None:
    elapsed = max(time.monotonic() - window_started_at, 0.0)
    if samples == 0:
        print(f"[{now()}] [nvidia-smi-gpu] {label} no samples collected in last {elapsed:.0f}s", flush=True)
        return

    if not peak_memory_by_device:
        print(f"[{now()}] [nvidia-smi-gpu] {label} no visible GPUs in last {elapsed:.0f}s", flush=True)
        return

    max_device = max(peak_memory_by_device, key=peak_memory_by_device.get)
    max_memory = peak_memory_by_device[max_device]
    gpu_util_available = bool(peak_gpu_by_device)
    device_summaries = []
    for device_id in sorted(peak_memory_by_device, key=lambda value: int(value) if value.isdigit() else value):
        gpu_util_max = peak_gpu_by_device.get(device_id)
        gpu_util_text = f"{gpu_util_max:.2f}%" if gpu_util_max is not None else "n/a"
        used_mib = peak_used_mib_by_device.get(device_id, 0.0)
        total_mib = total_mib_by_device.get(device_id, 0.0)
        device_summaries.append(
            f"gpu{device_id} mem_max={peak_memory_by_device[device_id]:.2f}% "
            f"mem_used_max={used_mib:.0f}MiB/{total_mib:.0f}MiB "
            f"gpu_util_max={gpu_util_text} "
            f"name={latest_names.get(device_id, 'unknown')}"
        )

    availability_text = "" if gpu_util_available else " gpu_utilization_metric=unavailable"
    print(
        f"[{now()}] [nvidia-smi-gpu] {label} "
        f"window_seconds={elapsed:.0f} samples={samples} "
        f"max_memory_utilization={max_memory:.2f}% max_memory_device=gpu{max_device}"
        f"{availability_text} "
        + " | ".join(device_summaries),
        flush=True,
    )


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monitor GPU peaks with nvidia-smi.")
    parser.add_argument("--label", default="job", help="Label to include in each log line.")
    parser.add_argument(
        "--sample-interval-seconds",
        type=positive_float,
        default=5.0,
        help="How often to sample nvidia-smi.",
    )
    parser.add_argument(
        "--report-interval-seconds",
        type=positive_float,
        default=60.0,
        help="How often to print peak GPU stats from the recent window.",
    )
    parser.add_argument(
        "--nvidia-smi-bin",
        default=os.environ.get("NVIDIA_SMI_BIN", "nvidia-smi"),
        help="Path to the nvidia-smi executable.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    args = build_parser().parse_args(argv)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    print(
        f"[{now()}] [nvidia-smi-gpu] {args.label} monitor started "
        f"sample_interval={args.sample_interval_seconds:g}s "
        f"report_interval={args.report_interval_seconds:g}s "
        f"nvidia_smi={args.nvidia_smi_bin}",
        flush=True,
    )

    window_started_at = time.monotonic()
    next_report_at = window_started_at + args.report_interval_seconds
    samples = 0
    peak_memory_by_device: dict[str, float] = {}
    peak_gpu_by_device: dict[str, float] = {}
    peak_used_mib_by_device: dict[str, float] = {}
    total_mib_by_device: dict[str, float] = {}
    latest_names: dict[str, str] = {}

    while RUNNING:
        try:
            infos = load_gpu_info(args.nvidia_smi_bin)
            for info in infos:
                latest_names[info.device_id] = info.name
                total_mib_by_device[info.device_id] = info.memory_total_mib
                peak_memory_by_device[info.device_id] = max(
                    peak_memory_by_device.get(info.device_id, 0.0),
                    info.memory_utilization_percent,
                )
                peak_used_mib_by_device[info.device_id] = max(
                    peak_used_mib_by_device.get(info.device_id, 0.0),
                    info.memory_used_mib,
                )
                if info.gpu_utilization_percent is not None:
                    peak_gpu_by_device[info.device_id] = max(
                        peak_gpu_by_device.get(info.device_id, 0.0),
                        info.gpu_utilization_percent,
                    )
            samples += 1
        except Exception as exc:  # noqa: BLE001 - keep monitor failures from killing the extraction job.
            print(f"[{now()}] [nvidia-smi-gpu] {args.label} sample failed: {exc!r}", file=sys.stderr, flush=True)

        current_time = time.monotonic()
        if current_time >= next_report_at:
            emit_window_report(
                label=args.label,
                window_started_at=window_started_at,
                samples=samples,
                peak_memory_by_device=peak_memory_by_device,
                peak_gpu_by_device=peak_gpu_by_device,
                peak_used_mib_by_device=peak_used_mib_by_device,
                total_mib_by_device=total_mib_by_device,
                latest_names=latest_names,
            )
            window_started_at = current_time
            next_report_at = current_time + args.report_interval_seconds
            samples = 0
            peak_memory_by_device.clear()
            peak_gpu_by_device.clear()
            peak_used_mib_by_device.clear()
            total_mib_by_device.clear()

        sleep_seconds = min(args.sample_interval_seconds, max(next_report_at - time.monotonic(), 0.1))
        time.sleep(sleep_seconds)

    emit_window_report(
        label=args.label,
        window_started_at=window_started_at,
        samples=samples,
        peak_memory_by_device=peak_memory_by_device,
        peak_gpu_by_device=peak_gpu_by_device,
        peak_used_mib_by_device=peak_used_mib_by_device,
        total_mib_by_device=total_mib_by_device,
        latest_names=latest_names,
    )
    print(f"[{now()}] [nvidia-smi-gpu] {args.label} monitor stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
