#!/usr/bin/env python3
"""Print one-minute GPU memory peaks using Erbium's GPU API."""

from __future__ import annotations

import argparse
import math
import signal
import sys
import time
from dataclasses import dataclass
from typing import Sequence


RUNNING = True


@dataclass(frozen=True)
class DeviceSample:
    device_id: int
    name: str
    gpu_utilization_percent: float
    memory_utilization_percent: float
    total_memory_gb: float


def stop(_signum: int, _frame: object) -> None:
    global RUNNING
    RUNNING = False


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def load_gpu_info() -> dict[int, DeviceSample]:
    try:
        from erbium.api import get_all_gpu_info
    except ImportError as exc:
        raise SystemExit(
            "Erbium is required for GPU monitoring. Install it with: "
            "pip install 'erbium @ git+https://github.com/ProjectNeura/Erbium'"
        ) from exc

    return {
        device_id: DeviceSample(
            device_id=device_id,
            name=info.name,
            gpu_utilization_percent=info.utilization_percent,
            memory_utilization_percent=info.memory_utilization_percent,
            total_memory_gb=info.total_memory_gb,
        )
        for device_id, info in get_all_gpu_info().items()
    }


def finite_or_zero(value: float) -> float:
    return value if math.isfinite(value) else 0.0


def emit_window_report(
    *,
    label: str,
    window_started_at: float,
    samples: int,
    peak_memory_by_device: dict[int, float],
    peak_gpu_by_device: dict[int, float],
    latest_names: dict[int, str],
) -> None:
    elapsed = max(time.monotonic() - window_started_at, 0.0)
    if samples == 0:
        print(f"[{now()}] [erbium-gpu] {label} no samples collected in last {elapsed:.0f}s", flush=True)
        return

    if not peak_memory_by_device:
        print(f"[{now()}] [erbium-gpu] {label} no visible GPUs in last {elapsed:.0f}s", flush=True)
        return

    max_device = max(peak_memory_by_device, key=peak_memory_by_device.get)
    max_memory = peak_memory_by_device[max_device]
    device_summaries = []
    for device_id in sorted(peak_memory_by_device):
        device_summaries.append(
            f"gpu{device_id} mem_max={peak_memory_by_device[device_id]:.2f}% "
            f"gpu_util_max={peak_gpu_by_device.get(device_id, 0.0):.2f}% "
            f"name={latest_names.get(device_id, 'unknown')}"
        )
    print(
        f"[{now()}] [erbium-gpu] {label} "
        f"window_seconds={elapsed:.0f} samples={samples} "
        f"max_memory_utilization={max_memory:.2f}% max_memory_device=gpu{max_device} "
        + " | ".join(device_summaries),
        flush=True,
    )


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monitor GPU memory peaks with Erbium.")
    parser.add_argument("--label", default="job", help="Label to include in each log line.")
    parser.add_argument(
        "--sample-interval-seconds",
        type=positive_float,
        default=5.0,
        help="How often to sample Erbium GPU stats.",
    )
    parser.add_argument(
        "--report-interval-seconds",
        type=positive_float,
        default=60.0,
        help="How often to print the max GPU memory utilization from the recent window.",
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
        f"[{now()}] [erbium-gpu] {args.label} monitor started "
        f"sample_interval={args.sample_interval_seconds:g}s "
        f"report_interval={args.report_interval_seconds:g}s",
        flush=True,
    )

    window_started_at = time.monotonic()
    next_report_at = window_started_at + args.report_interval_seconds
    samples = 0
    peak_memory_by_device: dict[int, float] = {}
    peak_gpu_by_device: dict[int, float] = {}
    latest_names: dict[int, str] = {}

    while RUNNING:
        try:
            infos = load_gpu_info()
            for device_id, info in infos.items():
                latest_names[device_id] = info.name
                peak_memory_by_device[device_id] = max(
                    peak_memory_by_device.get(device_id, 0.0),
                    finite_or_zero(info.memory_utilization_percent),
                )
                peak_gpu_by_device[device_id] = max(
                    peak_gpu_by_device.get(device_id, 0.0),
                    finite_or_zero(info.gpu_utilization_percent),
                )
            samples += 1
        except Exception as exc:  # noqa: BLE001 - keep the monitor from killing the main job.
            print(f"[{now()}] [erbium-gpu] {args.label} sample failed: {exc!r}", file=sys.stderr, flush=True)

        current_time = time.monotonic()
        if current_time >= next_report_at:
            emit_window_report(
                label=args.label,
                window_started_at=window_started_at,
                samples=samples,
                peak_memory_by_device=peak_memory_by_device,
                peak_gpu_by_device=peak_gpu_by_device,
                latest_names=latest_names,
            )
            window_started_at = current_time
            next_report_at = current_time + args.report_interval_seconds
            samples = 0
            peak_memory_by_device.clear()
            peak_gpu_by_device.clear()

        sleep_seconds = min(args.sample_interval_seconds, max(next_report_at - time.monotonic(), 0.1))
        time.sleep(sleep_seconds)

    emit_window_report(
        label=args.label,
        window_started_at=window_started_at,
        samples=samples,
        peak_memory_by_device=peak_memory_by_device,
        peak_gpu_by_device=peak_gpu_by_device,
        latest_names=latest_names,
    )
    print(f"[{now()}] [erbium-gpu] {args.label} monitor stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
