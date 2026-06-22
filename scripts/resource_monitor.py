#!/usr/bin/env python3
"""Print rolling and final CPU, CPU-memory, GPU, and GPU-memory utilization peaks."""

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
from pathlib import Path
from typing import Sequence


RUNNING = True


@dataclass(frozen=True)
class GpuSample:
    device_id: str
    name: str
    gpu_utilization_percent: float | None
    memory_utilization_percent: float
    memory_used_mib: float
    memory_total_mib: float


@dataclass(frozen=True)
class HostSample:
    cpu_usage_seconds: float | None
    memory_used_mib: float | None
    memory_limit_mib: float | None


@dataclass
class PeakStats:
    samples: int = 0
    cpu_utilization_percent: float | None = None
    memory_utilization_percent: float | None = None
    memory_used_mib: float | None = None
    memory_limit_mib: float | None = None
    gpu_utilization_by_device: dict[str, float] | None = None
    gpu_memory_percent_by_device: dict[str, float] | None = None
    gpu_memory_used_mib_by_device: dict[str, float] | None = None
    gpu_memory_total_mib_by_device: dict[str, float] | None = None
    gpu_names_by_device: dict[str, str] | None = None

    def __post_init__(self) -> None:
        self.gpu_utilization_by_device = {}
        self.gpu_memory_percent_by_device = {}
        self.gpu_memory_used_mib_by_device = {}
        self.gpu_memory_total_mib_by_device = {}
        self.gpu_names_by_device = {}

    def update_host(self, cpu_percent: float | None, sample: HostSample) -> None:
        if cpu_percent is not None:
            self.cpu_utilization_percent = max(self.cpu_utilization_percent or 0.0, cpu_percent)
        if sample.memory_used_mib is not None:
            self.memory_used_mib = max(self.memory_used_mib or 0.0, sample.memory_used_mib)
        if sample.memory_limit_mib is not None:
            self.memory_limit_mib = sample.memory_limit_mib
        if sample.memory_used_mib is not None and sample.memory_limit_mib and sample.memory_limit_mib > 0:
            memory_percent = 100.0 * sample.memory_used_mib / sample.memory_limit_mib
            self.memory_utilization_percent = max(self.memory_utilization_percent or 0.0, memory_percent)

    def update_gpus(self, samples: list[GpuSample]) -> None:
        assert self.gpu_utilization_by_device is not None
        assert self.gpu_memory_percent_by_device is not None
        assert self.gpu_memory_used_mib_by_device is not None
        assert self.gpu_memory_total_mib_by_device is not None
        assert self.gpu_names_by_device is not None
        for sample in samples:
            self.gpu_names_by_device[sample.device_id] = sample.name
            self.gpu_memory_total_mib_by_device[sample.device_id] = sample.memory_total_mib
            self.gpu_memory_percent_by_device[sample.device_id] = max(
                self.gpu_memory_percent_by_device.get(sample.device_id, 0.0),
                sample.memory_utilization_percent,
            )
            self.gpu_memory_used_mib_by_device[sample.device_id] = max(
                self.gpu_memory_used_mib_by_device.get(sample.device_id, 0.0),
                sample.memory_used_mib,
            )
            if sample.gpu_utilization_percent is not None:
                self.gpu_utilization_by_device[sample.device_id] = max(
                    self.gpu_utilization_by_device.get(sample.device_id, 0.0),
                    sample.gpu_utilization_percent,
                )

    def merge(self, other: "PeakStats") -> None:
        self.samples += other.samples
        if other.cpu_utilization_percent is not None:
            self.cpu_utilization_percent = max(self.cpu_utilization_percent or 0.0, other.cpu_utilization_percent)
        if other.memory_used_mib is not None:
            self.memory_used_mib = max(self.memory_used_mib or 0.0, other.memory_used_mib)
        if other.memory_limit_mib is not None:
            self.memory_limit_mib = other.memory_limit_mib
        if other.memory_utilization_percent is not None:
            self.memory_utilization_percent = max(self.memory_utilization_percent or 0.0, other.memory_utilization_percent)
        assert other.gpu_utilization_by_device is not None
        assert other.gpu_memory_percent_by_device is not None
        assert other.gpu_memory_used_mib_by_device is not None
        assert other.gpu_memory_total_mib_by_device is not None
        assert other.gpu_names_by_device is not None
        for device_id, value in other.gpu_utilization_by_device.items():
            assert self.gpu_utilization_by_device is not None
            self.gpu_utilization_by_device[device_id] = max(self.gpu_utilization_by_device.get(device_id, 0.0), value)
        for device_id, value in other.gpu_memory_percent_by_device.items():
            assert self.gpu_memory_percent_by_device is not None
            self.gpu_memory_percent_by_device[device_id] = max(
                self.gpu_memory_percent_by_device.get(device_id, 0.0),
                value,
            )
        for device_id, value in other.gpu_memory_used_mib_by_device.items():
            assert self.gpu_memory_used_mib_by_device is not None
            self.gpu_memory_used_mib_by_device[device_id] = max(
                self.gpu_memory_used_mib_by_device.get(device_id, 0.0),
                value,
            )
        assert self.gpu_memory_total_mib_by_device is not None
        assert self.gpu_names_by_device is not None
        self.gpu_memory_total_mib_by_device.update(other.gpu_memory_total_mib_by_device)
        self.gpu_names_by_device.update(other.gpu_names_by_device)


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


def format_percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}%"


def format_mib(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f}MiB"


def cgroup_paths() -> list[Path]:
    paths: list[Path] = []
    cgroup_file = Path("/proc/self/cgroup")
    if not cgroup_file.exists():
        return paths
    for line in cgroup_file.read_text(encoding="utf-8").splitlines():
        parts = line.split(":", 2)
        if len(parts) != 3:
            continue
        controllers = [item for item in parts[1].split(",") if item]
        relative = parts[2].lstrip("/")
        if parts[0] == "0":
            paths.append(Path("/sys/fs/cgroup") / relative)
        for controller in controllers:
            paths.append(Path("/sys/fs/cgroup") / controller / relative)
            paths.append(Path("/sys/fs/cgroup") / relative)
    return paths


def find_cgroup_file(names: Sequence[str]) -> Path | None:
    seen: set[Path] = set()
    for start in cgroup_paths():
        current = start
        while True:
            for name in names:
                candidate = current / name
                if candidate not in seen and candidate.exists():
                    return candidate
                seen.add(candidate)
            if current == current.parent:
                break
            current = current.parent
    return None


def read_cpu_usage_seconds() -> float | None:
    cpu_stat = find_cgroup_file(("cpu.stat",))
    if cpu_stat is not None:
        for line in cpu_stat.read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[0] == "usage_usec":
                return float(parts[1]) / 1_000_000.0

    cpuacct_usage = find_cgroup_file(("cpuacct.usage",))
    if cpuacct_usage is not None:
        return float(cpuacct_usage.read_text(encoding="utf-8").strip()) / 1_000_000_000.0
    return None


def read_memory_limit_mib(cli_limit_mib: float | None) -> float | None:
    if cli_limit_mib is not None and cli_limit_mib > 0:
        return cli_limit_mib

    for env_name in ("SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU"):
        value = os.environ.get(env_name)
        if value and value.isdigit():
            parsed = float(value)
            if env_name == "SLURM_MEM_PER_CPU":
                cpus = parse_optional_float(os.environ.get("SLURM_CPUS_PER_TASK", ""))
                parsed *= cpus or 1.0
            return parsed

    limit_file = find_cgroup_file(("memory.max", "memory.limit_in_bytes"))
    if limit_file is None:
        return None
    raw = limit_file.read_text(encoding="utf-8").strip()
    if raw == "max":
        return None
    limit_bytes = float(raw)
    if limit_bytes <= 0 or limit_bytes > 9e18:
        return None
    return limit_bytes / 1024.0 / 1024.0


def read_memory_used_mib() -> float | None:
    usage_file = find_cgroup_file(("memory.current", "memory.usage_in_bytes"))
    if usage_file is None:
        return None
    return float(usage_file.read_text(encoding="utf-8").strip()) / 1024.0 / 1024.0


def load_host_sample(memory_limit_mib: float | None) -> HostSample:
    return HostSample(
        cpu_usage_seconds=read_cpu_usage_seconds(),
        memory_used_mib=read_memory_used_mib(),
        memory_limit_mib=read_memory_limit_mib(memory_limit_mib),
    )


def load_gpu_samples(nvidia_smi_bin: str) -> list[GpuSample]:
    command = [
        nvidia_smi_bin,
        "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, text=True, capture_output=True)

    samples: list[GpuSample] = []
    reader = csv.reader(StringIO(result.stdout))
    for row in reader:
        if len(row) < 5:
            continue
        memory_used = parse_optional_float(row[3])
        memory_total = parse_optional_float(row[4])
        if memory_used is None or memory_total is None or memory_total <= 0:
            continue
        samples.append(
            GpuSample(
                device_id=row[0].strip(),
                name=row[1].strip(),
                gpu_utilization_percent=parse_optional_float(row[2]),
                memory_utilization_percent=100.0 * memory_used / memory_total,
                memory_used_mib=memory_used,
                memory_total_mib=memory_total,
            )
        )
    return samples


def sorted_device_ids(values: dict[str, float]) -> list[str]:
    return sorted(values, key=lambda value: int(value) if value.isdigit() else value)


def emit_report(label: str, kind: str, elapsed_seconds: float, peaks: PeakStats) -> None:
    assert peaks.gpu_utilization_by_device is not None
    assert peaks.gpu_memory_percent_by_device is not None
    assert peaks.gpu_memory_used_mib_by_device is not None
    assert peaks.gpu_memory_total_mib_by_device is not None
    assert peaks.gpu_names_by_device is not None

    gpu_util_max = max(peaks.gpu_utilization_by_device.values()) if peaks.gpu_utilization_by_device else None
    gpu_mem_max = max(peaks.gpu_memory_percent_by_device.values()) if peaks.gpu_memory_percent_by_device else None
    gpu_parts = []
    for device_id in sorted_device_ids(peaks.gpu_memory_percent_by_device):
        gpu_parts.append(
            f"gpu{device_id} "
            f"gpu_max={format_percent(peaks.gpu_utilization_by_device.get(device_id))} "
            f"gpu_mem_max={format_percent(peaks.gpu_memory_percent_by_device.get(device_id))} "
            f"gpu_mem_used_max={format_mib(peaks.gpu_memory_used_mib_by_device.get(device_id))}/"
            f"{format_mib(peaks.gpu_memory_total_mib_by_device.get(device_id))} "
            f"name={peaks.gpu_names_by_device.get(device_id, 'unknown')}"
        )

    memory_used = format_mib(peaks.memory_used_mib)
    memory_limit = format_mib(peaks.memory_limit_mib)
    print(
        f"[{now()}] [{kind}] {label} "
        f"elapsed_seconds={elapsed_seconds:.0f} samples={peaks.samples} "
        f"max_cpu_utilization={format_percent(peaks.cpu_utilization_percent)} "
        f"max_cpu_memory_utilization={format_percent(peaks.memory_utilization_percent)} "
        f"max_cpu_memory_used={memory_used}/{memory_limit} "
        f"max_gpu_utilization={format_percent(gpu_util_max)} "
        f"max_gpu_memory_utilization={format_percent(gpu_mem_max)} "
        + (" | ".join(gpu_parts) if gpu_parts else "gpu=n/a"),
        flush=True,
    )


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def optional_positive_float(value: str) -> float | None:
    if value == "":
        return None
    return positive_float(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monitor CPU, CPU memory, GPU, and GPU memory peaks.")
    parser.add_argument("--label", default="job", help="Label to include in each log line.")
    parser.add_argument("--sample-interval-seconds", type=positive_float, default=5.0)
    parser.add_argument("--report-interval-seconds", type=positive_float, default=60.0)
    parser.add_argument("--cpu-capacity", type=positive_float, help="CPU cores represented by 100%% utilization.")
    parser.add_argument("--memory-limit-mib", type=optional_positive_float, help="Memory limit represented by 100%%.")
    parser.add_argument("--nvidia-smi-bin", default=os.environ.get("NVIDIA_SMI_BIN", "nvidia-smi"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)

    args = build_parser().parse_args(argv)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    cpu_capacity = args.cpu_capacity or parse_optional_float(os.environ.get("SLURM_CPUS_PER_TASK", "")) or os.cpu_count() or 1
    memory_limit_mib = read_memory_limit_mib(args.memory_limit_mib)
    print(
        f"[{now()}] [resource-monitor] {args.label} monitor started "
        f"sample_interval={args.sample_interval_seconds:g}s report_interval={args.report_interval_seconds:g}s "
        f"cpu_capacity={cpu_capacity:g} memory_limit={format_mib(memory_limit_mib)} "
        f"nvidia_smi={args.nvidia_smi_bin}",
        flush=True,
    )

    started_at = time.monotonic()
    window_started_at = started_at
    next_report_at = started_at + args.report_interval_seconds
    previous_host_sample: HostSample | None = None
    previous_sample_at: float | None = None
    window_peaks = PeakStats()
    global_peaks = PeakStats()

    while RUNNING:
        current_time = time.monotonic()
        try:
            host_sample = load_host_sample(memory_limit_mib)
            cpu_percent = None
            if previous_host_sample is not None and previous_sample_at is not None:
                if host_sample.cpu_usage_seconds is not None and previous_host_sample.cpu_usage_seconds is not None:
                    delta_cpu = max(host_sample.cpu_usage_seconds - previous_host_sample.cpu_usage_seconds, 0.0)
                    delta_time = max(current_time - previous_sample_at, 1e-6)
                    cpu_percent = 100.0 * delta_cpu / delta_time / cpu_capacity
            previous_host_sample = host_sample
            previous_sample_at = current_time

            gpu_samples = load_gpu_samples(args.nvidia_smi_bin)
            window_peaks.samples += 1
            window_peaks.update_host(cpu_percent, host_sample)
            window_peaks.update_gpus(gpu_samples)
        except Exception as exc:  # noqa: BLE001 - monitors should not kill the job they observe.
            print(f"[{now()}] [resource-monitor] {args.label} sample failed: {exc!r}", file=sys.stderr, flush=True)

        current_time = time.monotonic()
        if current_time >= next_report_at:
            elapsed = max(current_time - window_started_at, 0.0)
            emit_report(args.label, "resource-window", elapsed, window_peaks)
            global_peaks.merge(window_peaks)
            window_peaks = PeakStats()
            window_started_at = current_time
            next_report_at = current_time + args.report_interval_seconds

        sleep_seconds = min(args.sample_interval_seconds, max(next_report_at - time.monotonic(), 0.1))
        time.sleep(sleep_seconds)

    if window_peaks.samples:
        global_peaks.merge(window_peaks)
        emit_report(args.label, "resource-window", max(time.monotonic() - window_started_at, 0.0), window_peaks)
    emit_report(args.label, "resource-summary", max(time.monotonic() - started_at, 0.0), global_peaks)
    print(f"[{now()}] [resource-monitor] {args.label} monitor stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
