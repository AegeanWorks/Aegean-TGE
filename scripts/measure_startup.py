#!/usr/bin/env python3
"""Measure the per-invocation startup path of `tge generate`.

The engine prints one line per phase (`load`, `tokenizer`, `encode`, `prefill`,
`decode`). This helper runs the binary several times and reports the median,
minimum and maximum of every phase it sees, plus the wall time between process
start and exit. It is deliberately read-only: no model is modified and no cache
is written.

Usage:
    python scripts/measure_startup.py --model path/to/model.gguf --runs 5
    python scripts/measure_startup.py --model ... --prompt "hello" --vocabulary lazy
"""

from __future__ import annotations

import argparse
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

PHASE_PATTERN = re.compile(r"^([a-z][a-z0-9_-]*)\s*:\s*([0-9]+(?:\.[0-9]+)?)\s*ms\b")
TOKEN_PATTERN = re.compile(
    r"^(prefill|decode)\s*:\s*([0-9]+)\s+tokens?\s+in\s+([0-9]+(?:\.[0-9]+)?)\s*ms"
)


def default_binary() -> Path:
    """The release CLI, resolved relative to this script, never an absolute path."""
    return Path(__file__).resolve().parents[1] / "build" / "release" / "bin" / "tge.exe"


def run_once(command: list[str], timeout: float) -> tuple[dict[str, float], float, str]:
    """Run one process and return its phase timings, wall milliseconds and stderr."""
    started = time.perf_counter()
    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )
    wall_ms = (time.perf_counter() - started) * 1e3
    phases: dict[str, float] = {}
    for line in completed.stdout.splitlines():
        match = PHASE_PATTERN.match(line.strip())
        if match is not None:
            phases[match.group(1)] = float(match.group(2))
            continue
        match = TOKEN_PATTERN.match(line.strip())
        if match is not None:
            phases[f"{match.group(1)}_ms"] = float(match.group(3))
    if completed.returncode != 0:
        tail = "\n".join(completed.stderr.splitlines()[-8:])
        raise RuntimeError(f"tge exited {completed.returncode}: {tail}")
    return phases, wall_ms, completed.stderr


def report(label: str, values: list[float]) -> None:
    """Print one median/min/max row."""
    median = statistics.median(values)
    print(f"{label:<14} {median:9.2f} {min(values):9.2f} {max(values):9.2f}")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="GGUF model path")
    parser.add_argument("--prompt", default="The capital of France is", help="text prompt")
    parser.add_argument("--runs", type=int, default=5, help="process launches (default 5)")
    parser.add_argument("--threads", type=int, default=6, help="worker threads (default 6)")
    parser.add_argument("--n", type=int, default=1, help="generated tokens (default 1)")
    parser.add_argument(
        "--vocabulary",
        choices=("auto", "eager", "lazy"),
        default="auto",
        help="tokenizer parse strategy (default auto)",
    )
    parser.add_argument("--binary", type=Path, default=default_binary(), help="tge executable")
    parser.add_argument("--timeout", type=float, default=300.0, help="per-run timeout seconds")
    arguments = parser.parse_args(argv)

    if arguments.runs < 1:
        print("measure_startup: --runs must be at least 1", file=sys.stderr)
        return 2
    if not arguments.binary.is_file():
        print(f"measure_startup: no binary at {arguments.binary}", file=sys.stderr)
        return 2

    command = [
        str(arguments.binary),
        "generate",
        "--model",
        arguments.model,
        "--prompt",
        arguments.prompt,
        "--n",
        str(arguments.n),
        "--threads",
        str(arguments.threads),
        "--vocabulary",
        arguments.vocabulary,
    ]

    samples: list[dict[str, float]] = []
    walls: list[float] = []
    for index in range(arguments.runs):
        phases, wall_ms, _stderr = run_once(command, arguments.timeout)
        samples.append(phases)
        walls.append(wall_ms)
        print(f"run {index + 1}: " + ", ".join(f"{key}={value:.1f}" for key, value in phases.items()))

    keys = sorted({key for sample in samples for key in sample})
    print()
    print(f"{'phase (ms)':<14} {'median':>9} {'min':>9} {'max':>9}")
    for key in keys:
        values = [sample[key] for sample in samples if key in sample]
        if values:
            report(key, values)
    report("wall", walls)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
