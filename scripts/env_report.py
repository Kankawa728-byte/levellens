"""C1: print the hardware and software of the machine that runs the jobs.

    python scripts/env_report.py                  # a Markdown table on the screen
    python scripts/env_report.py --json out.json  # the same values as JSON

The table in `docs/compute.md` is this output. Every job script runs the command first, so each
log starts with the versions and with the line that says whether PyTorch sees a GPU.

Only the standard library is needed. Anything that is not installed is reported as "not installed"
instead of raising, so the command also works in a half-built environment.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

PACKAGES = ("torch", "transformers", "tokenizers", "datasets", "numpy", "pandas", "scikit-learn", "optuna", "lightgbm")
NVIDIA_QUERY = "name,driver_version,memory.total,compute_cap"
# Environment variables that change where a run reads and writes. Reported so that a log can be traced back.
WATCHED = ("LEVELLENS_DATA", "LEVELLENS_CACHE", "LEVELLENS_RESULTS", "LEVELLENS_DEVICE", "HF_HOME", "HF_HUB_OFFLINE")


def package_versions() -> dict[str, str]:
    """Return the installed version of every package of the project, or "not installed"."""
    found = {}
    for name in PACKAGES:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = "not installed"
    return found


def nvidia_smi() -> list[dict[str, str]]:
    """Return one entry per GPU as `nvidia-smi` reports it, or an empty list when there is no GPU."""
    if not shutil.which("nvidia-smi"):
        return []
    try:
        output = subprocess.run(
            ["nvidia-smi", f"--query-gpu={NVIDIA_QUERY}", "--format=csv,noheader"],
            capture_output=True, text=True, check=True, timeout=60,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    keys = NVIDIA_QUERY.split(",")
    return [dict(zip(keys, [field.strip() for field in line.split(",")])) for line in output.splitlines() if line.strip()]


def torch_report() -> dict:
    """Return what PyTorch says about itself and about the GPUs it can use."""
    try:
        import torch
    except ImportError:
        return {"installed": False}
    report = {
        "installed": True,
        "version": str(torch.__version__),
        "built_for_cuda": torch.version.cuda or "cpu build",
        "cuda_available": bool(torch.cuda.is_available()),
        "n_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpus": [],
    }
    for index in range(report["n_gpus"]):
        properties = torch.cuda.get_device_properties(index)
        report["gpus"].append({
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "capability": f"{properties.major}.{properties.minor}",
            "memory_gb": round(properties.total_memory / 1024**3, 1),
            "bf16": bool(torch.cuda.is_bf16_supported()),
        })
    return report


def collect() -> dict:
    """Return the whole report: machine, Python, packages, GPUs and the watched environment variables."""
    return {
        "host_kind": "GPU node" if nvidia_smi() else "CPU node",
        "slurm_job": os.environ.get("SLURM_JOB_ID", ""),
        "slurm_gpus": os.environ.get("SLURM_JOB_GRES", os.environ.get("SLURM_GPUS_ON_NODE", "")),
        "cpus": os.environ.get("SLURM_CPUS_PER_TASK", str(os.cpu_count())),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "packages": package_versions(),
        "nvidia_smi": nvidia_smi(),
        "torch": torch_report(),
        "environment": {name: os.environ.get(name, "") for name in WATCHED},
    }


def as_markdown(report: dict) -> str:
    """Return the report as the Markdown table that goes into `docs/compute.md`."""
    torch_part = report["torch"]
    gpu = (report["nvidia_smi"] or [{}])[0]
    rows = [
        ("Machine", report["host_kind"] + (f", Slurm job {report['slurm_job']}" if report["slurm_job"] else "")),
        ("GPU", gpu.get("name", "none")),
        ("GPU memory", gpu.get("memory.total", "n/a")),
        ("Compute capability", gpu.get("compute_cap", "n/a")),
        ("Driver", gpu.get("driver_version", "n/a")),
        ("CPU cores for this job", report["cpus"]),
        ("Operating system", report["platform"]),
        ("Python", report["python"]),
    ]
    if torch_part["installed"]:
        rows += [
            ("PyTorch", f"{torch_part['version']} (built for CUDA {torch_part['built_for_cuda']})"),
            ("PyTorch sees a GPU", "yes" if torch_part["cuda_available"] else "no"),
        ]
        if torch_part["gpus"]:
            first = torch_part["gpus"][0]
            rows.append(("bfloat16 supported", "yes" if first["bf16"] else "no"))
    else:
        rows.append(("PyTorch", "not installed"))
    rows += [(name, value) for name, value in report["packages"].items() if name != "torch"]

    lines = ["| What | Value |", "|---|---|"]
    lines += [f"| {name} | {value} |" for name, value in rows]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the command line. The exit code is 1 when a GPU was required but PyTorch cannot see one."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", type=Path, metavar="FILE", help="also write the values as JSON")
    parser.add_argument("--require-gpu", action="store_true", help="exit 1 unless PyTorch sees a GPU")
    args = parser.parse_args(argv)

    report = collect()
    print(as_markdown(report))
    set_variables = {name: value for name, value in report["environment"].items() if value}
    if set_variables:
        print("\nEnvironment:")
        for name, value in set_variables.items():
            print(f"  {name}={value}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"\nWritten to {args.json}")
    if args.require_gpu and not report["torch"].get("cuda_available"):
        print("\nFAIL: PyTorch does not see a GPU on this machine.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
