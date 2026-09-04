"""Modal helpers for running assignment 5 jobs.

Example usage:

import sys
from cs336_alignment.modal_utils import app, quote_command, submit_commands

def build_run_commands(args):
    # Suppose args.seeds = '0,1,2,3'
    return [
        [sys.executable, "-u", "scripts/grpo.py", "--seed", seed]
        for seed in args.seeds.split(',')
    ]

@app.local_entrypoint(name=...)
def modal_main(*argv: str) -> None:
    args = make_parser().parse_args(list(argv))
    commands = build_run_commands(args)
    submit_commands(commands)
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import modal


# Two GPUs are required: CUDA device 0 trains the policy and device 1 runs vLLM.
# Environment overrides make it easy to use a different GPU type or app name
# without changing source code.
GPU = os.environ.get("CS336_MODAL_GPU", "L40S:2")
MAX_CONTAINERS = int(os.environ.get("CS336_MODAL_MAX_CONTAINERS", "1"))
REMOTE_ROOT = "/root"
RUN_TIMEOUT_SECONDS = int(
    os.environ.get("CS336_MODAL_TIMEOUT_SECONDS", str(4 * 60 * 60))
)
WANDB_SECRET_NAME = "wandb-secret"
HF_CACHE_MOUNT_PATH = "/cache/huggingface"

app = modal.App(f"cs336-a5-rlvr")
wandb_secret = modal.Secret.from_name(WANDB_SECRET_NAME)
hf_cache_volume = modal.Volume.from_name(
    f"cs336-a5-huggingface-cache",
    create_if_missing=True,
)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.1-devel-ubuntu22.04",
        add_python="3.12",
    )
    .uv_sync(extras=["gpu"])
    .workdir(REMOTE_ROOT)
    .add_local_dir("cs336_alignment", f"{REMOTE_ROOT}/cs336_alignment")
    .add_local_dir("data/gsm8k", f"{REMOTE_ROOT}/data/gsm8k")
    .add_local_dir("scripts", f"{REMOTE_ROOT}/scripts")
    .add_local_file("pyproject.toml", f"{REMOTE_ROOT}/pyproject.toml")
    .add_local_file("uv.lock", f"{REMOTE_ROOT}/uv.lock")
)
for optional_file in ("AGENTS.md", "CLAUDE.md"):
    if Path(optional_file).is_file():
        image = image.add_local_file(
            optional_file,
            f"{REMOTE_ROOT}/{optional_file}",
        )


def quote_command(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


@app.function(
    image=image,
    gpu=GPU,
    timeout=RUN_TIMEOUT_SECONDS,
    max_containers=MAX_CONTAINERS,
    volumes={HF_CACHE_MOUNT_PATH: hf_cache_volume},
    env={
        "HF_HOME": HF_CACHE_MOUNT_PATH,
        "HF_HUB_CACHE": f"{HF_CACHE_MOUNT_PATH}/hub",
        "TOKENIZERS_PARALLELISM": "false",
    },
    secrets=[wandb_secret],
)
def run_command(command: list[str]) -> str:
    command_str = quote_command(command)
    print(command_str, flush=True)
    try:
        subprocess.run(command, check=True)
    finally:
        hf_cache_volume.commit()
    return command_str


def submit_commands(commands: list[list[str]]) -> None:
    print(
        f"Submitting {len(commands)} Modal jobs "
        f"with max_containers={MAX_CONTAINERS}, gpu={GPU}, "
        f"timeout={RUN_TIMEOUT_SECONDS}s.",
        flush=True,
    )
    failures = []
    for idx, result in enumerate(run_command.map(commands, return_exceptions=True)):
        command = commands[idx]
        command_str = quote_command(command)
        if isinstance(result, BaseException):
            print(f"Failed: {command_str}", flush=True)
            print(f"Error: {result!r}", flush=True)
            failures.append(command_str)
        else:
            print(f"Completed: {result}", flush=True)

    if failures:
        print(f"{len(failures)} of {len(commands)} Modal jobs failed.", flush=True)
        raise SystemExit(1)
