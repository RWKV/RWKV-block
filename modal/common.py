import os
from pathlib import PurePosixPath
from typing import Union
import modal
import pathlib

# Get the current script's directory
MODAL_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(MODAL_SCRIPT_DIR)

# Workspace file filter
def workspace_file_filter(path):
    # Relative path from the project directory
    relative_path = os.path.relpath(path, PROJECT_DIR)
    split_path = relative_path.split(os.sep)
    # Ignore any folders with . in its name
    if any(part.startswith(".") for part in split_path):
        return True # True, means filter out
    
    # Ignore if any folders or files start or end with __
    if any(part.startswith("__") or part.endswith("__") for part in split_path):
        return True
    
    # Ignore .ipynb files
    if relative_path.endswith(".ipynb"):
        return True
    
    # Ignore if it starts with test_ext, or modal or specs
    ignore_proj_dirs = ["test_ext", "modal", "specs"]
    if split_path[0] in ignore_proj_dirs:
        return True
    
    # For note boook directory, ignore output and checkpoint directories
    # in any subdirectory
    if split_path[0] == "notebook":
        if any(part in ["output", "checkpoint"] for part in split_path[1:]):
            return True
        
    # print(f"Adding {relative_path} to the container")
    return False

# Model ENV variable to use
runtime_env = dict(
    HF_HUB_ENABLE_HF_TRANSFER="1",
    LOG_LEVEL="INFO",
    
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
    ZERO_INIT="1",
)
# Scan all local environment variables, and add them to the runtime env
# if they start with MODAL_
for key, value in os.environ.items():
    if key.startswith("MODAL_"):
        # Add to the runtime env
        runtime_env[key] = value

# Set HF_HUB_ENABLE_HF_TRANSFER if its configured
if "HF_HUB_ENABLE_HF_TRANSFER" in os.environ:
    runtime_env["HF_HUB_ENABLE_HF_TRANSFER"] = os.environ["HF_HUB_ENABLE_HF_TRANSFER"]

# Pytorch environment
TRAINER_BASE_IMAGE = modal.Image.from_registry(
    "pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel"
).apt_install(
    "git"
).pip_install(
    "huggingface_hub",
    "hf-transfer",
    "transformers",
    "wandb",
    "datasets",
    "accelerate",
).run_commands(
    # We use the lm-eval-harness repo directly, instead from our local repo
    # to reduce the "file sync" time, as this is a really large repo
    "mkdir -p /workspace/test_ext && cd /workspace/test_ext && git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness.git && cd lm-evaluation-harness && pip install -e .",
).add_local_file(
    # Add the main project requirements file to the container
    os.path.join(PROJECT_DIR, "requirements.txt"),
    remote_path="/workspace/requirements.txt",
    copy=True
).run_commands(
    "cd /workspace/ && pip install -r requirements.txt",
).env(runtime_env).entrypoint([])

# HF Cache Volume
HF_CACHE_VOLUME = modal.Volume.from_name("hf-cache-volume", create_if_missing=True)

# Baseline VOLUME_CONFIG
BASE_VOLUME_CONFIG: dict[Union[str, PurePosixPath], modal.Volume] = {
    "/root/.cache/huggingface": HF_CACHE_VOLUME
}

# Trainer env image, including the files
TRAINER_ENV_IMAGE = TRAINER_BASE_IMAGE.add_local_python_source("common").add_local_dir(
    # Add the local directory to the container
    PROJECT_DIR,
    # The path inside the container where the local directory will be mounted
    remote_path="/workspace",
    # Ignore files that are not needed in the container
    # This is a list of file patterns to ignore when syncing files from the host to the container.
    # It can be a list of strings or a single string.
    ignore=workspace_file_filter
)