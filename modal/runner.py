from pathlib import PurePosixPath
from typing import Union
import modal, os
from common import TRAINER_ENV_IMAGE, BASE_VOLUME_CONFIG, HF_CACHE_VOLUME

# Timing constrataints
MINUTES = 60  # seconds
HOURS = 60 * MINUTES

# Get time out hours from env
MODAL_TIMEOUT_HOURS = float(os.environ.get("MODAL_TIMEOUT_HOURS", 24))

# Get the system type (CPU or GPU)
MODAL_SYSTEM_TYPE = os.environ.get("MODAL_SYSTEM_TYPE", "CPU")

# Compute the GPU to use based on the system type
if MODAL_SYSTEM_TYPE == "CPU":
    gpu_to_use = None
elif MODAL_SYSTEM_TYPE == "1xL4":
    gpu_to_use = "L4:1"
elif MODAL_SYSTEM_TYPE == "2xL4":
    gpu_to_use = "L4:2"
elif MODAL_SYSTEM_TYPE == "4xL4":
    gpu_to_use = "L4:4"
elif MODAL_SYSTEM_TYPE == "4xA100":
    gpu_to_use = "A100-80GB:4"
elif MODAL_SYSTEM_TYPE == "2xA100":
    gpu_to_use = "A100-80GB:2"
elif MODAL_SYSTEM_TYPE == "8xH100":
    gpu_to_use = "H100:8"
elif MODAL_SYSTEM_TYPE == "4xH100":
    gpu_to_use = "H100:4"
elif MODAL_SYSTEM_TYPE == "2xH100":
    gpu_to_use = "H100:2"
elif MODAL_SYSTEM_TYPE == "1xH100":
    gpu_to_use = "H100:1"
else:
    raise ValueError(f"Unknown system type {MODAL_SYSTEM_TYPE}")

# Modal CPU / RAM allocation
MODAL_VCPU = float( os.environ.get("MODAL_VCPU", "0.125") )
MODAL_RAM_GB = float( os.environ.get("MODAL_RAM_GB", "0.25") )

# Prepare the output/checkpoint volumes, if configured
MODAL_WORKING_VOLUME_PATH = os.environ.get("MODAL_WORKING_VOLUME_PATH", False)
MODAL_WORKING_VOLUME_CHECKPOINT = os.environ.get("MODAL_WORKING_VOLUME_CHECKPOINT", False)

# Output and checkpoint volumes to use
OUTPUT_VOLUME = None
CHECKPOINT_VOLUME = None

# Clone the BASE_VOLUME_CONFIG
VOLUME_CONFIG = BASE_VOLUME_CONFIG.copy()

if MODAL_WORKING_VOLUME_PATH == "":
    MODAL_WORKING_VOLUME_PATH = False

if MODAL_WORKING_VOLUME_PATH:
    # Convert the path to a name (replace / with -)
    MODAL_WORKING_VOLUME_NAME_PREFIX = os.environ.get("MODAL_WORKING_VOLUME_NAME_PREFIX", MODAL_WORKING_VOLUME_PATH.replace("/", "-"))

    # Create a volume for the output/checkpoint
    OUTPUT_VOLUME = modal.Volume.from_name(
        f"{MODAL_WORKING_VOLUME_NAME_PREFIX}-output",
        create_if_missing=True
    )

    # Checkpoint volume
    CHECKPOINT_VOLUME = modal.Volume.from_name(
        f"{MODAL_WORKING_VOLUME_NAME_PREFIX}-checkpoint",
        create_if_missing=True
    )

    # Add the volumes to the config
    VOLUME_CONFIG[f"/workspace/{MODAL_WORKING_VOLUME_PATH}/output"] = OUTPUT_VOLUME
    VOLUME_CONFIG[f"/workspace/{MODAL_WORKING_VOLUME_PATH}/checkpoint"] = CHECKPOINT_VOLUME

    # VOLUME_CONFIG = {
    #     "/root/.cache/huggingface": HF_CACHE_VOLUME,
    #     f"/workspace/{MODAL_WORKING_VOLUME_PATH}/output": OUTPUT_VOLUME,
    #     f"/workspace/{MODAL_WORKING_VOLUME_PATH}/checkpoint": CHECKPOINT_VOLUME
    # }

else:
    # Disable checkpointing
    MODAL_WORKING_VOLUME_CHECKPOINT = False

# App operator
app = modal.App(
    name=f"layerwise-trainer-{MODAL_SYSTEM_TYPE}",
    # Set the image to use for the app
    image=TRAINER_ENV_IMAGE,
    # # Disable local source packages
    # include_source=True
)

# Secrets list
secret_list = []

# Get the wandb secret if possible
try:
    wandb_secret = modal.Secret.from_name("wandb-secret")
    secret_list.append(wandb_secret)
except Exception as e:
    print(f"Skipping wandb : Failed to get wandb secret.")
    pass

@app.function(
    # Volume config
    volumes=VOLUME_CONFIG,
    # Get secrets from the environment
    secrets=secret_list,
    # Set the timeout for the app
    timeout=int(MODAL_TIMEOUT_HOURS * HOURS),
    # GPU allocation
    gpu=gpu_to_use,
    # Number of times for container reuse
    max_inputs=1,
    # CPU and RAM minimum allocation
    cpu=(MODAL_VCPU),
    memory=int(MODAL_RAM_GB*1024),
)
def run_cmd(
    cmd_args = ["echo", "hello world"],
    cwd_path = "/workspace/"
):
    """
    Run a command inside a folder, within the modal environment
    """
    import subprocess

    # If command is a string, convert to a list via JSON
    if isinstance(cmd_args, str):
        import json
        cmd_args = json.loads(cmd_args)

    # Run with error code propagation
    if exit_code := subprocess.call(cmd_args, cwd=cwd_path):
        raise RuntimeError(f"Command '{cmd_args}' failed with exit code {exit_code}")
    
    # Check against ENV variables for VOL_COMMIT_HF_CACHE
    if "MODAL_COMMIT_HF_CACHE" in os.environ:
        print("## Updating HF_CACHE_VOLUME")
        HF_CACHE_VOLUME.commit()

    if MODAL_WORKING_VOLUME_CHECKPOINT is not False:
        print("## Updating CHECKPOINT volume")
        CHECKPOINT_VOLUME.commit()
        print("## Updating OUTPUT volume")
        OUTPUT_VOLUME.commit()

@app.local_entrypoint()
def main(
    cmd_args = ["echo", "hello world"],
    cwd_path = "/workspace/"
):
    run_cmd.remote(
        cmd_args=cmd_args,
        cwd_path=cwd_path
    )