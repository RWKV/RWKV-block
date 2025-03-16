#!/bin/bash

# Check if a followup command and/or args are provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <hf_model_name>"
    exit 1
fi

# Get the current file directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MODAL_COMMIT_HF_CACHE=1 "$SCRIPT_DIR/x-run.sh" huggingface-cli download "$1"

echo "-----------------------"
echo "| Note: Preloaded Hugging Face model $1"
echo "-----------------------"