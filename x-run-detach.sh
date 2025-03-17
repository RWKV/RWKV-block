#!/bin/bash

# Check if a followup command and/or args are provided
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <command> [args...]"
    exit 1
fi

# Check if `modal environment list` executes and returns a non-zero exit code
HAS_MODAL_SETUP=0
if modal environment list > /dev/null 2>&1; then
    HAS_MODAL_SETUP=1
fi

# If MODAL_ENVIRONMENT is not set, assume no modal
if [ -z "$MODAL_ENVIRONMENT" ]; then
    echo "-----------------------"
    echo "| Note: No modal environment found."
    echo "-----------------------"
    HAS_MODAL_SETUP=0
elif [ -n "$MODAL_DISABLE" ]; then
    echo "-----------------------"
    echo "| Note: Modal is disabled."
    echo "-----------------------"
    HAS_MODAL_SETUP=0
fi

# If no modal environment is set up, run the provided command and args
if [ $HAS_MODAL_SETUP -eq 0 ]; then
    echo "-----------------------"
    echo "| Note: No modal setup found. Running command directly."
    echo "-----------------------"
    "$@"
    exit $?
fi

# Get the current file directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Get the current working directory
CURRENT_DIR="$(pwd)"

# Get the command arguments into an array
CMD_ARGS=("$@")

# For every argument, replace instances of the script dir matches with "/workspace"
# preserve any traling string values after the script dir
for i in "${!CMD_ARGS[@]}"; do
    # Check if the argument starts with the script directory
    if [[ "${CMD_ARGS[$i]}" == "$SCRIPT_DIR" ]]; then
        CMD_ARGS[$i]="/workspace"
    elif [[ "${CMD_ARGS[$i]}" == "$SCRIPT_DIR"* ]]; then
        # Replace the script directory with "/workspace"
        CMD_ARGS[$i]="${CMD_ARGS[$i]#$SCRIPT_DIR/}"
        CMD_ARGS[$i]="/workspace/${CMD_ARGS[$i]}"
    fi
done

# Convert the command and its args into a JSON string array
CMD_JSON="$(printf '%s\n' "${CMD_ARGS[@]}" | jq -R . | jq -s .)"

# Get the relative path of the current directory, from the script directory
# if the current directory starts with the script directory, remove it
if [[ "$CURRENT_DIR" == "$SCRIPT_DIR" ]]; then
    RELATIVE_PATH=""
elif [[ "$CURRENT_DIR" == "$SCRIPT_DIR"* ]]; then
    RELATIVE_PATH="${CURRENT_DIR#$SCRIPT_DIR/}"
else
    # Otherwise, use the current directory as is
    RELATIVE_PATH="$CURRENT_DIR"
fi

# # Print the relative path
# echo "Relative path: $RELATIVE_PATH"

# Set the CWD to the SCRIPT_DIR
cd "$SCRIPT_DIR"

# # Log file to use
# LOG_FILE="./modal-logs/run-$(date +%s).log"
# mkdir -p "./modal-logs"

# Run the command in a modal environment
TERM=dumb modal run -d modal/runner.py --cmd-args "$CMD_JSON" --cwd-path "/workspace/$RELATIVE_PATH" 

# > "$LOG_FILE" 2>&1 &

# # Get the PID of the last background command
# PID=$!

# # Tail the log file in the background, until the command finishes
# tail -f "$LOG_FILE" &
# TAIL_PID=$!

# # Wait for the command to finish
# wait $PID
# # Get the exit code of the command
# EXIT_CODE=$?

# # Cleanup
# sleep 1
# kill $TAIL_PID
    
# # Print the exit code
# echo "-----------------------"
# echo "| Note: Command finished with exit code $EXIT_CODE"
# echo "| Log file: $LOG_FILE"
# echo "-----------------------"

# exit $EXIT_CODE