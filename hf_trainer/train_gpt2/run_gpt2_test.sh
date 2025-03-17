#!/bin/bash
# Simple script to run the Hugging Face Trainer CLI with GPT2 test configuration

# Get the absolute path to the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

# Set up Python path
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Get the path to the test configuration
CONFIG_PATH="$PROJECT_ROOT/test/hf_trainer/train_gpt2/test_config.jsonc"

# Create a temporary output directory
OUTPUT_DIR="$PROJECT_ROOT/test/hf_trainer/train_gpt2/output"
mkdir -p "$OUTPUT_DIR"

echo "Running GPT2 test with configuration: $CONFIG_PATH"
echo "Output directory: $OUTPUT_DIR"

# Run the trainer
python "$PROJECT_ROOT/hf_trainer/hf_trainer.py" \
  --config "$CONFIG_PATH" \
  --output_dir "$OUTPUT_DIR" 

# Check the exit code
if [ $? -eq 0 ]; then
  echo "Test completed successfully!"
  echo "Output files:"
  ls -la "$OUTPUT_DIR"
else
  echo "Test failed!"
fi
