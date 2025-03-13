# Hugging Face Trainer Tests

This directory contains a test for the Hugging Face Trainer CLI tool.

## Test Files

- `test_config.jsonc`: Configuration file for testing the trainer with GPT2
- `run_gpt2_test.sh`: Shell script to run the trainer with the test configuration

## Running the Test

To run the test, make sure you have the required dependencies installed:

```bash
pip install torch transformers datasets tqdm hjson
```

Then run the shell script:

```bash
# Make sure the script is executable
chmod +x run_gpt2_test.sh

# Run the test
./run_gpt2_test.sh
```

The script will:
1. Set up the Python path
2. Create a temporary output directory
3. Run the trainer with the test configuration
4. Display the results

## Test Configuration

The test configuration uses:

- A small GPT2 model (`gpt2`)
- A small subset of the WikiText-2 dataset
- Limited training steps and epochs for quick testing
- Small batch size and context length for efficient testing

You can modify the `test_config.jsonc` file to test different configurations.

## Output

The test will create a `test_output` directory containing:
- Model checkpoints
- Optimizer state
- Tokenizer files

You can examine these files to verify that the trainer is working correctly.
