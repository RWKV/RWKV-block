# Hugging Face Trainer CLI Tool

A command-line interface for training Hugging Face models using the TextDatasetStreamer for efficient dataset handling.

## Features

- Support for all Hugging Face models (load remotely or locally)
- BF16 precision training
- Gradient accumulation and microbatching
- Configurable optimizers and learning rate schedulers
- Checkpointing and resuming training
- Optional Weights & Biases integration for logging
- Flexible dataset configuration with TextDatasetStreamer

## Installation

Ensure you have the required dependencies:

```bash
pip install torch transformers datasets tqdm wandb hjson
```

## Usage

```bash
python hf_trainer.py --config your_config.jsonc
```

### Command-line Arguments

- `--config`: Path to the JSONC/JSON configuration file (required)
- `--output_dir`: Directory to save model checkpoints (overrides config file)
- `--resume_from_checkpoint`: Path to a checkpoint to resume training from
- `--seed`: Random seed (overrides config file)
- `--no_wandb`: Disable Weights & Biases logging even if enabled in config
- `--debug`: Enable debug mode (more verbose logging)

## Configuration File

The configuration file uses JSONC format (JSON with Comments). The hjson library is used for parsing, which allows for a more flexible JSON syntax with comments. See `sample_config.jsonc` for a complete example.

### Model Configuration

```hjson
"model": {
  "hf_model_path": "gpt2",  // HF model name or local path
  "use_bf16": true,
  "load_in_8bit": false,
  "load_in_4bit": false,
  "force_cpu": false,  // Set to true to force CPU usage even if CUDA is available
  
  // Model-specific arguments passed directly to from_pretrained
  "model_args": {
    "trust_remote_code": true,
    // Add any other model-specific arguments here
  }
}
```

### Training Configuration

```hjson
"training": {
  "learning_rate": 5e-5,
  "weight_decay": 0.01,
  "num_train_epochs": 3,
  "gradient_accumulation_steps": 4,
  "warmup_steps": 500,
  "max_steps": -1,  // -1 means train for num_train_epochs
  "save_steps": 1000,
  "save_total_limit": 3,
  "logging_steps": 100,
  "microbatch_size": 1,  // Number of samples processed at once
  "output_dir": "./output"
}
```

### Optimizer Configuration

```hjson
"optimizer": {
  "name": "adamw",  // Options: adamw, adafactor, sgd, adam
  "beta1": 0.9,
  "beta2": 0.999,
  "epsilon": 1e-8
}
```

### Scheduler Configuration

```hjson
"scheduler": {
  "name": "cosine",  // Options: linear, cosine, constant, constant_with_warmup
  "num_warmup_steps": 500
}
```

### Dataset Configuration

```hjson
"dataset": {
  "packing_batch_size": 8,
  "packing_context_length": 4096,
  "shuffle_buffer_size": 100,
  "seed": 42,
  "datasets": [
    {
      "hf_path": "wikitext",
      "hf_args": {
        "name": "wikitext-103-v1",
        "split": "train"
      },
      "template": "{text}",
      "weight": 1.0
    },
    {
      "hf_path": "Anthropic/hh-rlhf",
      "hf_args": {
        "split": "train"
      },
      "template": "### Human: {human}\n\n### Assistant: {assistant}",
      "weight": 2.0
    }
  ]
}
```

### Logging Configuration

```hjson
"logging": {
  "wandb": {
    "enabled": false,
    "project": "hf-trainer",
    "name": "training-run-1",
    "entity": null,  // Your W&B username or team name
    "tags": ["gpt2", "finetune"]
  }
}
```

## Dataset Templates

The TextDatasetStreamer uses a template system to format dataset rows before tokenization. Templates use an enhanced version of Python's string formatting syntax with named placeholders corresponding to dataset column names.

### Template Features

1. Standard placeholders: `{text}`, `{url}`
2. Nested dictionary access using dot notation: `{metadata.author}`, `{header.title}`
3. Array indexing: `{questions[0].text}`, `{options[2]}`
4. Function calls: `{to_letter(score)}` - converts numbers to letters (A, B, C, ...)
   - Optional case parameter: `{to_letter(score, case='lower')}` for lowercase letters

### Template Examples

- Basic text: `"{text}"`
- With metadata: `"Title: {title}\n\nContent: {content}"`
- With URL: `"---\nurl: {url}\n---\n{text}"`
- Instruction format: `"### Instruction:\n{instruction}\n\n### Response:\n{response}"`
- Nested dictionary access: `"Author: {metadata.author}\nPublished: {metadata.date}"`
- Array indexing: `"Question: {questions[0].text}\nOptions: {options[0]}, {options[1]}, {options[2]}"`
- Numeric to letter conversion: `"Answer: {to_letter(answer_index)}"` (0 → A, 1 → B, etc.)
- Lowercase letters: `"Choice: {to_letter(choice, case='lower')}"` (0 → a, 1 → b, etc.)

## Examples

### Training GPT-2 on WikiText

```bash
python hf_trainer.py --config sample_config.jsonc
```

### Resuming Training from a Checkpoint

```bash
python hf_trainer.py --config sample_config.jsonc --resume_from_checkpoint ./output/step-1000
```

### Using a Different Output Directory

```bash
python hf_trainer.py --config sample_config.jsonc --output_dir ./my_model_checkpoints
```

### Disabling W&B Logging

```bash
python hf_trainer.py --config sample_config.jsonc --no_wandb
```
