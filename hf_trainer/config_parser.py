"""
Configuration Parser for Hugging Face Trainer

This module provides functions for parsing and validating configuration files
for the Hugging Face Trainer CLI tool.
"""

import os
from typing import Dict, Any, List, Optional

def parse_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse and process the configuration dictionary.
    
    Args:
        config: The raw configuration dictionary
        
    Returns:
        Processed configuration dictionary with defaults applied
    """
    # Create a deep copy to avoid modifying the original
    processed_config = config.copy()
    
    # Apply defaults for model configuration
    if "model" not in processed_config:
        processed_config["model"] = {}
    
    model_config = processed_config["model"]
    model_config.setdefault("hf_model_path", "gpt2")
    model_config.setdefault("use_bf16", True)
    model_config.setdefault("load_in_8bit", False)
    model_config.setdefault("load_in_4bit", False)
    model_config.setdefault("force_cpu", False)
    
    # Support for model-specific arguments
    model_config.setdefault("model_args", {})
    
    # Apply defaults for training configuration
    if "training" not in processed_config:
        processed_config["training"] = {}
    
    training_config = processed_config["training"]
    training_config.setdefault("gradient_accumulation_steps", 4)
    training_config.setdefault("training_steps", 10000)  # Total number of training steps
    training_config.setdefault("save_steps", 1000)
    training_config.setdefault("save_total_limit", 3)
    training_config.setdefault("logging_steps", 100)
    training_config.setdefault("microbatch_size", 1)
    training_config.setdefault("output_dir", "./output")
    
    # Apply defaults for optimizer configuration
    if "optimizer" not in processed_config:
        processed_config["optimizer"] = {}
    
    optimizer_config = processed_config["optimizer"]
    optimizer_config.setdefault("name", "adamw")
    optimizer_config.setdefault("max_learning_rate", 5e-5)
    optimizer_config.setdefault("min_learning_rate", 1e-6)
    optimizer_config.setdefault("warmup_steps", 500)
    optimizer_config.setdefault("scheduler", "cosine")
    optimizer_config.setdefault("weight_decay", 0.01)
    optimizer_config.setdefault("beta1", 0.9)
    optimizer_config.setdefault("beta2", 0.999)
    optimizer_config.setdefault("epsilon", 1e-8)
    
    # Apply defaults for dataset configuration
    if "dataset" not in processed_config:
        processed_config["dataset"] = {}
    
    dataset_config = processed_config["dataset"]
    dataset_config.setdefault("packing_batch_size", 8)
    dataset_config.setdefault("packing_context_length", 4096)
    dataset_config.setdefault("shuffle_buffer_size", 100)
    dataset_config.setdefault("seed", 42)
    
    # Ensure datasets is a list
    if "datasets" not in dataset_config:
        dataset_config["datasets"] = []
    
    # Apply defaults for logging configuration
    if "logging" not in processed_config:
        processed_config["logging"] = {}
    
    logging_config = processed_config["logging"]
    if "wandb" not in logging_config:
        logging_config["wandb"] = {"enabled": False}
    else:
        logging_config["wandb"].setdefault("enabled", False)
    
    return processed_config

def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate the configuration dictionary.
    
    Args:
        config: The configuration dictionary to validate
        
    Raises:
        ValueError: If the configuration is invalid
    """
    # Validate model configuration
    if "model" not in config:
        raise ValueError("Model configuration is missing")
    
    model_config = config["model"]
    if "hf_model_path" not in model_config:
        raise ValueError("Model name or path is missing")
    
    # Validate training configuration
    if "training" not in config:
        raise ValueError("Training configuration is missing")
    
    training_config = config["training"]
    if training_config.get("training_steps", 0) <= 0:
        raise ValueError("training_steps must be positive")
    
    # Validate dataset configuration
    if "dataset" not in config:
        raise ValueError("Dataset configuration is missing")
    
    dataset_config = config["dataset"]
    if "datasets" not in dataset_config or not dataset_config["datasets"]:
        raise ValueError("At least one dataset must be specified")
    
    for i, ds_config in enumerate(dataset_config["datasets"]):
        if "hf_path" not in ds_config:
            raise ValueError(f"Dataset {i} is missing hf_path")
    
    # Validate optimizer and scheduler configuration
    if "optimizer" in config:
        optimizer_config = config["optimizer"]
        valid_optimizers = ["adamw", "adafactor", "sgd", "adam"]
        if optimizer_config.get("name") not in valid_optimizers:
            raise ValueError(f"Invalid optimizer: {optimizer_config.get('name')}. Valid options are: {', '.join(valid_optimizers)}")
        
        valid_schedulers = ["linear", "cosine", "constant", "constant_with_warmup"]
        if optimizer_config.get("scheduler") not in valid_schedulers:
            raise ValueError(f"Invalid scheduler: {optimizer_config.get('scheduler')}. Valid options are: {', '.join(valid_schedulers)}")
        
        if optimizer_config.get("max_learning_rate", 0) <= 0:
            raise ValueError("max_learning_rate must be positive")
    
    # Validate logging configuration
    if "logging" in config and "wandb" in config["logging"]:
        wandb_config = config["logging"]["wandb"]
        if wandb_config.get("enabled", False):
            if "project" not in wandb_config:
                raise ValueError("W&B project name is required when W&B logging is enabled")
