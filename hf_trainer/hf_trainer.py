#!/usr/bin/env python
"""
Hugging Face Trainer CLI Tool

This script provides a command-line interface for training Hugging Face models
using the TextDatasetStreamer for efficient dataset handling.
"""

import os
import sys
import argparse
import json
import logging
import torch
from pathlib import Path
from typing import Dict, Any, Optional

# Try to import hjson, fallback to json if not available
try:
    import hjson
except ImportError:
    hjson = None
    print("Warning: hjson not installed, falling back to standard json. Install with: pip install hjson")

# Import local modules
from text_dataset_streamer import TextDatasetStreamer
from config_parser import parse_config, validate_config
from model_utils import load_model_and_tokenizer, save_model_checkpoint
from training_loop import train_model
from logging_utils import setup_logging, setup_wandb

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train a Hugging Face model with TextDatasetStreamer")
    
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to the JSONC/JSON configuration file"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default=None,
        help="Directory to save model checkpoints (overrides config file)"
    )
    
    parser.add_argument(
        "--resume_from_checkpoint", 
        type=str, 
        default=None,
        help="Path to a checkpoint to resume training from"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None,
        help="Random seed (overrides config file)"
    )
    
    parser.add_argument(
        "--no_wandb", 
        action="store_true",
        help="Disable Weights & Biases logging even if enabled in config"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode (more verbose logging)"
    )
    
    parser.add_argument(
        "--train_wkv_state",
        action="store_true",
        help="Train only WKV state (automatically sets freeze_full_weights=True and init_wkv_state=True)"
    )
    
    parser.add_argument(
        "--save_full_weights",
        action="store_true",
        help="Save all model weights instead of only trainable weights"
    )
    
    parser.add_argument(
        "--hf_model_path",
        type=str,
        default=None,
        help="Path to the Hugging Face model (overrides config file)"
    )
    
    return parser.parse_args()

def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load and parse the configuration file.
    
    Args:
        config_path: Path to the configuration file (JSONC or JSON)
        
    Returns:
        Dict containing the parsed configuration
    """
    with open(config_path, 'r') as f:
        config_str = f.read()
    
    # Try to parse as JSONC (using hjson library), fall back to JSON
    if hjson:
        try:
            return hjson.loads(config_str)
        except Exception as e:
            print(f"Warning: Failed to parse as JSONC, trying standard JSON: {e}")
            return json.loads(config_str)
    else:
        return json.loads(config_str)

def override_config_with_args(config: Dict[str, Any], args) -> Dict[str, Any]:
    """
    Override configuration values with command-line arguments.
    
    Args:
        config: The loaded configuration dictionary
        args: Parsed command-line arguments
        
    Returns:
        Updated configuration dictionary
    """
    # Create a deep copy to avoid modifying the original
    updated_config = config.copy()
    
    # Override output directory if specified
    if args.output_dir:
        if "training" not in updated_config:
            updated_config["training"] = {}
        updated_config["training"]["output_dir"] = args.output_dir
    
    # Override seed if specified
    if args.seed is not None:
        if "dataset" not in updated_config:
            updated_config["dataset"] = {}
        updated_config["dataset"]["seed"] = args.seed
    
    # Disable W&B if --no_wandb is specified
    if args.no_wandb and "logging" in updated_config and "wandb" in updated_config["logging"]:
        updated_config["logging"]["wandb"]["enabled"] = False
    
    # Override model path if specified
    if args.hf_model_path:
        if "model" not in updated_config:
            updated_config["model"] = {}
        updated_config["model"]["hf_model_path"] = args.hf_model_path
    
    return updated_config

def main():
    """Main entry point for the CLI tool."""
    args = parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logger = setup_logging(log_level)
    
    logger.info(f"Loading configuration from {args.config}")
    
    try:
        # Load and parse configuration
        config = load_config(args.config)
        
        # Override with command-line arguments
        config = override_config_with_args(config, args)
        
        # Handle WKV state training mode
        if args.train_wkv_state:
            if "model" not in config:
                config["model"] = {}
            config["model"]["freeze_full_weights"] = True
            config["model"]["init_wkv_state"] = True
            logger.info("WKV state training mode enabled: Setting freeze_full_weights=True and init_wkv_state=True")
        
        # Log save mode
        if args.save_full_weights:
            logger.info("Full weights saving mode enabled: All model weights will be saved in checkpoints")
        else:
            logger.info("Trainable weights saving mode enabled: Only trainable weights will be saved in checkpoints")
        
        # Log model path override if specified
        if args.hf_model_path:
            logger.info(f"Model path override: Using model from {args.hf_model_path}")
        
        # Validate configuration
        validate_config(config)
        
        # Create output directory if it doesn't exist
        output_dir = config.get("training", {}).get("output_dir", "./output")
        os.makedirs(output_dir, exist_ok=True)
        
        # Set up W&B logging if enabled
        if config.get("logging", {}).get("wandb", {}).get("enabled", False) and not args.no_wandb:
            setup_wandb(config["logging"]["wandb"])
        
        # Check for CUDA availability
        if torch.cuda.is_available():
            logger.info(f"CUDA is available. Using device: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("CUDA is not available. Training will be slow on CPU.")
        
        # Load model and tokenizer
        model, tokenizer = load_model_and_tokenizer(config["model"])
        
        # Create dataset
        dataset = TextDatasetStreamer(
            dataset_configs=config["dataset"]["datasets"],
            tokenizer=tokenizer,
            packing_batch_size=config["dataset"].get("packing_batch_size", 8),
            packing_context_length=config["dataset"].get("packing_context_length", 4096),
            shuffle_buffer_size=config["dataset"].get("shuffle_buffer_size", 100),
            seed=config["dataset"].get("seed", 42),
            rank=0,  # Single GPU training
            world_size=1  # Single GPU training
        )
        
        # Start training
        train_model(
            model=model,
            tokenizer=tokenizer,
            dataset=dataset,
            config=config,
            resume_from=args.resume_from_checkpoint,
            save_full_weights=args.save_full_weights
        )
        
        logger.info("Training completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
