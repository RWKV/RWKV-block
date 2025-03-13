"""
Model Utilities for Hugging Face Trainer

This module provides utilities for loading, saving, and managing Hugging Face models.
"""

import os
import logging
import torch
from typing import Dict, Any, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

logger = logging.getLogger(__name__)

def load_model_and_tokenizer(
    model_config: Dict[str, Any]
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a Hugging Face model and tokenizer based on the provided configuration.
    
    Args:
        model_config: Model configuration dictionary
        
    Returns:
        Tuple of (model, tokenizer)
    """
    name_or_path = model_config["name_or_path"]
    logger.info(f"Loading model and tokenizer from {name_or_path}")
    
    # Extract model-specific arguments
    model_args = model_config.get("model_args", {})
    
    # Determine precision settings
    use_bf16 = model_config.get("use_bf16", True)
    load_in_8bit = model_config.get("load_in_8bit", False)
    load_in_4bit = model_config.get("load_in_4bit", False)
    
    # Set torch dtype based on precision settings
    torch_dtype = torch.bfloat16 if use_bf16 else torch.float32
    
    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(name_or_path)
    
    # Ensure the tokenizer has padding token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            logger.warning("Tokenizer has no pad_token or eos_token, setting pad_token to [PAD]")
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    
    # Load model with appropriate settings
    model_load_args = {
        "torch_dtype": torch_dtype,
        **model_args  # Include any model-specific arguments
    }
    
    # Add quantization settings if specified
    if load_in_8bit:
        model_load_args["load_in_8bit"] = True
    elif load_in_4bit:
        model_load_args["load_in_4bit"] = True
    
    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        name_or_path,
        **model_load_args
    )
    
    # Move model to GPU if available
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Log model and tokenizer information
    logger.info(f"Model loaded: {model.__class__.__name__}")
    logger.info(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, tokenizer

def save_model_checkpoint(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    output_dir: str,
    step: Optional[int] = None,
    epoch: Optional[int] = None,
    is_final: bool = False
) -> str:
    """
    Save a model checkpoint.
    
    Args:
        model: The model to save
        tokenizer: The tokenizer to save
        output_dir: Base directory for saving checkpoints
        step: Current training step (optional)
        epoch: Current epoch (optional)
        is_final: Whether this is the final checkpoint
        
    Returns:
        Path to the saved checkpoint
    """
    # Determine checkpoint directory name
    if is_final:
        checkpoint_dir = os.path.join(output_dir, "final")
    elif step is not None:
        checkpoint_dir = os.path.join(output_dir, f"step-{step}")
    elif epoch is not None:
        checkpoint_dir = os.path.join(output_dir, f"epoch-{epoch}")
    else:
        checkpoint_dir = os.path.join(output_dir, "checkpoint")
    
    # Create directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    logger.info(f"Saving model checkpoint to {checkpoint_dir}")
    
    # Save model and tokenizer
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    
    return checkpoint_dir

def load_checkpoint(
    checkpoint_path: str,
    model_config: Dict[str, Any]
) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Load a model and tokenizer from a checkpoint.
    
    Args:
        checkpoint_path: Path to the checkpoint directory
        model_config: Model configuration dictionary
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    
    # Update the model path in the config
    updated_config = model_config.copy()
    updated_config["name_or_path"] = checkpoint_path
    
    # Load model and tokenizer
    return load_model_and_tokenizer(updated_config)
