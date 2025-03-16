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
    hf_model_path = model_config["hf_model_path"]
    logger.info(f"Loading model and tokenizer from {hf_model_path}")
    
    # Extract model-specific arguments
    model_args = model_config.get("model_args", {})
    
    # Determine precision settings
    use_amp_bf16 = model_config.get("use_amp_bf16", False)
    load_in_8bit = model_config.get("load_in_8bit", False)
    load_in_4bit = model_config.get("load_in_4bit", False)
    
    # Set torch dtype based on precision settings
    torch_dtype = torch.bfloat16 if use_amp_bf16 else torch.float32
    
    # Load tokenizer first
    tokenizer = AutoTokenizer.from_pretrained(hf_model_path)
    
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
        hf_model_path,
        **model_load_args
    )
    
    # Check if we should force CPU usage
    force_cpu = model_config.get("force_cpu", False)
    
    # Move model to GPU if available and not forcing CPU
    if torch.cuda.is_available() and not force_cpu:
        model = model.cuda()
        logger.info("Using CUDA for training")
    else:
        if force_cpu:
            logger.info("Forcing CPU usage as specified in config")
        else:
            logger.info("CUDA not available, using CPU for training")
    
    # Log model and tokenizer information
    logger.info(f"Model loaded: {model.__class__.__name__}")
    logger.info(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Log all parameter names and their requires_grad status
    logger.info("Parameter trainability status:")
    non_trainable_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            non_trainable_params.append(name)
            logger.info(f"  {name}: requires_grad=False")
    
    if non_trainable_params:
        logger.info(f"Non-trainable parameters: {', '.join(non_trainable_params)}")
    else:
        logger.info("All parameters are trainable")

    # Count the trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Throw if there are no trainable parameters
    if trainable_params == 0:
        raise ValueError("No trainable parameters found in the model. Check your model configuration.")
    
    return model, tokenizer

def save_model_checkpoint(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    output_dir: str,
    step: Optional[int] = None,
    epoch: Optional[int] = None,
    is_final: bool = False,
    save_full_weights: bool = False
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
        save_full_weights: Whether to save all weights or only trainable weights (default: False)
        
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
    
    if save_full_weights:
        logger.info(f"Saving full model weights to {checkpoint_dir}")
        # Save all weights
        model.save_pretrained(checkpoint_dir, state_dict=model.state_dict())
    else:
        # Get a set of names of parameters that require gradients
        trainable_param_names = set()
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_param_names.add(name)
        
        # Save only trainable weights by filtering state_dict using the parameter names
        state_dict = model.state_dict()
        trainable_state_dict = {}
        
        # Note: For GPT-2 models, lm_head.weight is typically tied to the embedding weights
        # but appears separately in the state_dict. It may not be directly accessible through
        # named_parameters(), which is why it might show as "missing" (148/149 parameters).
        # This is expected behavior for GPT-2 models.
        
        for k, v in state_dict.items():
            # Check if this key exactly matches a parameter name or
            # if it's a key that contains a trainable parameter name
            # (handles cases where state_dict keys might have prefixes)
            if k in trainable_param_names or any(param_name in k for param_name in trainable_param_names):
                trainable_state_dict[k] = v
        
        # Count parameters
        total_params = len(state_dict)
        trainable_params = len(trainable_state_dict)
        
        logger.info(f"Saving trainable weights to {checkpoint_dir} ({trainable_params}/{total_params} parameters)")
        model.save_pretrained(checkpoint_dir, state_dict=trainable_state_dict)
    
    # -- We do not need to save the tokenizer
    # tokenizer.save_pretrained(checkpoint_dir)
    
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
    updated_config["hf_model_path"] = checkpoint_path
    
    # Load model and tokenizer
    return load_model_and_tokenizer(updated_config)
