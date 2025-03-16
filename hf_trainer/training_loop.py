"""
Training Loop Implementation for Hugging Face Trainer

This module provides the main training loop implementation for the Hugging Face Trainer CLI tool.
"""

import os
import time
import math
import logging
import torch
from typing import Dict, Any, Optional, Tuple, List
from tqdm import tqdm
from transformers import (
    PreTrainedModel, 
    PreTrainedTokenizer,
    get_scheduler,
    AdamW,
    Adafactor
)

from text_dataset_streamer import TextDatasetStreamer
from model_utils import save_model_checkpoint, load_checkpoint
from logging_utils import log_metrics

logger = logging.getLogger(__name__)

def create_optimizer(
    model: PreTrainedModel,
    config: Dict[str, Any]
) -> torch.optim.Optimizer:
    """
    Create an optimizer based on the configuration.
    
    Args:
        model: The model to optimize
        config: Configuration dictionary
        
    Returns:
        Optimizer instance
    """
    optimizer_config = config.get("optimizer", {})
    optimizer_name = optimizer_config.get("name", "adamw").lower()
    
    # Get parameters that require gradients
    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    
    # Get optimizer hyperparameters
    lr = config["training"].get("learning_rate", 5e-5)
    weight_decay = config["training"].get("weight_decay", 0.01)
    
    if optimizer_name == "adamw":
        beta1 = optimizer_config.get("beta1", 0.9)
        beta2 = optimizer_config.get("beta2", 0.999)
        epsilon = optimizer_config.get("epsilon", 1e-8)
        
        optimizer = AdamW(
            params_to_optimize,
            lr=lr,
            betas=(beta1, beta2),
            eps=epsilon,
            weight_decay=weight_decay
        )
    elif optimizer_name == "adafactor":
        optimizer = Adafactor(
            params_to_optimize,
            lr=lr,
            weight_decay=weight_decay,
            scale_parameter=optimizer_config.get("scale_parameter", True),
            relative_step=optimizer_config.get("relative_step", False),
            warmup_init=optimizer_config.get("warmup_init", False)
        )
    elif optimizer_name == "sgd":
        momentum = optimizer_config.get("momentum", 0.9)
        
        optimizer = torch.optim.SGD(
            params_to_optimize,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
    elif optimizer_name == "adam":
        beta1 = optimizer_config.get("beta1", 0.9)
        beta2 = optimizer_config.get("beta2", 0.999)
        epsilon = optimizer_config.get("epsilon", 1e-8)
        
        optimizer = torch.optim.Adam(
            params_to_optimize,
            lr=lr,
            betas=(beta1, beta2),
            eps=epsilon,
            weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    
    return optimizer

def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: Dict[str, Any],
    num_training_steps: int
) -> Optional[torch.optim.lr_scheduler.LRScheduler]:
    """
    Create a learning rate scheduler based on the configuration.
    
    Args:
        optimizer: The optimizer
        config: Configuration dictionary
        num_training_steps: Total number of training steps
        
    Returns:
        Scheduler instance or None if not configured
    """
    scheduler_config = config.get("scheduler", {})
    scheduler_name = scheduler_config.get("name", "cosine").lower()
    
    # Get warmup steps
    num_warmup_steps = scheduler_config.get(
        "num_warmup_steps", 
        config["training"].get("warmup_steps", 0)
    )
    
    # Create scheduler
    scheduler = get_scheduler(
        name=scheduler_name,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    return scheduler

def train_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    dataset: TextDatasetStreamer,
    config: Dict[str, Any],
    resume_from: Optional[str] = None,
    save_full_weights: bool = False
) -> PreTrainedModel:
    """
    Train a model using the provided dataset and configuration.
    
    Args:
        model: The model to train
        tokenizer: The tokenizer
        dataset: TextDatasetStreamer instance
        config: Configuration dictionary
        resume_from: Path to a checkpoint to resume from (optional)
        save_full_weights: Whether to save all weights or only trainable weights (default: False)
        
    Returns:
        Trained model
    """
    # Extract training configuration
    training_config = config["training"]
    
    # Set up training parameters
    num_train_epochs = training_config.get("num_train_epochs", 3)
    max_steps = training_config.get("max_steps", -1)
    gradient_accumulation_steps = training_config.get("gradient_accumulation_steps", 4)
    microbatch_size = training_config.get("microbatch_size", 1)
    
    # Set up logging and checkpointing parameters
    logging_steps = training_config.get("logging_steps", 100)
    save_steps = training_config.get("save_steps", 1000)
    save_total_limit = training_config.get("save_total_limit", 3)
    output_dir = training_config.get("output_dir", "./output")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up W&B if enabled
    wandb_run = None
    if config.get("logging", {}).get("wandb", {}).get("enabled", False):
        try:
            import wandb
            wandb_run = wandb.run
        except ImportError:
            logger.warning("W&B logging enabled but wandb not installed")
    
    # Calculate total training steps
    packing_batch_size = config["dataset"].get("packing_batch_size", 8)
    total_batch_size = packing_batch_size * gradient_accumulation_steps
    
    if max_steps > 0:
        num_training_steps = max_steps
    else:
        # For streaming datasets, we need to estimate the number of steps
        # based on the number of epochs and an estimate of dataset size
        estimated_dataset_size = config.get("estimated_dataset_size", 1000000)
        num_training_steps = math.ceil(
            estimated_dataset_size * num_train_epochs / total_batch_size
        )
    
    logger.info(f"Total training steps: {num_training_steps}")
    logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info(f"Microbatch size: {microbatch_size}")
    logger.info(f"Effective batch size: {total_batch_size}")
    
    # Create optimizer and scheduler
    optimizer = create_optimizer(model, config)
    scheduler = create_scheduler(optimizer, config, num_training_steps)
    
    # Resume from checkpoint if specified
    global_step = 0
    if resume_from:
        # Load checkpoint state
        checkpoint_state_path = os.path.join(resume_from, "optimizer.pt")
        if os.path.exists(checkpoint_state_path):
            logger.info(f"Loading optimizer and scheduler states from {checkpoint_state_path}")
            checkpoint_state = torch.load(checkpoint_state_path)
            global_step = checkpoint_state.get("global_step", 0)
            optimizer.load_state_dict(checkpoint_state.get("optimizer_state_dict"))
            if scheduler and "scheduler_state_dict" in checkpoint_state:
                scheduler.load_state_dict(checkpoint_state["scheduler_state_dict"])
    
    # Set up training state
    model.train()
    completed_steps = global_step
    
    # Track checkpoints to keep
    checkpoints = []
    
    # Initialize progress bar
    progress_bar = tqdm(
        range(num_training_steps),
        desc="Training",
        initial=global_step,
        disable=False
    )
    
    # Training loop
    logger.info("Starting training loop")
    
    # Initialize metrics
    tr_loss = 0.0
    logging_loss = 0.0
    
    # Get dataset iterator
    dataset_iterator = iter(dataset)
    
    # Main training loop
    while completed_steps < num_training_steps:
        accumulated_loss = 0.0
        
        # Gradient accumulation loop
        for _ in range(gradient_accumulation_steps):
            try:
                batch = next(dataset_iterator)
            except StopIteration:
                # This shouldn't happen with TextDatasetStreamer as it reshuffles automatically
                logger.warning("Dataset iterator exhausted, reshuffling")
                dataset_iterator = iter(dataset)
                batch = next(dataset_iterator)
            
            # Get input_ids tensor
            input_ids = batch["input_ids"]
            
            # Process in microbatches
            microbatch_losses = []
            for mb_start in range(0, input_ids.size(0), microbatch_size):
                mb_end = min(mb_start + microbatch_size, input_ids.size(0))
                microbatch = input_ids[mb_start:mb_end].to(model.device)
                
                # Forward pass
                # Use torch.amp.autocast with 'cpu' device type when CUDA is not available or force_cpu is True
                force_cpu = config["model"].get("force_cpu", False)
                device_type = 'cuda' if (torch.cuda.is_available() and not force_cpu) else 'cpu'
                with torch.amp.autocast(device_type=device_type, enabled=config["model"].get("use_bf16", True)):
                    outputs = model(
                        input_ids=microbatch,
                        labels=microbatch,
                        return_dict=True
                    )
                
                # Get loss
                loss = outputs.loss
                
                # Scale loss for gradient accumulation
                scaled_loss = loss / gradient_accumulation_steps
                
                # Backward pass
                scaled_loss.backward()
                
                # Track loss
                microbatch_losses.append(loss.detach().float())
            
            # Calculate average loss for this batch
            batch_loss = torch.stack(microbatch_losses).mean().item()
            accumulated_loss += batch_loss
        
        # Update weights
        optimizer.step()
        if scheduler:
            scheduler.step()
        optimizer.zero_grad()
        
        # Update metrics
        tr_loss += accumulated_loss / gradient_accumulation_steps
        
        # Update progress
        completed_steps += 1
        progress_bar.update(1)
        
        # Log metrics
        if completed_steps % logging_steps == 0:
            # Calculate average loss
            avg_loss = (tr_loss - logging_loss) / logging_steps
            logging_loss = tr_loss
            
            # Log metrics
            metrics = {"loss": avg_loss}
            if scheduler:
                metrics["learning_rate"] = scheduler.get_last_lr()[0]
            
            log_metrics(metrics, step=completed_steps, wandb_run=wandb_run)
        
        # Save checkpoint
        if completed_steps % save_steps == 0:
            checkpoint_dir = save_model_checkpoint(
                model=model,
                tokenizer=tokenizer,
                output_dir=output_dir,
                step=completed_steps,
                save_full_weights=save_full_weights
            )
            
            # Save optimizer and scheduler states
            checkpoint_state = {
                "global_step": completed_steps,
                "optimizer_state_dict": optimizer.state_dict(),
            }
            if scheduler:
                checkpoint_state["scheduler_state_dict"] = scheduler.state_dict()
            
            torch.save(
                checkpoint_state,
                os.path.join(checkpoint_dir, "optimizer.pt")
            )
            
            # Add to checkpoints list
            checkpoints.append(checkpoint_dir)
            
            # Remove old checkpoints if needed
            if save_total_limit > 0 and len(checkpoints) > save_total_limit:
                # Remove oldest checkpoint
                checkpoint_to_remove = checkpoints.pop(0)
                logger.info(f"Removing old checkpoint: {checkpoint_to_remove}")
                try:
                    import shutil
                    shutil.rmtree(checkpoint_to_remove)
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint: {e}")
    
    # Save final checkpoint
    final_checkpoint_dir = save_model_checkpoint(
        model=model,
        tokenizer=tokenizer,
        output_dir=output_dir,
        is_final=True,
        save_full_weights=save_full_weights
    )
    
    # Save optimizer and scheduler states
    final_checkpoint_state = {
        "global_step": completed_steps,
        "optimizer_state_dict": optimizer.state_dict(),
    }
    if scheduler:
        final_checkpoint_state["scheduler_state_dict"] = scheduler.state_dict()
    
    torch.save(
        final_checkpoint_state,
        os.path.join(final_checkpoint_dir, "optimizer.pt")
    )
    
    logger.info(f"Training completed. Final model saved to {final_checkpoint_dir}")
    
    return model
