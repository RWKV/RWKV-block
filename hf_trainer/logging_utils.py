"""
Logging Utilities for Hugging Face Trainer

This module provides utilities for setting up logging and integrating with Weights & Biases.
"""

import os
import logging
import sys
from typing import Dict, Any, Optional

# Configure the root logger
def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        level: Logging level (default: INFO)
        
    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicate logs
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Add formatter to handler
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    return logger

def setup_wandb(wandb_config: Dict[str, Any]) -> Optional[Any]:
    """
    Set up Weights & Biases logging.
    
    Args:
        wandb_config: W&B configuration dictionary
        
    Returns:
        W&B run object if successful, None otherwise
    """
    if not wandb_config.get("enabled", False):
        return None
    
    try:
        import wandb
    except ImportError:
        logging.warning("Weights & Biases (wandb) not installed. Install with: pip install wandb")
        return None
    
    # Extract W&B configuration
    project = wandb_config.get("project", "hf-trainer")
    name = wandb_config.get("name", None)
    entity = wandb_config.get("entity", None)
    tags = wandb_config.get("tags", [])
    
    # Initialize W&B
    logging.info(f"Initializing Weights & Biases: project={project}, name={name}")
    
    run = wandb.init(
        project=project,
        name=name,
        entity=entity,
        tags=tags,
        config=wandb_config.get("config", {})
    )
    
    return run

def log_metrics(metrics: Dict[str, float], step: Optional[int] = None, wandb_run=None) -> None:
    """
    Log metrics to console and optionally to W&B.
    
    Args:
        metrics: Dictionary of metrics to log
        step: Current step (optional)
        wandb_run: W&B run object (optional)
    """
    # Format metrics for logging
    metrics_str = ", ".join([f"{k}: {v:.6f}" for k, v in metrics.items()])
    step_str = f" (Step: {step})" if step is not None else ""
    
    # Log to console
    logging.info(f"Metrics{step_str}: {metrics_str}")
    
    # Log to W&B if available
    if wandb_run is not None:
        wandb_run.log(metrics, step=step)
