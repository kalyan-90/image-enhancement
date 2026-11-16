"""
Unified checkpoint loading, saving, and validation utility.
Supports both state_dict and full-checkpoint dicts with automatic device mapping.
"""

import os
import torch


def load_checkpoint(model, checkpoint_path, optimizer=None, device="cpu"):
    """
    Safely loads weights into model from checkpoint_path.
    Handles flat state dicts and dictionary wrappers seamlessly.
    
    Returns:
        dict with metadata (epoch, validation_loss) if available, else empty dict.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    metadata = {}
    
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        metadata["epoch"] = checkpoint.get("epoch", None)
        metadata["validation_loss"] = checkpoint.get("validation_loss", None)
        if optimizer is not None and "optimizer_state_dict" in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            except Exception as e:
                print(f"Warning: Optimizer state dict not loaded: {e}")
    elif isinstance(checkpoint, dict):
        model.load_state_dict(checkpoint)
    else:
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")
    
    return metadata


def save_checkpoint(model, path, optimizer=None, epoch=None, validation_loss=None, extra_data=None):
    """
    Saves a comprehensive checkpoint with model state, optimizer state, epoch, and metrics.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict()
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        payload["epoch"] = epoch
    if validation_loss is not None:
        payload["validation_loss"] = validation_loss
    if extra_data is not None:
        payload["extra_data"] = extra_data
        
    torch.save(payload, path)
