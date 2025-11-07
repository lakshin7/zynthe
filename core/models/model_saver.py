

import os
import torch
from typing import Any, Optional

def save_checkpoint(model: Any, optimizer: Optional[Any], path: str):
    """
    Save a PyTorch checkpoint.
    Args:
        model: The model to save (should have state_dict()).
        optimizer: The optimizer to save (should have state_dict()), or None.
        path: Path to save the checkpoint file.
    """
    checkpoint = {'model_state_dict': model.state_dict()}
    if optimizer is not None:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")

def load_checkpoint(model: Any, optimizer: Optional[Any], path: str, map_location: Optional[Any] = None):
    """
    Load a PyTorch checkpoint.
    Args:
        model: The model to load state_dict into.
        optimizer: The optimizer to load state_dict into, or None.
        path: Path to the checkpoint file.
        map_location: Device mapping for torch.load.
    Returns:
        The loaded checkpoint dict.
    """
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    print(f"Checkpoint loaded from {path}")
    return checkpoint

def save_model(model: Any, path: str, tokenizer: Optional[Any] = None):
    """
    Save a Hugging Face model (and optionally tokenizer) to a directory.
    Args:
        model: Hugging Face PreTrainedModel.
        path: Directory to save model.
        tokenizer: Hugging Face PreTrainedTokenizer, optional.
    """
    os.makedirs(path, exist_ok=True)
    model.save_pretrained(path)
    print(f"Model saved to {path}")
    if tokenizer is not None:
        tokenizer.save_pretrained(path)
        print(f"Tokenizer saved to {path}")

def load_model(model_class: Any, path: str, tokenizer_class: Optional[Any] = None):
    """
    Load a Hugging Face model (and optionally tokenizer) from a directory.
    Args:
        model_class: Hugging Face PreTrainedModel class (e.g., BertForSequenceClassification).
        path: Directory to load model from.
        tokenizer_class: Hugging Face PreTrainedTokenizer class, optional.
    Returns:
        model, tokenizer (if tokenizer_class is given, else None)
    """
    model = model_class.from_pretrained(path)
    print(f"Model loaded from {path}")
    tokenizer = None
    if tokenizer_class is not None:
        tokenizer = tokenizer_class.from_pretrained(path)
        print(f"Tokenizer loaded from {path}")
    return model, tokenizer