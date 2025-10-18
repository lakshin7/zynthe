import torch
from torch.utils.data import Dataset, DataLoader
import json

class JsonlDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        self.samples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                self.samples.append(data)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample['text']
        label = sample['label']
        
        # Tokenize with explicit parameters to avoid model incompatibility issues
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item

def create_dataloaders(cfg, tokenizer):
    """Create dataloaders with proper tokenizer configuration for model compatibility."""
    
    # Get max_length from config, with fallback
    max_length = cfg.get("model", {}).get("max_length", 128)
    
    train_dataset = JsonlDataset(cfg["data"]["train_path"], tokenizer, max_length=max_length)
    val_dataset = JsonlDataset(cfg["data"]["val_path"], tokenizer, max_length=max_length)
    batch_size = cfg["train"]["batch_size"]

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,  # Avoid multiprocessing issues on Mac
        pin_memory=False  # Not needed for MPS
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=False
    )

    return train_loader, val_loader


def get_imdb_dataloaders(train_path: str, val_path: str, tokenizer, batch_size: int = 8, max_length: int = 128):
    """
    Backward-compatible wrapper used by examples to build IMDB dataloaders directly.
    """
    train_dataset = JsonlDataset(train_path, tokenizer, max_length=max_length)
    val_dataset = JsonlDataset(val_path, tokenizer, max_length=max_length)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )
    return train_loader, val_loader
