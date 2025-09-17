import torch
from torch.utils.data import Dataset, DataLoader
import json

class JsonlDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.samples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                self.samples.append(data)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample['text']
        label = sample['label']
        encoding = self.tokenizer(text, truncation=True, padding='max_length', return_tensors='pt')
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label)
        return item

def create_dataloaders(cfg, tokenizer):
    train_dataset = JsonlDataset(cfg["data"]["train_path"], tokenizer)
    val_dataset = JsonlDataset(cfg["data"]["val_path"], tokenizer)
    batch_size = cfg["train"]["batch_size"]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
