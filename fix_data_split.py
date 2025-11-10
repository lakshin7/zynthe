#!/usr/bin/env python3
"""
Fix Data Split - Create Proper Train/Val Split
==============================================

This script:
1. Reads the original data
2. Shuffles with a fixed seed
3. Splits 80/20 into train/val
4. Ensures NO overlap
5. Maintains class balance (stratified split)
"""

import json
import random
from pathlib import Path
from collections import Counter

# Set random seed for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Configuration
ORIGINAL_FILE = "data/imdb_train.jsonl"  # Use this as source
TRAIN_OUTPUT = "data/imdb_train_fixed.jsonl"
VAL_OUTPUT = "data/imdb_val_fixed.jsonl"
SPLIT_RATIO = 0.8  # 80% train, 20% val

print("="*80)
print("FIXING DATA SPLIT")
print("="*80)

# 1. Load all data
print(f"\n1. Loading data from {ORIGINAL_FILE}...")
all_data = []
with open(ORIGINAL_FILE, 'r') as f:
    for line in f:
        data = json.loads(line.strip())
        all_data.append(data)

print(f"   Total samples: {len(all_data)}")

# 2. Check class distribution
labels = [item['label'] for item in all_data]
label_counts = Counter(labels)
print(f"\n2. Class distribution:")
for label, count in sorted(label_counts.items()):
    print(f"   Class {label}: {count} ({count/len(all_data)*100:.1f}%)")

# 3. Stratified split (maintain class balance)
print(f"\n3. Performing stratified split ({SPLIT_RATIO*100:.0f}/{(1-SPLIT_RATIO)*100:.0f})...")

# Group by label
data_by_label = {}
for item in all_data:
    label = item['label']
    if label not in data_by_label:
        data_by_label[label] = []
    data_by_label[label].append(item)

# Shuffle each class
for label in data_by_label:
    random.shuffle(data_by_label[label])

# Split each class
train_data = []
val_data = []

for label, items in data_by_label.items():
    split_idx = int(len(items) * SPLIT_RATIO)
    train_data.extend(items[:split_idx])
    val_data.extend(items[split_idx:])
    print(f"   Class {label}: {split_idx} train, {len(items) - split_idx} val")

# Shuffle combined datasets
random.shuffle(train_data)
random.shuffle(val_data)

print(f"\n   Final counts:")
print(f"   Train: {len(train_data)}")
print(f"   Val:   {len(val_data)}")

# 4. Verify no overlap
print(f"\n4. Verifying no overlap...")
train_texts = set([item['text'] for item in train_data])
val_texts = set([item['text'] for item in val_data])
overlap = train_texts.intersection(val_texts)

if len(overlap) > 0:
    print(f"   ❌ ERROR: Found {len(overlap)} overlapping samples!")
    print(f"   This should not happen - check the code")
else:
    print(f"   ✓ No overlap - split is clean!")

# 5. Save files
print(f"\n5. Saving new splits...")
print(f"   Train: {TRAIN_OUTPUT}")
print(f"   Val:   {VAL_OUTPUT}")

# Create backup of original files
original_train = Path("data/imdb_train.jsonl")
original_val = Path("data/imdb_val.jsonl")

if original_train.exists():
    backup_train = Path("data/imdb_train.jsonl.backup")
    print(f"   Creating backup: {backup_train}")
    original_train.rename(backup_train)

if original_val.exists():
    backup_val = Path("data/imdb_val.jsonl.backup")
    print(f"   Creating backup: {backup_val}")
    original_val.rename(backup_val)

# Write new files
with open(TRAIN_OUTPUT, 'w') as f:
    for item in train_data:
        f.write(json.dumps(item) + '\n')

with open(VAL_OUTPUT, 'w') as f:
    for item in val_data:
        f.write(json.dumps(item) + '\n')

print(f"\n✓ Done!")

# 6. Verify class balance in new splits
print(f"\n6. Verifying class balance in new splits...")
train_labels = Counter([item['label'] for item in train_data])
val_labels = Counter([item['label'] for item in val_data])

print(f"\n   Train distribution:")
for label, count in sorted(train_labels.items()):
    print(f"      Class {label}: {count} ({count/len(train_data)*100:.1f}%)")

print(f"\n   Val distribution:")
for label, count in sorted(val_labels.items()):
    print(f"      Class {label}: {count} ({count/len(val_data)*100:.1f}%)")

print("\n" + "="*80)
print("DATA SPLIT FIXED!")
print("="*80)
print(f"\nNext steps:")
print(f"1. The old files have been backed up:")
print(f"   - data/imdb_train.jsonl.backup")
print(f"   - data/imdb_val.jsonl.backup")
print(f"\n2. New properly split files created:")
print(f"   - data/imdb_train_fixed.jsonl (renamed to imdb_train.jsonl)")
print(f"   - data/imdb_val_fixed.jsonl (renamed to imdb_val.jsonl)")
print(f"\n3. Re-run your training to see realistic performance!")
print("="*80)

# Rename to final names
Path(TRAIN_OUTPUT).rename("data/imdb_train.jsonl")
Path(VAL_OUTPUT).rename("data/imdb_val.jsonl")
print(f"\n✓ Files renamed to final names")
