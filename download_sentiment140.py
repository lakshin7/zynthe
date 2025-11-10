"""
Download and prepare Twitter Sentiment140 dataset from Kaggle.

This script:
1. Downloads the Sentiment140 dataset from Kaggle
2. Preprocesses and converts to JSONL format
3. Splits into train/val sets
4. Saves to data/ directory

Dataset: https://www.kaggle.com/datasets/kazanova/sentiment140
"""

import os
import json
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

print("=" * 80)
print("TWITTER SENTIMENT140 DATASET DOWNLOADER")
print("=" * 80)

# Configuration
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

TRAIN_OUTPUT = DATA_DIR / "sentiment140_train.jsonl"
VAL_OUTPUT = DATA_DIR / "sentiment140_val.jsonl"

# Sample size for testing (use None for full dataset)
SAMPLE_SIZE = 10000  # Use 10k samples for fast testing (full dataset is 1.6M)
VAL_SIZE = 0.1  # 10% for validation

print(f"\nConfiguration:")
print(f"  Output directory: {DATA_DIR}")
print(f"  Sample size: {SAMPLE_SIZE if SAMPLE_SIZE else 'Full dataset'}")
print(f"  Validation split: {VAL_SIZE * 100}%")

# Step 1: Check if Kaggle dataset exists
KAGGLE_FILE = DATA_DIR / "training.1600000.processed.noemoticon.csv"

if not KAGGLE_FILE.exists():
    print(f"\n❌ Kaggle dataset not found at: {KAGGLE_FILE}")
    print("\nTo download the dataset:")
    print("1. Install kaggle CLI: pip install kaggle")
    print("2. Set up Kaggle API credentials: https://www.kaggle.com/docs/api")
    print("3. Run: kaggle datasets download -d kazanova/sentiment140 -p data/")
    print("4. Unzip: unzip data/sentiment140.zip -d data/")
    print("\nAlternatively, download manually from:")
    print("https://www.kaggle.com/datasets/kazanova/sentiment140")
    print("\nOr use the IMDB dataset that's already in your data/ directory:")
    print("  data/imdb_train.jsonl")
    print("  data/imdb_val.jsonl")
    exit(1)

print(f"\n✅ Found Kaggle dataset: {KAGGLE_FILE}")

# Step 2: Load and preprocess
print("\n📥 Loading dataset...")

# Columns: target, ids, date, flag, user, text
# target: 0 = negative, 4 = positive
try:
    df = pd.read_csv(
        KAGGLE_FILE,
        encoding='latin-1',
        names=['target', 'ids', 'date', 'flag', 'user', 'text'],
        usecols=['target', 'text']  # Only need these columns
    )
    print(f"✅ Loaded {len(df):,} samples")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    exit(1)

# Step 3: Sample if needed
if SAMPLE_SIZE and SAMPLE_SIZE < len(df):
    print(f"\n🎲 Sampling {SAMPLE_SIZE:,} examples...")
    df = df.sample(n=SAMPLE_SIZE, random_state=42)
    print(f"✅ Sampled dataset size: {len(df):,}")

# Step 4: Convert labels (0 = negative, 4 = positive → 0, 1)
print("\n🔄 Converting labels...")
df['label'] = (df['target'] == 4).astype(int)
df = df[['text', 'label']]

print(f"  Negative samples: {(df['label'] == 0).sum():,}")
print(f"  Positive samples: {(df['label'] == 1).sum():,}")

# Step 5: Clean text
print("\n🧹 Cleaning text...")
df['text'] = df['text'].str.strip()
df = df[df['text'].str.len() > 0]  # Remove empty
print(f"✅ Clean dataset size: {len(df):,}")

# Step 6: Train/val split
print(f"\n✂️  Splitting into train/val ({(1-VAL_SIZE)*100:.0f}/{VAL_SIZE*100:.0f})...")
train_df, val_df = train_test_split(
    df,
    test_size=VAL_SIZE,
    random_state=42,
    stratify=df['label']  # Maintain class balance
)

print(f"  Train samples: {len(train_df):,}")
print(f"  Val samples: {len(val_df):,}")

# Step 7: Save to JSONL
print(f"\n💾 Saving to JSONL format...")

def save_jsonl(df, output_path):
    """Save dataframe to JSONL format."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for _, row in df.iterrows():
            entry = {
                'text': row['text'],
                'label': int(row['label'])
            }
            f.write(json.dumps(entry) + '\n')

try:
    save_jsonl(train_df, TRAIN_OUTPUT)
    print(f"  ✅ Saved: {TRAIN_OUTPUT}")
    
    save_jsonl(val_df, VAL_OUTPUT)
    print(f"  ✅ Saved: {VAL_OUTPUT}")
except Exception as e:
    print(f"  ❌ Error saving files: {e}")
    exit(1)

# Step 8: Verify files
print(f"\n✅ Dataset preparation complete!")
print(f"\nOutput files:")
print(f"  📄 {TRAIN_OUTPUT} ({len(train_df):,} samples)")
print(f"  📄 {VAL_OUTPUT} ({len(val_df):,} samples)")

# Show sample
print(f"\n📋 Sample from training set:")
sample = train_df.head(3)
for idx, row in sample.iterrows():
    label_str = "positive" if row['label'] == 1 else "negative"
    text_preview = row['text'][:80] + "..." if len(row['text']) > 80 else row['text']
    print(f"  [{label_str}] {text_preview}")

print("\n" + "=" * 80)
print("READY TO TRAIN!")
print("=" * 80)
print("\nNext steps:")
print("  1. Review the config: configs/m2_test.yaml")
print("  2. Run training: python app/main.py --config configs/m2_test.yaml")
print("  or")
print("  2. Run CLI: python -m app.main distill --config configs/m2_test.yaml")
print("=" * 80)
