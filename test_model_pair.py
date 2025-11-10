#!/usr/bin/env python3
"""
Quick Model Pair Tester
Generate configurations and test different teacher-student model pairs.
"""

import sys
import os
from pathlib import Path
import argparse
import yaml

# Add project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Model pair presets optimized for Mac M2
PRESETS = {
    "tinybert": {
        "name": "BERT → TinyBERT (7.5x compression)",
        "teacher": "bert-base-uncased",
        "student": "huawei-noah/TinyBERT_General_4L_312D",
        "tokenizer": "bert-base-uncased",
        "batch_size": 4,
        "epochs": 3,
        "temperature": 3.0,
        "alpha": 0.3,
        "memory": "8GB",
        "expected_compression": "7.5x",
        "expected_drop": "2-4%"
    },
    "distilbert": {
        "name": "BERT → DistilBERT (1.64x compression)",
        "teacher": "bert-base-uncased",
        "student": "distilbert-base-uncased",
        "tokenizer": "distilbert-base-uncased",
        "batch_size": 8,
        "epochs": 3,
        "temperature": 1.5,
        "alpha": 0.4,
        "memory": "8GB",
        "expected_compression": "1.64x",
        "expected_drop": "1-3%"
    },
    "roberta": {
        "name": "RoBERTa → DistilRoBERTa (1.5x compression)",
        "teacher": "roberta-base",
        "student": "distilroberta-base",
        "tokenizer": "roberta-base",
        "batch_size": 8,
        "epochs": 4,
        "temperature": 2.5,
        "alpha": 0.4,
        "memory": "16GB",
        "expected_compression": "1.5x",
        "expected_drop": "1-3%"
    },
    "electra": {
        "name": "ELECTRA-base → ELECTRA-small (7.8x compression)",
        "teacher": "google/electra-base-discriminator",
        "student": "google/electra-small-discriminator",
        "tokenizer": "google/electra-base-discriminator",
        "batch_size": 6,
        "epochs": 3,
        "temperature": 2.5,
        "alpha": 0.4,
        "memory": "8-16GB",
        "expected_compression": "7.8x",
        "expected_drop": "2-5%"
    },
    "bert-large": {
        "name": "BERT-large → BERT-base (3x compression)",
        "teacher": "bert-large-uncased",
        "student": "bert-base-uncased",
        "tokenizer": "bert-large-uncased",
        "batch_size": 6,
        "epochs": 3,
        "temperature": 2.0,
        "alpha": 0.5,
        "memory": "16GB+",
        "expected_compression": "3.0x",
        "expected_drop": "1-2%"
    }
}


def print_presets():
    """Print available model pair presets."""
    print("\n" + "="*70)
    print("🎯 Available Model Pairs for Mac M2")
    print("="*70 + "\n")
    
    for key, preset in PRESETS.items():
        print(f"📦 {key.upper()}")
        print(f"   Name:        {preset['name']}")
        print(f"   Teacher:     {preset['teacher']}")
        print(f"   Student:     {preset['student']}")
        print(f"   Memory:      {preset['memory']}")
        print(f"   Compression: {preset['expected_compression']}")
        print(f"   Accuracy Drop: {preset['expected_drop']}")
        print()


def generate_config(preset_name: str, output_path: str) -> dict:
    """Generate configuration for a model pair."""
    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(PRESETS.keys())}")
    
    preset = PRESETS[preset_name]
    
    config = {
        "train": {
            "epochs": preset["epochs"],
            "batch_size": preset["batch_size"],
            "lr": 2e-5,
            "grad_accum_steps": 2 if preset["batch_size"] <= 4 else 1,
            "mixed_precision": False,
            "early_stop_patience": 2,
            "save_best_only": True,
            "log_interval": 10
        },
        "model": {
            "name": preset["teacher"],
            "student_name": preset["student"],
            "type": "transformer",
            "tokenizer_name": preset["tokenizer"],
            "max_length": 128,
            "num_labels": 2
        },
        "distillation": {
            "method": "kd_hinton",
            "temperature": preset["temperature"],
            "alpha": preset["alpha"]
        },
        "data": {
            "train_path": "data/imdb_train.jsonl",
            "val_path": "data/imdb_val.jsonl"
        },
        "device": {
            "prefer_mps": True,
            "prefer_cuda": False
        },
        "output_root": "experiments",
        "seed": 42,
        "quantization": {
            "enable": True,
            "mode": "ptq"
        },
        "similarity_transfer": True,
        "evaluate": True,
        "compare_models": True,
        "explainability": {
            "enable_shap": False,
            "enable_lime": False
        }
    }
    
    # Save config
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, sort_keys=False, default_flow_style=False)
    
    return preset


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate and test different teacher-student model pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available model pairs
  python test_model_pair.py --list

  # Generate TinyBERT config (recommended for high compression)
  python test_model_pair.py tinybert

  # Generate RoBERTa config (recommended for quality)
  python test_model_pair.py roberta

  # Generate and save to custom path
  python test_model_pair.py electra --output configs/my_electra.yaml

  # After generation, run training:
  python app/main.py --config configs/test_tinybert.yaml
        """
    )
    
    parser.add_argument(
        "preset",
        nargs="?",
        choices=list(PRESETS.keys()),
        help="Model pair preset to use"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available presets"
    )
    parser.add_argument(
        "--output",
        "-o",
        help="Output config file path (default: configs/test_<preset>.yaml)"
    )
    
    args = parser.parse_args()
    
    # List presets
    if args.list:
        print_presets()
        return
    
    # Check if preset provided
    if not args.preset:
        parser.print_help()
        print("\n❌ Error: Please specify a preset or use --list to see available options")
        sys.exit(1)
    
    # Generate output path
    if args.output:
        output_path = args.output
    else:
        output_path = f"configs/test_{args.preset}.yaml"
    
    # Generate config
    print("\n" + "="*70)
    print(f"🔧 Generating Configuration: {args.preset.upper()}")
    print("="*70)
    
    preset = generate_config(args.preset, output_path)
    
    print(f"\n✅ Configuration generated successfully!")
    print(f"\n📋 Model Pair Details:")
    print(f"   Name:          {preset['name']}")
    print(f"   Teacher:       {preset['teacher']}")
    print(f"   Student:       {preset['student']}")
    print(f"   Tokenizer:     {preset['tokenizer']}")
    print(f"   Batch Size:    {preset['batch_size']}")
    print(f"   Epochs:        {preset['epochs']}")
    print(f"   Temperature:   {preset['temperature']}")
    print(f"   Alpha:         {preset['alpha']}")
    print(f"\n📊 Expected Results:")
    print(f"   Compression:   {preset['expected_compression']}")
    print(f"   Accuracy Drop: {preset['expected_drop']}")
    print(f"   Memory Req:    {preset['memory']} Mac M2")
    print(f"\n💾 Config saved to: {output_path}")
    print(f"\n🚀 Next Steps:")
    print(f"   1. Run training:  python app/main.py --config {output_path}")
    print(f"   2. Wait for completion (~15-30 minutes)")
    print(f"   3. Check results: cd experiments/$(ls -t experiments | head -1)/comparison")
    print(f"   4. View report:   cat COMPARISON_REPORT.md")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
