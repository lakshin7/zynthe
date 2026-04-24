#!/usr/bin/env bash
# scripts/setup-venv.sh - Virtual environment setup for Zynthe
set -euo pipefail

echo "Setting up Zynthe virtual environment..."

# Check if .venv exists
if [ -d ".venv" ]; then
    echo "Virtual environment already exists. Removing..."
    rm -rf .venv
fi

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install core dependencies (CPU-only for Latitude 7490)
echo "Installing core dependencies..."
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install transformers datasets accelerate
pip install pyyaml omegaconf python-dotenv
pip install numpy pandas scikit-learn
pip install matplotlib seaborn tqdm rich
pip install pytest flake8 mypy

# Install optional dependencies
pip install onnx onnxruntime

echo "Virtual environment setup complete!"
echo "Activate with: source .venv/bin/activate"
