# Knowledge Distillation Toolkit - Quick Start Guide

This guide will help you get started with the Knowledge Distillation Toolkit on your Mac M2 system.

## Prerequisites

- Mac with Apple Silicon (M2/M3/M4)
- Python 3.8+ (recommended: Python 3.10 or 3.11)
- At least 8GB of RAM available

## Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd knowledge-distillation-toolkit
   ```

2. **Create a virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### 1. Basic Knowledge Distillation

Run your first distillation experiment using the default configuration:

```bash
python app/main.py --config configs/default.yaml
```

This will:
- Use **BERT-base** as the teacher model (110M parameters)
- Use **DistilBERT** as the student model (66M parameters)
- Train on IMDB sentiment classification data
- Use Apple Silicon GPU (MPS) automatically

### 2. Advanced Configuration

For more sophisticated distillation with multiple strategies:

```bash
python app/main.py --config configs/advanced.yaml
```

This uses multi-stage distillation with attention transfer and similarity matching.

### 3. Configuration Override

You can override any configuration parameter:

```bash
python app/main.py --config configs/default.yaml --override train.epochs=1 train.batch_size=4
```

## Model Configurations

### Default Setup (Recommended for Mac M2)
- **Teacher**: `bert-base-uncased` (110M params)
- **Student**: `distilbert-base-uncased` (66M params)
- **Memory**: ~4GB GPU memory required
- **Training time**: ~15 minutes for 3 epochs

### Advanced Setup
- **Teacher**: `microsoft/DialoGPT-medium` (117M params)
- **Student**: `microsoft/DialoGPT-small` (117M params)
- **Memory**: ~6GB GPU memory required
- **Training time**: ~25 minutes for 5 epochs

### Lightweight Setup (for 8GB Mac M2)
Edit `configs/default.yaml`:
```yaml
model:
  name: "distilbert-base-uncased"        # Teacher: DistilBERT
  student_name: "prajjwal1/bert-tiny"    # Student: BERT-tiny (4M params)
  type: "transformer"

train:
  batch_size: 16                         # Can use larger batch with tiny model
  epochs: 5
```

## Understanding the Output

After training, you'll see output like:
```
✅ Config loaded successfully
Experiment ID: 20240101T120000Z_abcd1234
Loading teacher model 'bert-base-uncased'...
Loading student model 'distilbert-base-uncased'...
[Model Summary] Teacher model: 110M parameters
[Model Summary] Student model: 66M parameters
[INFO] Starting training...
Epoch 1/3: 100%|██████████| 625/625 [03:45<00:00, loss=0.6234]
Validation: accuracy=0.8567, f1=0.8543
PTQ applied using mode: float16 on device mps
Training completed successfully!
```

## File Structure After Training

```
experiments/20240101T120000Z_abcd1234/
├── config.yaml              # Resolved configuration
├── student_model/           # Trained student model
│   ├── pytorch_model.bin
│   ├── config.json
│   └── tokenizer files...
├── training_metrics.json    # Training history
├── evaluation_report.json   # Final evaluation metrics
└── model_comparison.png     # Performance comparison chart
```

## Common Issues and Solutions

### 1. Memory Issues
If you get OOM (Out of Memory) errors:
```yaml
train:
  batch_size: 4              # Reduce batch size
  grad_accum_steps: 2        # Maintain effective batch size
```

### 2. Slow Training
Enable gradient checkpointing in advanced config:
```yaml
device:
  memory_management:
    gradient_checkpointing: true
```

### 3. MPS Compatibility Issues
If MPS fails, the system automatically falls back to CPU:
```yaml
device:
  prefer_mps: true
  fallback_cpu: true
```

## Next Steps

1. **Experiment with different models**: Try RoBERTa, ELECTRA, or other transformer models
2. **Custom data**: Replace IMDB data with your own dataset (see `data/preprocess.py`)
3. **Advanced distillation**: Explore attention transfer and feature matching
4. **Deployment**: Export models using the packaging system
5. **Evaluation**: Run comprehensive benchmarks

## Performance Expectations (Mac M2)

| Model Pair | Memory Usage | Training Time (3 epochs) | Student Accuracy |
|------------|--------------|--------------------------|------------------|
| BERT → DistilBERT | ~4GB | 15 minutes | ~89% of teacher |
| RoBERTa → DistilRoBERTa | ~5GB | 18 minutes | ~91% of teacher |
| BERT → BERT-tiny | ~2GB | 8 minutes | ~85% of teacher |

## Getting Help

- Check `docs/overview.md` for detailed architecture information
- See `examples/` for code examples
- Run tests with: `python -m pytest test/`
- Check logs in experiment directories for debugging

## Troubleshooting

### Installation Issues
```bash
# If transformers installation fails
pip install --upgrade pip
pip install torch torchvision torchaudio
pip install transformers datasets

# If MPS is not available
python -c "import torch; print(torch.backends.mps.is_available())"
```

### Model Loading Issues
Make sure you have internet connection for first-time model downloads. Models are cached in `~/.cache/huggingface/`.

### Configuration Errors
Validate your config:
```bash
python -c "from core.config.config_manager import ConfigManager; cm = ConfigManager('configs/default.yaml'); print('Config valid!')"
```
