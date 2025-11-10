# 🤖 Teacher Model Agent

**Intelligent automatic teacher model selection and management for knowledge distillation.**

## Overview

The Teacher Model Agent is an AI-powered system that automatically:
- 🔍 **Detects** your task type from data (sentiment analysis, classification, QA, etc.)
- 📋 **Recommends** the best teacher models based on your needs
- 📥 **Downloads** and configures the optimal teacher
- ✅ **Validates** teacher quality before distillation
- 🎯 **Integrates** seamlessly with the distillation pipeline

## Why Use the Agent?

### Problem: Manual Teacher Selection is Hard
- Which model is best for sentiment analysis? BERT? RoBERTa? DistilBERT?
- How do I know if a teacher is good enough?
- What if I have limited compute resources?
- Do I need to fine-tune the teacher first?

### Solution: Let the Agent Decide! 🚀
```python
from core.agents import quick_teacher_setup

# One line - that's it!
result = quick_teacher_setup(data_samples)
teacher = result['model']
tokenizer = result['tokenizer']
```

The agent automatically:
1. Analyzes your data to detect the task
2. Recommends the best teacher models
3. Loads and validates the teacher
4. Returns everything ready to use

## Quick Start

### 1. Basic Usage (One-Liner)

```python
from core.agents import quick_teacher_setup

# Load your training data
data_samples = [
    {"text": "This movie was great!", "label": "positive"},
    {"text": "I hated it", "label": "negative"},
    # ... more samples
]

# Let the agent handle everything
result = quick_teacher_setup(
    data_samples=data_samples,
    resource_constraint="medium"  # low, medium, or high
)

teacher_model = result['model']
tokenizer = result['tokenizer']
print(f"✅ Using: {result['model_name']}")
```

### 2. Get Recommendations First

```python
from core.agents import TeacherModelAgent

agent = TeacherModelAgent(device="cuda")

# Detect task
task = agent.detect_task_from_data(data_samples)

# Get recommendations
recommendations = agent.recommend_teachers(
    task=task,
    dataset_size=len(data_samples),
    resource_constraint="medium"
)

# See all options
for i, rec in enumerate(recommendations, 1):
    print(f"{i}. {rec.model_name}")
    print(f"   Confidence: {rec.confidence:.0%}")
    print(f"   Size: {rec.estimated_size}")
    print(f"   Reasoning: {rec.reasoning}")
    print()

# Load the best one
best = recommendations[0]
model, tokenizer = agent.load_teacher(best.model_name, task_type=task)
```

### 3. Integrated with Training Pipeline

The agent is automatically integrated into `core/models/model_loader.py`:

```python
# In your config (configs/default.yaml):
# Just remove or comment out the teacher model name!

model:
  # name: "bert-base-uncased"  # Comment this out
  student_name: "distilbert-base-uncased"
  type: "transformer"

# The agent will automatically:
# - Detect you have no teacher specified
# - Analyze your data
# - Select the best teacher
# - Load and validate it
```

Or explicitly enable in code:

```python
from core.models.model_loader import load_models

teacher, student, tokenizer = load_models(
    cfg, 
    device="cuda",
    use_agent=True,  # Enable agent
    data_samples=your_data  # Optional: provide samples
)
```

## Features

### 🔍 Smart Task Detection

The agent analyzes your data to automatically detect:
- **Sentiment Analysis** (positive/negative/neutral labels)
- **Text Classification** (general multi-class)
- **Question Answering** (context + question format)
- **Named Entity Recognition** (entity tags)
- **Causal Language Modeling** (text generation)

### 📋 Intelligent Recommendations

The agent considers multiple factors:
- **Task Type**: Different models excel at different tasks
- **Resource Constraints**: Prefer smaller models on limited hardware
- **Dataset Size**: Small datasets may need fine-tuned teachers
- **Confidence Scores**: Agent rates how confident it is about each choice

### 🎯 Teacher Catalog

Built-in knowledge of proven teacher models:

**Sentiment Analysis / Text Classification:**
- `bert-base-uncased` (110M params) - Industry standard, excellent accuracy
- `roberta-base` (125M params) - Improved BERT training, better on sentiment
- `distilbert-base-uncased` (66M params) - Faster, smaller, good for tight resources

**Question Answering:**
- `bert-large-uncased-whole-word-masking-finetuned-squad` - State-of-art QA

**Text Generation:**
- `gpt2` (117M params) - Excellent text generation
- `gpt2-medium` (345M params) - Better quality, more parameters

### ✅ Quality Validation

The agent can validate teacher quality:

```python
result = agent.auto_select_and_load(
    data_samples=data,
    validate=True  # Enable validation
)

print(f"Accuracy: {result['validation']['accuracy']:.2%}")
print(f"Passed: {result['validation']['passed']}")
print(f"Recommendation: {result['validation']['recommendation']}")
```

If the teacher isn't good enough, the agent tells you to fine-tune it first!

## Resource Constraints

Tell the agent about your hardware:

### Low Resources (CPU, limited RAM)
```python
quick_teacher_setup(data, resource_constraint="low")
# Agent prefers: DistilBERT (66M), BERT-base (110M)
```

### Medium Resources (Single GPU)
```python
quick_teacher_setup(data, resource_constraint="medium")
# Agent prefers: BERT-base, RoBERTa-base
```

### High Resources (Multi-GPU, HPC)
```python
quick_teacher_setup(data, resource_constraint="high")
# Agent prefers: RoBERTa-large, BERT-large
```

## Examples

See `examples/teacher_agent_demo.py` for comprehensive examples:

```bash
python examples/teacher_agent_demo.py
```

This runs 5 complete examples:
1. **Quick Setup** - One-line teacher loading
2. **Detailed Recommendations** - See all options before choosing
3. **Custom Resource Constraints** - Compare LOW vs HIGH resources
4. **Teacher Validation** - Check teacher quality
5. **Save Report** - Export recommendations to JSON

## API Reference

### `quick_teacher_setup()`

**Simplest way to get started:**

```python
quick_teacher_setup(
    data_samples: List[Dict[str, Any]],  # Your training data
    device: Optional[str] = None,         # "cuda", "mps", "cpu", or auto
    resource_constraint: str = "medium"   # "low", "medium", or "high"
) -> Dict[str, Any]
```

**Returns:**
```python
{
    "model": teacher_model,           # PyTorch model ready to use
    "tokenizer": tokenizer,           # HuggingFace tokenizer
    "model_name": "bert-base-uncased", # Model identifier
    "task": "sentiment_analysis",     # Detected task
    "device": "cuda",                 # Device it's loaded on
    "recommendation": TeacherRecommendation(...),  # Full details
    "validation": {                   # Quality check results
        "accuracy": 0.87,
        "passed": True,
        "samples_tested": 100,
        "recommendation": "Teacher is good!"
    }
}
```

### `TeacherModelAgent` Class

**Full control over the agent:**

```python
agent = TeacherModelAgent(device="cuda")

# Methods:
agent.detect_task_from_data(samples)        # Detect task type
agent.recommend_teachers(task, ...)         # Get recommendations
agent.load_teacher(model_name, task)        # Load specific model
agent.validate_teacher(model, tokenizer, samples)  # Check quality
agent.auto_select_and_load(samples)         # Do everything automatically
agent.save_recommendation_report(recs, path)  # Save report
```

### `TeacherRecommendation` Dataclass

```python
@dataclass
class TeacherRecommendation:
    model_name: str              # "bert-base-uncased"
    confidence: float            # 0.95 (0-1)
    reasoning: str               # "Industry standard, excellent accuracy"
    estimated_size: str          # "110M params"
    task_fit: float              # 0.95 (0-1)
    download_size: str           # "440MB"
    requires_finetuning: bool    # False
```

## Integration with Existing Code

### Option 1: Automatic (Config-Based)

Just remove the teacher model from your config:

```yaml
# configs/my_config.yaml
model:
  # name: "bert-base-uncased"  # Remove this line
  student_name: "distilbert-base-uncased"
  type: "transformer"
```

The agent activates automatically when `model.name` is missing!

### Option 2: Explicit (Code-Based)

```python
from core.models.model_loader import load_models

teacher, student, tokenizer = load_models(
    cfg, 
    use_agent=True,
    data_samples=train_data[:50]  # Provide samples for analysis
)
```

### Option 3: Standalone

Use the agent completely separately:

```python
from core.agents import TeacherModelAgent

agent = TeacherModelAgent()
result = agent.auto_select_and_load(your_data)

# Use result['model'] and result['tokenizer'] however you want
```

## Configuration Options

### In Python

```python
result = quick_teacher_setup(
    data_samples=data,
    device="cuda",                      # Force specific device
    resource_constraint="low",          # Prefer smaller models
)

# Or with full agent:
agent = TeacherModelAgent(device="mps")
recs = agent.recommend_teachers(
    task="sentiment_analysis",
    dataset_size=1000,                  # Small dataset
    resource_constraint="medium"
)
```

### In Config (Enable/Disable)

```yaml
# configs/default.yaml
model:
  name: null  # Set to null to trigger agent
  # OR remove 'name' entirely
  # OR comment it out
  student_name: "distilbert-base-uncased"
```

## Advantages

✅ **No Manual Selection** - Agent picks the best teacher automatically  
✅ **Task-Aware** - Recommends models proven for your specific task  
✅ **Resource-Aware** - Considers your hardware constraints  
✅ **Quality Validation** - Checks teacher performance before using it  
✅ **Easy Integration** - Works with existing training pipeline  
✅ **Transparent** - Shows reasoning for each recommendation  
✅ **Extensible** - Easy to add new teacher models to catalog  

## Future Enhancements

🚀 **Planned features:**
- [ ] Multi-task teacher selection (handle multiple tasks)
- [ ] Automatic ensemble teacher creation (combine multiple teachers)
- [ ] Fine-tuning scheduler (when to fine-tune teacher)
- [ ] Cost estimation (download size, inference time, training time)
- [ ] Teacher compression (quantize large teachers automatically)
- [ ] Cloud model catalog (fetch from HuggingFace Model Hub)
- [ ] Performance prediction (estimate accuracy before downloading)

## Troubleshooting

### Agent picks the wrong teacher?

```python
# See all recommendations:
recs = agent.recommend_teachers(task, dataset_size, resource_constraint)
for rec in recs:
    print(f"{rec.model_name}: {rec.reasoning}")

# Load a specific one:
model, tokenizer = agent.load_teacher(recs[1].model_name, task)
```

### Need to add a new teacher model?

Edit `core/agents/teacher_agent.py` and add to `TEACHER_CATALOG`:

```python
TEACHER_CATALOG = {
    "sentiment_analysis": [
        {
            "name": "your-custom-model",
            "size": "200M params",
            "download": "800MB",
            "confidence": 0.90,
            "reason": "Your reasoning here"
        },
        # ... existing models
    ]
}
```

### Teacher validation fails?

```python
result = agent.auto_select_and_load(data, validate=True)

if not result['validation']['passed']:
    print("Teacher needs fine-tuning!")
    # Set train_teacher: true in your config
    # Or fine-tune manually before distillation
```

## Credits

The Teacher Model Agent is part of the **Zynthe Knowledge Distillation Toolkit**.

Built with ❤️ to make knowledge distillation accessible to everyone!

---

**Quick Links:**
- [Main README](../README.md)
- [Examples](../examples/teacher_agent_demo.py)
- [API Docs](../docs/)
- [Training Guide](../docs/quickstart.md)
