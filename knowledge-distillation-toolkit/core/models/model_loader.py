import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core.config.config_manager import ConfigManager
def get_device(device_str=None):
    if device_str:
        device = torch.device(device_str)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device

def load_models(cfg: "ConfigManager", device=None):
    """Load teacher and student models based on ConfigManager.
    
    Args:
        cfg: ConfigManager instance with resolved configuration
        device: Target device (auto-detected if None)
        
    Returns:
        tuple: (teacher_model, student_model, tokenizer)
    """
    device = get_device(device)

    model_cfg = cfg.get("model", {})
    teacher_name = model_cfg.get("name")
    student_name = model_cfg.get("student_name", teacher_name)  # Default to teacher if not specified
    tokenizer_name = model_cfg.get("tokenizer_name", teacher_name)  # Default to teacher tokenizer
    model_type = model_cfg.get("type", "").lower()

    if not teacher_name:
        raise ValueError("Teacher model name must be specified in config.model.name")

    # Determine model class based on type
    if model_type == "causallm":
        ModelClass = AutoModelForCausalLM
        model_kwargs = {}
    elif model_type in ["transformer", "sequenceclassification"]:
        ModelClass = AutoModelForSequenceClassification
        # Binary classification by default with explicit label mappings
        # This ensures the model's classifier head aligns with dataset labels
        model_kwargs = {
            "num_labels": 2,
            "label2id": {"negative": 0, "positive": 1},
            "id2label": {0: "negative", 1: "positive"}
        }
    else:
        ModelClass = AutoModel
        model_kwargs = {}

    print(f"Loading teacher model '{teacher_name}' with type '{model_type}' on device {device}")
    teacher_model = ModelClass.from_pretrained(teacher_name, **model_kwargs)
    teacher_model.to(device)

    print(f"Loading student model '{student_name}' with type '{model_type}' on device {device}")
    student_model = ModelClass.from_pretrained(student_name, **model_kwargs)
    student_model.to(device)

    print(f"Loading tokenizer '{tokenizer_name}'")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    # Add padding token if not present (needed for some models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return teacher_model, student_model, tokenizer

def model_summary(model):
    name = getattr(model.config, "name_or_path", "unknown_model")
    model_type = getattr(model.config, "model_type", "unknown_type")
    param_count = sum(p.numel() for p in model.parameters())
    device = next(model.parameters()).device
    return {"name": name, "type": model_type, "parameters": param_count, "device": str(device)}
