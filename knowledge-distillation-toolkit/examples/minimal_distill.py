
import logging
import os
import sys
import torch

# Ensure project root is in sys.path for 'core' imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
	sys.path.insert(0, project_root)

from core.config.config_manager import ConfigManager
from core.models.model_loader import load_models
from core.models.model_wrapper import ModelWrapper
from core.distillers.kd_hinton import KDHintonDistiller

def main():
	logging.basicConfig(level=logging.INFO)
	logger = logging.getLogger("minimal_distill")
	config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../configs/default.yaml'))
	cfg = ConfigManager(config_path=config_path)
	device = cfg.device()
	logger.info(f"Using device: {device}")

	# Load teacher, student, tokenizer
	teacher, student, tokenizer = load_models(cfg, device=device)
	logger.info("Models loaded.")

	# Wrap student for enterprise utilities
	student_wrapper = ModelWrapper(student, device=device, tokenizer=tokenizer)

	# Dummy data (batch of 2, sequence length 8)
	input_ids = torch.randint(0, 100, (2, 8)).to(device)
	labels = torch.randint(0, 2, (2,)).to(device)

	# Forward pass
	with torch.no_grad():
		teacher_logits = teacher(input_ids).logits
	student_logits = student_wrapper.forward(input_ids).logits

	# Distillation loss
	distiller = KDHintonDistiller(temperature=2.0, alpha=0.5)
	loss = distiller.compute_loss(student_logits, teacher_logits, labels)
	logger.info(f"Distillation loss: {loss.item():.4f}")

	# Save student model
	student_wrapper.save("/tmp/minimal_student_model")
	logger.info("Student model saved to /tmp/minimal_student_model")

if __name__ == "__main__":
	main()
