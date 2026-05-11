"""
Model Validator - Pre-flight model compatibility checker
Validates models before downloading/loading them
"""

from __future__ import annotations


import torch
from typing import Dict, List, Optional, Any
from huggingface_hub import HfApi, model_info
from huggingface_hub.errors import RepositoryNotFoundError, HfHubHTTPError
import logging
import os

logger = logging.getLogger(__name__)


class ModelValidator:
    """
    Validates models before training:
    - Checks if model exists on HuggingFace Hub
    - Validates device compatibility (CUDA/MPS/CPU)
    - Checks model architecture support
    - Suggests alternatives if incompatible
    """

    # Known models with device limitations
    DEVICE_LIMITATIONS = {
        # Models that require specific hardware
        "facebook/opt-66b": ["cuda"],  # Too large for MPS/CPU
        "EleutherAI/gpt-neox-20b": ["cuda"],
        "bigscience/bloom-176b": ["cuda"],
        # Models with MPS issues (Metal Performance Shaders)
        "microsoft/phi-2": ["cuda", "cpu"],  # MPS compatibility issues
        # Models requiring specific CUDA compute capability
        "nvidia/megatron-gpt2-345m": ["cuda>=7.0"],
    }

    # Recommended model pairs (teacher → compatible students)
    RECOMMENDED_PAIRS = {
        "bert-base-uncased": [
            "distilbert-base-uncased",
            "google/mobilebert-uncased",
            "huawei-noah/TinyBERT_General_4L_312D",
            "prajjwal1/bert-tiny",
            "nreimers/MiniLM-L6-H384-uncased",
        ],
        "roberta-base": ["distilroberta-base", "sentence-transformers/all-MiniLM-L6-v2"],
        "albert-base-v2": ["albert/albert-tiny-v2", "albert/albert-small-v2"],
        "microsoft/deberta-base": ["microsoft/deberta-v3-small", "microsoft/deberta-v3-xsmall"],
        "xlm-roberta-base": ["sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"],
    }

    def __init__(self, hf_token: Optional[str] = None):
        """
        Initialize model validator.

        Args:
            hf_token: HuggingFace authentication token
        """
        self.hf_token = hf_token or os.getenv("HF_TOKEN")
        self.api = HfApi(token=self.hf_token)
        self.available_device = self._detect_available_device()

    def _detect_available_device(self) -> str:
        """Detect the best available device on this system."""
        if torch.cuda.is_available():
            # Get CUDA compute capability
            major, minor = torch.cuda.get_device_capability()
            return f"cuda>={major}.{minor}"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def validate_model(self, model_id: str, role: str = "teacher") -> Dict[str, Any]:
        """
        Validate a single model.

        Args:
            model_id: HuggingFace model ID (e.g., 'bert-base-uncased')
            role: 'teacher' or 'student'

        Returns:
            Validation report with status and details
        """
        report: Dict[str, Any] = {
            "model_id": model_id,
            "role": role,
            "exists": False,
            "accessible": False,
            "device_compatible": False,
            "architecture_supported": False,
            "size_mb": None,
            "params_millions": None,
            "model_type": None,
            "pipeline_tag": None,
            "device_requirements": [],
            "errors": [],
            "warnings": [],
            "alternatives": [],
        }

        try:
            # Check if model exists
            logger.info(f"Validating model: {model_id}")
            info = model_info(model_id, token=self.hf_token)
            report["exists"] = True
            report["accessible"] = True

            # Extract model info
            report["model_type"] = getattr(info, "pipeline_tag", "unknown")
            report["pipeline_tag"] = getattr(info, "pipeline_tag", None)

            # Get model size (if available)
            # Try multiple approaches to get model size
            size_found = False

            try:
                if hasattr(info, "safetensors") and info.safetensors:
                    # Approach 1: safetensors is a dict with 'total' key
                    if isinstance(info.safetensors, dict) and "total" in info.safetensors:
                        total_size = info.safetensors["total"]
                        report["size_mb"] = total_size / (1024 * 1024)
                        size_found = True
                    # Approach 2: safetensors has 'parameters' dict
                    elif hasattr(info.safetensors, "parameters"):
                        try:
                            params = getattr(info.safetensors, "parameters", {})
                            if isinstance(params, dict):
                                total_size = 0
                                for file_info in params.values():
                                    if isinstance(file_info, dict):
                                        size = file_info.get("size", 0)
                                        total_size += size or 0  # Handle None
                                    elif hasattr(file_info, "size"):
                                        size = getattr(file_info, "size", 0)
                                        total_size += size or 0  # Handle None
                                if total_size > 0:
                                    report["size_mb"] = total_size / (1024 * 1024)
                                    size_found = True
                        except Exception as e:
                            logger.debug(f"Could not parse safetensors.parameters: {e}")
            except Exception as e:
                logger.debug(f"Could not parse safetensors: {e}")

            # Fallback: estimate from model files
            if not size_found:
                if hasattr(info, "siblings") and info.siblings:
                    try:
                        total_size = sum(
                            (getattr(sibling, "size", 0) or 0)  # Handle None values
                            for sibling in info.siblings
                            if hasattr(sibling, "rfilename")
                            and (
                                sibling.rfilename.endswith(".bin")
                                or sibling.rfilename.endswith(".safetensors")
                            )
                        )
                        if total_size > 0:
                            report["size_mb"] = total_size / (1024 * 1024)
                    except Exception as e:
                        logger.debug(f"Could not calculate size from siblings: {e}")

            # Check device compatibility
            device_check = self._check_device_compatibility(model_id, report["size_mb"])  # type: ignore[arg-type,operator]
            report["device_compatible"] = device_check["compatible"]
            report["device_requirements"] = device_check["requirements"]

            if not device_check["compatible"]:
                report["errors"].append(  # type: ignore[union-attr,attr-defined]
                    f"Model requires {', '.join(device_check['requirements'])} "
                    f"but you have {self.available_device}"
                )
                report["alternatives"] = self._suggest_alternatives(model_id, role)

            # Check architecture support
            arch_check = self._check_architecture_support(model_id)
            report["architecture_supported"] = arch_check["supported"]

            if not arch_check["supported"]:
                report["errors"].append(  # type: ignore[union-attr,attr-defined]
                    f"Architecture {arch_check['architecture']} not fully supported for distillation"
                )

            # Warnings
            if report["size_mb"] and report["size_mb"] > 5000:  # type: ignore[comparison,operator]  # > 5GB
                report["warnings"].append(  # type: ignore[union-attr,attr-defined]
                    f"Large model ({report['size_mb']:.1f}MB). "
                    "May require significant memory and download time."
                )

            if not report["pipeline_tag"] or report["pipeline_tag"] not in [
                "text-classification",
                "token-classification",
                "question-answering",
                "text-generation",
                "fill-mask",
            ]:
                report["warnings"].append(  # type: ignore[union-attr,attr-defined]
                    f"Model pipeline tag '{report['pipeline_tag']}' may not be ideal for distillation"
                )  # type: ignore[union-attr,attr-defined]

        except RepositoryNotFoundError:
            report["errors"].append(f"Model '{model_id}' not found on HuggingFace Hub")
            report["alternatives"] = self._suggest_alternatives(model_id, role)

        except HfHubHTTPError as e:
            if e.response.status_code == 401:
                report["errors"].append(
                    "Model is private or gated. Please provide a valid HuggingFace token."
                )
            elif e.response.status_code == 403:
                report["errors"].append(
                    "Access forbidden. You may need to accept model terms on HuggingFace."
                )
            else:
                report["errors"].append(f"HTTP error: {str(e)}")

        except Exception as e:
            report["errors"].append(f"Validation failed: {str(e)}")
            logger.error(f"Model validation error for {model_id}: {e}")

        return report

    def _check_device_compatibility(
        self, model_id: str, size_mb: Optional[float] = None
    ) -> Dict[str, Any]:
        """Check if model is compatible with available device."""
        result = {
            "compatible": True,
            "requirements": ["cpu"],  # All models work on CPU
            "optimal_device": self.available_device,
        }

        # Check known limitations
        if model_id in self.DEVICE_LIMITATIONS:
            required_devices = self.DEVICE_LIMITATIONS[model_id]
            result["requirements"] = required_devices

            # Check if our device matches
            if self.available_device == "cpu":
                result["compatible"] = "cpu" in required_devices
            elif self.available_device == "mps":
                result["compatible"] = "mps" in required_devices or "cpu" in required_devices
            elif self.available_device.startswith("cuda"):
                # Check CUDA compute capability if specified
                for req in required_devices:
                    if req.startswith("cuda>="):
                        required_version = float(req.split(">=")[1])
                        available_version = float(self.available_device.split(">=")[1])
                        result["compatible"] = available_version >= required_version
                        break
                else:
                    result["compatible"] = "cuda" in required_devices

        # Heuristic: very large models likely need CUDA
        if size_mb and size_mb > 10000:  # > 10GB
            if self.available_device in ["cpu", "mps"]:
                result["compatible"] = False
                result["requirements"] = ["cuda"]

        # Additional heuristic: very large models in name
        if any(size in model_id.lower() for size in ["xl", "xxl", "20b", "66b", "176b"]):
            if self.available_device != "cuda":
                result["compatible"] = False
                result["requirements"] = ["cuda"]

        return result

    def _check_architecture_support(self, model_id: str) -> Dict[str, Any]:
        """Check if model architecture is supported for distillation."""
        result = {"supported": True, "architecture": "unknown", "confidence": "high"}

        # Heuristic architecture detection
        supported_archs = [
            "bert",
            "roberta",
            "albert",
            "distilbert",
            "electra",
            "deberta",
            "xlm",
            "gpt2",
            "gpt-neo",
            "opt",
            "t5",
            "mobilebert",
            "tinybert",
            "minilm",
        ]

        model_lower = model_id.lower()
        for arch in supported_archs:
            if arch in model_lower:
                result["architecture"] = arch
                return result

        # Unknown architecture - still allow but warn
        result["architecture"] = "unknown"
        result["confidence"] = "low"
        result["supported"] = True  # Allow but warn

        return result

    def _suggest_alternatives(self, model_id: str, role: str) -> List[Dict[str, str]]:
        """Suggest alternative models if validation fails."""
        alternatives = []

        # If it's a teacher, suggest from recommended pairs
        if role == "teacher":
            for teacher, students in self.RECOMMENDED_PAIRS.items():
                alternatives.append(
                    {
                        "model_id": teacher,
                        "reason": f"Popular teacher model with {len(students)} compatible students",
                        "device_compatible": True,
                    }
                )
                if len(alternatives) >= 3:
                    break

        # If it's a student, suggest lightweight alternatives
        else:
            alternatives = [
                {
                    "model_id": "distilbert-base-uncased",
                    "reason": "Lightweight, widely compatible, good performance",
                    "device_compatible": True,
                },
                {
                    "model_id": "google/mobilebert-uncased",
                    "reason": "Mobile-optimized, very fast inference",
                    "device_compatible": True,
                },
                {
                    "model_id": "prajjwal1/bert-tiny",
                    "reason": "Tiny model, extreme compression",
                    "device_compatible": True,
                },
            ]

        return alternatives[:3]  # type: ignore[return-value]

    def validate_pair(self, teacher_id: str, student_id: str) -> Dict[str, Any]:
        """
        Validate teacher-student pair compatibility.

        Args:
            teacher_id: Teacher model ID
            student_id: Student model ID

        Returns:
            Comprehensive validation report
        """
        report: Dict[str, Any] = {
            "teacher": self.validate_model(teacher_id, "teacher"),
            "student": self.validate_model(student_id, "student"),
            "pair_compatible": True,
            "issues": [],
            "warnings": [],
            "recommendations": [],
        }

        # Check individual models
        teacher_ok = (
            report["teacher"]["exists"]  # type: ignore[truthy-iterable,attr-defined,index]
            and report["teacher"]["accessible"]
            and report["teacher"]["device_compatible"]
        )

        student_ok = (
            report["student"]["exists"]  # type: ignore[truthy-iterable,attr-defined,index]
            and report["student"]["accessible"]
            and report["student"]["device_compatible"]
        )

        if not teacher_ok or not student_ok:
            report["pair_compatible"] = False
            report["issues"].append("One or both models failed validation")
            return report

        # Check architecture compatibility
        teacher_arch = report["teacher"]["architecture_supported"]  # type: ignore[truthy-iterable,attr-defined,index]
        student_arch = report["student"]["architecture_supported"]  # type: ignore[truthy-iterable,attr-defined,index]

        if not teacher_arch or not student_arch:
            report["warnings"].append(
                "One or both models have unknown architecture. "
                "Distillation may require custom configuration."
            )

        # Check size difference (student should be smaller)
        if (
            report["teacher"]["size_mb"]
            and report["student"]["size_mb"]  # type: ignore[truthy-iterable,attr-defined,index]
            and report["student"]["size_mb"] >= report["teacher"]["size_mb"]
        ):
            report["warnings"].append(
                f"Student ({report['student']['size_mb']:.1f}MB) is larger than "
                f"teacher ({report['teacher']['size_mb']:.1f}MB). "
                "Consider using a smaller student model."
            )

        # Calculate compression ratio
        if report["teacher"]["size_mb"] and report["student"]["size_mb"]:  # type: ignore[truthy-iterable,attr-defined,index]
            compression = report["teacher"]["size_mb"] / report["student"]["size_mb"]
            report["compression_ratio"] = f"{compression:.1f}x"

            if compression < 1.5:
                report["warnings"].append(  # type: ignore[union-attr,attr-defined]
                    f"Low compression ratio ({compression:.1f}x). "
                    "Consider using a smaller student for better efficiency gains."
                )

        # Success recommendations
        if report["pair_compatible"]:
            report["recommendations"].append(  # type: ignore[union-attr,attr-defined]
                f" Both models validated successfully on {self.available_device}"
            )
            if "compression_ratio" in report:
                report["recommendations"].append(  # type: ignore[union-attr,attr-defined]
                    f" Good compression: {report['compression_ratio']}"
                )

        return report


# Convenience function
def validate_models(
    teacher_id: str, student_id: str, hf_token: Optional[str] = None
) -> Dict[str, Any]:
    """
    Quick validation of teacher-student pair.

    Args:
        teacher_id: Teacher model ID
        student_id: Student model ID
        hf_token: HuggingFace token (optional)

    Returns:
        Validation report
    """
    validator = ModelValidator(hf_token=hf_token)
    return validator.validate_pair(teacher_id, student_id)
