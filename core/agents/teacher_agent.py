"""
Teacher Model Agent - Intelligent Teacher Selection and Management

This agent automatically:
1. Analyzes the task and dataset
2. Recommends optimal teacher models
3. Downloads and configures the teacher
4. Optionally fine-tunes the teacher
5. Validates teacher quality before distillation
"""

import torch
from transformers import (
    AutoModel, AutoTokenizer, AutoModelForSequenceClassification,
    AutoModelForCausalLM
)
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class TeacherRecommendation:
    """Recommendation for a teacher model"""
    model_name: str
    confidence: float  # 0-1, how confident we are this is good
    reasoning: str
    estimated_size: str  # e.g., "110M params"
    task_fit: float  # 0-1, how well it fits the task
    download_size: str  # e.g., "440MB"
    requires_finetuning: bool


class TeacherModelAgent:
    """
    Intelligent agent that selects and manages teacher models.
    
    Features:
    - Task detection (sentiment, NER, QA, etc.)
    - Model recommendation based on task
    - Automatic download and setup
    - Teacher quality validation
    - Optional fine-tuning
    """
    
    # Pre-defined teacher models for common tasks
    TEACHER_CATALOG = {
        "sentiment_analysis": [
            {
                "name": "bert-base-uncased",
                "size": "110M params",
                "download": "440MB",
                "confidence": 0.95,
                "reason": "BERT-base: Industry standard, excellent accuracy, well-documented"
            },
            {
                "name": "roberta-base",
                "size": "125M params",
                "download": "500MB",
                "confidence": 0.90,
                "reason": "RoBERTa: Improved BERT training, better on sentiment tasks"
            },
            {
                "name": "distilbert-base-uncased",
                "size": "66M params",
                "download": "268MB",
                "confidence": 0.75,
                "reason": "DistilBERT: Faster, smaller, good for tight resources"
            }
        ],
        "text_classification": [
            {
                "name": "bert-base-uncased",
                "size": "110M params",
                "download": "440MB",
                "confidence": 0.95,
                "reason": "BERT-base: Excellent general classification performance"
            },
            {
                "name": "roberta-large",
                "size": "355M params",
                "download": "1.4GB",
                "confidence": 0.85,
                "reason": "RoBERTa-large: Highest accuracy, needs more resources"
            }
        ],
        "question_answering": [
            {
                "name": "bert-large-uncased-whole-word-masking-finetuned-squad",
                "size": "340M params",
                "download": "1.3GB",
                "confidence": 0.95,
                "reason": "BERT-large fine-tuned on SQuAD: State-of-art QA"
            }
        ],
        "causal_lm": [
            {
                "name": "gpt2",
                "size": "117M params",
                "download": "548MB",
                "confidence": 0.90,
                "reason": "GPT-2: Excellent text generation, widely used"
            },
            {
                "name": "gpt2-medium",
                "size": "345M params",
                "download": "1.5GB",
                "confidence": 0.85,
                "reason": "GPT-2 Medium: Better quality, more parameters"
            }
        ]
    }
    
    def __init__(self, device: Optional[str] = None):
        """Initialize the teacher agent"""
        self.device = self._detect_device(device)
        logger.info(f"🤖 Teacher Agent initialized on device: {self.device}")
    
    def _detect_device(self, device_str: Optional[str] = None) -> torch.device:
        """Auto-detect best device"""
        if device_str:
            return torch.device(device_str)
        elif torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    def detect_task_from_data(self, data_samples: List[Dict[str, Any]]) -> str:
        """
        Analyze data samples to detect the task type.
        
        Args:
            data_samples: List of data samples (dictionaries with text/label)
            
        Returns:
            Task type string
        """
        if not data_samples:
            return "text_classification"  # Default
        
        sample = data_samples[0]
        
        # Check for sentiment analysis indicators
        if "label" in sample:
            label = str(sample["label"]).lower()
            if label in ["positive", "negative", "neutral", "0", "1"]:
                logger.info("🔍 Detected task: Sentiment Analysis")
                return "sentiment_analysis"
        
        # Check for QA indicators
        if "question" in sample or "context" in sample:
            logger.info("🔍 Detected task: Question Answering")
            return "question_answering"
        
        # Check for NER indicators
        if "entities" in sample or "ner_tags" in sample:
            logger.info("🔍 Detected task: Named Entity Recognition")
            return "ner"
        
        # Default to text classification
        logger.info("🔍 Detected task: Text Classification (default)")
        return "text_classification"
    
    def recommend_teachers(
        self,
        task: str,
        dataset_size: Optional[int] = None,
        resource_constraint: str = "medium"  # low, medium, high
    ) -> List[TeacherRecommendation]:
        """
        Recommend teacher models for the given task.
        
        Args:
            task: Task type (sentiment_analysis, text_classification, etc.)
            dataset_size: Number of training samples (helps decide if fine-tuning needed)
            resource_constraint: 'low', 'medium', or 'high' computing resources
            
        Returns:
            List of TeacherRecommendation objects, sorted by confidence
        """
        candidates = self.TEACHER_CATALOG.get(task, self.TEACHER_CATALOG["text_classification"])
        
        recommendations = []
        for candidate in candidates:
            # Adjust confidence based on resource constraints
            confidence: float = candidate["confidence"]  # type: ignore[assignment]
            if resource_constraint == "low":
                # Prefer smaller models on low resources
                if "66M" in str(candidate["size"]) or "110M" in str(candidate["size"]):
                    confidence *= 1.1  # Boost smaller models
                elif "340M" in str(candidate["size"]) or "355M" in str(candidate["size"]):
                    confidence *= 0.7  # Penalize large models
            elif resource_constraint == "high":
                # Prefer larger models on high resources
                if "340M" in str(candidate["size"]) or "355M" in str(candidate["size"]):
                    confidence *= 1.1
            
            # Determine if fine-tuning is needed
            requires_finetuning = dataset_size and dataset_size < 5000
            
            rec = TeacherRecommendation(
                model_name=str(candidate["name"]),  # type: ignore[arg-type]
                confidence=min(float(confidence), 1.0),  # Cap at 1.0
                reasoning=str(candidate["reason"]),  # type: ignore[arg-type]
                estimated_size=str(candidate["size"]),  # type: ignore[arg-type]
                task_fit=float(candidate["confidence"]),  # type: ignore[arg-type]
                download_size=str(candidate["download"]),  # type: ignore[arg-type]
                requires_finetuning=bool(requires_finetuning) if requires_finetuning else False
            )
            recommendations.append(rec)
        
        # Sort by confidence
        recommendations.sort(key=lambda x: x.confidence, reverse=True)
        
        return recommendations
    
    def load_teacher(
        self,
        model_name: str,
        task_type: str = "sequence_classification",
        num_labels: int = 2,
        **kwargs
    ) -> Tuple[Any, Any]:
        """
        Load a teacher model and tokenizer.
        
        Args:
            model_name: HuggingFace model name
            task_type: Type of task
            num_labels: Number of output labels
            **kwargs: Additional model arguments
            
        Returns:
            (model, tokenizer) tuple
        """
        logger.info(f"📥 Loading teacher model: {model_name}")
        
        try:
            # Load tokenizer first
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model based on task type
            if task_type in ["sequence_classification", "sentiment_analysis", "text_classification"]:
                model_kwargs = {
                    "num_labels": num_labels,
                    "label2id": kwargs.get("label2id", {"negative": 0, "positive": 1}),
                    "id2label": kwargs.get("id2label", {0: "negative", 1: "positive"})
                }
                model = AutoModelForSequenceClassification.from_pretrained(
                    model_name, **model_kwargs
                )
            elif task_type in ["causal_lm", "text_generation"]:
                model = AutoModelForCausalLM.from_pretrained(model_name)
            else:
                model = AutoModel.from_pretrained(model_name)
            
            model.to(self.device)
            logger.info(f"✅ Teacher loaded successfully on {self.device}")
            
            return model, tokenizer
            
        except Exception as e:
            logger.error(f"❌ Failed to load teacher {model_name}: {e}")
            raise
    
    def validate_teacher(
        self,
        model: Any,
        tokenizer: Any,
        validation_samples: List[Dict[str, Any]],
        min_accuracy: float = 0.7
    ) -> Dict[str, Any]:
        """
        Validate teacher model quality on validation samples.
        
        Args:
            model: Teacher model
            tokenizer: Tokenizer
            validation_samples: List of validation samples
            min_accuracy: Minimum required accuracy
            
        Returns:
            Validation results dictionary
        """
        logger.info("🔍 Validating teacher model quality...")
        
        model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for sample in validation_samples[:100]:  # Test on first 100
                text = sample.get("text", "")
                label = sample.get("label", 0)
                
                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=128
                ).to(self.device)
                
                outputs = model(**inputs)
                pred = torch.argmax(outputs.logits, dim=-1).item()
                
                if isinstance(label, str):
                    label = 1 if label.lower() == "positive" else 0
                
                if pred == label:
                    correct += 1
                total += 1
        
        accuracy = correct / total if total > 0 else 0.0
        
        result = {
            "accuracy": accuracy,
            "passed": accuracy >= min_accuracy,
            "samples_tested": total,
            "recommendation": "Teacher is good!" if accuracy >= min_accuracy else "Consider fine-tuning teacher"
        }
        
        logger.info(f"📊 Teacher Validation: {accuracy:.2%} accuracy on {total} samples")
        
        return result
    
    def auto_select_and_load(
        self,
        data_samples: List[Dict[str, Any]],
        resource_constraint: str = "medium",
        num_labels: int = 2,
        validate: bool = True
    ) -> Dict[str, Any]:
        """
        Automatically detect task, select best teacher, load and validate.
        
        This is the main entry point for the agent!
        
        Args:
            data_samples: Training/validation data samples
            resource_constraint: 'low', 'medium', or 'high'
            num_labels: Number of classification labels
            validate: Whether to validate teacher quality
            
        Returns:
            Dictionary with model, tokenizer, and metadata
        """
        logger.info("🤖 Starting automatic teacher selection...")
        
        # Step 1: Detect task
        task = self.detect_task_from_data(data_samples)
        
        # Step 2: Get recommendations
        recommendations = self.recommend_teachers(
            task=task,
            dataset_size=len(data_samples),
            resource_constraint=resource_constraint
        )
        
        logger.info(f"📋 Found {len(recommendations)} teacher candidates:")
        for i, rec in enumerate(recommendations[:3], 1):
            logger.info(f"  {i}. {rec.model_name} (confidence: {rec.confidence:.0%})")
            logger.info(f"     {rec.reasoning}")
        
        # Step 3: Load the best teacher
        best_teacher = recommendations[0]
        logger.info(f"✨ Selected: {best_teacher.model_name}")
        
        model, tokenizer = self.load_teacher(
            model_name=best_teacher.model_name,
            task_type=task,
            num_labels=num_labels
        )
        
        # Step 4: Validate (optional)
        validation_result = None
        if validate and len(data_samples) > 0:
            validation_result = self.validate_teacher(
                model, tokenizer, data_samples[:100]
            )
        
        return {
            "model": model,
            "tokenizer": tokenizer,
            "model_name": best_teacher.model_name,
            "task": task,
            "recommendation": best_teacher,
            "validation": validation_result,
            "device": str(self.device)
        }
    
    def save_recommendation_report(
        self,
        recommendations: List[TeacherRecommendation],
        output_path: Path
    ):
        """Save teacher recommendations to a report file"""
        report = {
            "recommendations": [
                {
                    "model_name": rec.model_name,
                    "confidence": rec.confidence,
                    "reasoning": rec.reasoning,
                    "size": rec.estimated_size,
                    "download_size": rec.download_size,
                    "task_fit": rec.task_fit,
                    "requires_finetuning": rec.requires_finetuning
                }
                for rec in recommendations
            ]
        }
        
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"💾 Saved recommendation report to {output_path}")


def quick_teacher_setup(
    data_samples: List[Dict[str, Any]],
    device: Optional[str] = None,
    resource_constraint: str = "medium"
) -> Dict[str, Any]:
    """
    Quick one-line teacher setup for easy integration.
    
    Example:
        result = quick_teacher_setup(data_samples)
        teacher = result['model']
        tokenizer = result['tokenizer']
    """
    agent = TeacherModelAgent(device=device)
    return agent.auto_select_and_load(
        data_samples=data_samples,
        resource_constraint=resource_constraint
    )
