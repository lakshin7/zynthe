"""
Zynthé EvalX - Extended Metrics Module
Advanced distillation-specific metrics and KPIs
"""

from __future__ import annotations


import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, List, Any
import time
import logging

logger = logging.getLogger(__name__)


class DistillationMetrics:
    """Distillation-specific metrics for knowledge retention analysis."""

    @staticmethod
    def kl_divergence(
        teacher_logits: torch.Tensor, student_logits: torch.Tensor, temperature: float = 1.0
    ) -> float:
        """
        Compute KL divergence between teacher and student predictions.
        Lower = better knowledge transfer.

        Args:
            teacher_logits: Teacher model logits [batch, num_classes]
            student_logits: Student model logits [batch, num_classes]
            temperature: Temperature for softening distributions

        Returns:
            KL divergence value
        """
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)

        kl_div = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
        return kl_div.item() * (temperature**2)

    @staticmethod
    def js_divergence(teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> float:
        """
        Jensen-Shannon divergence (symmetric version of KL).
        More stable metric for distribution comparison.
        """
        teacher_probs = F.softmax(teacher_logits, dim=-1)
        student_probs = F.softmax(student_logits, dim=-1)

        # Compute midpoint distribution
        m = 0.5 * (teacher_probs + student_probs)

        # KL(teacher || m) + KL(student || m)
        kl_teacher = F.kl_div(torch.log(m), teacher_probs, reduction="batchmean")
        kl_student = F.kl_div(torch.log(m), student_probs, reduction="batchmean")

        js_div = 0.5 * (kl_teacher + kl_student)
        return js_div.item()

    @staticmethod
    def prediction_agreement(teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> float:
        """
        Percentage of samples where teacher and student agree on prediction.
        Higher = better mimicry.
        """
        teacher_preds = torch.argmax(teacher_logits, dim=-1)
        student_preds = torch.argmax(student_logits, dim=-1)

        agreement = (teacher_preds == student_preds).float().mean()
        return agreement.item()

    @staticmethod
    def confidence_correlation(teacher_logits: torch.Tensor, student_logits: torch.Tensor) -> float:
        """
        Correlation between teacher and student confidence levels.
        1.0 = perfect correlation, 0.0 = no correlation.
        """
        teacher_conf = F.softmax(teacher_logits, dim=-1).max(dim=-1)[0]
        student_conf = F.softmax(student_logits, dim=-1).max(dim=-1)[0]

        # Pearson correlation
        teacher_mean = teacher_conf.mean()
        student_mean = student_conf.mean()

        numerator = ((teacher_conf - teacher_mean) * (student_conf - student_mean)).sum()
        denominator = torch.sqrt(
            ((teacher_conf - teacher_mean) ** 2).sum() * ((student_conf - student_mean) ** 2).sum()
        )

        if denominator == 0:
            return 0.0

        correlation = numerator / denominator
        return correlation.item()


class FeatureSimilarity:
    """Feature-space similarity metrics."""

    @staticmethod
    def cosine_similarity(teacher_features: torch.Tensor, student_features: torch.Tensor) -> float:
        """
        Cosine similarity between teacher and student feature representations.
        1.0 = identical direction, 0.0 = orthogonal, -1.0 = opposite.
        """
        # Flatten if needed
        if teacher_features.dim() > 2:
            teacher_features = teacher_features.flatten(1)
            student_features = student_features.flatten(1)

        # Compute cosine similarity
        similarity = F.cosine_similarity(teacher_features, student_features, dim=-1)
        return similarity.mean().item()

    @staticmethod
    def l2_distance(teacher_features: torch.Tensor, student_features: torch.Tensor) -> float:
        """
        L2 distance between feature representations.
        Lower = better feature mimicry.
        """
        if teacher_features.dim() > 2:
            teacher_features = teacher_features.flatten(1)
            student_features = student_features.flatten(1)

        distance = torch.norm(teacher_features - student_features, p=2, dim=-1)
        return distance.mean().item()

    @staticmethod
    def feature_correlation(
        teacher_features: torch.Tensor, student_features: torch.Tensor
    ) -> float:
        """
        Correlation between teacher and student feature activations.
        """
        if teacher_features.dim() > 2:
            teacher_features = teacher_features.flatten(1)
            student_features = student_features.flatten(1)

        # Compute correlation coefficient
        t_mean = teacher_features.mean(dim=1, keepdim=True)
        s_mean = student_features.mean(dim=1, keepdim=True)

        t_centered = teacher_features - t_mean
        s_centered = student_features - s_mean

        numerator = (t_centered * s_centered).sum(dim=1)
        denominator = torch.sqrt((t_centered**2).sum(dim=1) * (s_centered**2).sum(dim=1))

        correlation = numerator / (denominator + 1e-8)
        return correlation.mean().item()


class CompressionAwareScore:
    """Compression-Aware Score (CAS) - Unified metric for model selection."""

    @staticmethod
    def compute_cas(
        accuracy: float,
        teacher_params: int,
        student_params: int,
        teacher_latency: float,
        student_latency: float,
        alpha: float = 0.6,
        beta: float = 0.2,
        gamma: float = 0.2,
    ) -> Dict[str, float]:
        """
        Compute Compression-Aware Score.

        CAS = α·Accuracy - β·SizeRatio - γ·LatencyRatio

        Args:
            accuracy: Student model accuracy (0-1)
            teacher_params: Teacher parameter count
            student_params: Student parameter count
            teacher_latency: Teacher inference latency (ms)
            student_latency: Student inference latency (ms)
            alpha: Weight for accuracy (importance)
            beta: Weight for size penalty
            gamma: Weight for latency penalty

        Returns:
            Dictionary with CAS and components
        """
        size_ratio = student_params / teacher_params
        latency_ratio = student_latency / teacher_latency

        cas = alpha * accuracy - beta * size_ratio - gamma * latency_ratio

        return {
            "cas": cas,
            "accuracy": accuracy,
            "size_ratio": size_ratio,
            "latency_ratio": latency_ratio,
            "compression_factor": teacher_params / student_params,
            "speedup": teacher_latency / student_latency,
            "efficiency_score": accuracy / size_ratio,  # Accuracy per parameter ratio
        }

    @staticmethod
    def rank_models(models: List[Dict]) -> List[Dict]:
        """
        Rank multiple student models by CAS.

        Args:
            models: List of dicts with 'name', 'accuracy', 'params', 'latency'

        Returns:
            Sorted list of models with CAS scores
        """
        teacher_params = models[0].get("teacher_params", models[0]["params"])
        teacher_latency = models[0].get("teacher_latency", models[0]["latency"])

        for model in models:
            if "cas" not in model:
                cas_result = CompressionAwareScore.compute_cas(
                    accuracy=model["accuracy"],
                    teacher_params=teacher_params,
                    student_params=model["params"],
                    teacher_latency=teacher_latency,
                    student_latency=model["latency"],
                )
                model.update(cas_result)

        # Sort by CAS descending
        return sorted(models, key=lambda x: x["cas"], reverse=True)


class DistillationEfficacyIndex:
    """Distillation Efficacy Index (DEI) - Holistic distillation quality metric."""

    @staticmethod
    def compute_dei(
        teacher_acc: float,
        student_acc: float,
        teacher_params: int,
        student_params: int,
        retention_bonus: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Compute Distillation Efficacy Index.

        DEI = (Acc_student / Acc_teacher) × (Params_teacher / Params_student) × (1 + retention_bonus)

        Higher DEI = better distillation (maintained accuracy with compression)

        Args:
            teacher_acc: Teacher accuracy
            student_acc: Student accuracy
            teacher_params: Teacher parameter count
            student_params: Student parameter count
            retention_bonus: Bonus for knowledge retention metrics (0-0.5)

        Returns:
            DEI and components
        """
        accuracy_retention = student_acc / (teacher_acc + 1e-8)
        compression_ratio = teacher_params / student_params

        dei = accuracy_retention * compression_ratio * (1.0 + retention_bonus)

        efficiency_rating = "Excellent" if dei > 1.5 else "Good" if dei > 1.0 else "Fair"

        return {
            "dei": float(dei),
            "accuracy_retention": float(accuracy_retention),
            "compression_ratio": float(compression_ratio),
            "teacher_acc": float(teacher_acc),
            "student_acc": float(student_acc),
            "accuracy_drop": float(teacher_acc - student_acc),
            "efficiency_rating": efficiency_rating,
        }


class LossComponentTracker:
    """Track individual loss components during training."""

    def __init__(self):
        self.history = {
            "total_loss": [],
            "kd_loss": [],
            "ce_loss": [],
            "feature_loss": [],
            "attention_loss": [],
            "kl_divergence": [],
            "prediction_agreement": [],
            "confidence_correlation": [],
        }

    def update(self, epoch: int, **kwargs):
        """Update loss components for current epoch."""
        for key, value in kwargs.items():
            if key in self.history:
                self.history[key].append(value)

    def get_trends(self) -> Dict[str, Dict[str, float]]:
        """Analyze trends in loss components."""
        trends = {}

        for component, values in self.history.items():
            if len(values) < 2:
                continue

            # Compute trend
            initial = values[0] if values[0] != 0 else 1e-8
            final = values[-1]
            improvement = ((initial - final) / initial) * 100

            # Compute stability (coefficient of variation)
            mean_val = np.mean(values)
            std_val = np.std(values)
            stability = 1.0 - (std_val / (mean_val + 1e-8))

            trends[component] = {
                "initial": initial,
                "final": final,
                "improvement_pct": improvement,
                "stability": stability,
                "trend": (
                    "improving"
                    if improvement > 5
                    else "stable" if improvement > -5 else "degrading"
                ),
            }

        return trends

    def export(self) -> Dict:
        """Export complete history."""
        return {"history": self.history, "trends": self.get_trends()}


class PerformanceProfiler:
    """Profile model inference performance."""

    @staticmethod
    def profile_inference(
        model: torch.nn.Module,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        device: str,
        num_runs: int = 100,
        warmup: int = 10,
    ) -> Dict[str, float]:
        """
        Profile model inference latency and throughput.

        Args:
            model: Model to profile
            input_ids: Sample input
            attention_mask: Attention mask
            device: Device (cpu/cuda/mps)
            num_runs: Number of inference runs
            warmup: Warmup runs (excluded from timing)

        Returns:
            Performance metrics
        """
        model.eval()

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(input_ids=input_ids, attention_mask=attention_mask)

        # Synchronize if GPU
        if device in ["cuda", "mps"]:
            torch.cuda.synchronize() if device == "cuda" else None

        # Profile
        latencies = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = model(input_ids=input_ids, attention_mask=attention_mask)

                if device in ["cuda", "mps"]:
                    torch.cuda.synchronize() if device == "cuda" else None

                end = time.perf_counter()
                latencies.append((end - start) * 1000)  # Convert to ms

        latencies = np.array(latencies)

        return {
            "mean_latency_ms": float(np.mean(latencies)),
            "median_latency_ms": float(np.median(latencies)),
            "std_latency_ms": float(np.std(latencies)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
            "throughput_samples_per_sec": float(1000.0 / np.mean(latencies)),
            "batch_size": int(input_ids.size(0)),
        }

    @staticmethod
    def compare_models(teacher_profile: Dict, student_profile: Dict) -> Dict:
        """Compare performance between teacher and student."""
        speedup = teacher_profile["mean_latency_ms"] / student_profile["mean_latency_ms"]
        throughput_improvement = (
            student_profile["throughput_samples_per_sec"]
            / teacher_profile["throughput_samples_per_sec"]
        )

        return {
            "speedup": speedup,
            "throughput_improvement": throughput_improvement,
            "teacher_latency_ms": teacher_profile["mean_latency_ms"],
            "student_latency_ms": student_profile["mean_latency_ms"],
            "latency_reduction_pct": (
                (teacher_profile["mean_latency_ms"] - student_profile["mean_latency_ms"])
                / teacher_profile["mean_latency_ms"]
            )
            * 100,
        }


# Convenience function for complete evaluation
def compute_extended_metrics(
    teacher_logits: torch.Tensor,
    student_logits: torch.Tensor,
    teacher_features: Optional[torch.Tensor] = None,
    student_features: Optional[torch.Tensor] = None,
    temperature: float = 2.0,
) -> Dict[str, float]:
    """
    Compute all extended distillation metrics.

    Args:
        teacher_logits: Teacher output logits
        student_logits: Student output logits
        teacher_features: Teacher intermediate features (optional)
        student_features: Student intermediate features (optional)
        temperature: Temperature for KL divergence

    Returns:
        Dictionary of all metrics
    """
    metrics = {}

    # Distillation metrics
    dist_metrics = DistillationMetrics()
    metrics["kl_divergence"] = dist_metrics.kl_divergence(
        teacher_logits, student_logits, temperature
    )
    metrics["js_divergence"] = dist_metrics.js_divergence(teacher_logits, student_logits)
    metrics["prediction_agreement"] = dist_metrics.prediction_agreement(
        teacher_logits, student_logits
    )
    metrics["confidence_correlation"] = dist_metrics.confidence_correlation(
        teacher_logits, student_logits
    )

    # Feature similarity (if features provided)
    if teacher_features is not None and student_features is not None:
        feat_sim = FeatureSimilarity()
        metrics["feature_cosine_similarity"] = feat_sim.cosine_similarity(
            teacher_features, student_features
        )
        metrics["feature_l2_distance"] = feat_sim.l2_distance(teacher_features, student_features)
        metrics["feature_correlation"] = feat_sim.feature_correlation(
            teacher_features, student_features
        )

    return metrics


if __name__ == "__main__":
    # Example usage
    logger.info("Zynthé EvalX - Extended Metrics Module")
    logger.info("=" * 60)
    # Simulate teacher and student outputs
    teacher_logits = torch.randn(32, 10)
    student_logits = teacher_logits + torch.randn(32, 10) * 0.1

    # Compute metrics
    metrics = compute_extended_metrics(teacher_logits, student_logits)

    logger.info("\n[INFO] Distillation Metrics:")
    for key, value in metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    # Compute CAS
    logger.info("\n[SCORE] Compression-Aware Score:")
    cas = CompressionAwareScore.compute_cas(
        accuracy=0.945,
        teacher_params=125_000_000,
        student_params=82_000_000,
        teacher_latency=45.2,
        student_latency=28.5,
    )
    for key, value in cas.items():
        logger.info(f"  {key}: {value:.4f}")
    # Compute DEI
    logger.info("\n[TARGET] Distillation Efficacy Index:")
    dei = DistillationEfficacyIndex.compute_dei(
        teacher_acc=0.96, student_acc=0.945, teacher_params=125_000_000, student_params=82_000_000
    )
    for key, value in dei.items():
        logger.info(f"  {key}: {value}")
