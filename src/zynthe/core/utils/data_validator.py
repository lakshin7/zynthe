"""
Data Validation and Leakage Detection
=====================================

Automatic detection of:
- Train/Val data overlap (data leakage)
- Class imbalance issues
- Data quality problems
- Overfitting/Underfitting signals

Author: Zynthé Team
"""

import json
from pathlib import Path
from collections import Counter
from typing import Dict, List, Optional, Any
import logging

LOG = logging.getLogger(__name__)


class DataLeakageDetector:
    """Detect data leakage between train and validation sets."""
    
    @staticmethod
    def check_overlap(
        train_texts: List[str],
        val_texts: List[str],
        check_prefixes: bool = True,
        prefix_length: int = 100
    ) -> Dict[str, Any]:
        """
        Check for overlap between train and validation datasets.
        
        Args:
            train_texts: List of training texts
            val_texts: List of validation texts
            check_prefixes: Also check for near-duplicates using prefixes
            prefix_length: Length of prefix to check for near-duplicates
            
        Returns:
            Dict with overlap statistics and samples
        """
        LOG.info("Checking for data leakage between train and validation sets...")
        
        # Check exact matches
        train_set = set(train_texts)
        val_set = set(val_texts)
        exact_overlap = train_set.intersection(val_set)
        
        results = {
            'train_size': len(train_texts),
            'val_size': len(val_texts),
            'train_unique': len(train_set),
            'val_unique': len(val_set),
            'exact_overlap_count': len(exact_overlap),
            'exact_overlap_pct': (len(exact_overlap) / len(val_set) * 100) if val_set else 0,
            'has_exact_leakage': len(exact_overlap) > 0,
            'exact_overlap_samples': list(exact_overlap)[:5]  # First 5 examples
        }
        
        # Check near-duplicates using prefixes
        if check_prefixes:
            train_prefixes = set([t[:prefix_length] for t in train_texts])
            val_prefixes = set([t[:prefix_length] for t in val_texts])
            prefix_overlap = train_prefixes.intersection(val_prefixes)
            
            # Exclude exact matches we already found
            near_duplicates = prefix_overlap - exact_overlap
            
            results['prefix_overlap_count'] = len(near_duplicates)
            results['prefix_overlap_pct'] = (len(near_duplicates) / len(val_set) * 100) if val_set else 0
            results['has_near_duplicates'] = len(near_duplicates) > 0
        
        # Log results
        if results['has_exact_leakage']:
            LOG.error(f"[FAIL] DATA LEAKAGE DETECTED! {results['exact_overlap_count']} samples "
                     f"({results['exact_overlap_pct']:.2f}%) overlap between train and val")
        else:
            LOG.info(" No exact overlap detected between train and val")
        
        return results
    
    @staticmethod
    def check_class_balance(
        labels: List[int],
        dataset_name: str = "dataset"
    ) -> Dict[str, Any]:
        """
        Check class balance in dataset.
        
        Args:
            labels: List of labels
            dataset_name: Name for logging
            
        Returns:
            Dict with class distribution statistics
        """
        label_counts = Counter(labels)
        total = len(labels)
        
        # Calculate imbalance ratio (max_class / min_class)
        if len(label_counts) > 1:
            max_count = max(label_counts.values())
            min_count = min(label_counts.values())
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        else:
            imbalance_ratio = 1.0
        
        results = {
            'total_samples': total,
            'num_classes': len(label_counts),
            'class_distribution': dict(label_counts),
            'class_percentages': {k: v/total*100 for k, v in label_counts.items()},
            'imbalance_ratio': imbalance_ratio,
            'is_balanced': imbalance_ratio < 2.0,  # Less than 2x difference
            'is_severely_imbalanced': imbalance_ratio > 5.0
        }
        
        # Log warnings
        if results['is_severely_imbalanced']:
            LOG.warning(f"  {dataset_name}: SEVERE class imbalance detected! "
                       f"Ratio: {imbalance_ratio:.2f}:1")
            LOG.warning("   Consider using class weights or oversampling")
        elif not results['is_balanced']:
            LOG.warning(f"  {dataset_name}: Class imbalance detected. Ratio: {imbalance_ratio:.2f}:1")
        
        return results


class OverfitUnderfitDetector:
    """Detect overfitting and underfitting during training."""
    
    @staticmethod
    def analyze_training_curves(
        train_losses: List[float],
        val_losses: List[float],
        train_metrics: Optional[List[float]] = None,
        val_metrics: Optional[List[float]] = None,
        metric_name: str = "accuracy"
    ) -> Dict[str, Any]:
        """
        Analyze training curves to detect overfitting/underfitting.
        
        Args:
            train_losses: Training losses per epoch
            val_losses: Validation losses per epoch
            train_metrics: Training metrics (optional)
            val_metrics: Validation metrics (optional)
            metric_name: Name of the metric
            
        Returns:
            Dict with overfitting/underfitting analysis
        """
        if len(train_losses) < 2 or len(val_losses) < 2:
            return {'status': 'insufficient_data', 'message': 'Need at least 2 epochs'}
        
        # Calculate metrics
        latest_train_loss = train_losses[-1]
        latest_val_loss = val_losses[-1]
        loss_gap = latest_val_loss - latest_train_loss
        loss_gap_pct = (loss_gap / latest_train_loss * 100) if latest_train_loss > 0 else 0
        
        # Check if validation loss is increasing (overfitting signal)
        val_loss_trend = "increasing" if len(val_losses) >= 3 and val_losses[-1] > val_losses[-3] else "decreasing"
        
        # Check if both losses are high (underfitting signal)
        high_loss_threshold = 0.5  # Adjust based on your task
        both_losses_high = latest_train_loss > high_loss_threshold and latest_val_loss > high_loss_threshold
        
        # Determine status
        status = "healthy"
        confidence = 0.0
        recommendations = []
        
        if loss_gap_pct > 20 and val_loss_trend == "increasing":
            status = "overfitting"
            confidence = min(loss_gap_pct / 50, 1.0)  # 50% gap = 100% confidence
            recommendations.extend([
                "Reduce model complexity or use regularization",
                "Add dropout or increase dropout rate",
                "Use early stopping",
                "Increase training data or use data augmentation",
                "Reduce training epochs"
            ])
        elif loss_gap_pct > 15:
            status = "mild_overfitting"
            confidence = loss_gap_pct / 30
            recommendations.extend([
                "Monitor closely - slight overfitting detected",
                "Consider adding regularization",
                "Early stopping may help"
            ])
        elif both_losses_high:
            status = "underfitting"
            confidence = 0.7
            recommendations.extend([
                "Increase model capacity",
                "Train for more epochs",
                "Reduce regularization",
                "Check if learning rate is too low",
                "Verify data quality and preprocessing"
            ])
        elif latest_val_loss < latest_train_loss:
            status = "suspicious"
            confidence = 0.5
            recommendations.append("Validation loss lower than training loss - check for data leakage!")
        
        results = {
            'status': status,
            'confidence': confidence,
            'train_loss': latest_train_loss,
            'val_loss': latest_val_loss,
            'loss_gap': loss_gap,
            'loss_gap_pct': loss_gap_pct,
            'val_loss_trend': val_loss_trend,
            'recommendations': recommendations
        }
        
        # Add metric analysis if provided
        if train_metrics and val_metrics and len(train_metrics) >= 2 and len(val_metrics) >= 2:
            latest_train_metric = train_metrics[-1]
            latest_val_metric = val_metrics[-1]
            metric_gap = latest_train_metric - latest_val_metric
            metric_gap_pct = (metric_gap / latest_train_metric * 100) if latest_train_metric > 0 else 0
            
            results[f'train_{metric_name}'] = latest_train_metric
            results[f'val_{metric_name}'] = latest_val_metric
            results[f'{metric_name}_gap'] = metric_gap
            results[f'{metric_name}_gap_pct'] = metric_gap_pct
            
            # If train metric much higher than val metric -> overfitting
            if metric_gap_pct > 10 and status == "healthy":
                status = "mild_overfitting"
                results['status'] = status
        
        return results
    
    @staticmethod
    def get_training_health_summary(
        train_losses: List[float],
        val_losses: List[float],
        metrics_history: Dict[str, List[float]]
    ) -> str:
        """
        Generate a human-readable summary of training health.
        
        Args:
            train_losses: Training losses
            val_losses: Validation losses
            metrics_history: Dict of metric name -> values per epoch
            
        Returns:
            Formatted summary string
        """
        analysis = OverfitUnderfitDetector.analyze_training_curves(
            train_losses, val_losses
        )
        
        status_emoji = {
            'healthy': '',
            'mild_overfitting': '',
            'overfitting': '[FAIL]',
            'underfitting': '[FAIL]',
            'suspicious': ''
        }
        
        summary = f"\n{'='*80}\n"
        summary += "TRAINING HEALTH SUMMARY\n"
        summary += f"{'='*80}\n\n"
        summary += f"Status: {status_emoji.get(analysis['status'], '?')} {analysis['status'].upper().replace('_', ' ')}\n"
        
        # Enhanced confidence message with context
        confidence_pct = analysis['confidence']*100
        num_epochs = len(val_losses)
        if confidence_pct == 0.0 and num_epochs < 5:
            summary += f"Confidence: {confidence_pct:.1f}% (Low - need 5+ epochs for statistical confidence, currently {num_epochs})\n\n"
        else:
            summary += f"Confidence: {confidence_pct:.1f}%\n\n"
        
        summary += "Latest Metrics:\n"
        summary += f"  Train Loss: {analysis['train_loss']:.4f}\n"
        summary += f"  Val Loss:   {analysis['val_loss']:.4f}\n"
        summary += f"  Gap:        {analysis['loss_gap']:.4f} ({analysis['loss_gap_pct']:.1f}%)\n"
        summary += f"  Trend:      {analysis['val_loss_trend']}\n\n"
        
        if analysis['recommendations']:
            summary += "Recommendations:\n"
            for i, rec in enumerate(analysis['recommendations'], 1):
                summary += f"  {i}. {rec}\n"
        
        summary += f"\n{'='*80}\n"
        
        return summary


class DataValidator:
    """Main validator class combining all validation checks."""
    
    @staticmethod
    def validate_dataset_split(
        train_data: List[Dict[str, Any]],
        val_data: List[Dict[str, Any]],
        text_key: str = 'text',
        label_key: str = 'label'
    ) -> Dict[str, Any]:
        """
        Comprehensive validation of train/val split.
        
        Args:
            train_data: Training dataset
            val_data: Validation dataset
            text_key: Key for text field
            label_key: Key for label field
            
        Returns:
            Dict with all validation results
        """
        LOG.info("="*80)
        LOG.info("DATA VALIDATION REPORT")
        LOG.info("="*80)
        
        results: Dict[str, Any] = {
            'validation_passed': True,
            'errors': [],
            'warnings': []
        }
        
        # Extract texts and labels
        train_texts = [item[text_key] for item in train_data if text_key in item]
        val_texts = [item[text_key] for item in val_data if text_key in item]

        # Advanced preprocessing adds Instruction/Response templating.
        # Strip potential dynamic prefixes to reduce false positives in leakage (keep core body after last Response marker).
        def _normalize_for_leakage(t: str) -> str:
            if '### Response:' in t:
                # keep prompt+response but collapse whitespace
                return ' '.join(t.split())
            return ' '.join(t.split())
        train_texts_norm = list(map(_normalize_for_leakage, train_texts))
        val_texts_norm = list(map(_normalize_for_leakage, val_texts))
        train_labels = [item[label_key] for item in train_data if label_key in item]
        val_labels = [item[label_key] for item in val_data if label_key in item]
        
        # 1. Check for data leakage
        leakage_results = DataLeakageDetector.check_overlap(train_texts_norm, val_texts_norm)
        results['leakage'] = leakage_results
        
        if leakage_results['has_exact_leakage']:
            results['validation_passed'] = False
            results['errors'].append(  # type: ignore[union-attr,attr-defined]
                f"DATA LEAKAGE: {leakage_results['exact_overlap_count']} samples overlap!"
            )
        
        # 2. Check class balance
        train_balance = DataLeakageDetector.check_class_balance(train_labels, "Training set")
        val_balance = DataLeakageDetector.check_class_balance(val_labels, "Validation set")
        
        results['train_balance'] = train_balance
        results['val_balance'] = val_balance
        
        if train_balance['is_severely_imbalanced']:
            results['warnings'].append(  # type: ignore[union-attr,attr-defined]
                f"Severe class imbalance in training set (ratio: {train_balance['imbalance_ratio']:.2f}:1)"
            )
        
        # 3. Check dataset sizes
        if len(val_data) < 50:
            results['warnings'].append(  # type: ignore[union-attr,attr-defined]
                f"Validation set very small ({len(val_data)} samples) - metrics may be unreliable"
            )
        
        if len(train_data) < 100:
            results['warnings'].append(  # type: ignore[union-attr,attr-defined]
                f"Training set very small ({len(train_data)} samples) - model may underfit"
            )
        
        # Log summary
        LOG.info("\nValidation Summary:")
        LOG.info(f"  Train: {len(train_data)} samples, {train_balance['num_classes']} classes")
        LOG.info(f"  Val:   {len(val_data)} samples, {val_balance['num_classes']} classes")
        LOG.info(f"  Leakage: {'[FAIL] DETECTED' if leakage_results['has_exact_leakage'] else ' None'}")
        LOG.info(f"  Status: {' PASSED' if results['validation_passed'] else '[FAIL] FAILED'}")
        
        if results['errors']:
            LOG.error("\nErrors:")
            for error in results['errors']:
                LOG.error(f"  - {error}")
        
        if results['warnings']:
            LOG.warning("\nWarnings:")
            for warning in results['warnings']:
                LOG.warning(f"  - {warning}")
        
        LOG.info("="*80)
        
        return results

    # ------------------ PREPROCESSING VALIDATION ------------------
    @staticmethod
    def validate_preprocessing_sample(sample: Dict[str, Any]) -> List[str]:
        """Validate a single preprocessed sample, returning list of errors."""
        errors = []
        required = ["input_ids", "attention_mask", "labels"]
        for key in required:
            if key not in sample:
                errors.append(f"Missing key: {key}")
        # Empty input_ids check
        ids = sample.get("input_ids")
        if ids is not None:
            try:
                length = ids.size(0) if hasattr(ids, 'size') else len(ids)
                if length == 0:
                    errors.append("Empty input_ids")
            except Exception:
                errors.append("input_ids length check failed")
        label = sample.get("labels")
        if label is None:
            errors.append("Missing labels")
        return errors

    @staticmethod
    def validate_preprocessing_batch(batch: List[Dict[str, Any]], max_errors: int = 5) -> Dict[str, Any]:
        """Validate a batch of samples quickly."""
        error_samples = 0
        total = len(batch)
        collected_errors = []
        for i, sample in enumerate(batch[:50]):  # limit
            errs = DataValidator.validate_preprocessing_sample(sample)
            if errs:
                error_samples += 1
                collected_errors.append({"index": i, "errors": errs})
                if error_samples >= max_errors:
                    break
        return {
            "total_checked": min(50, total),
            "error_samples": error_samples,
            "details": collected_errors
        }

    @staticmethod
    def assert_preprocessing_ok(train_dataset, val_dataset) -> None:
        """Run lightweight validation on first few samples of train/val datasets."""
        train_batch = [train_dataset[i] for i in range(min(20, len(train_dataset)))]
        val_batch = [val_dataset[i] for i in range(min(20, len(val_dataset)))]
        train_report = DataValidator.validate_preprocessing_batch(train_batch)
        val_report = DataValidator.validate_preprocessing_batch(val_batch)
        if train_report["error_samples"] > 0:
            raise RuntimeError(f"Preprocessing errors in training data: {train_report}")
        if val_report["error_samples"] > 0:
            raise RuntimeError(f"Preprocessing errors in validation data: {val_report}")
        LOG.info(f"Preprocessing validation passed: train_checked={train_report['total_checked']} val_checked={val_report['total_checked']}")
    
    @staticmethod
    def save_validation_report(
        results: Dict[str, Any],
        output_path: Path
    ) -> None:
        """Save validation report to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        LOG.info(f"Validation report saved to {output_path}")
