"""
Data Inspector - Dataset Schema Validation & Task Detection
============================================================

Validates dataset format and detects task type:
1. Schema validation (text/image/audio/video/multimodal)
2. Task type detection (classification/regression/generation/QA)
3. Data distribution analysis
4. Batch size recommendations
5. Preprocessing requirements
6. Data augmentation suggestions

Supports:
- Text: Classification, NER, QA, Generation, Translation
- Vision: Classification, Detection, Segmentation, Captioning
- Audio: Classification, ASR, Speaker Recognition
- Multimodal: VQA, Image Captioning, Video QA
"""

from typing import Dict, List, Tuple, Optional, Any, Union
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from collections import Counter, defaultdict
import warnings


class DataInspector:
    """
    Comprehensive dataset inspector and validator.
    
    Analyzes datasets to extract:
    - Data type and modality
    - Task type
    - Distribution statistics
    - Format validation
    - Batch size recommendations
    """
    
    def __init__(self, dataset: Optional[Dataset] = None, config: Optional[Dict] = None):
        """
        Initialize data inspector.
        
        Args:
            dataset: PyTorch Dataset object
            config: Configuration dictionary with dataset info
        """
        self.dataset = dataset
        self.config = config or {}
        self.inspection_results = {}
    
    def validate(self) -> Dict[str, Any]:
        """
        Run full validation and analysis.
        
        Returns:
            Comprehensive data inspection report
        """
        report = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'recommendations': []
        }
        
        if self.dataset is not None:
            # Analyze dataset
            report['dataset_info'] = self._analyze_dataset()
            report['data_type'] = self._detect_data_type()
            report['task_type'] = self._detect_task_type()
            report['statistics'] = self._compute_statistics()
            report['batch_recommendations'] = self._recommend_batch_size()
            
            # Validate schema
            schema_valid, schema_errors = self._validate_schema()
            if not schema_valid:
                report['is_valid'] = False
                report['errors'].extend(schema_errors)
            
            # Check class imbalance
            imbalance_warnings = self._check_class_imbalance()
            report['warnings'].extend(imbalance_warnings)
            
            # Suggest preprocessing
            report['preprocessing'] = self._suggest_preprocessing()
            
        elif self.config:
            # Validate from config only
            report['config_info'] = self._analyze_config()
            report['data_type'] = self.config.get('data_type', 'unknown')
            report['task_type'] = self.config.get('task_type', 'unknown')
        
        else:
            report['is_valid'] = False
            report['errors'].append("No dataset or config provided")
        
        return report
    
    def _analyze_dataset(self) -> Dict[str, Any]:
        """
        Analyze dataset basic properties.
        
        Returns:
            Dictionary with dataset info
        """
        info = {
            'size': len(self.dataset) if self.dataset is not None and hasattr(self.dataset, '__len__') else 'unknown',  # type: ignore[arg-type]
            'type': type(self.dataset).__name__,
            'sample_structure': None
        }
        
        # Inspect first sample
        try:
            if self.dataset is not None and hasattr(self.dataset, '__len__') and len(self.dataset) > 0:  # type: ignore[arg-type]
                sample = self.dataset[0]
                info['sample_structure'] = self._analyze_sample_structure(sample)
        except Exception as e:
            info['sample_error'] = str(e)
        
        return info
    
    def _analyze_sample_structure(self, sample: Any) -> Dict[str, Any]:
        """
        Analyze structure of a single sample.
        
        Args:
            sample: Dataset sample
            
        Returns:
            Structure description
        """
        structure = {
            'type': type(sample).__name__,
            'fields': {}
        }
        
        # Handle different sample formats
        if isinstance(sample, dict):
            # Dictionary format (common in HuggingFace datasets)
            for key, value in sample.items():
                structure['fields'][key] = {
                    'type': type(value).__name__,
                    'shape': self._get_shape(value),
                    'dtype': self._get_dtype(value)
                }
        
        elif isinstance(sample, (tuple, list)):
            # Tuple/List format (common in PyTorch datasets)
            for i, item in enumerate(sample):
                structure['fields'][f'item_{i}'] = {
                    'type': type(item).__name__,
                    'shape': self._get_shape(item),
                    'dtype': self._get_dtype(item)
                }
        
        elif isinstance(sample, torch.Tensor):
            # Single tensor
            structure['fields']['data'] = {
                'type': 'Tensor',
                'shape': list(sample.shape),
                'dtype': str(sample.dtype)
            }
        
        return structure
    
    def _get_shape(self, data: Any) -> Optional[List[int]]:
        """Get shape of data if applicable."""
        if isinstance(data, torch.Tensor):
            return list(data.shape)
        elif isinstance(data, np.ndarray):
            return list(data.shape)
        elif isinstance(data, (list, tuple)):
            return [len(data)]
        elif isinstance(data, str):
            return [len(data)]  # String length
        return None
    
    def _get_dtype(self, data: Any) -> Optional[str]:
        """Get data type."""
        if isinstance(data, torch.Tensor):
            return str(data.dtype)
        elif isinstance(data, np.ndarray):
            return str(data.dtype)
        elif isinstance(data, (int, float, bool, str)):
            return type(data).__name__
        return None
    
    def _detect_data_type(self) -> str:
        """
        Detect data modality.
        
        Returns:
            Data type: text, vision, audio, video, multimodal, unknown
        """
        if self.dataset is None:
            return self.config.get('data_type', 'unknown')
        
        try:
            sample = self.dataset[0]
            
            # Check for common patterns
            if isinstance(sample, dict):
                keys = set(sample.keys())
                
                # Text indicators
                if any(k in keys for k in ['input_ids', 'attention_mask', 'text', 'sentence']):
                    return 'text'
                
                # Vision indicators
                if any(k in keys for k in ['pixel_values', 'image', 'img']):
                    # Check if also has text (multimodal)
                    if any(k in keys for k in ['input_ids', 'text', 'caption']):
                        return 'multimodal'
                    return 'vision'
                
                # Audio indicators
                if any(k in keys for k in ['audio', 'waveform', 'spectrogram']):
                    return 'audio'
                
                # Video indicators
                if any(k in keys for k in ['video', 'frames']):
                    return 'video'
            
            elif isinstance(sample, (tuple, list)):
                # Check first element
                first_item = sample[0]
                
                if isinstance(first_item, torch.Tensor):
                    shape = first_item.shape
                    
                    # 1D: likely audio or sequence
                    if len(shape) == 1 or (len(shape) == 2 and shape[0] == 1):
                        return 'audio' if shape[-1] > 1000 else 'text'
                    
                    # 2D: likely text embeddings or grayscale image
                    elif len(shape) == 2:
                        if shape[0] > 50 and shape[1] > 50:
                            return 'vision'
                        return 'text'
                    
                    # 3D: likely image (C, H, W) or sequence (B, L, D)
                    elif len(shape) == 3:
                        if shape[0] <= 4:  # Likely channels
                            return 'vision'
                        return 'text'
                    
                    # 4D: likely video (T, C, H, W) or batch of images
                    elif len(shape) == 4:
                        return 'video'
        
        except Exception as e:
            warnings.warn(f"Error detecting data type: {e}")
        
        return 'unknown'
    
    def _detect_task_type(self) -> str:
        """
        Detect task type.
        
        Returns:
            Task type: classification, regression, generation, qa, detection, etc.
        """
        if self.dataset is None:
            return self.config.get('task_type', 'unknown')
        
        try:
            # Sample multiple items to infer task
            dataset_len = len(self.dataset) if hasattr(self.dataset, '__len__') else 10  # type: ignore[arg-type]
            samples = [self.dataset[i] for i in range(min(10, dataset_len))]
            
            # Check labels/targets
            if isinstance(samples[0], dict):
                keys = samples[0].keys()
                
                # Classification indicators
                if 'labels' in keys or 'label' in keys:
                    label_key = 'labels' if 'labels' in keys else 'label'
                    labels = [s[label_key] for s in samples]
                    
                    # Check if labels are categorical
                    if all(isinstance(l, (int, np.integer)) or 
                          (isinstance(l, torch.Tensor) and l.numel() == 1) for l in labels):
                        unique_labels = set()
                        for l in labels:
                            if isinstance(l, torch.Tensor):
                                unique_labels.add(l.item())
                            else:
                                unique_labels.add(l)
                        
                        if len(unique_labels) < dataset_len * 0.5:  # Likely classification
                            return 'classification'
                        else:  # Too many unique values, likely regression or generation
                            return 'regression'
                
                # Generation indicators
                if any(k in keys for k in ['decoder_input_ids', 'target_ids']):
                    return 'generation'
                
                # QA indicators
                if any(k in keys for k in ['question', 'answer', 'context']):
                    return 'question_answering'
                
                # Detection indicators
                if any(k in keys for k in ['boxes', 'bboxes', 'annotations']):
                    return 'object_detection'
                
                # Segmentation indicators
                if any(k in keys for k in ['mask', 'segmentation']):
                    return 'segmentation'
            
            elif isinstance(samples[0], (tuple, list)) and len(samples[0]) >= 2:
                # Check second element (likely label)
                labels = [s[1] for s in samples]
                
                # Categorical classification
                if all(isinstance(l, (int, np.integer)) for l in labels):
                    return 'classification'
                
                # Regression
                elif all(isinstance(l, (float, np.floating)) for l in labels):
                    return 'regression'
        
        except Exception as e:
            warnings.warn(f"Error detecting task type: {e}")
        
        return 'unknown'
    
    def _compute_statistics(self) -> Dict[str, Any]:
        """
        Compute dataset statistics.
        
        Returns:
            Dictionary with statistics
        """
        if self.dataset is None:
            return {}
        
        stats = {
            'num_samples': len(self.dataset) if hasattr(self.dataset, '__len__') else 0,  # type: ignore[arg-type]
            'class_distribution': None,
            'input_statistics': {}
        }
        
        try:
            # Sample subset for statistics
            dataset_len = len(self.dataset) if hasattr(self.dataset, '__len__') else 0  # type: ignore[arg-type]
            sample_size = min(1000, dataset_len)
            indices = np.random.choice(dataset_len, sample_size, replace=False)
            
            # Collect labels if available
            labels = []
            for idx in indices:
                sample = self.dataset[int(idx)]
                
                if isinstance(sample, dict) and 'labels' in sample:
                    label = sample['labels']
                    if isinstance(label, torch.Tensor):
                        labels.append(label.item() if label.numel() == 1 else label.tolist())
                    else:
                        labels.append(label)
                
                elif isinstance(sample, (tuple, list)) and len(sample) >= 2:
                    label = sample[1]
                    if isinstance(label, torch.Tensor):
                        labels.append(label.item() if label.numel() == 1 else label.tolist())
                    else:
                        labels.append(label)
            
            # Compute class distribution
            if labels:
                # Flatten if necessary
                flat_labels = []
                for l in labels:
                    if isinstance(l, list):
                        flat_labels.extend(l)
                    else:
                        flat_labels.append(l)
                
                counter = Counter(flat_labels)
                stats['class_distribution'] = dict(counter)
                stats['num_classes'] = len(counter)
                
                # Compute imbalance ratio
                if len(counter) > 1:
                    max_count = max(counter.values())
                    min_count = min(counter.values())
                    stats['imbalance_ratio'] = max_count / min_count
        
        except Exception as e:
            warnings.warn(f"Error computing statistics: {e}")
        
        return stats
    
    def _validate_schema(self) -> Tuple[bool, List[str]]:
        """
        Validate dataset schema.
        
        Returns:
            Tuple of (is_valid, list of errors)
        """
        if self.dataset is None:
            return True, []
        
        errors = []
        
        try:
            # Check if dataset is iterable
            if not hasattr(self.dataset, '__getitem__'):
                errors.append("Dataset does not implement __getitem__")
                return False, errors
            
            # Check if dataset has length
            if not hasattr(self.dataset, '__len__'):
                errors.append("Dataset does not implement __len__")
            else:
                dataset_len = len(self.dataset)  # type: ignore[arg-type]
                
                # Try to get first sample
                if dataset_len == 0:
                    errors.append("Dataset is empty")
                    return False, errors
            
            sample = self.dataset[0]
            
            # Validate sample structure consistency
            if hasattr(self.dataset, '__len__') and len(self.dataset) > 1:  # type: ignore[arg-type]
                try:
                    sample2 = self.dataset[1]
                    
                    # Check if structure matches
                    if type(sample) != type(sample2):
                        errors.append("Inconsistent sample types across dataset")
                    
                    if isinstance(sample, dict) and isinstance(sample2, dict):
                        if set(sample.keys()) != set(sample2.keys()):
                            errors.append("Inconsistent dictionary keys across samples")
                
                except Exception as e:
                    errors.append(f"Error accessing second sample: {e}")
        
        except Exception as e:
            errors.append(f"Schema validation error: {e}")
            return False, errors
        
        return len(errors) == 0, errors
    
    def _check_class_imbalance(self) -> List[str]:
        """
        Check for class imbalance.
        
        Returns:
            List of warnings
        """
        warnings_list = []
        
        stats = self.inspection_results.get('statistics', self._compute_statistics())
        
        if 'class_distribution' in stats and stats['class_distribution']:
            imbalance_ratio = stats.get('imbalance_ratio', 1.0)
            
            if imbalance_ratio > 10:
                warnings_list.append(
                    f"Severe class imbalance detected (ratio: {imbalance_ratio:.1f}:1). "
                    f"Consider using class weights or oversampling."
                )
            elif imbalance_ratio > 3:
                warnings_list.append(
                    f"Moderate class imbalance detected (ratio: {imbalance_ratio:.1f}:1). "
                    f"May benefit from class weighting."
                )
            
            # Check for very small classes
            class_dist = stats['class_distribution']
            total_samples = sum(class_dist.values())
            
            for cls, count in class_dist.items():
                if count / total_samples < 0.01:  # Less than 1%
                    warnings_list.append(
                        f"Class {cls} has very few samples ({count}, {count/total_samples*100:.1f}%). "
                        f"May cause training issues."
                    )
        
        return warnings_list
    
    def _recommend_batch_size(self) -> Dict[str, Any]:
        """
        Recommend batch size based on dataset characteristics.
        
        Returns:
            Batch size recommendations
        """
        recommendations = {
            'min_batch_size': 8,
            'optimal_batch_size': 32,
            'max_batch_size': 128,
            'reasoning': []
        }
        
        if self.dataset is None:
            return recommendations
        
        try:
            dataset_size = len(self.dataset) if hasattr(self.dataset, '__len__') else 0  # type: ignore[arg-type]
            
            # Adjust based on dataset size
            if dataset_size < 100:
                recommendations['optimal_batch_size'] = 8
                recommendations['max_batch_size'] = 16
                recommendations['reasoning'].append("Small dataset: using smaller batches")
            
            elif dataset_size < 1000:
                recommendations['optimal_batch_size'] = 16
                recommendations['max_batch_size'] = 32
                recommendations['reasoning'].append("Medium dataset: moderate batch size")
            
            else:
                recommendations['optimal_batch_size'] = 32
                recommendations['max_batch_size'] = 64
                recommendations['reasoning'].append("Large dataset: can use larger batches")
            
            # Adjust based on data type
            data_type = self._detect_data_type()
            
            if data_type == 'vision':
                recommendations['optimal_batch_size'] = min(recommendations['optimal_batch_size'], 16)
                recommendations['reasoning'].append("Vision data: reducing batch size for memory")
            
            elif data_type == 'video':
                recommendations['optimal_batch_size'] = min(recommendations['optimal_batch_size'], 8)
                recommendations['reasoning'].append("Video data: small batches for memory constraints")
            
            elif data_type == 'multimodal':
                recommendations['optimal_batch_size'] = min(recommendations['optimal_batch_size'], 8)
                recommendations['reasoning'].append("Multimodal data: small batches recommended")
        
        except Exception as e:
            warnings.warn(f"Error recommending batch size: {e}")
        
        return recommendations
    
    def _suggest_preprocessing(self) -> Dict[str, List[str]]:
        """
        Suggest preprocessing steps based on data characteristics.
        
        Returns:
            Dictionary with preprocessing suggestions
        """
        suggestions = {
            'required': [],
            'recommended': [],
            'optional': []
        }
        
        data_type = self._detect_data_type()
        task_type = self._detect_task_type()
        
        # Text preprocessing
        if data_type == 'text':
            suggestions['required'].append("Tokenization (using teacher tokenizer)")
            suggestions['required'].append("Padding to max length or dynamic padding")
            suggestions['recommended'].append("Lowercase normalization")
            suggestions['optional'].append("Special token handling")
        
        # Vision preprocessing
        elif data_type == 'vision':
            suggestions['required'].append("Resize to model input size")
            suggestions['required'].append("Normalize with ImageNet stats or custom")
            suggestions['recommended'].append("Random horizontal flip (training)")
            suggestions['optional'].append("Color jitter for augmentation")
            suggestions['optional'].append("Random crop (training)")
        
        # Audio preprocessing
        elif data_type == 'audio':
            suggestions['required'].append("Resample to model sample rate")
            suggestions['required'].append("Convert to spectrogram or mel-spectrogram")
            suggestions['recommended'].append("Normalize amplitude")
            suggestions['optional'].append("Time stretching augmentation")
        
        # Multimodal preprocessing
        elif data_type == 'multimodal':
            suggestions['required'].append("Process both modalities separately")
            suggestions['required'].append("Align sequences if needed")
            suggestions['recommended'].append("Cross-modal augmentation")
        
        # Task-specific preprocessing
        if task_type == 'classification':
            suggestions['recommended'].append("Class weight computation for imbalance")
        
        elif task_type == 'generation':
            suggestions['required'].append("Prepare decoder inputs")
            suggestions['recommended'].append("Teacher forcing during training")
        
        return suggestions
    
    def _analyze_config(self) -> Dict[str, Any]:
        """
        Analyze configuration for dataset information.
        
        Returns:
            Configuration analysis
        """
        info = {}
        
        # Extract dataset config
        data_cfg = self.config.get('data', {})
        
        info['dataset_path'] = data_cfg.get('dataset_path', 'not specified')
        info['train_file'] = data_cfg.get('train_file', None)
        info['val_file'] = data_cfg.get('val_file', None)
        info['batch_size'] = data_cfg.get('batch_size', 'not specified')
        info['num_workers'] = data_cfg.get('num_workers', 0)
        
        return info
    
    def generate_report(self) -> str:
        """
        Generate human-readable data inspection report.
        
        Returns:
            Formatted report string
        """
        report = self.validate()
        
        lines = [
            "=" * 70,
            "DATA INSPECTION REPORT",
            "=" * 70,
            ""
        ]
        
        # Validation status
        status = "✓ Valid" if report['is_valid'] else "✗ Invalid"
        lines.append(f"Validation Status: {status}")
        lines.append("")
        
        # Dataset info
        if 'dataset_info' in report:
            info = report['dataset_info']
            lines.extend([
                "DATASET INFORMATION",
                "-" * 70,
                f"Type: {info['type']}",
                f"Size: {info['size']}",
                ""
            ])
            
            if 'sample_structure' in info and info['sample_structure']:
                lines.append("Sample Structure:")
                struct = info['sample_structure']
                for field, props in struct.get('fields', {}).items():
                    lines.append(f"  {field}:")
                    lines.append(f"    Type: {props.get('type', 'unknown')}")
                    if props.get('shape'):
                        lines.append(f"    Shape: {props['shape']}")
                    if props.get('dtype'):
                        lines.append(f"    Dtype: {props['dtype']}")
                lines.append("")
        
        # Data and task type
        lines.extend([
            f"Data Type: {report.get('data_type', 'unknown')}",
            f"Task Type: {report.get('task_type', 'unknown')}",
            ""
        ])
        
        # Statistics
        if 'statistics' in report:
            stats = report['statistics']
            lines.extend([
                "STATISTICS",
                "-" * 70,
                f"Number of Samples: {stats.get('num_samples', 'unknown')}",
            ])
            
            if 'class_distribution' in stats and stats['class_distribution']:
                lines.append(f"Number of Classes: {stats.get('num_classes', 'unknown')}")
                lines.append(f"Imbalance Ratio: {stats.get('imbalance_ratio', 1.0):.2f}:1")
                lines.append("")
                lines.append("Class Distribution:")
                for cls, count in sorted(stats['class_distribution'].items())[:10]:
                    percentage = (count / stats['num_samples']) * 100
                    lines.append(f"  Class {cls}: {count} ({percentage:.1f}%)")
                lines.append("")
        
        # Batch recommendations
        if 'batch_recommendations' in report:
            batch = report['batch_recommendations']
            lines.extend([
                "BATCH SIZE RECOMMENDATIONS",
                "-" * 70,
                f"Optimal: {batch['optimal_batch_size']}",
                f"Range: {batch['min_batch_size']}-{batch['max_batch_size']}",
                ""
            ])
            
            if batch.get('reasoning'):
                lines.append("Reasoning:")
                for reason in batch['reasoning']:
                    lines.append(f"  • {reason}")
                lines.append("")
        
        # Preprocessing
        if 'preprocessing' in report:
            prep = report['preprocessing']
            lines.extend([
                "PREPROCESSING RECOMMENDATIONS",
                "-" * 70
            ])
            
            if prep.get('required'):
                lines.append("Required:")
                for step in prep['required']:
                    lines.append(f"  ✓ {step}")
                lines.append("")
            
            if prep.get('recommended'):
                lines.append("Recommended:")
                for step in prep['recommended']:
                    lines.append(f"  → {step}")
                lines.append("")
            
            if prep.get('optional'):
                lines.append("Optional:")
                for step in prep['optional']:
                    lines.append(f"  ○ {step}")
                lines.append("")
        
        # Errors and warnings
        if report.get('errors'):
            lines.append("ERRORS:")
            for error in report['errors']:
                lines.append(f"  ✗ {error}")
            lines.append("")
        
        if report.get('warnings'):
            lines.append("WARNINGS:")
            for warning in report['warnings']:
                lines.append(f"  ⚠ {warning}")
            lines.append("")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
