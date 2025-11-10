from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch

import evaluation.metrics

try:  # Optional explainability dependencies
    from core.explainability.lime_explainer import LimeTextExplainerWrapper
except Exception:  # pragma: no cover - optional dependency may be missing
    LimeTextExplainerWrapper = None  # type: ignore

try:
    from core.explainability.shap_explainer import SHAPExplainer
except Exception:  # pragma: no cover - optional dependency may be missing
    SHAPExplainer = None  # type: ignore


LOG = logging.getLogger(__name__)

class Evaluator:
    def __init__(
        self,
        model,
        dataloader,
        tokenizer,
        device,
        loss_fn=None,
        task_type: str = "classification",
        explainer=None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        update_frequency: int = 10,
        explainability_config: Optional[Mapping[str, Any]] = None,
        enable_runtime_profiling: bool = True,
        store_batch_text: bool = True,
    ):
        """
        task_type: 'classification', 'multi_label', or 'regression'
        progress_callback: Optional callback for real-time progress updates (WebSocket streaming)
        update_frequency: Update progress every N batches (default: 10)
        """
        self.model = model
        self.dataloader = dataloader
        self.tokenizer = tokenizer
        self.device = device
        self.loss_fn = loss_fn
        self.task_type = task_type
        self.explainer = explainer

        # Real-time progress streaming for UI
        self.progress_callback = progress_callback
        self.update_frequency = update_frequency
        self.explainability_config = dict(explainability_config or {})
        self.enable_runtime_profiling = enable_runtime_profiling
        self.store_batch_text = store_batch_text

        self._lime_explainer: Optional[Any] = None
        self._shap_explainer: Optional[Any] = None

    @staticmethod
    def _normalise_text(sample: Any) -> Optional[str]:
        if sample is None:
            return None
        if isinstance(sample, str):
            return sample
        if isinstance(sample, Mapping):
            for key in ("text", "content", "sentence", "prompt", "input"):
                if key in sample:
                    candidate = Evaluator._normalise_text(sample[key])
                    if candidate:
                        return candidate
            try:
                joined = " ".join(str(value) for value in sample.values() if value is not None)
                if joined:
                    return joined
            except Exception:
                pass
            return str(sample)
        if isinstance(sample, Sequence) and not isinstance(sample, (bytes, bytearray)):
            parts = [Evaluator._normalise_text(item) for item in sample]
            filtered = [part for part in parts if part]
            if filtered:
                return " ".join(filtered)
            return None
        return str(sample)

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        total_samples = 0
        all_preds: List[Any] = []
        all_labels: List[Any] = []
        batch_losses: List[float] = []
        all_probabilities: List[np.ndarray] = []
        confidence_samples: List[float] = []
        latency_ms: List[float] = []
        all_texts: List[str] = []

        device_str = str(self.device)
        if isinstance(self.device, torch.device):
            device_str = self.device.type if self.device.index is None else f"{self.device.type}:{self.device.index}"
        peak_memory_bytes: List[int] = []
        if self.enable_runtime_profiling and torch.cuda.is_available() and "cuda" in device_str:
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                LOG.debug("Could not reset CUDA peak memory stats", exc_info=True)
        
        # Calculate total batches for progress tracking
        total_batches = len(self.dataloader) if hasattr(self.dataloader, '__len__') else None
        batch_count = 0

        with torch.no_grad():
            for batch in self.dataloader:
                raw_input_ids = None
                raw_attention_mask = None

                # Handle batch being dict or tuple/list
                if isinstance(batch, dict):
                    raw_input_ids = batch['input_ids']
                    input_ids = raw_input_ids.to(self.device)
                    raw_attention_mask = batch.get('attention_mask', None)
                    attention_mask = raw_attention_mask.to(self.device) if raw_attention_mask is not None else None
                    labels = batch['labels'].to(self.device)
                elif isinstance(batch, (tuple, list)):
                    input_ids, labels = batch[:2]
                    raw_input_ids = input_ids
                    input_ids = input_ids.to(self.device)
                    labels = labels.to(self.device)
                    if len(batch) > 2 and batch[2] is not None:
                        raw_attention_mask = batch[2]
                        attention_mask = raw_attention_mask.to(self.device)
                    else:
                        attention_mask = None
                else:
                    raise ValueError("Unsupported batch format")

                forward_start = time.perf_counter()
                try:
                    outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                except RuntimeError as e:
                    # For PTQ or quantized models on MPS/CPU, fallback to CPU evaluation
                    device_lower = str(self.device).lower()
                    if 'mps' in device_lower or 'cpu' in device_lower:
                        cpu_input = input_ids.cpu()
                        cpu_attention = attention_mask.cpu() if attention_mask is not None else None
                        outputs = self.model(input_ids=cpu_input, attention_mask=cpu_attention)
                        input_ids = cpu_input
                        attention_mask = cpu_attention
                    else:
                        raise e
                batch_latency = (time.perf_counter() - forward_start) * 1000.0
                if self.enable_runtime_profiling:
                    latency_ms.append(batch_latency)
                    if torch.cuda.is_available() and "cuda" in device_str:
                        try:
                            peak_memory_bytes.append(torch.cuda.max_memory_allocated())
                        except Exception:
                            LOG.debug("Could not read CUDA memory stats", exc_info=True)

                logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

                if self.loss_fn is not None:
                    if self.task_type == 'regression':
                        loss = self.loss_fn(logits.squeeze(), labels.float())
                    else:
                        loss = self.loss_fn(logits, labels)
                    total_loss += loss.item() * input_ids.size(0)
                    batch_losses.append(loss.item())
                total_samples += input_ids.size(0)

                # Predictions
                if self.task_type == 'classification':
                    preds = torch.argmax(logits, dim=-1)
                elif self.task_type == 'multi_label':
                    preds = (torch.sigmoid(logits) > 0.5).int()
                else:  # regression
                    preds = logits

                prob_tensor = None
                if self.task_type == 'classification':
                    prob_tensor = torch.softmax(logits, dim=-1)
                elif self.task_type == 'multi_label':
                    prob_tensor = torch.sigmoid(logits)

                if prob_tensor is not None:
                    prob_cpu = prob_tensor.detach().cpu().numpy()
                    if self.task_type == 'classification':
                        all_probabilities.append(prob_cpu)
                    conf_tensor = (
                        prob_tensor.max(dim=-1)[0]
                        if self.task_type == 'classification'
                        else prob_tensor.mean(dim=-1)
                    )
                    confidence_samples.extend(conf_tensor.detach().cpu().tolist())

                if self.store_batch_text and raw_input_ids is not None and hasattr(self.tokenizer, "batch_decode"):
                    try:
                        decode_source = raw_input_ids.detach().cpu() if hasattr(raw_input_ids, "detach") else raw_input_ids
                        decoded_batch = self.tokenizer.batch_decode(decode_source, skip_special_tokens=True)
                        all_texts.extend(decoded_batch)
                    except Exception:
                        LOG.debug("Failed to decode batch for evaluation explainability", exc_info=True)

                # Safely convert preds and labels to list
                if isinstance(preds, torch.Tensor):
                    preds = preds.detach().cpu().tolist()
                elif isinstance(preds, np.ndarray):
                    preds = preds.tolist()
                if isinstance(labels, torch.Tensor):
                    labels = labels.detach().cpu().tolist()
                elif isinstance(labels, np.ndarray):
                    labels = labels.tolist()

                if isinstance(preds, list):
                    all_preds.extend(preds)
                else:
                    all_preds.append(preds)
                if isinstance(labels, list):
                    all_labels.extend(labels)
                else:
                    all_labels.append(labels)
                
                batch_count += 1
                
                # ========== REAL-TIME PROGRESS STREAMING ==========
                # Send progress updates to UI via WebSocket
                if self.progress_callback and batch_count % self.update_frequency == 0:
                    # Calculate running accuracy for classification tasks
                    running_accuracy = None
                    if self.task_type in ['classification', 'multi_label'] and len(all_preds) > 0 and len(all_labels) > 0:
                        try:
                            correct = sum(p == l for p, l in zip(all_preds, all_labels))
                            running_accuracy = correct / len(all_preds)
                        except Exception:
                            running_accuracy = None

                    running_confidence = float(np.mean(confidence_samples)) if confidence_samples else None
                    rolling_loss = (total_loss / total_samples) if total_samples > 0 and self.loss_fn else None
                    avg_latency = float(np.mean(latency_ms)) if latency_ms else None

                    progress_payload = {
                        'type': 'evaluation_progress',
                        'stage': 'evaluation',
                        'batch': batch_count,
                        'total_batches': total_batches,
                        'progress': (batch_count / total_batches * 100) if total_batches else None,
                        'samples_processed': total_samples,
                        'current_accuracy': running_accuracy,
                        'current_loss': rolling_loss,
                        'mean_confidence': running_confidence,
                        'latest_latency_ms': latency_ms[-1] if latency_ms else None,
                        'avg_latency_ms': avg_latency,
                    }

                    try:
                        self.progress_callback(progress_payload)
                    except Exception as e:
                        LOG.warning("Progress callback failed: %s", e)
                # ================================================

        avg_loss = (total_loss / total_samples) if self.loss_fn is not None and total_samples > 0 else None

        pred_probs_np: Optional[np.ndarray] = None
        if self.task_type == 'classification' and all_probabilities:
            try:
                pred_probs_np = np.concatenate(all_probabilities, axis=0)
            except ValueError:
                LOG.warning("Could not concatenate probability tensors; skipping probability-based metrics")

        metrics: Dict[str, Any] = {}
        if self.task_type in ['classification', 'multi_label']:
            metrics = evaluation.metrics.compute_all_metrics(
                all_preds,
                all_labels,
                pred_probs=pred_probs_np,
            )

        result: Dict[str, Any] = {
            'loss': avg_loss,
            'preds': all_preds,
            'labels': all_labels,
            'all_preds': all_preds,
            'all_labels': all_labels,
            'num_samples': total_samples,
            'metrics': metrics,
        }

        if self.task_type in ['classification', 'multi_label']:
            result.update({
                'accuracy': metrics.get('accuracy'),
                'f1': metrics.get('f1'),
                'precision': metrics.get('precision'),
                'recall': metrics.get('recall'),
            })

        if pred_probs_np is not None:
            result['probabilities'] = pred_probs_np.tolist()

        runtime_stats = self._summarize_runtime(latency_ms, peak_memory_bytes, total_samples, batch_count)
        if runtime_stats:
            result['runtime'] = runtime_stats

        diagnostics = self._build_diagnostics(metrics, avg_loss, batch_losses, confidence_samples, pred_probs_np)
        if diagnostics:
            result['diagnostics'] = diagnostics

        explainability = self._generate_explainability(all_texts, all_preds, all_labels, pred_probs_np)
        if explainability:
            result['explainability'] = explainability

        if self.explainer is not None:
            self.explainer.visualize(all_preds, all_labels)
        return result

    def _summarize_runtime(
        self,
        latencies: Sequence[float],
        memory_bytes: Sequence[int],
        total_samples: int,
        total_batches: int,
    ) -> Optional[Dict[str, Any]]:
        if not latencies:
            return None
        lat_array = np.asarray(latencies, dtype=np.float64)
        runtime: Dict[str, Any] = {
            'mean_ms': float(lat_array.mean()),
            'median_ms': float(np.median(lat_array)),
            'p95_ms': float(np.percentile(lat_array, 95)),
            'p99_ms': float(np.percentile(lat_array, 99)),
            'max_ms': float(lat_array.max()),
            'min_ms': float(lat_array.min()),
            'batches': len(latencies),
        }
        total_time = lat_array.sum() / 1000.0
        if total_samples > 0 and total_time > 0:
            runtime['throughput_samples_per_s'] = float(total_samples / total_time)
        runtime['batches_completed'] = total_batches

        if memory_bytes:
            mem_array = np.asarray(memory_bytes, dtype=np.float64)
            runtime['peak_memory_mb'] = float(mem_array.max() / (1024 ** 2))
        return runtime

    def _build_diagnostics(
        self,
        metrics: Mapping[str, Any],
        avg_loss: Optional[float],
        batch_losses: Sequence[float],
        confidence_samples: Sequence[float],
        probabilities: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        diagnostics: Dict[str, Any] = {}
        warnings: List[str] = []

        if metrics:
            core_metrics = [metrics.get('accuracy'), metrics.get('precision'), metrics.get('recall'), metrics.get('f1')]
            numeric_vals = [float(val) for val in core_metrics if isinstance(val, (int, float, np.floating))]
            if numeric_vals:
                spread = max(numeric_vals) - min(numeric_vals)
                diagnostics['core_metric_spread'] = float(spread)
                if spread < 0.01:
                    warnings.append('metrics_overlap')

        if avg_loss is not None:
            diagnostics['eval_loss'] = float(avg_loss)

        if batch_losses:
            losses = np.asarray(batch_losses, dtype=np.float64)
            diagnostics['loss_std'] = float(losses.std())
            diagnostics['loss_min'] = float(losses.min())
            diagnostics['loss_max'] = float(losses.max())

        if confidence_samples:
            conf_arr = np.asarray(confidence_samples, dtype=np.float64)
            diagnostics['confidence_mean'] = float(conf_arr.mean())
            diagnostics['confidence_std'] = float(conf_arr.std())
            diagnostics['confidence_min'] = float(conf_arr.min())
            diagnostics['confidence_max'] = float(conf_arr.max())
            if conf_arr.mean() > 0.95 and conf_arr.std() < 0.02:
                warnings.append('confidence_collapse')

        if probabilities is not None and probabilities.ndim >= 1:
            try:
                probs = probabilities
                if probabilities.ndim == 1:
                    probs = probabilities.reshape(-1, 1)
                entropy = -np.sum(probs * np.log(np.clip(probs, 1e-9, 1.0)), axis=1)
                diagnostics['prediction_entropy_mean'] = float(np.mean(entropy))
                diagnostics['prediction_entropy_std'] = float(np.std(entropy))
                if np.mean(entropy) < 0.2:
                    warnings.append('low_entropy_predictions')
            except Exception:
                LOG.debug("Failed to compute prediction entropy", exc_info=True)

        diagnostics['warnings'] = warnings
        return diagnostics

    def _generate_explainability(
        self,
        texts: Sequence[str],
        preds: Sequence[Any],
        labels: Sequence[Any],
        probabilities: Optional[np.ndarray],
    ) -> Dict[str, Any]:
        if not self.explainability_config:
            return {}
        if self.task_type != 'classification':
            return {}
        if not texts:
            return {}

        methods_cfg = self.explainability_config.get('methods', ['lime', 'shap'])
        if isinstance(methods_cfg, str):
            methods = [methods_cfg.lower()]
        else:
            methods = [str(method).lower() for method in methods_cfg]
        max_samples = int(self.explainability_config.get('num_samples', 4))
        if max_samples <= 0:
            return {}

        num_items = min(len(texts), len(preds), len(labels))
        if num_items == 0:
            return {}

        confidence_scores: Optional[np.ndarray] = None
        if probabilities is not None:
            try:
                if probabilities.ndim == 2 and probabilities.shape[0] >= num_items:
                    confidence_scores = probabilities[:num_items].max(axis=1)
                elif probabilities.ndim == 1 and probabilities.shape[0] >= num_items:
                    confidence_scores = probabilities[:num_items]
            except Exception:
                LOG.debug("Failed to compute confidence scores for explainability", exc_info=True)

        sample_indices = self._select_explainability_indices(preds, labels, num_items, confidence_scores, max_samples)
        if not sample_indices:
            return {}

        class_names = self.explainability_config.get('class_names') or self.explainability_config.get('label_names')
        if isinstance(class_names, Mapping):
            class_names = list(class_names.values())
        elif isinstance(class_names, (list, tuple)):
            class_names = list(class_names)
        elif class_names is not None:
            class_names = [str(class_names)]
        if class_names is None and probabilities is not None and probabilities.ndim == 2:
            class_names = [f'class_{i}' for i in range(probabilities.shape[1])]

        samples_payload: List[Dict[str, Any]] = []
        for idx in sample_indices:
            if idx >= len(texts):
                continue
            sample_confidence = None
            if confidence_scores is not None and idx < confidence_scores.shape[0]:
                sample_confidence = float(confidence_scores[idx])
            prediction = preds[idx]
            try:
                prediction_int = int(prediction)
            except Exception:
                prediction_int = None
            text_value = self._normalise_text(texts[idx])
            if not text_value:
                LOG.debug("Skipping explainability sample %s due to missing text", idx)
                continue
            samples_payload.append(
                {
                    'index': idx,
                    'text': text_value,
                    'label': labels[idx] if idx < len(labels) else None,
                    'prediction': prediction,
                    'prediction_index': prediction_int,
                    'confidence': sample_confidence,
                }
            )

        explainability: Dict[str, Any] = {'samples': samples_payload}

        if 'lime' in methods and LimeTextExplainerWrapper is not None:
            lime_results: List[Dict[str, Any]] = []
            if self._lime_explainer is None and class_names is not None:
                try:
                    self._lime_explainer = LimeTextExplainerWrapper(self.model, self.tokenizer, list(class_names))
                except Exception as exc:
                    LOG.warning("Failed to initialise LIME explainer: %s", exc)
            if self._lime_explainer is not None:
                num_features = int(self.explainability_config.get('lime_num_features', 8))
                for sample in samples_payload:
                    label_index = sample.get('prediction_index')
                    try:
                        explanation = self._lime_explainer.explain(sample['text'], num_features=num_features)
                        top_features = self._lime_explainer.visualize(explanation, num_features=num_features, label=label_index)
                        lime_results.append(
                            {
                                'index': sample['index'],
                                'prediction': sample['prediction'],
                                'top_features': top_features,
                            }
                        )
                    except Exception:
                        LOG.debug("LIME explanation failed for index %s", sample['index'], exc_info=True)
            if lime_results:
                explainability['lime'] = lime_results

        if 'shap' in methods and SHAPExplainer is not None:
            shap_results: List[Dict[str, Any]] = []
            if self._shap_explainer is None:
                try:
                    self._shap_explainer = SHAPExplainer(self.model, self.tokenizer, device=str(self.device))
                except Exception as exc:
                    LOG.warning("Failed to initialise SHAP explainer: %s", exc)
            if self._shap_explainer is not None:
                sample_texts = [sample['text'] for sample in samples_payload]
                if sample_texts:
                    try:
                        shap_values = self._shap_explainer.explain(sample_texts)
                        shap_summary = self._shap_explainer.summarize(
                            shap_values,
                            top_k=int(self.explainability_config.get('shap_top_k', 10)),
                        )
                        for sample, contribs in zip(samples_payload, shap_summary):
                            shap_results.append(
                                {
                                    'index': sample['index'],
                                    'prediction': sample['prediction'],
                                    'top_features': contribs,
                                }
                            )
                    except Exception:
                        LOG.debug("SHAP explanation failed", exc_info=True)
            if shap_results:
                explainability['shap'] = shap_results

        if len(explainability) == 1 and not explainability.get('samples'):
            return {}
        return explainability

    def _select_explainability_indices(
        self,
        preds: Sequence[Any],
        labels: Sequence[Any],
        num_items: int,
        confidence_scores: Optional[np.ndarray],
        max_samples: int,
    ) -> List[int]:
        indices: List[int] = []
        misclassified = [idx for idx in range(num_items) if preds[idx] != labels[idx]]
        indices.extend(misclassified[:max_samples])

        if confidence_scores is not None and len(indices) < max_samples:
            uncertain_sorted = list(np.argsort(confidence_scores))
            for idx in uncertain_sorted:
                if idx not in indices:
                    indices.append(int(idx))
                if len(indices) >= max_samples:
                    break

        if len(indices) < max_samples:
            for idx in range(num_items):
                if idx not in indices:
                    indices.append(idx)
                if len(indices) >= max_samples:
                    break

        return indices[:max_samples]

    def run_all(self):
        """
        Run comprehensive evaluation and return metrics.
        This method provides compatibility with the expected interface.
        """
        return self.evaluate()