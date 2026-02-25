# app/main.py
from __future__ import annotations

import importlib
import logging
from pathlib import Path
import sys
import os
import typer
import numpy as np
from typing import Any, Dict, List, Optional, Sequence

# Set default opt-outs for optional heavy vision deps when running text pipelines.
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("TRANSFORMERS_NO_PIL", "1")

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import argparse
from core.config.config_manager import ConfigManager, ConfigError

# Import model loader and summary utilities
from core.models.model_loader import load_models, model_summary

# Optional: nicer prints if you have rich installed; not required
try:
    from rich import print as rprint
except Exception:
    rprint = print

app = typer.Typer(name="zyn", help="Zynthe / Knowledge-distillation Toolkit CLI")

LOG = logging.getLogger("zyn")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

def convert_to_serializable(obj):
    """
    Convert numpy arrays and other non-JSON-serializable objects to JSON-serializable types.
    
    Args:
        obj: Object to convert (can be dict, list, numpy array, etc.)
    
    Returns:
        JSON-serializable version of the object
    """
    if isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_serializable(item) for item in obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):
        # Handle all numpy scalar types
        return obj.item()
    else:
        return obj

def parse_args():
    parser = argparse.ArgumentParser(description="Knowledge Distillation Toolkit")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config YAML file"
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Config overrides in KEY=VALUE format, e.g. train.lr=0.001"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "train", "infer"],
        help="Run mode. auto => infer when --load-model-dir is set, otherwise train"
    )
    parser.add_argument(
        "--load-model-dir",
        type=str,
        default=None,
        help="Path to saved student model directory for reuse/inference"
    )
    parser.add_argument(
        "--load-checkpoint-path",
        type=str,
        default=None,
        help="Path to training checkpoint (.pt) to resume from"
    )
    parser.add_argument(
        "--checkpoint-non-strict",
        action="store_true",
        help="Load checkpoint with strict=False (safer compatibility fallback)"
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        help="Save final model artifacts for future reuse"
    )
    parser.add_argument(
        "--no-save-model",
        action="store_false",
        dest="save_model",
        help="Disable final model artifact saving"
    )
    parser.add_argument(
        "--save-model-dir",
        type=str,
        default=None,
        help="Optional output directory for saved model artifacts"
    )
    parser.add_argument(
        "--save-checkpoint",
        action="store_true",
        help="Save trainer checkpoint after run"
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Optional checkpoint output path (default: experiment_dir/checkpoints/latest.pt)"
    )
    parser.add_argument(
        "--infer-text",
        nargs="*",
        default=[],
        help="One or more texts for inference"
    )
    parser.add_argument(
        "--infer-file",
        type=str,
        default=None,
        help="Text file path (one input per line) for inference"
    )
    parser.add_argument(
        "--infer-batch-size",
        type=int,
        default=8,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--infer-output",
        type=str,
        default=None,
        help="Optional JSON output path for inference predictions"
    )
    parser.set_defaults(save_model=True)
    return parser.parse_args()

def parse_overrides(override_list):
    """Convert list of KEY=VALUE strings to nested dictionary."""
    overrides = {}
    for override in override_list:
        if '=' not in override:
            print(f"WARNING: Invalid override format '{override}', expected KEY=VALUE")
            continue
        
        key, value = override.split('=', 1)
        
        # Try to parse value as appropriate type
        try:
            # Try int first
            if '.' not in value and value.isdigit():
                value = int(value)
            # Try float
            elif value.replace('.', '').replace('-', '').isdigit():
                value = float(value)
            # Try boolean
            elif value.lower() in ('true', 'false'):
                value = value.lower() == 'true'
        except ValueError:
            # Keep as string if parsing fails
            pass
        
        # Handle nested keys like "train.epochs" 
        keys = key.split('.')
        current = overrides
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        current[keys[-1]] = value
    
    return overrides


from pathlib import Path
from core.config.config_manager import ConfigManager

def load_config(path: str | Path, overrides: dict | None = None):
    """
    Load a YAML/JSON config using core.config.config_manager.ConfigManager.

    Updated for current ConfigManager API:
      - __init__(config_path, defaults_path=None, overrides=None, experiments_root="experiments")
      - config is resolved automatically and available via .resolved_config
    """
    cm = ConfigManager(config_path=str(path), overrides=overrides)
    return cm, cm.resolved_config


def _use_teacher_agent(config: Dict[str, Any]) -> bool:
    """Resolve whether teacher-agent based selection should be used."""
    agentic_cfg = config.get('agentic', {}) if isinstance(config, dict) else {}
    return bool(agentic_cfg.get('enable_teacher_agent', False))


def resolve_label_names(config: Dict[str, Any], dataset: Any | None = None) -> List[str]:
    """Best-effort resolution of label names for visualization artifacts."""

    def _coerce_sequence(source: Any) -> List[str]:
        if isinstance(source, dict):
            try:
                ordered = sorted(source.items(), key=lambda item: int(item[0]))
            except (TypeError, ValueError):
                ordered = list(source.items())
            return [str(value) for _, value in ordered]
        if isinstance(source, (list, tuple)):
            return [str(value) for value in source]
        if source is not None:
            return [str(source)]
        return []

    candidates: List[Any] = [
        config.get('data', {}).get('label_names'),
        config.get('model', {}).get('label_names'),
        config.get('evaluation', {}).get('label_names'),
    ]

    for candidate in candidates:
        names = _coerce_sequence(candidate)
        if names:
            return names

    if dataset is not None and hasattr(dataset, 'label_counts'):
        try:
            counts = dataset.label_counts()  # type: ignore[attr-defined]
        except Exception:
            counts = {}
        if counts:
            ordered_indices = sorted(counts.keys())
            return [f"Class {idx}" for idx in ordered_indices]

    return []


def _select_explainability_indices(
    predictions: Sequence[Any], labels: Sequence[Any], max_samples: int
) -> List[int]:
    """Prefer misclassified examples but ensure deterministic coverage."""
    limit = max(max_samples, 0)
    if limit == 0:
        return []

    misclassified = [
        idx
        for idx, (pred, label) in enumerate(zip(predictions, labels))
        if pred != label
    ]
    selected: List[int] = misclassified[:limit]

    if len(selected) < limit:
        for idx in range(min(len(predictions), len(labels))):
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= limit:
                break

    return selected[:limit]


def generate_explainability_reports(
    student_model,
    tokenizer,
    cfg_manager: ConfigManager,
    val_dataset: Any,
    eval_results: Dict[str, Any],
    output_dir: Path,
    class_labels: Sequence[str],
) -> Dict[str, str]:
    """Generate optional LIME/SHAP artifacts with safe device fallbacks."""

    explain_cfg = cfg_manager.resolved_config.get('explainability', {}) or {}
    enable_lime = bool(explain_cfg.get('enable_lime', False))
    enable_shap = bool(explain_cfg.get('enable_shap', False))
    if not (enable_lime or enable_shap):
        return {}

    if val_dataset is None or not hasattr(val_dataset, 'samples'):
        print("[EXPLAIN] Validation dataset samples unavailable; skipping explainability.")
        return {}

    predictions = eval_results.get('student', {}).get('predictions') or []
    labels = (
        eval_results.get('labels')
        or eval_results.get('true_labels')
        or []
    )
    if not predictions or not labels:
        print("[EXPLAIN] Missing predictions or labels; skipping explainability.")
        return {}

    max_samples = int(explain_cfg.get('num_samples', 4))
    sample_indexes = _select_explainability_indices(predictions, labels, max_samples)
    if not sample_indexes:
        print("[EXPLAIN] No samples selected for explainability.")
        return {}

    samples = getattr(val_dataset, 'samples', [])
    if not samples:
        print("[EXPLAIN] Validation dataset missing 'samples' attribute; skipping.")
        return {}

    def _label_name(index: Optional[int]) -> Optional[str]:
        if index is None:
            return None
        if 0 <= index < len(class_labels):
            return str(class_labels[index])
        return f"Class {index}"

    sample_records: List[Dict[str, Any]] = []
    for idx in sample_indexes:
        if idx >= len(samples):
            continue
        raw_sample = samples[idx]
        text_value = raw_sample.get('text') if isinstance(raw_sample, dict) else None
        text_value = text_value or ''

        try:
            label_idx = int(labels[idx])
        except Exception:
            label_idx = None
        try:
            pred_idx = int(predictions[idx])
        except Exception:
            pred_idx = None

        sample_records.append(
            {
                'index': idx,
                'text': text_value,
                'label': label_idx,
                'label_name': _label_name(label_idx),
                'prediction': pred_idx,
                'prediction_name': _label_name(pred_idx),
            }
        )

    if not sample_records:
        return {}

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch required for explainability
        print(f"[EXPLAIN] PyTorch unavailable ({exc}); skipping explainability.")
        return {}

    def _batch_predict(texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, getattr(student_model.config, 'num_labels', 1)))
        device = next(student_model.parameters()).device
        encoded = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = student_model(**encoded)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
            probs = torch.softmax(logits, dim=-1)
        return probs.detach().cpu().numpy()

    sample_texts = [record['text'] for record in sample_records]
    probabilities = _batch_predict(sample_texts)
    for record, prob in zip(sample_records, probabilities):
        record['probabilities'] = [float(p) for p in prob]
        record['confidence'] = float(max(prob)) if prob.size else None

    explain_dir = output_dir / "explainability"
    explain_dir.mkdir(exist_ok=True)

    artifacts: Dict[str, str] = {}

    num_labels = getattr(student_model.config, 'num_labels', None)
    if num_labels is None:
        num_labels = probabilities.shape[1] if probabilities.size else len(class_labels) or 2
    class_names = list(class_labels) if class_labels else [f"Class {i}" for i in range(num_labels)]

    try:
        requested_device = explain_cfg.get('device') or str(cfg_manager.device())
        explain_device = torch.device(requested_device)
    except Exception:
        explain_device = next(student_model.parameters()).device

    cpu_device = torch.device('cpu')
    original_device = next(student_model.parameters()).device

    def _move_student(target: torch.device) -> None:
        current_device = next(student_model.parameters()).device
        if current_device == target:
            return
        student_model.to(target)
        student_model.eval()

    def _should_retry_on_cpu(exc: Exception) -> bool:
        message = str(exc).lower()
        return ('mps' in message or 'metal' in message) and ('out of memory' in message or 'oom' in message)

    try:
        # LIME explanations -------------------------------------------------
        if enable_lime:
            from core.explainability.lime_explainer import LimeTextExplainerWrapper
            import json

            lime_num_features = int(explain_cfg.get('lime_num_features', 8))
            devices_to_try = [explain_device]
            if explain_device.type != 'cpu':
                devices_to_try.append(cpu_device)

            lime_explanations: Optional[List[List[Any]]] = None
            lime_active: Optional[LimeTextExplainerWrapper] = None

            for device_option in devices_to_try:
                lime_candidate: Optional[LimeTextExplainerWrapper] = None
                try:
                    _move_student(device_option)
                    lime_candidate = LimeTextExplainerWrapper(
                        student_model,
                        tokenizer,
                        class_names=list(class_names),
                        device=device_option,
                    )
                    lime_explanations = []
                    for record in sample_records:
                        explanation = lime_candidate.explain(record['text'], num_features=lime_num_features)
                        label_idx = record['prediction'] if record['prediction'] is not None else None
                        weights = lime_candidate.visualize(
                            explanation,
                            num_features=lime_num_features,
                            label=label_idx,
                        )
                        lime_explanations.append(weights)
                    lime_active = lime_candidate
                    break
                except RuntimeError as exc:
                    if device_option != cpu_device and _should_retry_on_cpu(exc):
                        print(f"[EXPLAIN] LIME encountered MPS issue: {exc}. Falling back to CPU.")
                        lime_explanations = None
                        if lime_candidate is not None:
                            lime_candidate.restore()
                        continue
                    print(f"[EXPLAIN] LIME generation failed on {device_option}: {exc}")
                    lime_explanations = None
                except Exception as exc:
                    print(f"[EXPLAIN] LIME generation error on {device_option}: {exc}")
                    lime_explanations = None
                finally:
                    if lime_candidate is not None and lime_candidate is not lime_active:
                        lime_candidate.restore()

            if lime_explanations and lime_active is not None:
                lime_records: List[Dict[str, Any]] = []
                for record, weights in zip(sample_records, lime_explanations):
                    enriched = dict(record)
                    enriched['lime_weights'] = weights
                    lime_records.append(enriched)

                lime_path = explain_dir / "lime_explanations.json"
                with lime_path.open('w', encoding='utf-8') as handle:
                    json.dump(lime_records, handle, indent=2, ensure_ascii=False)
                artifacts['lime'] = str(lime_path)

                lime_active.restore()

        # SHAP explanations -------------------------------------------------
        if enable_shap:
            from core.explainability.shap_explainer import SHAPExplainer
            import json

            shap_top_k = int(explain_cfg.get('shap_top_k', 10))
            background_size = int(explain_cfg.get('shap_background_size', 10))
            shap_payload: List[Dict[str, Any]] = []
            devices_to_try = [explain_device]
            if explain_device.type != 'cpu':
                devices_to_try.append(cpu_device)

            shap_values = None
            shap_active: Optional[SHAPExplainer] = None

            for device_option in devices_to_try:
                shap_candidate: Optional[SHAPExplainer] = None
                try:
                    _move_student(device_option)
                    shap_candidate = SHAPExplainer(
                        model=student_model,
                        tokenizer=tokenizer,
                        device=device_option,
                        background_size=background_size,
                    )
                    limited_texts = sample_texts[:max_samples]
                    shap_values = shap_candidate.explain(limited_texts)
                    if shap_values is not None:
                        shap_active = shap_candidate
                        break
                except RuntimeError as exc:
                    if device_option != cpu_device and _should_retry_on_cpu(exc):
                        print(f"[EXPLAIN] SHAP encountered MPS issue: {exc}. Falling back to CPU.")
                        shap_values = None
                        if shap_candidate is not None:
                            shap_candidate.restore()
                        continue
                    print(f"[EXPLAIN] SHAP generation failed on {device_option}: {exc}")
                    shap_values = None
                except Exception as exc:
                    print(f"[EXPLAIN] SHAP generation error on {device_option}: {exc}")
                    shap_values = None
                finally:
                    if shap_candidate is not None and shap_candidate is not shap_active:
                        shap_candidate.restore()

            if shap_values is not None and shap_active is not None:
                try:
                    shap_plot_path = explain_dir / "shap_summary.png"
                    shap_active.visualize(shap_values, show=False, save_path=str(shap_plot_path))
                    artifacts['shap_plot'] = str(shap_plot_path)
                except Exception as exc:
                    print(f"[EXPLAIN] Failed to render SHAP plot: {exc}")

                shap_summary = shap_active.summarize(shap_values, top_k=shap_top_k)
                if shap_summary:
                    for record, contribs in zip(sample_records, shap_summary):
                        enriched = dict(record)
                        enriched['top_features'] = contribs
                        shap_payload.append(enriched)
                    if shap_payload:
                        shap_json_path = explain_dir / "shap_contributions.json"
                        with shap_json_path.open('w', encoding='utf-8') as handle:
                            json.dump(shap_payload, handle, indent=2, ensure_ascii=False)
                        artifacts['shap'] = str(shap_json_path)

                shap_active.restore()
    finally:
        try:
            _move_student(original_device)
        except Exception as exc:
            print(f"[EXPLAIN] Failed to restore model device: {exc}")

    return artifacts


@app.command()
def distill(
    config: str = typer.Option("configs/default.yaml", help="Path to distill config"),
    dry_run: bool = typer.Option(False, "--dry-run", "-n", help="Show config and exit"),
):
    """
    Run a distillation pipeline.

    This command loads models, dataloaders, and runs MultiStageDistiller.
    """
    cm, cfg = load_config(config)
    rprint(f"[bold green]Loaded distill config:[/bold green] {config}")

    if dry_run:
        rprint("[yellow]Dry-run: printing parsed config and exiting[/yellow]")
        rprint(cfg)
        raise typer.Exit()

    try:
        # Enable HuggingFace download progress tracking
        import os
        os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'  # Ensure standard download for progress tracking
        
        # Install download progress hooks
        from core.utils.download_monitor import install_progress_hooks
        install_progress_hooks()
        
        # Load models with progress tracking
        rprint("[bold blue]📥 Downloading/Loading Models...[/bold blue]")
        print("[PROGRESS] stage=downloading_teacher progress=0.0 message=Starting model downloads")
        
        teacher, student, tokenizer = load_models(cm, cm.device(), use_agent=_use_teacher_agent(cfg))
        
        print("[PROGRESS] stage=downloading_student progress=1.0 message=All models loaded successfully")
        rprint(f"[green]✓ Teacher loaded: {model_summary(teacher)['name']}[/green]")
        rprint(f"[green]✓ Student loaded: {model_summary(student)['name']}[/green]")
        
        # Load dataloaders
        rprint("[bold blue]📚 Loading dataloaders...[/bold blue]")
        print("[PROGRESS] stage=loading_data progress=0.0 message=Preparing training data")
        from data.dataloaders import create_dataloaders
        train_loader, val_loader = create_dataloaders(cfg, tokenizer)
        print("[PROGRESS] stage=loading_data progress=1.0 message=Dataloaders created successfully")
        rprint(f"[green]✓ Dataloaders created[/green]")
        
        # Create multi-stage distiller
        rprint("[bold blue]⚙️  Initializing MultiStageDistiller...[/bold blue]")
        print("[PROGRESS] stage=initializing progress=0.5 message=Setting up distillation pipeline")
        from core.distillers.multi_stage_distiller import MultiStageDistiller
        distiller = MultiStageDistiller(
            teacher=teacher,
            student=student,
            config=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            device=str(cm.device())
        )
        print("[PROGRESS] stage=initializing progress=1.0 message=Distiller initialized successfully")
        rprint(f"[green]✓ Distiller initialized[/green]")
        
        # Run distillation
        rprint("[bold blue]🚀 Starting distillation...[/bold blue]")
        print("[PROGRESS] stage=training progress=0.0 message=Beginning knowledge distillation")
        report = distiller.run()
        print("[PROGRESS] stage=complete progress=1.0 message=Training completed successfully")
        
        rprint("\n[bold green]🎉 Distillation completed successfully![/bold green]")
        if 'summary' in report:
            rprint(f"Total stages: {report['summary'].get('total_stages', 0)}")
            rprint(f"Final accuracy gain: {report['summary'].get('total_accuracy_gain', 0):.2f}%")
        
    except Exception as exc:
        LOG.exception("Failed to run distillation: %s", exc)
        print("[PROGRESS] stage=failed progress=0.0 message=Training failed")
        rprint("[red]Distillation failed — check logs for details.[/red]")
        raise typer.Exit(code=1)


@app.command()
def quantize(
    config: str = typer.Option("configs/quant.yaml", help="Path to quantization config"),
    mode: str = typer.Option("ptq", help="Quantization mode: ptq or qat"),
):
    """
    Run quantization workflows (PTQ or QAT).
    Expects core.quant.ptq.PTQRunner and core.quant.qat.QATRunner entry classes (or similar).
    """
    cm, cfg = load_config(config)
    rprint(f"[bold green]Loaded quant config:[/bold green] {config}")
    try:
        if mode == "ptq":
            mod = importlib.import_module("core.quant.ptq")
            Runner = getattr(mod, "PTQRunner")
        elif mode == "qat":
            mod = importlib.import_module("core.quant.qat")
            Runner = getattr(mod, "QATRunner")
        else:
            rprint(f"[red]Unknown quant mode: {mode}[/red]")
            raise typer.Exit(code=2)

        runner = Runner(cfg)
        runner.run()
    except Exception as exc:
        LOG.exception("Quantization failed: %s", exc)
        rprint("[red]Quantization failed — check logs for details.[/red]")
        raise typer.Exit(code=1)


@app.command()
def evaluate(
    model: str = typer.Option(..., help="Path to the model to evaluate"),
    config: str = typer.Option("configs/eval.yaml", help="Path to evaluation config"),
):
    """
    Run evaluation harness for given model.
    Expects evaluation.evaluator.Evaluator to exist with interface:
      Evaluator(model_path: str, config: dict).run_all()
    """
    cm, cfg = load_config(config)
    rprint(f"[bold green]Evaluating[/bold green] model: {model}")

    try:
        mod = importlib.import_module("evaluation.evaluator")
        Evaluator = getattr(mod, "Evaluator")
        ev = Evaluator(model_path=model, config=cfg)
        ev.run_all()
    except Exception as exc:
        LOG.exception("Evaluation failed: %s", exc)
        rprint("[red]Evaluation failed — check logs for details.[/red]")
        raise typer.Exit(code=1)


@app.command()
def pkg(
    model: str = typer.Option(..., help="Path to trained model"),
    out: str = typer.Option("exports/", help="Output folder for packaged artifact"),
):
    """
    Package a model for deployment (GGUF / safetensors / manifest).
    Expects core.pkg.exporter.Exporter with .export(model_path, out_dir) signature.
    """
    rprint(f"[bold green]Packaging[/bold green] {model} → {out}")
    try:
        mod = importlib.import_module("core.pkg.exporter")
        Exporter = getattr(mod, "Exporter")
        exporter = Exporter()
        exporter.export(model, out)
    except Exception as exc:
        LOG.exception("Packaging failed: %s", exc)
        rprint("[red]Packaging failed — check logs for details.[/red]")
        raise typer.Exit(code=1)


def main():
    args = parse_args()

    try:
        # ========================================================================
        # PHASE 0: Environment & Configuration Setup
        # ========================================================================
        print("\n" + "="*70)
        print("PHASE 0: Environment & Configuration Setup")
        print("="*70 + "\n")
        
        # Parse overrides from command line arguments
        overrides_dict = parse_overrides(args.override)
        
        cfg_manager = ConfigManager(
            config_path=args.config,
            overrides=overrides_dict
        )

        resolved_mode = (args.mode or "auto").lower()
        if resolved_mode == "auto":
            resolved_mode = "infer" if args.load_model_dir else "train"

        def _collect_inference_texts() -> List[str]:
            texts: List[str] = []
            if args.infer_text:
                texts.extend([str(t) for t in args.infer_text if str(t).strip()])
            if args.infer_file:
                file_path = Path(args.infer_file)
                if file_path.exists():
                    with file_path.open("r", encoding="utf-8") as handle:
                        for line in handle:
                            line = line.strip()
                            if line:
                                texts.append(line)
                else:
                    print(f"[WARN] Inference file not found: {file_path}")
            return texts

        if resolved_mode == "infer":
            print("\n" + "="*70)
            print("INFERENCE MODE")
            print("="*70 + "\n")
            from core.inference.predict import StudentInference
            from core.models.model_saver import load_model
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            import json

            if args.load_model_dir:
                student, tokenizer, _ = load_model(
                    AutoModelForSequenceClassification,
                    args.load_model_dir,
                    AutoTokenizer,
                    map_location=str(cfg_manager.device()),
                )
                print(f"✅ Loaded model from artifacts: {args.load_model_dir}")
            else:
                print("ℹ️  No --load-model-dir provided; loading from config model settings")
                _, student, tokenizer = load_models(
                    cfg_manager,
                    cfg_manager.device(),
                    use_agent=False,
                )

            predictor = StudentInference(
                model=student,
                tokenizer=tokenizer,
                config=cfg_manager.resolved_config,
                device=cfg_manager.device(),
            )

            infer_texts = _collect_inference_texts()
            if not infer_texts:
                print("[WARN] No inference inputs provided (--infer-text/--infer-file). Exiting.")
                return

            preds = predictor.predict(infer_texts, batch_size=max(1, int(args.infer_batch_size)))
            print(f"✅ Inference completed for {len(preds)} samples")
            for idx, row in enumerate(preds[:5], 1):
                print(f"  [{idx}] label={row.get('label')} prob={row.get('prob'):.4f} text={row.get('text')[:80]}")

            infer_output = args.infer_output or str(Path(cfg_manager.experiment_dir) / "inference_predictions.json")
            out_path = Path(infer_output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with out_path.open("w", encoding="utf-8") as handle:
                json.dump(convert_to_serializable(preds), handle, indent=2, ensure_ascii=False)
            print(f"✅ Inference results saved to: {out_path}")
            return

        # Unified training runtime (Phase 1 orchestration spine)
        from app.runtime import RuntimeOptions, UnifiedTrainingRuntime

        runtime = UnifiedTrainingRuntime()
        runtime_options = RuntimeOptions(
            config_path=args.config,
            overrides=overrides_dict,
            load_model_dir=args.load_model_dir,
            load_checkpoint_path=args.load_checkpoint_path,
            checkpoint_non_strict=args.checkpoint_non_strict,
            save_model=args.save_model,
            save_model_dir=args.save_model_dir,
            save_checkpoint=args.save_checkpoint,
            checkpoint_path=args.checkpoint_path,
            use_teacher_agent=_use_teacher_agent(cfg_manager.resolved_config),
        )
        runtime_result = runtime.run(runtime_options)
        if runtime_result.success:
            print("✅ Unified runtime completed")
            print(f"Engine: {runtime_result.engine}")
            print(f"Experiment: {runtime_result.experiment_id}")
            if runtime_result.manifest_path:
                print(f"Manifest: {runtime_result.manifest_path}")
            if runtime_result.warnings:
                print("Warnings:")
                for warning in runtime_result.warnings:
                    print(f"  • {warning}")
            return

        print("❌ Unified runtime failed")
        for error in runtime_result.errors:
            print(f"  • {error}")
        raise RuntimeError("Unified runtime execution failed")

        print("✅ Config loaded successfully")
        print("Experiment ID:", cfg_manager.experiment_id)
        print("Paths:", cfg_manager.paths)
        print("Resolved config:", dict(cfg_manager.resolved_config))

        # ========================================================================
        # PHASE 1: Preflight Analysis & Model Loading
        # ========================================================================
        print("\n" + "="*70)
        print("PHASE 1: Preflight Analysis & Model Loading")
        print("="*70 + "\n")
        
        # 1.1 Config Validation (NEW - Critical!)
        print("📋 Step 1.1: Validating configuration...")
        from core.preflight.analyser import validate_config_only
        
        config_validation = validate_config_only(cfg_manager.resolved_config)
        
        if not config_validation['is_valid']:
            print("\n❌ Config validation failed. Cannot proceed.\n")
            print("ERRORS:")
            for error in config_validation['errors']:
                print(f"  • {error}")
            if config_validation['warnings']:
                print("\nWARNINGS:")
                for warning in config_validation['warnings']:
                    print(f"  • {warning}")
            return
        
        print("✅ Configuration validated successfully\n")

        # 1.2 Load models and tokenizer
        print("📋 Step 1.2: Loading models...")
        teacher, student, tokenizer = load_models(
            cfg_manager,
            cfg_manager.device(),
            use_agent=_use_teacher_agent(cfg_manager.resolved_config),
        )

        if args.load_model_dir:
            from core.models.model_saver import load_model
            from transformers import AutoModelForSequenceClassification, AutoModelForCausalLM, AutoTokenizer
            train_engine_preview = str(cfg_manager.resolved_config.get("train", {}).get("engine", "legacy")).lower()
            task_type_preview = str(cfg_manager.resolved_config.get("distillation", {}).get("task_type", "")).lower()
            is_causal_lm_preview = train_engine_preview in {"causal_lm_core", "causal_lm", "gpt_core"} or task_type_preview in {"causal_lm", "gpt", "language_modeling"}
            model_class = AutoModelForCausalLM if is_causal_lm_preview else AutoModelForSequenceClassification
            loaded_student, loaded_tokenizer, loaded_meta = load_model(
                model_class,
                args.load_model_dir,
                AutoTokenizer,
                map_location=str(cfg_manager.device()),
            )
            student = loaded_student
            tokenizer = loaded_tokenizer or tokenizer
            print(f"✅ Reused student model from: {args.load_model_dir}")
            if loaded_meta:
                print(f"ℹ️  Loaded metadata keys: {list(loaded_meta.keys())}")

        print("[Model Summary] Teacher model:")
        print(model_summary(teacher))
        print("[Model Summary] Student model:")
        print(model_summary(student))
        print(f"Tokenizer loaded: {type(tokenizer).__name__}")

        # ========================================================================
        # PHASE 2: Dataset Preparation
        # ========================================================================
        print("\n" + "="*70)
        print("PHASE 2: Dataset Preparation")
        print("="*70 + "\n")
        
        from data.dataloaders import create_dataloaders
        train_loader, val_loader = create_dataloaders(cfg_manager.resolved_config, tokenizer)
        print(f"✅ Train loader: {len(train_loader)} batches")
        print(f"✅ Val loader: {len(val_loader)} batches")

        # 1.3 & 1.4: Preflight Analysis (Model Compatibility) - Optional but recommended
        # Uncomment below to enable full preflight analysis with model inspection
        # from core.preflight.analyser import run_preflight_check
        # print("\n📋 Step 1.3-1.4: Running preflight analysis...")
        # preflight_report = run_preflight_check(
        #     teacher_model=teacher,
        #     student_model=student,
        #     dataset=train_loader.dataset,
        #     config=cfg_manager.resolved_config,
        #     save_report=True,
        #     output_dir=str(Path(cfg_manager.paths['logs']) / "preflight")
        # )
        # if not preflight_report['can_proceed']:
        #     print("\n❌ Preflight checks failed.")
        #     return

        # ========================================================================
        # PHASE 3-4: Distillation Training
        # ========================================================================
        print("\n" + "="*70)
        print("PHASE 3-4: Distillation Engine & Training")
        print("="*70 + "\n")
        
        train_engine = str(cfg_manager.resolved_config.get("train", {}).get("engine", "legacy")).lower()
        stable_candidate_mode = train_engine == "causal_lm_core_stable"
        use_causal_lm_core = train_engine in {"causal_lm_core", "causal_lm", "gpt_core", "causal_lm_core_stable"}

        if use_causal_lm_core:
            from core.distillers.causal_lm import RegressionGate, SafeCausalLMTrainer

            if stable_candidate_mode:
                print("🛡️  Running RegressionGate for causal_lm_core_stable...")
                gate = RegressionGate.from_mapping(cfg_manager.resolved_config.get("train", {}))
                gate_report = gate.run(
                    teacher=teacher,
                    student=student,
                    tokenizer=tokenizer,
                    config=cfg_manager.resolved_config,
                    device=cfg_manager.device(),
                    experiment_dir=cfg_manager.experiment_dir,
                    train_loader=train_loader,
                )
                if not gate_report.passed:
                    raise RuntimeError(
                        "RegressionGate failed for causal_lm_core_stable: "
                        f"reasons={gate_report.reasons}, "
                        f"max_token_loss_diff={gate_report.max_token_loss_diff:.6f}, "
                        f"max_grad_norm_diff={gate_report.max_grad_norm_diff:.6f}"
                    )
                print("✅ RegressionGate passed")

            trainer = SafeCausalLMTrainer(
                teacher=teacher,
                student=student,
                tokenizer=tokenizer,
                config=cfg_manager.resolved_config,
                device=cfg_manager.device(),
                experiment_dir=cfg_manager.experiment_dir,
            )
            print("✅ Using stable Causal-LM core trainer")
        else:
            from training.trainer import Trainer
            trainer = Trainer(
                teacher=teacher,
                student=student,
                tokenizer=tokenizer,
                config=cfg_manager.resolved_config,
                device=cfg_manager.device(),
                experiment_dir=cfg_manager.experiment_dir
            )

        if args.load_checkpoint_path:
            if use_causal_lm_core and hasattr(trainer, "resume_from_checkpoint"):
                resume_report = trainer.resume_from_checkpoint(args.load_checkpoint_path)
                print(f"✅ Loaded checkpoint from: {args.load_checkpoint_path}")
                if resume_report.get("warning"):
                    print(f"⚠️  {resume_report['warning']}")
            else:
                from core.models.model_saver import load_checkpoint
                _, checkpoint_meta = load_checkpoint(
                    model=trainer.student,
                    optimizer=trainer.optimizer,
                    path=args.load_checkpoint_path,
                    scheduler=trainer.scheduler,
                    scaler=trainer.scaler,
                    map_location=str(cfg_manager.device()),
                    strict=not args.checkpoint_non_strict,
                )
                print(f"✅ Loaded checkpoint from: {args.load_checkpoint_path}")
                if checkpoint_meta is not None:
                    print(f"ℹ️  Checkpoint epoch={checkpoint_meta.epoch}, step={checkpoint_meta.global_step}")
        elif use_causal_lm_core and hasattr(trainer, "resume_from_latest"):
            latest_resume = trainer.resume_from_latest()
            if latest_resume is not None:
                print("✅ Auto-resumed from latest valid checkpoint")
                if latest_resume.get("warning"):
                    print(f"⚠️  {latest_resume['warning']}")

        print("[INFO] Starting training...")
        trainer.fit(train_loader, val_loader)

        if args.save_model:
            from core.models.model_saver import save_model
            model_output_dir = args.save_model_dir or str(Path(cfg_manager.experiment_dir) / "student_model")
            metadata = {
                "experiment_id": cfg_manager.experiment_id,
                "device": str(cfg_manager.device()),
                "mode": resolved_mode,
                "source_config": args.config,
            }
            save_model(
                model=trainer.student,
                path=model_output_dir,
                tokenizer=tokenizer,
                metadata=metadata,
                use_safetensors=True,
            )
            print(f"✅ Saved reusable student model to: {model_output_dir}")

        if args.save_checkpoint:
            checkpoint_out = args.checkpoint_path or str(Path(cfg_manager.experiment_dir) / "checkpoints" / "latest.pt")
            if use_causal_lm_core and hasattr(trainer, "save_explicit_checkpoint"):
                out_path = trainer.save_explicit_checkpoint(checkpoint_out)
                print(f"✅ Saved checkpoint to: {out_path}")
            else:
                from core.models.model_saver import save_checkpoint, CheckpointMetadata
                ckpt_meta = CheckpointMetadata(
                    stage="training_complete",
                    epoch=len(getattr(trainer, 'train_losses', [])),
                    global_step=0,
                    best_metric=float(max(getattr(trainer, 'metrics_history', {}).get('accuracy', [0.0]) or [0.0])),
                    metrics={
                        "best_val_loss": float(getattr(trainer, 'best_val_loss', 0.0)),
                    },
                    extras={
                        "experiment_id": cfg_manager.experiment_id,
                        "config_path": args.config,
                    },
                )
                save_checkpoint(
                    model=trainer.student,
                    optimizer=trainer.optimizer,
                    path=checkpoint_out,
                    scheduler=trainer.scheduler,
                    scaler=trainer.scaler,
                    metadata=ckpt_meta,
                )
                print(f"✅ Saved checkpoint to: {checkpoint_out}")

        # ========================================================================
        # PHASE 6: Quantization
        # ========================================================================
        quantized_model = None
        if cfg_manager.resolved_config.get("quantization", {}).get("enable", False):
            print("\n" + "="*70)
            print("PHASE 6: Quantization")
            print("="*70 + "\n")
            
            from core.quant.ptq import apply_ptq
            runtime_device = cfg_manager.device()
            cfg_mode = str(cfg_manager.resolved_config.get("quantization", {}).get("mode", "ptq")).lower()

            # Reasoning logic
            if runtime_device == "mps" and cfg_mode == "dynamic":
                rprint("[yellow]MPS detected: Dynamic int8 PTQ not supported. Switching to Float16 PTQ.[/yellow]")
                mode = "float16"
            else:
                mode_map = {"ptq": "dynamic", "dynamic": "dynamic", "float16": "float16", "fp16": "float16", "static": "static", "int8_static": "static"}
                mode = mode_map.get(cfg_mode, "dynamic")

            quantized_model = apply_ptq(student, runtime_device, mode=mode)
            rprint(f"[green]PTQ applied using mode: {mode} on device {runtime_device}[/green]")

        # ========================================================================
        # PHASE 5: Evaluation (with Extended Metrics & DEI/CAS)
        # ========================================================================
        print("\n" + "="*70)
        print("PHASE 5: Evaluation (Extended Metrics)")
        print("="*70 + "\n")
        
        # Final evaluation with DualEvaluator
        evaluate_enabled = cfg_manager.resolved_config.get("evaluate", True)
        metrics: Dict[str, Any] = {}
        extended_metrics: Dict[str, Any] = {}
        cas_rating: Optional[str] = None  # Initialize for later use in summary
        teacher_metrics: Dict[str, Any] = {}  # Initialize for later use in summary
        dei_results: Dict[str, Any] = {}  # Initialize for later use in summary
        cas_results: Dict[str, Any] = {}  # Initialize for later use in summary
        eval_results: Dict[str, Any] = {}
        class_labels: List[str] = []
        
        if evaluate_enabled:
            try:
                from evaluation.evaluator_extended import DualEvaluator
                from evaluation.metrics_extended import DistillationEfficacyIndex, CompressionAwareScore
                from pathlib import Path
                import json
                
                model_to_eval = quantized_model if quantized_model is not None else student
                import torch
                device_obj = torch.device(str(cfg_manager.device()))
                model_to_eval = model_to_eval.to(device_obj)  # type: ignore[arg-type]
                
                rprint("[bold blue]Running dual evaluation (Teacher vs Student)...[/bold blue]")
                
                # Use DualEvaluator for side-by-side comparison
                dual_evaluator = DualEvaluator(
                    teacher=teacher,
                    student=model_to_eval,
                    dataloader=val_loader,
                    device=str(cfg_manager.device())
                )
                
                eval_results = dual_evaluator.evaluate()
                class_labels = resolve_label_names(
                    cfg_manager.resolved_config,
                    getattr(val_loader, 'dataset', None),
                )
                
                # Extract metrics
                metrics = eval_results.get('student', {})
                teacher_metrics = eval_results.get('teacher', {})
                extended_metrics = eval_results.get('extended', {})
                
                # Helper to format metrics safely
                def fmt(value, format_spec='.4f'):
                    """Format value or return N/A if not available"""
                    if value == 'N/A' or value is None:
                        return 'N/A'
                    try:
                        return f"{value:{format_spec}}"
                    except (ValueError, TypeError):
                        return str(value)
                
                # Display results
                rprint(f"[bold green]Teacher Metrics:[/bold green]")
                rprint(f"  Accuracy: {fmt(teacher_metrics.get('accuracy', 'N/A'))}")
                rprint(f"  F1 Score: {fmt(teacher_metrics.get('f1', 'N/A'))}")
                rprint(f"  Loss: {fmt(teacher_metrics.get('loss', 'N/A'))}")
                
                rprint(f"\n[bold green]Student Metrics:[/bold green]")
                rprint(f"  Accuracy: {fmt(metrics.get('accuracy', 'N/A'))}")
                rprint(f"  F1 Score: {fmt(metrics.get('f1', 'N/A'))}")
                rprint(f"  Loss: {fmt(metrics.get('loss', 'N/A'))}")
                
                rprint(f"\n[bold cyan]Extended Distillation Metrics:[/bold cyan]")
                rprint(f"  KL Divergence: {fmt(extended_metrics.get('kl_divergence', 'N/A'))}")
                rprint(f"  JS Divergence: {fmt(extended_metrics.get('js_divergence', 'N/A'))}")
                rprint(f"  Prediction Agreement: {fmt(extended_metrics.get('prediction_agreement', 'N/A'), '.2%')}")
                rprint(f"  Confidence Correlation: {fmt(extended_metrics.get('confidence_correlation', 'N/A'))}")
                
                # Compute DEI & CAS
                teacher_params = sum(p.numel() for p in teacher.parameters())
                student_params = sum(p.numel() for p in model_to_eval.parameters())
                
                teacher_acc = teacher_metrics.get('accuracy', 0.0)
                student_acc = metrics.get('accuracy', 0.0)
                
                dei_results = DistillationEfficacyIndex.compute_dei(
                    teacher_acc=teacher_acc,
                    student_acc=student_acc,
                    teacher_params=teacher_params,
                    student_params=student_params
                )
                
                # CAS requires latency measurements, use placeholder values
                cas_results = CompressionAwareScore.compute_cas(
                    accuracy=student_acc,
                    teacher_params=teacher_params,
                    student_params=student_params,
                    teacher_latency=1.0,  # Placeholder - would measure in production
                    student_latency=0.5   # Placeholder - would measure in production
                )
                
                # Add rating to CAS based on score (store as separate key to avoid type issues)
                cas_score = cas_results['cas']
                if cas_score > 0.35:
                    cas_rating = 'Excellent'
                elif cas_score > 0.25:
                    cas_rating = 'Very Good'
                elif cas_score > 0.15:
                    cas_rating = 'Good'
                elif cas_score > 0.05:
                    cas_rating = 'Fair'
                else:
                    cas_rating = 'Poor'
                # Note: Not adding rating to cas_results dict due to type constraints
                
                # Display DEI & CAS
                rprint(f"\n[bold magenta]Distillation Efficacy Index (DEI):[/bold magenta]")
                rprint(f"  DEI Score: {dei_results['dei']:.4f}")
                rprint(f"  Rating: {dei_results['efficiency_rating']}")
                rprint(f"  Accuracy Retention: {dei_results['accuracy_retention']:.2%}")
                rprint(f"  Compression Ratio: {dei_results['compression_ratio']:.2f}x")
                
                rprint(f"\n[bold magenta]Compression-Aware Score (CAS):[/bold magenta]")
                rprint(f"  CAS Score: {cas_results['cas']:.4f}")
                rprint(f"  Rating: {cas_rating}")
                
                # Save extended evaluation results
                extended_eval_path = Path(cfg_manager.experiment_dir) / "extended_evaluation.json"
                with open(extended_eval_path, 'w') as f:
                    # Convert all metrics to JSON-serializable format
                    serializable_data = convert_to_serializable({
                        'teacher': teacher_metrics,
                        'student': metrics,
                        'extended_metrics': extended_metrics,
                        'dei': dei_results,
                        'cas': cas_results
                    })
                    json.dump(serializable_data, f, indent=2)
                
                print(f"✅ Extended evaluation saved to: {extended_eval_path}")
                
            except Exception as e:
                LOG.exception("Evaluation after training failed: %s", e)
                rprint("[red]Evaluation after training failed — check logs for details.[/red]")
        
        # ========================================================================
        # PHASE 9: Visualization & Showcasing
        # ========================================================================
        print("\n" + "="*70)
        print("PHASE 9: Visualization & Showcasing")
        print("="*70 + "\n")
        
        # Calculate model parameters (always available after training)
        teacher_params = sum(p.numel() for p in teacher.parameters())
        student_params = sum(p.numel() for p in student.parameters())
        compression_ratio = teacher_params / student_params
        
        viz_enabled = cfg_manager.resolved_config.get("visualization", {}).get("enable", True)
        if viz_enabled:
            try:
                from evaluation.visualizer import plot_training_curves
                from evaluation.metrics import plot_metrics
                from pathlib import Path
                
                viz_dir = Path(cfg_manager.experiment_dir) / "visualizations"
                viz_dir.mkdir(exist_ok=True)
                
                print("📊 Generating visualizations...")
                
                # Training curves already saved by trainer, but let's confirm
                curves_path = Path(cfg_manager.experiment_dir) / "training_curves.png"
                if curves_path.exists():
                    print(f"✅ Training curves: {curves_path}")
                else:
                    print("⚠️  Training curves not found (may not have been generated)")

                explain_artifacts: Dict[str, str] = {}
                student_metrics_dir: Optional[Path] = None
                teacher_metrics_dir: Optional[Path] = None

                if evaluate_enabled and metrics:
                    student_metrics_dir = viz_dir / "student_metrics"
                    student_metrics_dir.mkdir(exist_ok=True)
                    student_metric_payload = metrics.get('metrics', metrics)
                    plot_metrics(student_metric_payload, str(student_metrics_dir), labels=class_labels or None)

                    if teacher_metrics:
                        teacher_metrics_dir = viz_dir / "teacher_metrics"
                        teacher_metrics_dir.mkdir(exist_ok=True)
                        teacher_metric_payload = teacher_metrics.get('metrics', teacher_metrics)
                        plot_metrics(teacher_metric_payload, str(teacher_metrics_dir), labels=class_labels or None)

                    explain_artifacts = generate_explainability_reports(
                        student_model=student,
                        tokenizer=tokenizer,
                        cfg_manager=cfg_manager,
                        val_dataset=getattr(val_loader, 'dataset', None),
                        eval_results=eval_results,
                        output_dir=viz_dir,
                        class_labels=class_labels,
                    )
                
                # Create comparison visualization if we have teacher and student
                try:
                    import matplotlib.pyplot as plt
                    
                    # Model size comparison
                    teacher_params = sum(p.numel() for p in teacher.parameters())
                    student_params = sum(p.numel() for p in student.parameters())
                    compression_ratio = teacher_params / student_params
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    models = ['Teacher', 'Student']
                    params = [teacher_params / 1e6, student_params / 1e6]  # Convert to millions
                    colors = ['#3498db', '#2ecc71']
                    
                    bars = ax.bar(models, params, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
                    ax.set_ylabel('Parameters (Millions)', fontsize=12, fontweight='bold')
                    ax.set_title(f'Model Size Comparison\nCompression Ratio: {compression_ratio:.2f}x', 
                                fontsize=14, fontweight='bold')
                    ax.grid(axis='y', alpha=0.3, linestyle='--')
                    
                    # Add value labels on bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.1f}M',
                               ha='center', va='bottom', fontweight='bold')
                    
                    plt.tight_layout()
                    comparison_path = viz_dir / "model_comparison.png"
                    plt.savefig(comparison_path, dpi=150, bbox_inches='tight')
                    plt.close()
                    print(f"✅ Model comparison: {comparison_path}")
                    
                except Exception as viz_e:
                    print(f"⚠️  Could not create model comparison: {viz_e}")
                
                # Summary report with extended metrics
                summary_path = Path(cfg_manager.experiment_dir) / "EXPERIMENT_SUMMARY.md"
                with open(summary_path, 'w') as f:
                    resolved_cfg = cfg_manager.resolved_config
                    f.write(f"# Experiment Summary\n\n")
                    f.write(f"**Experiment ID**: {cfg_manager.experiment_id}\n\n")
                    f.write(f"**Date**: {Path(cfg_manager.experiment_dir).name.split('_')[0]}\n\n")
                    f.write(f"## Models\n\n")
                    f.write(f"- **Teacher**: {resolved_cfg.get('model', {}).get('name')} ({teacher_params/1e6:.1f}M params)\n")
                    f.write(f"- **Student**: {resolved_cfg.get('model', {}).get('student_name')} ({student_params/1e6:.1f}M params)\n")
                    f.write(f"- **Compression**: {compression_ratio:.2f}x\n\n")
                    f.write(f"## Training\n\n")
                    f.write(f"- **Epochs**: {resolved_cfg.get('train', {}).get('epochs')}\n")
                    f.write(f"- **Batch Size**: {resolved_cfg.get('train', {}).get('batch_size')}\n")
                    f.write(f"- **Learning Rate**: {resolved_cfg.get('train', {}).get('lr')}\n")
                    f.write(f"- **Device**: {cfg_manager.device()}\n\n")
                    f.write(f"## Results\n\n")
                    if evaluate_enabled and 'metrics' in locals():
                        f.write(f"### Standard Metrics\n\n")
                        
                        # Safely format student metrics
                        student_acc = metrics.get('accuracy', 'N/A')
                        student_f1 = metrics.get('f1', 'N/A')
                        f.write(f"- **Student Accuracy**: {student_acc if student_acc == 'N/A' else f'{student_acc:.4f}'}\n")
                        f.write(f"- **Student F1 Score**: {student_f1 if student_f1 == 'N/A' else f'{student_f1:.4f}'}\n")
                        
                        if 'teacher_metrics' in locals():
                            teacher_acc = teacher_metrics.get('accuracy', 'N/A')
                            teacher_f1 = teacher_metrics.get('f1', 'N/A')
                            f.write(f"- **Teacher Accuracy**: {teacher_acc if teacher_acc == 'N/A' else f'{teacher_acc:.4f}'}\n")
                            f.write(f"- **Teacher F1 Score**: {teacher_f1 if teacher_f1 == 'N/A' else f'{teacher_f1:.4f}'}\n")
                        
                        f.write(f"\n### Extended Distillation Metrics\n\n")
                        if 'extended_metrics' in locals() and extended_metrics:
                            kl_div = extended_metrics.get('kl_divergence', 'N/A')
                            js_div = extended_metrics.get('js_divergence', 'N/A')
                            pred_agree = extended_metrics.get('prediction_agreement', 'N/A')
                            conf_corr = extended_metrics.get('confidence_correlation', 'N/A')
                            
                            f.write(f"- **KL Divergence**: {kl_div if kl_div == 'N/A' else f'{kl_div:.4f}'}\n")
                            f.write(f"- **JS Divergence**: {js_div if js_div == 'N/A' else f'{js_div:.4f}'}\n")
                            f.write(f"- **Prediction Agreement**: {pred_agree if pred_agree == 'N/A' else f'{pred_agree:.2%}'}\n")
                            f.write(f"- **Confidence Correlation**: {conf_corr if conf_corr == 'N/A' else f'{conf_corr:.4f}'}\n")
                        
                        f.write(f"\n### Distillation Quality Scores\n\n")
                        if 'dei_results' in locals():
                            f.write(f"- **DEI Score**: {dei_results['dei']:.4f} ({dei_results['efficiency_rating']})\n")
                            f.write(f"  - Accuracy Retention: {dei_results['accuracy_retention']:.2%}\n")
                            f.write(f"  - Compression Ratio: {dei_results['compression_ratio']:.2f}x\n")
                        if 'cas_results' in locals() and cas_rating:
                            f.write(f"- **CAS Score**: {cas_results['cas']:.4f} ({cas_rating})\n")
                    
                    f.write(f"\n## Artifacts\n\n")
                    f.write(f"- Teacher model: `teacher_model/`\n")
                    f.write(f"- Student model: `student_model/`\n")
                    f.write(f"- Training curves: `training_curves.png`\n")
                    f.write(f"- Extended metrics: `extended_metrics.json`\n")
                    f.write(f"- Extended evaluation: `extended_evaluation.json`\n")
                    if evaluate_enabled and student_metrics_dir is not None:
                        student_cm = student_metrics_dir / "confusion_matrix.png"
                        if student_cm.exists():
                            f.write(
                                f"- Student confusion matrix: `{student_cm.relative_to(cfg_manager.experiment_dir)}`\n"
                            )
                    if evaluate_enabled and teacher_metrics_dir is not None:
                        teacher_cm = teacher_metrics_dir / "confusion_matrix.png"
                        if teacher_cm.exists():
                            f.write(
                                f"- Teacher confusion matrix: `{teacher_cm.relative_to(cfg_manager.experiment_dir)}`\n"
                            )
                    if explain_artifacts:
                        for label, key in (
                            ("LIME explanations", "lime"),
                            ("SHAP contributions", "shap"),
                            ("SHAP summary plot", "shap_plot"),
                        ):
                            artifact_path = explain_artifacts.get(key)
                            if artifact_path:
                                artifact_obj = Path(artifact_path)
                                if artifact_obj.exists():
                                    f.write(
                                        f"- {label}: `{artifact_obj.relative_to(cfg_manager.experiment_dir)}`\n"
                                    )
                    f.write(f"- Model comparison: `visualizations/model_comparison.png`\n")
                    f.write(f"- Config: `resolved_config.yaml`\n")
                
                print(f"✅ Experiment summary: {summary_path}")
                
                rprint(f"\n[bold green]🎉 Complete! All artifacts saved to:[/bold green]")
                rprint(f"[cyan]{cfg_manager.experiment_dir}[/cyan]")
                
            except Exception as e:
                LOG.exception("Visualization failed: %s", e)
                rprint("[yellow]⚠️  Visualization failed — check logs for details.[/yellow]")
        else:
            print("ℹ️  Visualization disabled in config")
        
        # ========================================================================
        # PHASE 8: Reporting (Summary)
        # ========================================================================
        print("\n" + "="*70)
        print("✅ ALL PHASES COMPLETE")
        print("="*70 + "\n")
        rprint(f"[bold green]Training completed successfully![/bold green]")
        rprint(f"[bold]Experiment directory:[/bold] [cyan]{cfg_manager.experiment_dir}[/cyan]")

    except ConfigError as e:
        print("❌ Config error:", e)
        
    except Exception as e:
        print("❌ Unexpected error:", e)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
