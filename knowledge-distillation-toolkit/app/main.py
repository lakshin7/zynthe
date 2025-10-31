# app/main.py
from __future__ import annotations

import importlib
import logging
from pathlib import Path
import sys
import typer

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

from core.config.config_manager import ConfigManager

app = typer.Typer(name="zyn", help="Zynthe / Knowledge-distillation Toolkit CLI")

LOG = logging.getLogger("zyn")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

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
        # Load models
        rprint("[bold blue]Loading models...[/bold blue]")
        teacher, student, tokenizer = load_models(cm, cm.device())
        rprint(f"[green]✓ Teacher loaded: {model_summary(teacher)['name']}[/green]")
        rprint(f"[green]✓ Student loaded: {model_summary(student)['name']}[/green]")
        
        # Load dataloaders
        rprint("[bold blue]Loading dataloaders...[/bold blue]")
        from data.dataloaders import create_dataloaders
        train_loader, val_loader = create_dataloaders(cfg, tokenizer)
        rprint(f"[green]✓ Dataloaders created[/green]")
        
        # Create multi-stage distiller
        rprint("[bold blue]Initializing MultiStageDistiller...[/bold blue]")
        from core.distillers.multi_stage_distiller import MultiStageDistiller
        distiller = MultiStageDistiller(
            teacher=teacher,
            student=student,
            config=cfg,
            train_loader=train_loader,
            val_loader=val_loader,
            device=cm.device()
        )
        rprint(f"[green]✓ Distiller initialized[/green]")
        
        # Run distillation
        rprint("[bold blue]Starting distillation...[/bold blue]")
        report = distiller.run()
        
        rprint("\n[bold green]🎉 Distillation completed successfully![/bold green]")
        if 'summary' in report:
            rprint(f"Total stages: {report['summary'].get('total_stages', 0)}")
            rprint(f"Final accuracy gain: {report['summary'].get('total_accuracy_gain', 0):.2f}%")
        
    except Exception as exc:
        LOG.exception("Failed to run distillation: %s", exc)
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
        teacher, student, tokenizer = load_models(cfg_manager, cfg_manager.device())
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
        
        from training.trainer import Trainer
        trainer = Trainer(
            teacher=teacher,
            student=student,
            tokenizer=tokenizer,
            config=cfg_manager.resolved_config,
            device=cfg_manager.device(),
            experiment_dir=cfg_manager.experiment_dir
        )
        print("[INFO] Starting training...")
        trainer.fit(train_loader, val_loader)

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
        metrics = {}
        extended_metrics = {}
        
        if evaluate_enabled:
            try:
                from evaluation.evaluator_extended import DualEvaluator
                from evaluation.metrics_extended import DistillationEfficacyIndex, CompressionAwareScore
                from pathlib import Path
                import json
                
                model_to_eval = quantized_model if quantized_model is not None else student
                model_to_eval = model_to_eval.to(cfg_manager.device())
                
                rprint("[bold blue]Running dual evaluation (Teacher vs Student)...[/bold blue]")
                
                # Use DualEvaluator for side-by-side comparison
                dual_evaluator = DualEvaluator(
                    teacher=teacher,
                    student=model_to_eval,
                    dataloader=val_loader,
                    device=cfg_manager.device()
                )
                
                eval_results = dual_evaluator.evaluate()
                
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
                
                # Add rating to CAS based on score
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
                cas_results['rating'] = cas_rating
                
                # Display DEI & CAS
                rprint(f"\n[bold magenta]Distillation Efficacy Index (DEI):[/bold magenta]")
                rprint(f"  DEI Score: {dei_results['dei']:.4f}")
                rprint(f"  Rating: {dei_results['efficiency_rating']}")
                rprint(f"  Accuracy Retention: {dei_results['accuracy_retention']:.2%}")
                rprint(f"  Compression Ratio: {dei_results['compression_ratio']:.2f}x")
                
                rprint(f"\n[bold magenta]Compression-Aware Score (CAS):[/bold magenta]")
                rprint(f"  CAS Score: {cas_results['cas']:.4f}")
                rprint(f"  Rating: {cas_results['rating']}")
                
                # Save extended evaluation results
                extended_eval_path = Path(cfg_manager.experiment_dir) / "extended_evaluation.json"
                with open(extended_eval_path, 'w') as f:
                    json.dump({
                        'teacher': teacher_metrics,
                        'student': metrics,
                        'extended_metrics': extended_metrics,
                        'dei': dei_results,
                        'cas': cas_results
                    }, f, indent=2)
                
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
                        if 'cas_results' in locals():
                            f.write(f"- **CAS Score**: {cas_results['cas']:.4f} ({cas_results['rating']})\n")
                    
                    f.write(f"\n## Artifacts\n\n")
                    f.write(f"- Teacher model: `teacher_model/`\n")
                    f.write(f"- Student model: `student_model/`\n")
                    f.write(f"- Training curves: `training_curves.png`\n")
                    f.write(f"- Extended metrics: `extended_metrics.json`\n")
                    f.write(f"- Extended evaluation: `extended_evaluation.json`\n")
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
