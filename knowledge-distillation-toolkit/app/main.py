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
        # PHASE 5 & 8: Evaluation & Reporting
        # ========================================================================
        print("\n" + "="*70)
        print("PHASE 5 & 8: Evaluation & Reporting")
        print("="*70 + "\n")
        
        rprint(f"[bold green]Training completed successfully! Check experiment directory: {cfg_manager.experiment_dir}[/bold green]")

        # Final evaluation
        evaluate_enabled = cfg_manager.resolved_config.get("evaluate", True)
        if evaluate_enabled:
            try:
                import evaluation.evaluator
                Evaluator = getattr(evaluation.evaluator, "Evaluator")
                model_to_eval = quantized_model if quantized_model is not None else student
                # Ensure model is on the correct device
                model_to_eval = model_to_eval.to(cfg_manager.device())
                evaluator = Evaluator(model=model_to_eval, tokenizer=tokenizer, dataloader=val_loader, device=cfg_manager.device())
                rprint("[bold blue]Running final evaluation on the model...[/bold blue]")
                metrics = evaluator.run_all()
                rprint(f"[bold green]Final evaluation metrics:[/bold green] {metrics}")
            except Exception as e:
                LOG.exception("Evaluation after training failed: %s", e)
                rprint("[red]Evaluation after training failed — check logs for details.[/red]")

    except ConfigError as e:
        print("❌ Config error:", e)
        
    except Exception as e:
        print("❌ Unexpected error:", e)

if __name__ == "__main__":
    main()
