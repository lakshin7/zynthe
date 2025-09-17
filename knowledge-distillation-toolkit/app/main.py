# app/main.py
from __future__ import annotations

import importlib
import logging
from pathlib import Path
import sys
import typer

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

    This command:
      1. loads the config,
      2. tries to import core.distillers.multi_stage_distiller.MultiStageDistiller (fallbacks allowed),
      3. instantiates it with the loaded config and calls .run()
    """
    cm, cfg = load_config(config)
    rprint(f"[bold green]Loaded distill config:[/bold green] {config}")

    if dry_run:
        rprint("[yellow]Dry-run: printing parsed config and exiting[/yellow]")
        rprint(cfg)
        raise typer.Exit()

    # Dynamic import so the CLI works even while modules are under development
    try:
        mod = importlib.import_module("core.distillers.multi_stage_distiller")
        Distiller = getattr(mod, "MultiStageDistiller", None) or getattr(mod, "Distiller", None)
        if Distiller is None:
            raise ImportError("No MultiStageDistiller/Distiller class found in module.")
        distiller = Distiller(cfg)
        distiller.run()
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
        cfg_manager = ConfigManager(
            config_path=args.config,
            overrides=args.override
        )
        print("✅ Config loaded successfully")
        print("Experiment ID:", cfg_manager.experiment_id)
        print("Paths:", cfg_manager.paths)
        print("Resolved config:", dict(cfg_manager.resolved_config))

        # --- Load models and tokenizer ---
        teacher, student, tokenizer = load_models(cfg_manager.resolved_config, cfg_manager.device())
        print("[Model Summary] Teacher model:")
        print(model_summary(teacher))
        print("[Model Summary] Student model:")
        print(model_summary(student))
        print(f"Tokenizer loaded: {type(tokenizer).__name__}")

        # --- Trainer initialization ---
        from training.trainer import Trainer
        from data.dataloaders import create_dataloaders
        train_loader, val_loader = create_dataloaders(cfg_manager.resolved_config, tokenizer)

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

        # --- Reasoning-based Quantization ---
        quantized_model = None
        if cfg_manager.resolved_config.get("quantization", {}).get("enable", False):
            from core.quant.ptq import apply_ptq
            runtime_device = cfg_manager.get_runtime().get("device", "cpu")
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

        # --- Final metrics visualization already handled by trainer ---
        rprint(f"[bold green]Training completed successfully! Check experiment directory: {cfg_manager.experiment_dir}[/bold green]")

        # --- Evaluation after training and quantization ---
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
