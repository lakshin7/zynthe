from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
import json

import typer
from rich import print as rprint

# Set default opt-outs for optional heavy vision deps when running text pipelines.
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("TRANSFORMERS_NO_PIL", "1")

# Add project root to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from core.config.config_manager import ConfigManager, ConfigError
from core.models.model_loader import load_models, model_summary
from app.cli_helpers import (
    convert_to_serializable,
    parse_overrides,
    load_config,
    detect_device,
    format_device_info,
    validate_config_path,
    use_teacher_agent,
)

app = typer.Typer(name="zyn", help="Zynthe / Knowledge-distillation Toolkit CLI")

LOG = logging.getLogger("zyn")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

@app.command("distill")
def distill(
    config: str = typer.Option("configs/default.yaml", help="Path to config YAML file"),
    override: List[str] = typer.Option([], help="Config overrides in KEY=VALUE format"),
    load_model_dir: Optional[str] = typer.Option(None, help="Path to saved student model directory for reuse"),
    load_checkpoint_path: Optional[str] = typer.Option(None, help="Path to training checkpoint to resume from"),
    checkpoint_non_strict: bool = typer.Option(False, help="Load checkpoint with strict=False"),
    save_model: bool = typer.Option(True, help="Save final model artifacts"),
    save_model_dir: Optional[str] = typer.Option(None, help="Optional output directory for saved model artifacts"),
    save_checkpoint: bool = typer.Option(False, help="Save trainer checkpoint after run"),
    checkpoint_path: Optional[str] = typer.Option(None, help="Optional checkpoint output path"),
):
    """Run the distillation pipeline."""
    from app.runtime import RuntimeOptions, UnifiedTrainingRuntime

    overrides_dict = parse_overrides(override)
    cfg_manager = ConfigManager(config_path=config, overrides=overrides_dict)

    runtime = UnifiedTrainingRuntime()
    runtime_options = RuntimeOptions(
        config_path=config,
        overrides=overrides_dict,
        load_model_dir=load_model_dir,
        load_checkpoint_path=load_checkpoint_path,
        checkpoint_non_strict=checkpoint_non_strict,
        save_model=save_model,
        save_model_dir=save_model_dir,
        save_checkpoint=save_checkpoint,
        checkpoint_path=checkpoint_path,
        use_teacher_agent=use_teacher_agent(cfg_manager.resolved_config),
    )
    
    try:
        runtime_result = runtime.run(runtime_options)
        if runtime_result.success:
            rprint("[bold green]✅ Unified runtime completed[/bold green]")
            rprint(f"Engine: {runtime_result.engine}")
            rprint(f"Experiment: {runtime_result.experiment_id}")
            if runtime_result.manifest_path:
                rprint(f"Manifest: {runtime_result.manifest_path}")
            if runtime_result.warnings:
                rprint("[yellow]Warnings:[/yellow]")
                for warning in runtime_result.warnings:
                    rprint(f"  • {warning}")
        else:
            rprint("[bold red]❌ Unified runtime failed[/bold red]")
            for error in runtime_result.errors:
                rprint(f"  • {error}")
            raise typer.Exit(code=1)
    except ConfigError as e:
        rprint(f"[bold red]❌ Config error:[/bold red] {e}")
        raise typer.Exit(code=1)
    except Exception as e:
        rprint(f"[bold red]❌ Unexpected error:[/bold red] {e}")
        import traceback
        traceback.print_exc()
        raise typer.Exit(code=1)


@app.command("evaluate")
def evaluate(
    config: str = typer.Option("configs/default.yaml", help="Path to config YAML file"),
    override: List[str] = typer.Option([], help="Config overrides in KEY=VALUE format"),
    load_model_dir: Optional[str] = typer.Option(None, help="Path to saved student model directory for inference"),
    infer_text: List[str] = typer.Option([], help="One or more texts for inference"),
    infer_file: Optional[str] = typer.Option(None, help="Text file path (one input per line) for inference"),
    infer_batch_size: int = typer.Option(8, help="Batch size for inference"),
    infer_output: Optional[str] = typer.Option(None, help="Optional JSON output path for inference predictions"),
):
    """Run standalone evaluation or inference."""
    from core.inference.predict import StudentInference
    from core.models.model_saver import load_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    overrides_dict = parse_overrides(override)
    cfg_manager = ConfigManager(config_path=config, overrides=overrides_dict)

    if load_model_dir:
        student, tokenizer, _ = load_model(
            AutoModelForSequenceClassification,
            load_model_dir,
            AutoTokenizer,
            map_location=str(cfg_manager.device()),
        )
        rprint(f"[green]✅ Loaded model from artifacts: {load_model_dir}[/green]")
    else:
        rprint("[blue]ℹ️ No --load-model-dir provided; loading from config model settings[/blue]")
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

    texts: List[str] = []
    if infer_text:
        texts.extend([str(t) for t in infer_text if str(t).strip()])
    if infer_file:
        file_path = Path(infer_file)
        if file_path.exists():
            with file_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        texts.append(line)
        else:
            rprint(f"[yellow][WARN] Inference file not found: {file_path}[/yellow]")
            
    if not texts:
        rprint("[yellow][WARN] No inference inputs provided. Exiting.[/yellow]")
        return

    preds = predictor.predict(texts, batch_size=max(1, infer_batch_size))
    rprint(f"[green]✅ Inference completed for {len(preds)} samples[/green]")
    for idx, row in enumerate(preds[:5], 1):
        print(f"  [{idx}] label={row.get('label')} prob={row.get('prob'):.4f} text={row.get('text')[:80]}")

    output_path = infer_output or str(Path(cfg_manager.experiment_dir) / "inference_predictions.json")
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(convert_to_serializable(preds), handle, indent=2, ensure_ascii=False)
    rprint(f"[green]✅ Inference results saved to: {out_path}[/green]")


@app.command("export")
def export(
    model_dir: str = typer.Argument(..., help="Path to the model directory to export"),
    format: str = typer.Option(
        "onnx",
        help="Export format(s): onnx, torchscript, safetensors, gguf, bitnet (comma-separated supported)",
    ),
    output_dir: Optional[str] = typer.Option(None, help="Directory to save the exported model"),
):
    """Export a trained model to various deployment formats."""
    from core.models.model_loader import ModelSaver
    from core.models.model_saver import load_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    
    rprint(f"[blue]Loading model from {model_dir}...[/blue]")
    model, tokenizer, _ = load_model(
        AutoModelForSequenceClassification,
        model_dir,
        AutoTokenizer,
        map_location="cpu",
    )
    
    out_dir = output_dir or f"{model_dir}_exported"
    rprint(f"[blue]Exporting to {format} format at {out_dir}...[/blue]")
    
    try:
        save_path = ModelSaver.export_for_deployment(
            model=model,
            tokenizer=tokenizer,
            save_dir=out_dir,
            export_format=format
        )
        rprint(f"[green]✅ Model successfully exported to: {save_path}[/green]")
        requested = [item.strip() for item in format.split(',') if item.strip()]
        if len(requested) > 1:
            rprint("[green]Produced export artifacts:[/green]")
            for item in requested:
                rprint(f"  - {Path(out_dir) / item.lower()}")
    except Exception as e:
        rprint(f"[red]❌ Export failed: {e}[/red]")
        raise typer.Exit(code=1)


@app.command("compare")
def compare(
    teacher_dir: str = typer.Argument(..., help="Path to teacher model directory"),
    student_dir: str = typer.Argument(..., help="Path to student model directory"),
    train_path: str = typer.Option("data/imdb_train.jsonl", help="Training JSONL path used for tokenizer-aligned loading"),
    val_path: str = typer.Option("data/imdb_val.jsonl", help="Validation JSONL path for comparison evaluation"),
    batch_size: int = typer.Option(8, help="Evaluation batch size"),
    max_length: int = typer.Option(128, help="Tokenizer max length"),
    output_dir: Optional[str] = typer.Option(None, help="Directory to store comparison outputs"),
    device: Optional[str] = typer.Option(None, help="Device override: cpu|cuda|mps"),
):
    """Compare teacher and student models."""
    import torch
    from data.dataloaders import get_imdb_dataloaders
    from evaluation.model_comparison import ModelComparator

    teacher_path = Path(teacher_dir)
    student_path = Path(student_dir)
    if not teacher_path.exists():
        rprint(f"[red]❌ Teacher path not found: {teacher_path}[/red]")
        raise typer.Exit(code=1)
    if not student_path.exists():
        rprint(f"[red]❌ Student path not found: {student_path}[/red]")
        raise typer.Exit(code=1)

    resolved_device = device
    if not resolved_device:
        if torch.cuda.is_available():
            resolved_device = "cuda"
        elif torch.backends.mps.is_available():
            resolved_device = "mps"
        else:
            resolved_device = "cpu"

    out_dir = output_dir or str(student_path.parent / "comparison")
    rprint(f"[blue]Loading models for comparison on {resolved_device}...[/blue]")

    try:
        comparator = ModelComparator(
            teacher_path=str(teacher_path),
            student_path=str(student_path),
            device=resolved_device,
            use_same_tokenizer=True,
        )

        _, val_loader = get_imdb_dataloaders(
            train_path=train_path,
            val_path=val_path,
            tokenizer=comparator.tokenizer,
            batch_size=max(1, batch_size),
            max_length=max(8, max_length),
        )

        teacher_results, student_results = comparator.compare_models(val_loader)
        comparator.visualize_comparison(teacher_results, student_results, save_dir=out_dir, show_plots=False)
        comparator.save_results(teacher_results, student_results, save_dir=out_dir)
        comparator.generate_report(teacher_results, student_results, save_dir=out_dir)

        rprint("[green]✅ Comparison complete[/green]")
        rprint(f"Output directory: {out_dir}")
        rprint(f"Teacher accuracy: {teacher_results.get('accuracy', 0.0):.4f}")
        rprint(f"Student accuracy: {student_results.get('accuracy', 0.0):.4f}")
    except Exception as e:
        rprint(f"[red]❌ Comparison failed: {e}[/red]")
        raise typer.Exit(code=1)


@app.command("info")
def info(
    config: str = typer.Option("configs/default.yaml", help="Path to config YAML file"),
    override: List[str] = typer.Option([], help="Config overrides in KEY=VALUE format"),
):
    """Print system state, loaded presets, and environment config."""
    overrides_dict = parse_overrides(override)
    cfg_manager = ConfigManager(config_path=config, overrides=overrides_dict)
    
    rprint("\n[bold]⚙️ Configuration Information[/bold]")
    rprint(f"Config path: [cyan]{config}[/cyan]")
    rprint(f"Experiment ID: [cyan]{cfg_manager.experiment_id}[/cyan]")
    rprint(f"Device: [cyan]{cfg_manager.device()}[/cyan]")
    rprint("\n[bold]Resolved Configuration:[/bold]")
    print(json.dumps(dict(cfg_manager.resolved_config), indent=2))
    
    try:
        rprint("\n[bold]Model Information[/bold]")
        teacher, student, tokenizer = load_models(
            cfg_manager,
            cfg_manager.device(),
            use_agent=False,
        )
        rprint("\n[Teacher Summary]")
        print(json.dumps(model_summary(teacher), indent=2))
        rprint("\n[Student Summary]")
        print(json.dumps(model_summary(student), indent=2))
    except Exception as e:
        rprint(f"[yellow]Could not load models for summary: {e}[/yellow]")


@app.command("smoke")
def smoke(
    epochs: int = typer.Option(1, help="Smoke test epochs"),
    batch_size: int = typer.Option(2, help="Smoke test batch size"),
    seq_len: int = typer.Option(32, help="Smoke test sequence length"),
    train_samples: int = typer.Option(16, help="Smoke test train samples"),
    val_samples: int = typer.Option(8, help="Smoke test val samples"),
    vocab_size: int = typer.Option(128, help="Smoke test vocab size"),
    hidden_size: int = typer.Option(64, help="Smoke test hidden size"),
):
    """Run a tiny CPU-only synthetic training smoke test."""
    import time
    import torch
    import torch.nn as nn
    
    rprint("\n[bold blue]" + "="*70 + "[/bold blue]")
    rprint("[bold blue]SMOKE MODE: No model downloads (CPU synthetic test)[/bold blue]")
    rprint("[bold blue]" + "="*70 + "[/bold blue]\n")

    torch.manual_seed(42)
    device = torch.device("cpu")

    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(vocab_size, hidden_size)
            self.head = nn.Linear(hidden_size, vocab_size)

        def forward(self, ids: torch.Tensor) -> torch.Tensor:
            return self.head(self.embed(ids))

    model = TinyLM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

    def make_data(n: int) -> torch.Tensor:
        return torch.randint(0, vocab_size, (n, seq_len), dtype=torch.long, device=device)

    train_ids = make_data(train_samples)
    val_ids = make_data(val_samples)

    def iter_batches(data: torch.Tensor, bs: int):
        for i in range(0, data.size(0), bs):
            yield data[i:i + bs]

    started = time.time()
    best_val = float("inf")
    train_history: List[float] = []
    val_history: List[float] = []

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        steps = 0
        for x in iter_batches(train_ids, batch_size):
            logits = model(x)
            shift_logits = logits[:, :-1, :].contiguous().view(-1, vocab_size)
            shift_labels = x[:, 1:].contiguous().view(-1)
            loss = nn.functional.cross_entropy(shift_logits, shift_labels)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item())
            steps += 1

        train_loss = epoch_loss / max(1, steps)
        train_history.append(train_loss)

        model.eval()
        with torch.no_grad():
            val_epoch = 0.0
            val_steps = 0
            for x in iter_batches(val_ids, batch_size):
                logits = model(x)
                shift_logits = logits[:, :-1, :].contiguous().view(-1, vocab_size)
                shift_labels = x[:, 1:].contiguous().view(-1)
                vloss = nn.functional.cross_entropy(shift_logits, shift_labels)
                val_epoch += float(vloss.item())
                val_steps += 1
            val_loss = val_epoch / max(1, val_steps)
            val_history.append(val_loss)
            best_val = min(best_val, val_loss)

        print(f"[SMOKE] epoch={epoch+1}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

    out_dir = Path("experiments") / "smoke_no_model"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "status": "ok",
        "mode": "smoke_no_model",
        "device": str(device),
        "epochs": epochs,
        "batch_size": batch_size,
        "seq_len": seq_len,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "train_loss": train_history,
        "val_loss": val_history,
        "best_val_loss": best_val,
        "duration_sec": round(time.time() - started, 3),
    }

    summary_path = out_dir / "smoke_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    rprint("\n[bold green]✅ Smoke test complete (no model downloads)[/bold green]")
    rprint(f"Summary: [cyan]{summary_path}[/cyan]")


if __name__ == "__main__":
    app()
