"""Example: run validation and determinism checks for SafeCausalLMTrainer.

This is a programmatic smoke entrypoint (no CLI wiring).
"""

from __future__ import annotations

import json
from pathlib import Path


from core.config.config_manager import ConfigManager
from core.distillers.causal_lm import run_checkpoint_stress_tests
from core.models.model_loader import load_models
from data.dataloaders import create_dataloaders


def main(config_path: str = "configs/causal_lm_core_minimal.yaml") -> None:
    cm = ConfigManager(config_path=config_path)
    cfg = cm.resolved_config
    device = cm.device()

    teacher, student, tokenizer = load_models(cm, device, use_agent=False)
    train_loader, val_loader = create_dataloaders(cfg, tokenizer)

    from core.distillers.causal_lm import SafeCausalLMTrainer

    trainer = SafeCausalLMTrainer(
        teacher=teacher,
        student=student,
        tokenizer=tokenizer,
        config=cfg,
        device=device,
        experiment_dir=cm.experiment_dir,
    )

    ckpt_report = run_checkpoint_stress_tests(device=device)
    print("checkpoint_stress:", json.dumps({
        "all_passed": ckpt_report.all_passed,
        "scenarios": [{"name": s.name, "passed": s.passed} for s in ckpt_report.scenarios],
    }, indent=2))

    det_report = trainer.run_determinism_verification(train_loader, compare_steps=5, tolerance=1e-7)
    print("determinism:", json.dumps(det_report.__dict__, indent=2))

    trainer.fit(train_loader, val_loader)
    health = trainer.build_training_health_report()
    print("health:", json.dumps(health.to_dict(), indent=2))

    report_path = Path(cm.experiment_dir) / "logs" / "causal_lm_validation_summary.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "checkpoint_stress": {
                    "all_passed": ckpt_report.all_passed,
                    "scenarios": [s.__dict__ for s in ckpt_report.scenarios],
                },
                "determinism": det_report.__dict__,
                "training_health": health.to_dict(),
            },
            handle,
            indent=2,
        )
    print(f"saved: {report_path}")


if __name__ == "__main__":
    main()
