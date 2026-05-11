"""Command-line interface for Zynthé."""

from __future__ import annotations

import argparse
import logging
import sys
from typing import List, Optional

from zynthe.app.runtime import RuntimeOptions, UnifiedTrainingRuntime
from zynthe.core.config.config_manager import ConfigManager
from zynthe.core.distillers.toolkit import DistillationToolkit
from zynthe.core.preflight.analyser import PreflightAnalyzer
from zynthe.evaluation.evaluator import Evaluator
from zynthe.evaluation.model_comparison import ModelComparator

logger = logging.getLogger(__name__)


def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _load_config(config_path: Optional[str]) -> dict:
    if not config_path:
        return ConfigManager().resolved_config
    return ConfigManager(config_path=config_path).resolved_config


def _resolve_overrides(args: argparse.Namespace) -> dict:
    overrides: dict = {}
    for pair in getattr(args, "override", []) or []:
        if "=" not in pair:
            raise ValueError(f"Override must be key=value, got: {pair}")
        key, value = pair.split("=", 1)
        overrides[key] = value
    return overrides


def cmd_train(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose)
    overrides = _resolve_overrides(args)
    options = RuntimeOptions(
        config_path=args.config,
        overrides=overrides,
        save_model_dir=args.output_dir,
    )
    runtime = UnifiedTrainingRuntime()
    result = runtime.run(options)
    if result.success:
        logger.info("Training completed successfully: %s", result.experiment_dir)
        return 0
    logger.error("Training failed: %s", result.errors)
    return 1


def cmd_preflight(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose)
    cfg = _load_config(args.config)
    overrides = _resolve_overrides(args)
    if overrides:
        cfg.update(overrides)
    analyzer = PreflightAnalyzer(config=cfg)
    report = analyzer.run_preflight(verbose=True)
    decision = report.get("decision", {})
    if decision.get("can_proceed", False):
        logger.info("Preflight check PASSED — ready to proceed.")
        return 0
    logger.error("Preflight check FAILED — blockers found.")
    for blocker in report.get("blockers", []):
        logger.error("  - %s", blocker)
    return 1


def cmd_evaluate(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose)
    logger.info("Evaluate command not yet fully implemented.")
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    _setup_logging(args.verbose)
    logger.info("Compare command not yet fully implemented.")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zynthe",
        description="Zynthé — Universal Knowledge Distillation Toolkit",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--override",
        "-o",
        action="append",
        default=[],
        help="Config override in key=value format (repeatable)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Run training / distillation")
    train_parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to save experiment outputs",
    )
    train_parser.set_defaults(func=cmd_train)

    preflight_parser = subparsers.add_parser("preflight", help="Run preflight checks")
    preflight_parser.set_defaults(func=cmd_preflight)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a model")
    eval_parser.set_defaults(func=cmd_evaluate)

    compare_parser = subparsers.add_parser("compare", help="Compare teacher vs student")
    compare_parser.set_defaults(func=cmd_compare)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
