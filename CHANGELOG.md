# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.6] - 2026-05-12
### Added
- GitHub Actions CI workflow for linting and tests.
- GitHub Actions Docs workflow for deploying MkDocs to GitHub Pages.
- Issue templates for bug reports and feature requests.
- `balanced.yaml` preset configuration matching the Kaggle SST-2 distillation example.
- Full ONNX export support using `optimum[onnxruntime]`.

## [0.2.5] - 2026-05-12
### Added
- Multi-stage distillation framework with `DistillationToolkit`.
- Support for Text, Code, Vision, and Multimodal models via `AdapterRegistry`.
- Dynamic learning rate (auto_lr) and Automatic Gradient Clipping (AGC).
- Hardware-agnostic preflight checks.
- Post-Training Quantization (PTQ) support.
