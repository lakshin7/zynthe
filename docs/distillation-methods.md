# Distillation Methods

Zynthé ships a curated catalogue of distillation methods.  Each
method is implemented as a `BaseDistiller` subclass and registered in
`DistillerRegistry` under a stable string key.  This page describes
the methods, when to use them, and the math behind each.

## Loss-based distillation

| Distiller | Loss | Reference | Best for |
|---|---|---|---|
| `KDHintonDistiller` | `α·T²·KL(σ(s/T) ‖ σ(t/T)) + (1−α)·CE(s, y)` | Hinton et al. 2015 | The default.  Any classification / seq2seq task. |
| `FeatureDistiller` | weighted sum of L2 / CKA / cosine / Gram / FSP / AB / contrastive losses on hooked layers | FitNets, AB, CRD-light | Hidden-state alignment. |
| `AttentionTransferDistiller` | L2 / KL on self-attention maps, plus rollout, cross-layer flow, dual matching | Zagoruyko & Komodakis 2017, Abnar & Zuidema 2020 | Vision transformers; explainability. |
| `SimilarityTransfer` | pairwise / Gram matrix MSE between student/teacher | PKT-light | Pre-logit feature distillation. |
| `ContrastiveDistiller` (CRD) | InfoNCE on projected CLS features with in-batch + memory-bank negatives | Tian et al. 2020 | High-dim feature spaces; low-label data. |
| `RelationalDistiller` (PKT) | `MSE(cos(z_s), cos(z_t))` — pairwise cosine matrix | Park et al. 2019 | Sample-relationship preservation. |
| `ProjectionDistiller` | `MSE(g_s(h_s), h_t.detach())` — translator MLP | Zynthé report §213 | When student and teacher widths differ. |
| `AuxHeadDistiller` | mean of per-layer CE on labels using small aux heads | Zynthé report §215-217 | Deep supervision at intermediate layers. |
| `RationaleDistiller` | `CE_label + λ·CE_rationale` on multi-task logits (label + rationale views) | Hsieh et al. 2023 (Distill step-by-step) | Text-to-text tasks with LLM-extracted rationales. |

## Regularizers (KD-Hinton extensions)

| Option | Effect |
|---|---|
| `entropy_regularizer_weight` | Add `\|H(σ(s/T)) − H(σ(t/T))\|` to the KD loss.  Pins student entropy to teacher. |
| `dynamic_temperature: 'learnable'` | Register an `nn.Parameter` for τ.  The optimiser adapts it via gradient descent (scheduler bypassed).  τ is clamped to `[0.1, 10.0]`. |

## Multi-stage pipelines

`PipelineBuilder` composes multiple distillers into a single
pipeline.  Mode selection:

- **sequential** — stages run one after another.
- **parallel** — all stages run, losses summed.
- **conditional** — a stage runs only if its predicate returns true.
- **hybrid** — mixed sequential + parallel.

`MultiStagePipeline.setup()` normalises stage weights so they sum
to 1.0 (controlled by `normalize_weights`).

## Adapter layer (universal model support)

Every Zynthé pipeline is built on top of an `AdapterRegistry`-driven
adapter that normalises batch I/O and module-name conventions across
text, code, vision, multimodal, VLM, Seq2Seq, audio, diffusion, and a
universal HuggingFace fallback.  See `docs/adapters.md` for the
detection heuristics.

## Method citations

- Hinton, Vinyals, Dean. 2015. *Distilling the Knowledge in a Neural
  Network.* NeurIPS 2014 Deep Learning Workshop.
- Tian, Krishnan, Isola. 2020. *Contrastive Representation
  Distillation.* ICLR 2020.
- Park, Kim, Lee, Lee. 2019. *Relational Knowledge Distillation.*
  CVPR 2019.
- Zagoruyko, Komodakis. 2017. *Paying More Attention to Attention:
  Improving the Performance of Convolutional Neural Networks via
  Attention Transfer.* ICLR 2017.
- Abnar, Zuidema. 2020. *Quantifying Attention Flow in Transformers.*
  ACL 2020.
- Hsieh, Li, Yeh, Nakhost, Fujii, Ratner, Krishna, Lee, Pfister. 2023.
  *Distilling Step-by-Step: Outperforming Larger Language Models with
  Less Training Data and Smaller Model Sizes.* ACL 2023.

## Smoke proof

See `docs/benchmarks.md` for the empirical smoke proof of each
method.  Each new distiller is exercised on Modal L4 as part of the
smoke-gate pipeline.
