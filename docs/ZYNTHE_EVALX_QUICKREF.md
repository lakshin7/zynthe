# Zynthe EvalX - Quick Reference Card

## 🎯 Core Metrics at a Glance

### Knowledge Transfer Quality
```python
from evaluation.metrics_extended import compute_extended_metrics

metrics = compute_extended_metrics(teacher_logits, student_logits)
```

| Metric | Good | Excellent | Outstanding |
|--------|------|-----------|-------------|
| KL Divergence | < 1.0 | < 0.5 | < 0.1 |
| Prediction Agreement | > 90% | > 95% | > 98% |
| Confidence Correlation | > 0.7 | > 0.9 | > 0.95 |

### Deployment Readiness
```python
from evaluation.metrics_extended import CompressionAwareScore, DistillationEfficacyIndex

cas = CompressionAwareScore.compute_cas(acc, t_params, s_params, t_lat, s_lat)
dei = DistillationEfficacyIndex.compute_dei(t_acc, s_acc, t_params, s_params)
```

| Score | Interpretation | Action |
|-------|----------------|--------|
| **DEI > 1.5** | 🌟 Outstanding | Deploy immediately |
| **DEI 1.0-1.5** | ✅ Excellent | Deploy with confidence |
| **DEI 0.8-1.0** | ✓ Good | Consider minor tuning |
| **CAS > 0.5** | 🌟 Excellent candidate | Production-ready |
| **CAS 0.3-0.5** | ✅ Very good | Deploy confidently |
| **CAS 0-0.3** | ✓ Good | Monitor in production |

---

## 🚀 Quick Usage

### 1. Test Extended Metrics (On Your Model)
```bash
python test_extended_metrics.py
```

### 2. Dual Evaluation (Teacher + Student)
```python
from evaluation.evaluator_extended import DualEvaluator

evaluator = DualEvaluator(teacher, student, val_loader, device)
results = evaluator.evaluate(profile_performance=True)

# Results structure:
{
    'teacher': {'accuracy', 'metrics', 'params'},
    'student': {'accuracy', 'metrics', 'params'},
    'extended_metrics': {'kl_divergence', 'js_divergence', ...},
    'dei': {'dei', 'efficiency_rating', ...},
    'cas': {'cas', 'efficiency_score', ...},
    'performance': {'speedup', 'latency_reduction_pct', ...}
}
```

### 3. Rank Multiple Students
```python
from evaluation.metrics_extended import CompressionAwareScore

models = [
    {'name': 'Student A', 'accuracy': 0.94, 'params': 80e6, 'latency': 30},
    {'name': 'Student B', 'accuracy': 0.91, 'params': 15e6, 'latency': 12},
]

ranked = CompressionAwareScore.rank_models(models)
# Returns sorted by CAS (highest first)
```

---

## 📊 Your Current Results

### Experiment: `20251023T175322Z_5285cff4`

| Component | Value | Status |
|-----------|-------|--------|
| **Accuracy Retention** | 98.9% | 🌟 Outstanding |
| **Compression** | 1.52x | ✅ Good |
| **Speedup** | 1.58x | ✅ Good |
| **KL Divergence** | 0.0283 | 🌟 Excellent |
| **DEI Score** | 1.5763 | 🌟 Outstanding |
| **CAS Score** | 0.3113 | ✅ Very Good |

**Verdict: Production-Ready** ✅

---

## 🔥 Advanced Features

### Performance Profiling
```python
from evaluation.metrics_extended import PerformanceProfiler

profile = PerformanceProfiler.profile_inference(
    model, input_ids, attention_mask, device, num_runs=100
)

print(f"Latency: {profile['mean_latency_ms']:.2f} ms")
print(f"P95: {profile['p95_latency_ms']:.2f} ms")
print(f"Throughput: {profile['throughput_samples_per_sec']:.1f} samples/sec")
```

### Curriculum Evaluation
```python
from evaluation.evaluator_extended import CurriculumEvaluator

evaluator = CurriculumEvaluator(model, device)
results = evaluator.evaluate_by_difficulty({
    'easy': easy_loader,
    'medium': medium_loader,
    'hard': hard_loader
})

print(f"Robustness: {results['summary']['robustness_score']:.4f}")
```

---

## 💡 Tuning Recommendations

### If KL Divergence > 1.0
```yaml
distillation:
  temperature: 3.0      # Increase (was 2.0)
  alpha: 0.6            # More weight on teacher (was 0.5)
```

### If Prediction Agreement < 90%
```yaml
train:
  epochs: 5             # More epochs
  lr: 2e-5              # Lower learning rate
distillation:
  temperature: 2.5      # Softer distributions
```

### If DEI < 1.0
- Consider smaller student model
- Increase distillation temperature
- Add feature-based distillation

### If CAS < 0.3
- Optimize student architecture
- Apply quantization (PTQ/QAT)
- Profile and optimize inference

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `evaluation/metrics_extended.py` | Extended metrics implementation |
| `evaluation/evaluator_extended.py` | Dual & curriculum evaluators |
| `test_extended_metrics.py` | Test script |
| `docs/ZYNTHE_EVALX_PHASE_A.md` | Full documentation |

---

## 🎓 Formulas

### DEI (Distillation Efficacy Index)
```
DEI = (Acc_student / Acc_teacher) × (Params_teacher / Params_student) × (1 + bonus)
```
Where bonus = prediction_agreement × 0.1

### CAS (Compression-Aware Score)
```
CAS = α·Accuracy - β·SizeRatio - γ·LatencyRatio
```
Defaults: α=0.6, β=0.2, γ=0.2

### KL Divergence
```
KL(P || Q) = Σ P(x) log(P(x) / Q(x))
```
Scaled by temperature² for distillation

---

## 🚀 Next Features (Coming Soon)

### Phase B: Benchmarking
- Cross-dataset evaluation
- Energy efficiency measurement
- Auto-report generation

### Phase C: Visual Intelligence
- Layer-wise similarity maps
- 3D feature projections
- Attention heatmaps
- Interactive dashboards

---

## ⚡ Pro Tips

1. **Always profile performance** - CAS needs accurate latency
2. **Track KL divergence during training** - Early indicator of quality
3. **Compare multiple students** - Use CAS ranking
4. **Test across difficulties** - Use curriculum evaluator
5. **Save DEI/CAS to experiment** - Track over time

---

## 📞 Integration Points

### With Training
```python
# In trainer.py evaluate()
from evaluation.metrics_extended import compute_extended_metrics

extended = compute_extended_metrics(teacher_logits, student_logits)
metrics['kl_divergence'] = extended['kl_divergence']
metrics['prediction_agreement'] = extended['prediction_agreement']
```

### With Reporting
```python
# In report generation
from evaluation.metrics_extended import DistillationEfficacyIndex

dei = DistillationEfficacyIndex.compute_dei(...)
report['dei_score'] = dei['dei']
report['efficiency_rating'] = dei['efficiency_rating']
```

---

## 🎉 Your Achievement

Your distillation:
- 🌟 **Outstanding DEI** (1.5763)
- ✅ **Very Good CAS** (0.3113)
- 🌟 **Excellent KL** (0.0283)
- ✅ **Production-Ready**

**Congratulations! Your model exceeds industry standards.** 🚀
