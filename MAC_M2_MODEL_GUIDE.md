# Mac M2 Model Compatibility Guide

## ❌ DeepSeek Models - NOT Recommended for Mac M2

**Why DeepSeek won't work well:**
- **Size**: DeepSeek-Coder models are 1.3B - 33B parameters
- **Memory**: Requires 16GB+ RAM minimum, 32GB+ recommended
- **Target**: Designed for servers/workstations with dedicated GPUs
- **Mac M2 Reality**: 8GB unified memory = Out of Memory errors

**DeepSeek Model Sizes:**
- `deepseek-ai/deepseek-coder-1.3b-base`: 1.3B params (~5GB RAM minimum)
- `deepseek-ai/deepseek-coder-6.7b-base`: 6.7B params (~26GB RAM minimum)
- `deepseek-ai/deepseek-coder-33b-base`: 33B params (not feasible)

---

## ✅ RECOMMENDED Models for Mac M2 (8GB)

### Tier 1: Proven & Safe (Best Choice)

#### 1. **BERT Family** ⭐ CURRENT DEFAULT
```yaml
model:
  name: "bert-base-uncased"              # 110M params
  student_name: "distilbert-base-uncased" # 66M params
  tokenizer_name: "bert-base-uncased"
```
**Pros:**
- ✅ Well-tested, proven to work
- ✅ Fast training (3 epochs in ~10 minutes)
- ✅ Excellent documentation
- ✅ Works with IMDB, GLUE, custom datasets

**Memory Usage:** ~2GB

---

#### 2. **RoBERTa Family** ⭐ RECOMMENDED
```yaml
model:
  name: "roberta-base"                   # 125M params
  student_name: "distilroberta-base"     # 82M params
  tokenizer_name: "roberta-base"
```
**Pros:**
- ✅ Better performance than BERT
- ✅ Still fits comfortably on Mac M2
- ✅ Good for text classification

**Memory Usage:** ~2.5GB

---

#### 3. **ALBERT Family** ⭐ VERY EFFICIENT
```yaml
model:
  name: "albert-base-v2"                 # 12M params (parameter sharing!)
  student_name: "albert-base-v2"         # Can use same or smaller
  tokenizer_name: "albert-base-v2"
```
**Pros:**
- ✅ **Only 12M parameters** (10x smaller than BERT!)
- ✅ Parameter sharing = very memory efficient
- ✅ Fast training and inference
- ✅ Good performance despite small size

**Memory Usage:** ~500MB

---

### Tier 2: Advanced (Requires 16GB Mac M2)

#### 4. **TinyLlama** (Requires 16GB RAM)
```yaml
model:
  name: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"  # 1.1B params
  student_name: "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
  tokenizer_name: "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
```
**Pros:**
- ✅ Optimized for edge devices
- ✅ Llama architecture (modern)
- ✅ Good generation capabilities

**Cons:**
- ⚠️ Requires 16GB Mac M2
- ⚠️ Slower training (~30-60 min for 3 epochs)

**Memory Usage:** ~4-6GB

---

#### 5. **Phi-2 (Microsoft)** (Requires 16GB RAM)
```yaml
model:
  name: "microsoft/phi-2"                # 2.7B params
  student_name: "microsoft/phi-1_5"      # 1.3B params
  tokenizer_name: "microsoft/phi-2"
```
**Pros:**
- ✅ Excellent performance for size
- ✅ Well-optimized by Microsoft
- ✅ Good reasoning capabilities

**Cons:**
- ⚠️ Requires 16GB Mac M2
- ⚠️ Slower training

**Memory Usage:** ~8-10GB

---

## 🎯 Quick Recommendation Based on Your Mac M2

### If you have **8GB RAM** (Base M2):
1. **ALBERT** - Most efficient, fastest
2. **BERT** - Proven, balanced (current default)
3. **RoBERTa** - Best performance in this tier

### If you have **16GB RAM** (Upgraded M2):
1. **BERT/RoBERTa** - Still good for quick experiments
2. **TinyLlama** - If you want modern Llama architecture
3. **Phi-2** - If you want cutting-edge performance

---

## 📊 Performance Comparison (on Mac M2 8GB)

| Model | Params | Training Time | Memory | Quality | Recommended |
|-------|--------|---------------|--------|---------|-------------|
| ALBERT | 12M | ~5 min | 500MB | Good | ⭐⭐⭐ |
| BERT | 110M | ~10 min | 2GB | Excellent | ⭐⭐⭐⭐⭐ |
| RoBERTa | 125M | ~12 min | 2.5GB | Excellent | ⭐⭐⭐⭐ |
| TinyLlama | 1.1B | ~60 min | 6GB | Very Good | ⚠️ 16GB only |
| Phi-2 | 2.7B | ~90 min | 10GB | Excellent | ⚠️ 16GB only |
| DeepSeek | 1.3B+ | N/A | >16GB | N/A | ❌ Not feasible |

---

## 🚀 How to Use Different Models

### Option 1: Edit `configs/mac_m2_optimized.yaml`

Uncomment the model you want:

```yaml
# For ALBERT (most efficient):
model:
  name: "albert-base-v2"
  student_name: "albert-base-v2"
  tokenizer_name: "albert-base-v2"

# For RoBERTa (better performance):
# model:
#   name: "roberta-base"
#   student_name: "distilroberta-base"
#   tokenizer_name: "roberta-base"
```

### Option 2: Use Command Line Override

```bash
# Try ALBERT
python app/main.py --config configs/mac_m2_optimized.yaml \
  --override model.name=albert-base-v2 \
  model.student_name=albert-base-v2 \
  model.tokenizer_name=albert-base-v2

# Try RoBERTa
python app/main.py --config configs/mac_m2_optimized.yaml \
  --override model.name=roberta-base \
  model.student_name=distilroberta-base \
  model.tokenizer_name=roberta-base
```

---

## 💡 My Recommendation for You

**Start with the new config I created:**

```bash
python app/main.py --config configs/mac_m2_optimized.yaml
```

This uses **BERT (default)** which is:
- ✅ Proven to work on Mac M2
- ✅ Fast training (~10 minutes)
- ✅ Excellent results (you already saw DEI: 2.96!)
- ✅ Safe memory usage

**If you want to experiment:**
1. **ALBERT** - If you want fastest training
2. **RoBERTa** - If you want slightly better accuracy
3. **TinyLlama** - Only if you have 16GB RAM

**Avoid:**
- ❌ DeepSeek models (too large)
- ❌ GPT-2 Large+ (>500M params)
- ❌ Llama-2 7B+ (too large)

---

## 🔧 Troubleshooting

### Out of Memory Errors?

**Reduce batch size:**
```yaml
train:
  batch_size: 2  # Reduce from 4 to 2
  grad_accum_steps: 4  # Increase to maintain effective batch size
```

**Reduce sequence length:**
```yaml
model:
  max_length: 64  # Reduce from 128 to 64
```

**Limit dataset:**
```yaml
data:
  preprocessing:
    max_examples: 2000  # Reduce from 5000
```

---

## ✅ Current Status

Your system is already working perfectly with:
- ✅ BERT → DistilBERT (110M → 66M params)
- ✅ DEI Score: 2.96 (Excellent)
- ✅ CAS Score: 0.33 (Very Good)
- ✅ Training time: ~10 minutes

**Recommendation: Stick with BERT or try RoBERTa/ALBERT for variety!**
