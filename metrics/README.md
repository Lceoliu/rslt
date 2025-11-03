# Model Diagnostic Metrics

Complete diagnostic toolkit for analyzing sign language translation model performance.

## Overview

This module provides comprehensive metrics to identify bottlenecks in the training pipeline:

- **Visual-Text Retrieval**: Tests if visual features have semantic meaning
- **Visual Ablation**: Checks if LLM actually uses visual input
- **Error Analysis**: Categorizes prediction errors by type
- **Data Quality**: Validates dataset integrity
- **Special Token Analysis**: Checks if special tokens learned meaningful representations

## Quick Start

### Run Full Diagnosis

```bash
# Basic usage
python diagnose_model.py \
    --checkpoint runs/20251003_125814/checkpoints \
    --config configs/train_default.yaml

# Save report to file
python diagnose_model.py \
    --checkpoint runs/20251003_125814/checkpoints \
    --config configs/train_default.yaml \
    --output diagnosis_report.txt

# Adjust sample count
python diagnose_model.py \
    --checkpoint runs/20251003_125814/checkpoints \
    --config configs/train_default.yaml \
    --max-samples 500
```

### Use Individual Metrics

```python
from metrics import (
    compute_retrieval_metrics,
    test_visual_importance,
    analyze_prediction_errors,
)

# Visual-Text retrieval
retrieval_results = compute_retrieval_metrics(model, val_loader, max_samples=500)
print(f"R@1: {retrieval_results['retrieval_r@1']:.2f}%")

# Visual ablation test
ablation_results = test_visual_importance(model, val_loader, max_samples=100)
print(f"Δ(Normal-Random): {ablation_results['delta_normal_random']:.3f}")

# Error analysis
error_results = analyze_prediction_errors(model, val_loader, max_samples=200)
print(f"Total fail: {error_results['total_fail_ratio']:.1f}%")
```

## Metrics Details

### 1. Visual-Text Retrieval (`visual_text_retrieval.py`)

**Purpose**: Test if visual encoder produces semantically meaningful features.

**Method**:
- Extract visual features (average pooled chunk tokens)
- Extract text embeddings (from LLM vocabulary)
- Compute cosine similarity matrix
- Measure retrieval accuracy (R@1, R@5, R@10)

**Interpretation**:
- ✅ R@1 > 30%: Visual features are good
- ⚠️ R@1 = 10-30%: Features are weak
- ❌ R@1 < 10%: Features have no semantic meaning

### 2. Visual Ablation (`visual_ablation.py`)

**Purpose**: Check if LLM actually uses visual information.

**Method**: Compare LLM loss with:
- Normal visual features
- Random visual features (same distribution)
- Zero visual features

**Interpretation**:
- ✅ Δ > 2.0: LLM strongly uses visual input
- ⚠️ Δ = 0.5-2.0: LLM partially uses visual
- ❌ Δ < 0.5: LLM ignores visual input

### 3. Error Analysis (`error_analysis.py`)

**Purpose**: Identify what type of errors the model makes.

**Categories**:
1. **Total Fail** (BLEU-1=0): Completely wrong predictions
2. **Keyword Only** (BLEU-1>0, BLEU-4=0): Has words but no phrases
3. **Partial Correct** (BLEU-4>0 but <20): Some phrases correct
4. **Good** (BLEU-4≥20): Mostly correct

**Interpretation**:
- High "Total Fail" → Visual encoder problem
- High "Keyword Only" → LLM needs more training or larger capacity
- High "Partial Correct" → Close to success, tune hyperparameters

### 4. Data Quality (`data_quality.py`)

**Purpose**: Ensure dataset is valid and diverse.

**Checks**:
- Unique text ratio (should be >80%)
- Vocabulary size (should be >500)
- Text length distribution
- Pose length statistics
- Oracle BLEU (should be ~100)

### 5. Special Token Analysis (`special_tokens.py`)

**Purpose**: (Deprecated) Earlier revisions used dedicated <BOC>/<EOC>/<BOT>/<EOT> tokens.

**Analysis**: The current prompt-prefix design no longer exposes those special tokens; the helper now reports that the check is not applicable.

**Expected**: N/A

## Example Report Output

```
================================================================================
  DIAGNOSTIC SUMMARY
================================================================================

⚠ CRITICAL ISSUES FOUND:
  1. LLM is not using visual information
  2. High rate of complete failures

💡 RECOMMENDATIONS:
  1. Check label masking in LLM_wrapper.py
  2. Increase visual encoder training
  3. Visual encoder is not extracting useful features

================================================================================
```

## Interpreting Results

### Scenario 1: Low BLEU but Good Retrieval

**Symptoms**:
- BLEU-4 = 7.4
- R@1 = 35% ✓
- Δ(Normal-Random) = 2.5 ✓

**Diagnosis**: Visual features are good, but LLM is not generating fluent text.

**Solutions**:
- Train LLM longer
- Use a larger LLM
- Reduce learning rate for LLM

### Scenario 2: Low BLEU and Poor Retrieval

**Symptoms**:
- BLEU-4 = 7.4
- R@1 = 8% ❌
- Δ(Normal-Random) = 0.3 ❌

**Diagnosis**: Visual encoder is failing to extract semantic features.

**Solutions**:
- Increase contrastive loss weight
- Improve GCN architecture
- Add more temporal modeling
- Check if data augmentation is too aggressive

### Scenario 3: Label Masking Bug

**Symptoms**:
- Train loss = 0.5-1.0 (too low!)
- Δ(Normal-Random) = 0.2 ❌
- Special tokens all similar (>0.9)

**Diagnosis**: Model is learning to predict fixed special tokens instead of text.

**Solutions**:
- Fix `_prefix_labels_from_ids()` to mask all prefix tokens
- Retrain from scratch

## Dependencies

- PyTorch
- sacrebleu (for BLEU computation)
- numpy

Install with:
```bash
pip install sacrebleu
```

## FAQ

**Q: How long does diagnosis take?**
A: ~5-10 minutes for 200 samples on a single GPU.

**Q: Can I run on CPU?**
A: Yes, use `--device cpu`, but it will be slower.

**Q: What if I don't have a checkpoint?**
A: You need a trained checkpoint. Use at least 1000 training steps.

**Q: Can I compare multiple checkpoints?**
A: Yes, run diagnosis on each and compare the reports.

## Integration with Training

Add diagnosis to your training script:

```python
# training/train_deepspeed.py

if global_step % 5000 == 0:
    # Run quick diagnosis
    from metrics import test_visual_importance
    ablation = test_visual_importance(engine.module, val_loader, max_samples=50)
    writer.add_scalar('diagnosis/delta_visual', ablation['delta_normal_random'], global_step)
```

## Citation

If you use these diagnostics in your research, please cite:
```
@misc{rslt_diagnostics,
  title={Diagnostic Metrics for Sign Language Translation},
  author={Your Name},
  year={2025}
}
```
