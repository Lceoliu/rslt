# Sign Language Translation - Abstract Pipeline

## Project Goal
Translate continuous sign language videos (represented as 2D keypoint sequences) into natural language text using a multi-stage deep learning pipeline combining spatial-temporal graph convolution, transformer encoding, and large language models.

---

## Input & Output

- **Input**: COCO WholeBody 2D keypoints `[T, 133, 3]`
  - 133 keypoints: body(17) + face(68) + hands(42) + full(6)
  - 3 channels: x, y, confidence

- **Output**: Natural language text describing the sign language content

---

## Network Architecture

### 1. Chunking & Normalization
```
Raw Video [T frames]
  ↓ [sliding window: size=32, stride=16]
Chunks [N, 32, 133, 3]
  ↓ [normalize per body part]
Normalized [N, 32, 133, 2]  # confidence dropped
```

### 2. Multi-Part Graph Convolutional Network
```
Input: [N, 32, 133, 2]
  ↓ [split into 5 body parts]
  ├─ body [17 joints] ──→ GCN ──→ [N, 32, 17, 256]
  ├─ face [68 joints] ──→ GCN ──→ [N, 32, 68, 256]
  ├─ left_hand [21]   ──→ GCN ──→ [N, 32, 21, 256]
  ├─ right_hand [21]  ──→ GCN ──→ [N, 32, 21, 256]
  └─ fullbody [133]   ──→ GCN ──→ [N, 32, 133, 256]
  ↓ [stack parts]
Output: [N, 5_parts, 32, 256]
```
**Key**: Each body part has an independent Uni-GCN backbone with temporal convolution (kernel=5) and adaptive graph learning.

### 3. Chunk Transformer Encoder
```
Input: [N, 5, 32, 256]
  ↓ [concat parts: 5×256=1280]
Features: [N, 32, 1280]
  ↓ [project to LLM dim]
Projected: [N, 32, 1024]
  ↓ [3-layer transformer + cross-attention pooling]
Output: [N, 10_tokens, 1024]
```
**Key**: Compress each 32-frame chunk into 10 fixed-length tokens matching LLM embedding dimension.

### 4. LLM Wrapper (Decoder-only)
```
Visual Tokens: [N, 10, 1024]
  ↓ [build sequence with special tokens]
Prefix: <BOC>[10 tokens]<EOC> ... <BOC>[10 tokens]<EOC>
  ↓ [concatenate with text]
Full Sequence: [prefix] + <BOT> + text_tokens + <EOT>
  ↓ [autoregressive forward]
Loss: Cross-entropy (prefix masked, text supervised)
```
**Key**:
- Special tokens `<BOC>/<EOC>/<BOT>/<EOT>` are trainable embeddings
- Only text portion contributes to loss (prefix labels = -100)
- Uses pre-trained causal LLM (e.g., Qwen2.5-0.5B)

---

## Training Strategy

### Two-Stage Approach

**Stage 1: Visual Pre-training (10 epochs)**
- Freeze LLM parameters
- Train visual encoder (GCN + Transformer) with learning rate `5e-5`
- Add contrastive loss (weight=0.5) between visual tokens and text embeddings
- Use all chunks (`min_reserved_ratio=1.0`)

**Stage 2: Joint Fine-tuning (15 epochs)**
- Unfreeze LLM
- Differential learning rates: visual `1e-5`, LLM `1e-6`
- Disable contrastive loss
- Random chunk trimming (`min_reserved_ratio=0.6`) for regularization
- Resume from Stage 1 checkpoint (load weights only, reset optimizer)

---

## Key Technologies

- **Framework**: PyTorch + DeepSpeed (ZeRO Stage 2, bf16)
- **Spatial-Temporal Modeling**: Uni-GCN with adaptive adjacency learning
- **Sequence Compression**: Transformer encoder with learnable query pooling
- **Language Generation**: Causal LLM with visual prefix conditioning
- **Training Optimization**: Two-stage strategy with contrastive pre-training

---

## Performance Considerations

- **Chunking**: Fixed-size windows (32 frames) enable batch processing and parallel GCN computation
- **Multi-part design**: Body parts processed independently, capturing fine-grained spatial patterns
- **Token compression**: 32 frames → 10 tokens reduces LLM input length by ~70%
- **Masking**: Explicit masks for padded chunks/frames ensure correct gradient flow
- **Differential LRs**: Prevent catastrophic forgetting of pre-trained LLM during fine-tuning

---

**Full Technical Details**: See `pipeline.md`
