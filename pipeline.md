# Sign Language Translation Pipeline

## Overview
- **Goal**: Stream chunked COCO WholeBody 2D keypoints through multi-part GCN and transformer encoder into a decoder-only LLM for sign-to-text translation.
- **Flow**: Dataset chunking → Multi-Part GCN → Chunk Transformer → LLM Wrapper
- **Training**: Two-stage strategy with DeepSpeed ZeRO, bf16, differential learning rates, and contrastive learning.

---

## 1. Dataset Pipeline (`dataset/my_dataset.py`)

### Input
- **Memory-mapped pose data**: `[Total_Frames, K=133, C=3]` where C = (x, y, conf)
- **JSON annotations**: text, gloss, frame spans per sample

### Augmentation (train only)
- **Speed warp**: Time-axis interpolation with factor ∈ [0.9, 1.1]
- **Joint masking**: Random zero-out with probability ~0.05

### Sliding Window Chunking
- **Parameters**:
  - `window=32`: Chunk length
  - `stride=16`: Sliding step
  - `pad_last=True`: Pad sequences to fit window alignment
  - `min_reserved_ratio=0.6-1.0`: Minimum fraction of chunks to keep (stage1: 1.0, stage2: 0.6)
- **Process**:
  1. Pad pose to length `T'` such that `(T' - window) % stride == 0`
  2. Calculate chunk count: `N = (T' - window) // stride + 1`
  3. Apply random chunk trimming (train only) based on `min_reserved_ratio`
  4. Unfold: `[T', K, C] → [N, window, K, C]`

### Normalization (`transform.py`)
- Split joints into parts: `body(17), face(68), left_hand(21), right_hand(21), fullbody(133)`
- Normalize each part independently (reference point + scale)
- Output: `Dict[part_name, Tensor[N, window, K_part, C]]`

### Collate Function (`my_collate_fn`)
- **Input**: Batch of samples with variable chunk counts
- **Process**:
  1. Find `max_chunks = max(N_i)` across batch
  2. Pad samples to `max_chunks` by repeating last chunk
  3. Stack parts along joint axis: `[N, window, K_part, C] → [N, window, sum_K, C]`
  4. Stack batch: `[B, N, window, sum_K, C]`
- **Output Shape**: `pose: [B, N, window=32, sum_K=133, C=2]` (conf dropped if `drop_conf=True`)
- **Metadata**:
  - `pose_len: [B]` - valid chunk counts per sample
  - `last_chunk_valid_len: [B]` - valid frames in last chunk (accounting for padding)
  - `part_lens: List[int]` - `[17, 68, 21, 21, 133]` for default parts
  - `parts: List[str]` - `['body', 'face', 'left_hand', 'right_hand', 'fullbody']`
  - `adjacency_matrix: Dict[str, Tensor]` - `{part: [K_part, K_part]}` adjacency matrices
  - `text: List[str]` - ground truth text, length B
  - `gloss: List[List[str]]` - gloss tokens (optional)

---

## 2. Multi-Part GCN Stage (`model/visual_encoder.py`, `model/parts_gcn.py`)

### Input
- `pose: [B, N, T=32, sum_K=133, C=2]`
- `part_lens: [17, 68, 21, 21, 133]`
- `pose_len: [B]` (valid chunk mask)
- `last_chunk_valid_len: [B]` (valid frames in last chunk)
- `adjacency: Dict[str, [K_part, K_part]]`

### Process
1. **Reshape**: Merge batch and chunk axes: `[B, N, T, sum_K, C] → [B*N, T, sum_K, C]`
2. **Split by parts**: Slice along joint axis using `part_lens`:
   - `body: [B*N, T=32, K_body=17, C=2]`
   - `face: [B*N, T=32, K_face=68, C=2]`
   - `left_hand: [B*N, T=32, K_lh=21, C=2]`
   - `right_hand: [B*N, T=32, K_rh=21, C=2]`
   - `fullbody: [B*N, T=32, K_full=133, C=2]`
3. **Per-part Uni-GCN** (`MultiPartGCNModel`):
   - Each part has independent GCN backbone (left/right hands can optionally share weights)
   - Temporal convolution kernel: `temporal_kernel=5`
   - GCN parameters:
     - `embed_dim=256` (output feature dimension)
     - `proj_dim=256` (adaptive graph projection dimension)
     - `adaptive=True` (learnable adjacency matrix)
   - Output per part: `[B*N, T=32, K_part, D=256]`
4. **Stack parts**: Concatenate along part dimension:
   - `[B*N, P=5, T=32, D=256]` where P = number of parts
5. **Apply chunk mask**: Zero out padded chunks using `pose_len`
6. **Apply frame mask**: Zero out padded frames in last chunk using `last_chunk_valid_len`

### Output
- `features: [B*N, P=5, T=32, D=256]` - multi-part temporal features

---

## 3. Chunk Transformer Encoder (`model/chunk_transformer.py`)

### Input
- `features: [B*N, P=5, T=32, D=256]`
- `chunk_mask: [B, N]` (derived from `pose_len`)

### Process
1. **Concatenate parts**: Flatten part dimension into feature dimension:
   - `[B*N, P=5, T=32, D=256] → [B*N, T=32, P*D=1280]`
2. **Input projection**: Linear layer projects to LLM dimension:
   - `[B*N, T=32, 1280] → [B*N, T=32, E=1024]` (E = LLM hidden size)
3. **Transformer encoder**:
   - `layers=3`, `heads=8`, `mlp_dim=512`, `dropout=0.1`
   - Apply self-attention with frame-level padding mask
   - Output: `[B*N, T=32, E=1024]`
4. **Token pooling**: Extract fixed `tokens_per_chunk=10` tokens:
   - Learnable query tokens: `[tokens_per_chunk=10, E=1024]`
   - Cross-attention: queries attend to encoder output
   - Final output: `[B*N, P_tok=10, E=1024]`

### Output
- `chunk_tokens: [B*N, P_tok=10, E=1024]` - compressed chunk representations

---

## 4. LLM Wrapper (`model/LLM_wrapper.py`)

### Input
- `chunk_tokens: [B*N, P_tok=10, E=1024]` (reshaped to `[B, N, P_tok=10, E=1024]`)
- `token_mask: [B, N, P_tok=10]` (bool mask for valid tokens)
- `texts: List[str]` (length B)

### Process (Training Forward Pass)

#### 4.1 Build Visual Prefix (per sample)
For each sample `i` with `n_i` valid chunks (from `pose_len[i]`):
1. **Extract valid chunks**: `chunk_tokens[i, :n_i, :, :]` → `[n_i, P_tok=10, E]`
2. **Build chunk sequences**: For each chunk `j`:
   ```
   <BOC>[10 tokens]<EOC>
   ```
   - `<BOC>`: Beginning-of-chunk token (trainable embedding, shape `[1, E]`)
   - `[10 tokens]`: Chunk embeddings from transformer `[P_tok=10, E]`
   - `<EOC>`: End-of-chunk token (trainable embedding, shape `[1, E]`)
3. **Concatenate all chunks**:
   ```
   prefix = [<BOC>, tok_1_1, ..., tok_1_10, <EOC>, <BOC>, tok_2_1, ..., tok_2_10, <EOC>, ..., <BOC>, tok_n_1, ..., tok_n_10, <EOC>]
   ```
   - Length: `n_i * (P_tok + 2) = n_i * 12`
   - Shape: `[n_i * 12, E]`
   - Prefix labels: All set to `-100` (ignored in loss)

#### 4.2 Build Text Suffix
1. **Tokenize text**: `text_ids = tokenizer(texts[i])` → `[L_text]` (max `max_text_len=128`)
2. **Add special tokens**:
   ```
   <BOT> text_tokens <EOT>
   ```
   - `<BOT>`: Beginning-of-text token
   - `<EOT>`: End-of-text token
   - Shape: `[L_text + 2, E]` after embedding
3. **Text labels**: `[text_ids, EOT_id]` (first token label is `-100`)

#### 4.3 Build Full Sequence
```
seq = [prefix, <BOT>, text_tokens, <EOT>]
```
- Total length: `n_i * 12 + L_text + 2`
- Shape: `[seq_len, E=1024]`
- Labels: `[-100 * (n_i * 12), -100, text_ids, EOT_id]`

#### 4.4 Batch Padding
- Pad sequences to `max_seq_len = max(seq_len_i)` across batch
- `inputs_embeds: [B, max_seq_len, E]`
- `labels: [B, max_seq_len]` (padded with `-100`)
- `attention_mask: [B, max_seq_len]` (causal mask + padding mask)

#### 4.5 LLM Forward
```python
outputs = model(
    inputs_embeds=inputs_embeds[:, :-1, :],  # shift right
    attention_mask=attention_mask[:, :-1],
    labels=labels[:, 1:],  # shift left for next-token prediction
)
loss = outputs.loss  # Cross-entropy averaged over non-padding tokens
```

### Output
- `outputs.loss` - Autoregressive cross-entropy loss (includes special tokens)
- `outputs.logits: [B, max_seq_len-1, vocab_size]` - Next-token predictions
- `labels: [B, max_seq_len]` - Target token IDs

---

## 5. Training Pipeline (`training/train_deepspeed.py`)

### Two-Stage Training Strategy

#### Stage 1: Visual Pre-training
- **Config**: `configs/train_default.yaml` + `configs/ds_config_stage1.json`
- **LLM**: Frozen (`freeze_lm=True`)
- **Learning rate**: `visual_lr=5e-5` (only visual encoder trained)
- **Loss**: `total_loss = LM_loss + 0.5 * contrastive_loss`
- **Contrastive learning**:
  - Positive pairs: Visual tokens vs. text embeddings (same sample)
  - Negative samples: Random `neg_k=64` tokens from vocabulary
  - Temperature: `tau=0.07`
  - Weight: `contrastive_alpha=0.5`
- **Data**: `min_reserved_ratio=1.0` (use all chunks)
- **Epochs**: 10
- **Save**: `runs/stage1/YYYYMMDD_HHMMSS/checkpoints/`

#### Stage 2: Joint Fine-tuning
- **Config**: `configs/train_default_stage2.yaml` + `configs/ds_config_stage2.json`
- **LLM**: Unfrozen (`freeze_lm=False`)
- **Learning rates** (differential):
  - `visual_lr=1e-5` (param group 0)
  - `llm_lr=1e-6` (param group 1)
- **Resume**: Load stage1 checkpoint with `resume_for_new_stage=True`
  - Loads model weights only (no optimizer/scheduler state)
- **Loss**: `total_loss = LM_loss` (contrastive disabled, `contrastive_alpha=0.0`)
- **Data**: `min_reserved_ratio=0.6` (random chunk trimming for regularization)
- **Epochs**: 15
- **Save**: `runs/YYYYMMDD_HHMMSS/checkpoints/`

### Loss Components

#### LM Loss (Autoregressive Cross-Entropy)
```python
loss = F.cross_entropy(
    logits.view(-1, vocab_size),  # [B * seq_len, V]
    labels.view(-1),               # [B * seq_len]
    ignore_index=-100,             # Skip padding and prefix
    reduction='mean'
)
```

#### Contrastive Loss (Stage 1 only)
For each sample with visual tokens `Y: [T_visual, E]` and text embeddings `P: [L_text, E]`:
1. Normalize: `Y = F.normalize(Y)`, `P = F.normalize(P)`
2. Sample negatives: `N: [K_neg=64, E]` from vocabulary (avoid `P` tokens)
3. Candidates: `C = [P; N]` → `[L_text + K_neg, E]`
4. Similarity: `logits = (Y @ C.T) / tau` → `[T_visual, L_text + K_neg]`
5. Loss: `-log_softmax(logits)[:, :L_text].mean()` (maximize positive similarity)

### DeepSpeed Configuration

#### Key Settings
- **Zero optimization**: Stage 2
- **Precision**: bf16
- **Gradient accumulation**: 1
- **Micro batch size**: 8
- **Gradient clipping**: 1.5 (stage1) / 1.0 (stage2)
- **Optimizer**: AdamW (betas=[0.9, 0.95], weight_decay=0.01)
  - **No `lr` in config** - uses param groups from code
- **Scheduler**: WarmupDecayLR
  - `warmup_num_steps=0.05` (5% of total steps, auto-computed)
  - `warmup_max_lr` = max(visual_lr, llm_lr) (auto-synced)
  - `total_num_steps="auto"` (epochs * steps_per_epoch)

#### Special Handling for Differential LRs
```python
# train_deepspeed.py lines 91-119
param_groups = [
    {"params": visual.parameters(), "lr": visual_lr, "initial_lr": visual_lr},
    {"params": llm.parameters(), "lr": llm_lr, "initial_lr": llm_lr},
]
engine, optimizer, _, scheduler = deepspeed.initialize(
    model=net, model_parameters=param_groups
)
_sync_param_group_lrs(engine, [visual_lr, llm_lr])  # Force sync after init
```

### Masking Rules

#### Chunk Mask
- Source: `pose_len: [B]` (valid chunk count per sample)
- Expand to: `[B, N, P_tok]` where `mask[i, j, :] = (j < pose_len[i])`
- Used by: Visual encoder, transformer, LLM prefix builder

#### Frame Mask (Last Chunk)
- Source: `last_chunk_valid_len: [B]` (valid frames in last chunk)
- Applied in: GCN forward pass to zero out padded frames
- Shape: `[B*N, T]` where last chunk of each sample is masked at `last_chunk_valid_len[i]`

#### Attention Mask (LLM)
- Causal mask: Standard autoregressive (lower triangular)
- Padding mask: Zero out positions beyond sequence length
- Combined: `attention_mask = causal_mask & padding_mask`

---

## 6. Inference (`model/LLM_wrapper.py::generate`)

### Process
1. Build visual prefix (same as training, without text)
2. Set `inputs_embeds = prefix` (shape `[B, n_i * 12, E]`)
3. Autoregressive generation:
   ```python
   outputs = model.generate(
       inputs_embeds=inputs_embeds,
       max_new_tokens=64,
       do_sample=True,
       temperature=0.3,
       top_k=10,
       pad_token_id=tokenizer.pad_token_id,
   )
   ```
4. Decode: `tokenizer.batch_decode(outputs, skip_special_tokens=True)`

### Output
- `List[str]` - Generated text per sample

---

## 7. Shape Reference Table

| Stage | Tensor | Shape | Description |
|-------|--------|-------|-------------|
| **Dataset** | pose (raw) | `[T, K=133, C=3]` | Memory-mapped keypoints (x, y, conf) |
| | pose (chunked) | `[N, window=32, K=133, C=3]` | After sliding window |
| | pose (normalized) | `Dict[str, [N, 32, K_part, C=2]]` | Per-part, conf dropped |
| **Collate** | pose | `[B, N, 32, sum_K=133, 2]` | Batched, parts stacked |
| | pose_len | `[B]` | Valid chunk counts |
| | last_chunk_valid_len | `[B]` | Valid frames in last chunk |
| | part_lens | `[17, 68, 21, 21, 133]` | Joint counts per part |
| | adjacency | `Dict[str, [K_part, K_part]]` | Per-part adjacency |
| **GCN Input** | pose (reshaped) | `[B*N, 32, sum_K=133, 2]` | Chunk axis merged |
| | per-part | `[B*N, 32, K_part, 2]` | Split by part_lens |
| **GCN Output** | features | `[B*N, P=5, 32, D=256]` | Multi-part temporal features |
| **Transformer Input** | concat | `[B*N, 32, P*D=1280]` | Parts concatenated |
| | projected | `[B*N, 32, E=1024]` | After input projection |
| **Transformer Output** | chunk_tokens | `[B*N, P_tok=10, E=1024]` | Pooled chunk tokens |
| | reshaped | `[B, N, 10, 1024]` | For LLM input |
| **LLM Input** | prefix (per sample) | `[n_i * 12, E=1024]` | Visual prefix with special tokens |
| | text | `[L_text, E=1024]` | Embedded text tokens |
| | full_seq | `[n_i*12 + L_text + 2, E]` | Concatenated sequence |
| | inputs_embeds | `[B, max_seq_len, E]` | Batched and padded |
| **LLM Output** | logits | `[B, max_seq_len-1, vocab_size]` | Next-token predictions |
| | labels | `[B, max_seq_len]` | Target IDs (-100 for padding/prefix) |

---

## 8. Configuration Reference

### Default Hyperparameters (from `configs/train_default.yaml`)

```yaml
# Window & Chunking
window: 32
stride: 16
pad_last: true
min_reserved_ratio: 1.0  # stage1: 1.0, stage2: 0.6

# Model Architecture
parts: ["body", "face", "left_hand", "right_hand", "fullbody"]
drop_conf: true
part_embed_dim: 256      # GCN output dimension
tokens_per_chunk: 10     # Transformer output tokens
uni_gcn:
  proj_dim: 256
  temporal_kernel: 5
  adaptive: true
  dropout: 0.0
chunk_transformer:
  layers: 3
  heads: 8
  dropout: 0.1
  mlp_dim: 512

# LLM
model_name_or_path: /workspace/Qwen  # Qwen2.5-0.5B or similar
max_text_len: 128
freeze_lm: true  # stage1: true, stage2: false
gradient_checkpointing: false

# Training
visual_lr: 5e-5  # stage1: 5e-5, stage2: 1e-5
llm_lr: 1e-6     # Only used in stage2
contrastive_alpha: 0.5  # stage1: 0.5, stage2: 0.0
contrastive_tau: 0.07
contrastive_neg_k: 64
```

---

## 9. Design Principles

1. **Metadata-driven**: `pose_len`, `part_lens`, `parts` are the single source of truth; no hard-coded dimensions
2. **Explicit masking**: Chunk and frame masks propagate through all stages; padding is always masked out in loss
3. **Special token training**: `<BOC>`, `<EOC>`, `<BOT>`, `<EOT>` are trainable embeddings included in loss computation
4. **Two-stage strategy**: Pre-train visual encoder with frozen LLM + contrastive loss, then jointly fine-tune with differential LRs
5. **Linus-style patches**: Small, direct code changes with clear intent; pipeline document updated alongside code

---

## 10. File Mapping

| Module | File Path | Key Classes/Functions |
|--------|-----------|----------------------|
| Dataset | `dataset/my_dataset.py` | `MyDataset`, `my_collate_fn` |
| Normalization | `dataset/transform.py` | `NormalizeProcessor` |
| Visual Encoder | `model/visual_encoder.py` | `VisualEncoder` |
| Multi-Part GCN | `model/parts_gcn.py` | `MultiPartGCNModel`, `UniGCNModule` |
| Chunk Transformer | `model/chunk_transformer.py` | `ChunkTokenEncoder` |
| LLM Wrapper | `model/LLM_wrapper.py` | `LLMWithVisualPrefix` |
| Training Loop | `training/train_deepspeed.py` | `VLLMTrainer`, `train()` |
| Config Loader | `model/config.py` | `load_config()` |
| Data Builder | `training/data.py` | `build_dataloaders()` |
| Utils | `training/utils.py` | `set_seed()`, `log_logits()` |

---

**Last Updated**: 2025-01-02
**Compatible Code Version**: Latest commit with two-stage training and contrastive learning
