# Sign Language Translation Pipeline

## Overview
- **Goal**: Stream chunked COCO WholeBody 2D keypoints through multi-part GCN and transformer encoder into a decoder-only LLM for sign-to-text translation.
- **Flow**: Dataset chunking → Multi-Part GCN → Chunk Transformer → LLM Wrapper
- **Training**: Two-stage strategy with DeepSpeed ZeRO, bf16, differential learning rates, and a powerful hierarchical contrastive learning approach.

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
- **Learning rate**: `visual_lr=5e-5` (trains visual encoder, summarizer, and projection heads)
- **Loss**: `total_loss = LM_loss + contrastive_alpha * contrastive_loss + diversity_alpha * decorrelation_loss`
- **Contrastive learning**: Hierarchical InfoNCE loss with in-batch negatives.
- **Decorrelation loss**: Penalizes pairwise similarity between visual tokens to prevent representation collapse.
- **Data**: `min_reserved_ratio=1.0` (use all chunks)
- **Epochs**: 10
- **Save**: `runs/stage1/YYYYMMDD_HHMMSS/checkpoints/`

#### Stage 2: Joint Fine-tuning
- **Config**: `configs/train_default_stage2.yaml` + `configs/ds_config_stage2.json`
- **LLM**: Unfrozen (`freeze_lm=False`)
- **Learning rates** (differential):
  - `visual_lr=1e-5`
  - `llm_lr=1e-6`
- **Resume**: Load stage1 checkpoint with `resume_for_new_stage=True`
- **Loss**: `total_loss = LM_loss` (contrastive and decorrelation disabled)
- **Data**: `min_reserved_ratio=0.6` (random chunk trimming for regularization)
- **Epochs**: 15
- **Save**: `runs/YYYYMMDD_HHMMSS/checkpoints/`

### Loss Components

#### LM Loss (Autoregressive Cross-Entropy)
Standard next-token prediction loss, ignoring padding and visual prefix tokens.

#### Contrastive Loss (Stage 1 only)
A symmetric InfoNCE loss designed to align visual and text representations in a dedicated projection space.
1.  **Hierarchical Summarization**:
    - For each sample, all valid visual tokens from all chunks are flattened into a single sequence `[L_v, E]`.
    - A `SequenceSummarizer` module (Transformer with a learnable summary token) processes this sequence to produce a single summary vector `s_v` representing the entire video.
    - The same process is applied to the text tokens to get a text summary vector `s_t`.
2.  **Projection**: `s_v` and `s_t` are projected into a lower-dimensional contrastive space using separate `ProjectionHead` MLPs.
3.  **In-Batch Negatives**:
    - In a distributed setting, projected vectors from all GPUs are gathered to form a global batch of size `B_global`.
    - A `[B_global, B_global]` similarity matrix is computed. For each sample, its corresponding sample is the positive pair, while all other `B_global - 1` samples in the batch act as hard negatives.
4.  **Loss Calculation**: A symmetric cross-entropy loss is computed over the similarity matrix, optimizing both visual-to-text and text-to-visual retrieval.

#### Decorrelation Loss (Stage 1 only)
To prevent representation collapse where all visual tokens become too similar.
1.  Collect all valid visual tokens `Y: [T_total, E]` from a batch.
2.  Normalize them and compute the pairwise cosine similarity matrix `S: [T_total, T_total]`.
3.  The loss is the mean of the squared off-diagonal elements of `S`. This penalizes any non-zero similarity, encouraging orthogonal features.

### DeepSpeed Configuration
(No changes to this section)

---

## 6. Inference (`model/LLM_wrapper.py::generate`)
(No changes to this section)

---

## 7. Shape Reference Table
(No changes to this section, as the core visual pipeline shapes remain the same)

---

## 8. Configuration Reference

### Default Hyperparameters (from `configs/train_default.yaml`)

```yaml
# ... (previous sections unchanged)

# LLM
model_name_or_path: /workspace/Qwen
# ...
freeze_lm: true
gradient_checkpointing: false

# New parameters for contrastive learning
contrastive_projection_dim: 256 # Output dim of the projection head

# Training
visual_lr: 5e-5
llm_lr: 1e-6
contrastive_alpha: 0.75
contrastive_tau: 0.07
diversity_alpha: 0.1 # Weight for the decorrelation loss
```

---

## 9. Design Principles

1.  **Metadata-driven**: `pose_len`, `part_lens`, `parts` are the single source of truth; no hard-coded dimensions.
2.  **Explicit masking**: Chunk and frame masks propagate through all stages; padding is always masked out in loss.
3.  **Hierarchical Representation**: A `SequenceSummarizer` creates a global representation from chunk tokens before contrastive loss, enabling better alignment.
4.  **Two-stage strategy**: Pre-train visual modules with a frozen LLM using a powerful contrastive loss, then jointly fine-tune.
5.  **Linus-style patches**: Small, direct code changes with clear intent; pipeline document updated alongside code.

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
| **Contrastive Modules** | `training/contrastive_modules.py` | `ProjectionHead`, `SequenceSummarizer` |
| Training Loop | `training/train_deepspeed.py` | `VLLMTrainer`, `train()` |
| Config Loader | `model/config.py` | `load_config()` |
| Data Builder | `training/data.py` | `build_dataloaders()` |
| Utils | `training/utils.py` | `set_seed()`, `log_logits()` |

---

**Last Updated**: 2025-10-09
**Compatible Code Version**: Commit with hierarchical contrastive learning and decorrelation loss.
