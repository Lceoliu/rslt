# Sign Language Translation Pipeline

## Overview
- **Goal**: Stream chunked COCO WholeBody 2D keypoints through multi-part GCN and lightweight temporal pooling into a decoder-only LLM for sign-to-text translation.
- **Flow**: Dataset chunking -> Multi-Part GCN -> Temporal downsampling -> Prompted LLM
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
  4. Unfold: `[T', K, C] -> [N, window, K, C]`

### Normalization (`transform.py`)
- Split joints into parts: `body(17), face(68), left_hand(21), right_hand(21)`, plus optional `fullbody`
- Interpolate low-confidence joints, but keep their confidence at zero
- Crop the body using a high-confidence bounding box, scale to `[-1, 1]`, and record the global scale/origin
- Reuse the body scale for every part; hands and face are re-centered on their anchor joints before scaling
- Zero-out joints whose confidence falls below the threshold after normalization
- Output: `Dict[part_name, Tensor[N, window, K_part, C]]`

### Collate Function (`my_collate_fn`)
- **Input**: Batch of samples with variable chunk counts
- **Process**:
  1. Find `max_chunks = max(N_i)` across batch
  2. Pad samples to `max_chunks` by repeating last chunk
  3. Stack parts along joint axis: `[N, window, K_part, C] -> [N, window, sum_K, C]`
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
1. **Reshape**: Merge batch and chunk axes: `[B, N, T, sum_K, C] -> [B*N, T, sum_K, C]`
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

## 3. Chunk Token Preparation (model/visual_encoder.py)

### Input
- eatures: [B*N, P=5, T=32, D=256]
- rame_mask: [B*N, T] (valid frames per chunk)
- chunk_mask: [B, N] (valid chunks per sample)

### Process
1. Flatten part dimension: [B*N, P, T, D] -> [B*N, T, P*D].
2. Temporal stride sampling (temporal_sampling_stride in config) reduces the length to T_s = T / stride.
3. Apply the frame mask, zeroing invalid positions, and project with a linear layer to the LLM hidden size: [B*N, T_s, E].
4. Reshape back to [B, N, T_s, E] and derive token_mask: [B, N, T_s] by combining frame and chunk masks.

### Output
- chunk_tokens: [B, N, T_s, E]
- token_mask: [B, N, T_s]
- chunk_mask: [B, N]

## 4. LLM Wrapper (model/LLM_wrapper.py)

### Input
- chunk_tokens: [B, N, T_s, E]
- token_mask: [B, N, T_s]
- texts: List[str]

### Training Forward Path
1. Flatten visual tokens per sample using token_mask to keep only valid embeddings.
2. Prepend a fixed prompt (default: \u8bf7\u5c06\u63a5\u4e0b\u6765\u7684\u624b\u8bed\u5185\u5bb9\u7ffb\u8bd1\u6210\u6587\u5b57\uff1a) by looking up its token embeddings.
3. Tokenize the ground-truth text (max_text_len) and embed it; append the model's EOS token.
4. Concatenate [prompt_embeds, visual_embeds, text_embeds, eos_embed].
5. Build labels: prefix positions (prompt + visual) are set to -100, text positions keep their token IDs, EOS label equals eos_id.
6. Pad batched sequences to build inputs_embeds, attention_mask, and labels, then call the causal LM.

### Inference
- Reuse the same prompt+visual prefix and call generate.
- Drop the prefix length from generated token IDs before decoding to text.

### Masking Notes
- token_mask ensures repeated/padded visual tokens never contribute to loss.
- The prompt keeps instructions explicit without trainable special token embeddings.

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
