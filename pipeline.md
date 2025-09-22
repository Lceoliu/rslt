**Pipeline Overview**
- Goal: stream COCO WholeBody 2D poses into a decoder‑only LLM for sign‑to‑text translation with low latency and KV‑cache reuse.
- Core idea: per‑chunk multi‑part GCN features → lightweight Transformer encoder → fixed P visual tokens → adapter to LLM hidden size → LLM trained with autoregressive loss, using cache across chunks.

**Conventions**
- Shapes:
  - Input parts `x_part`: `[B, T, V, C]`, `C∈{2,3}` (xy or xy+conf). Internally converted to `[B, C, T, V]` for backbones.
  - Chunked visual tokens: `[B, N, P, D]` where `N` chunks, `P` fixed tokens per chunk, `D` visual dim.
  - Adapter outputs: `[B, N, P, E]` where `E` is LLM hidden size.
  - Non‑streaming visual: `[B, D]` → adapter → `[B, P, E]`.
- Special tokens:
  - Visual scope: `<BOV> ... <EOV>`
  - Per‑chunk: `<BOC> [P visual tokens] <EOC>`
  - Text: `<BOT> text ... <EOT>`
- Masks: `pose_len` → per‑frame mask `[B, T]` (1=valid). Down/upsampled to match temporal resolution after backbone strides.
- Dtypes: embeddings and inputs are cast to the live dtype of layer weights (bf16/fp16/fp32) to avoid precision mismatches.

**Visual Encoder**
- Backbones (Torch‑only):
  - `AAGCNBackbone` and `STGCNBackbone` (in `model/backbones/*_minimal.py`).
  - Accept `x: [B, C, T, V]`, optional `mask: [B, T]`.
  - `return_seq=False` (default): mask‑aware spatiotemporal pooling → `[B, D]`.
  - `return_seq=True`: average over joints only, keep time → `[B, T’, D]` with invalid frames zeroed via mask.
- Per‑part processing:
  - `MultiPartGCNModel` builds one backbone per part (`body`, `face`, `left_hand`, `right_hand`, `fullbody`).
  - Adjacency from `NormalizeProcessor.gen_adjacency_matrix(split_part=True)`; dtype normalized to match inputs.

**Chunking and Per‑Chunk Encoding**
- Sliding window: `window`, `stride`, `drop_last` from `cfg.streaming`.
- For each chunk starting at `s`:
  - Slice each part `x_part[:, s:s+window]` and convert to `[B, C, t, V]`.
  - Build per‑chunk mask from `pose_len` shifted by `s`.
  - Run backbone with `return_seq=True` to get `[B, t’, D_part]` per part.
- Time‑step fusion across parts:
  - Flatten time into batch: each part `[B, t’, D_part]` → `[B*t’, D_part]`.
  - Fuse with one of:
    - `AttentionFusion`: project to `d_model`, self‑attend across parts, mean over parts, linear out → `[B*t’, D]`.
    - `ConcatMLPFusion`: concatenate features, MLP → `[B*t’, D]`.
  - Reshape back to `[B, t’, D]`.
- Chunk Transformer encoder (`ChunkTransformerEncoder`):
  - Sin‑cos positional encoding + 1× `TransformerEncoderLayer` (batch_first) on `[B, t’, D]`.
  - Linearly interpolate along time to fixed `P` tokens → `[B, P, D]`.
- Stacking all chunks gives `[B, N, P, D]` from `MultiPartGCNModel.encode_chunks`.

**Adapter (Visual → LLM)**
- `VisualAdapter` maps visual dim `D` to LLM hidden `E`:
  - `[B, D]` → `[B, P, E]` via `net` and reshape (for non‑streaming/global use).
  - `[B, N, D]` → `[B, N, P, E]` similarly.
  - `[B, N, P, D]` → `[B, N, P, E]` via per‑token `token_net` (keeps P).
- `P` is set by `cfg.llm.num_prefix_tokens` or `cfg.model.preset_len` (the latter overrides if provided).

**LLM Wrapper (KV‑Cache + Tokens)**
- `LLMWithVisualPrefix` wraps a HF CausalLM:
  - Tokenizer adds special tokens: `<BOV> <EOV> <BOC> <EOC> <BOT> <EOT>`; embeddings resized.
  - Streaming API:
    - `reset_prefix_cache()` clears cache and state.
    - `step_prefix(step[, is_last])` accepts `[B, P, E]` or `[B, E]` and feeds `[<BOV>?] <BOC> step <EOC> [<EOV if is_last>]` into the model with `use_cache=True`.
    - `loss_with_text(texts)` computes CE on `[<BOT> text <EOT>]` given current cache (no cache update).
    - `forward_stream(prefix_seq, texts, reduction)` loops chunks, calls `step_prefix(…, is_last=last)` then `loss_with_text`; `reduction ∈ {mean,last}`.
  - Non‑streaming loss:
    - For `[B, P, E]` or `[B, E]`, build `[<BOV> <BOC> prefix <EOC> <EOV> <BOT> text <EOT>]`, mask out non‑text positions in labels.
  - Generation:
    - `generate_from_prefix(...)` starts from cached visuals (ensures `<EOV>` locally if not present), feeds `<BOT>`, then greedy/sampling decode; early‑stop on `eos` or `<EOT>`.

**Trainer and Data Flow**
- `VLLMTrainer` composes: `embedder (MultiPart) → adapter → llm`.
- Streaming default: `encode_chunks` → `[B,N,P,D]` → adapter → `[B,N,P,E]` → LLM streaming loss.
- Batch format from `dataset.my_dataset`:
  - `{'pose': {part: Tensor[B,T,V,C]}, 'text': List[str], 'pose_len': Tensor[B]}`.
- Masks are built from `pose_len` per chunk and propagated through backbones and fusion.

**Configs**
- `cfg.model`:
  - `parts`, `backbone ∈ {aagcn, stgcn}`, `part_embed_dim`, `embed_dim` (visual `D`), `preset_len` (P), `fusion ∈ {attention, concat_mlp}`, `drop_conf`.
- `cfg.streaming`: `enabled`, `window`, `stride`, `drop_last`, `loss_reduction`.
  - Random CE skipping: `skip_loss_prob` (0..1), `keep_first` (bool), `always_keep_last` (bool).
- `cfg.llm`: `model_name_or_path`, `num_prefix_tokens` (P), `adapter_hidden`, `freeze_lm`, `gradient_checkpointing`, tokens (`bot_token`, etc.).
- `cfg.decoding`: `max_new_tokens`, `do_sample`, `temperature`, `top_k`.

**Training**
- `training/train_deepspeed.py`:
  - Builds trainer, aligns dataloader micro‑batch with DeepSpeed config, bf16/fp16 casting utility, tqdm + TensorBoard minimal metrics.
  - Streaming enabled by default; computes loss per chunk with optional random skipping, `reduction` from config (often `last`).
  - Checkpointing: periodic save, resume support, consistent `RUN_ID` across ranks, all‑reduce validation to avoid NCCL timeouts.
  - Validation sampling: randomly pick items; at chunk milestones, run generate to log qualitative predictions.

**Inference**
- `inference/run_infer.py`:
  - Loads trainer module and merges ZeRO shards for single‑GPU eval.
  - Encodes `[B=1, N, P, D]` → adapter → `[1, N, P, E]`; builds cache incrementally and decodes at milestone chunks.
  - Supports timing modes (`total` vs `llm`) and bf16 autocast.

**Design Rationale**
- Keeping P tokens per chunk:
  - Preserves richer intra‑chunk temporal structure compared to mean/pool.
  - Makes visual tokens natively match LLM prefix interface; `<BOC>/<EOC>` delimiters isolate chunks; `<BOV>/<EOV>` bracket the entire visual stream.
- Lightweight Transformer encoder:
  - One‑layer encoder is a good speed/quality trade‑off; sin‑cos PE keeps it stateless; interpolation to P guarantees fixed capacity.
- Time‑step fusion via batch flattening:
  - Reuses existing fusion modules without redesign; simple and efficient while remaining mask‑aware.
- Loss on `<BOT> text <EOT>`:
  - Single unified AR objective without frame‑level alignment; training stays simple and compatible with caching.
  - Randomly skipping CE on some chunks reduces compute and acts as regularization; always keeping first/last stabilizes training and preserves supervision anchors.

**Gotchas**
- Always cast inputs to layer weight dtype (esp. bf16) before matmul/embeddings.
- When sampling partial chunks for qualitative decode, `<EOV>` is auto‑appended locally to avoid corrupting the persistent cache.
- `pose_len`‑based masks must be down/upsampled to the backbone’s temporal stride before applying.
- Ensure `P` in `cfg.llm.num_prefix_tokens` matches `cfg.model.preset_len` (the model loader defaults `preset_len` from `llm.num_prefix_tokens` if unspecified).

**Change Policy**
- Any functional change to the visual encoder, adapter, tokens, caching, training, or inference must be reflected here. Keep sections synchronized with code entry points:
  - Visual: `model/backbones/*`, `model/parts_gcn.py`
  - Fusion: `model/fusion.py`
  - Adapter: `model/adapter.py`
  - LLM: `model/LLM_wrapper.py`
  - Trainer: `training/train_deepspeed.py`
  - Inference: `inference/run_infer.py`
