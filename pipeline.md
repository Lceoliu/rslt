**Pipeline Overview**
- Goal: stream chunked COCO WholeBody 2D keypoints through Uni-GCN parts and a transformer encoder before a decoder-only LLM for low-latency sign translation.
- Flow: dataset chunking -> MultiPart GCN (chunk axis folded into batch) -> chunk transformer -> LLM wrapper without KV cache during training.
- Shapes stay explicit at every stage so that downstream modules rely only on `pose`, `pose_len`, `parts`, and `part_lens` metadata.

**Dataset Pipeline**
- Data source: memory-mapped pose arrays plus JSON annotations that hold text, gloss, and frame spans.
- Augmentation (train only): optional speed warp and joint masking applied before normalisation.
- Sliding window: dataset owns `window`, `stride`, and `pad_last`; samples shorter than `window` are padded, and the tail is padded to complete the final chunk when required.
- Chunk trimming: enforce `min_reserved_ratio`, optionally drop a random suffix, then commit the chunk count so every consumer observes the same windows.
- Normalisation: split poses into parts (body, face, left_hand, right_hand, full-body) and normalise each slice before stacking.
- Collate: stack chunks to `pose[B, N_chunk, chunk_len, sum_K, C]`, repeat the last chunk when padding, and emit metadata that records part ordering and adjacency.

**Batch Format**
- `pose`: `torch.Tensor[B, N_chunk, chunk_len, sum_K, C]` where joints follow `parts` order.
- `pose_len`: `torch.Tensor[B]` giving the count of valid chunks per sample; anything beyond this index is padding.
- `text`: list of strings; `gloss`: list of gloss token lists or `None`; `gloss_len`: tensor when gloss exists.
- `parts`: list of part names; `part_lens`: list of joint counts aligned with `parts` (hands kept separate); `adjacency_matrix`: dict of `(K_part, K_part)` tensors for each part.
- Consumers must derive per-part tensors by slicing the joint axis with `part_lens`; never hard-code joint counts.

**Multi-Part GCN Stage**
- Merge the chunk axis into the batch: reshape to `[B * N_chunk, chunk_len, sum_K, C]` before splitting into parts.
- Each part runs its own Uni-GCN (or compatible) backbone over `[B * N_chunk, chunk_len, K_part, C]`. Left and right hands stay independent GCN streams, though weights can be optionally shared in code.
- The combined output keeps all parts: `[B * N_chunk, part_count, chunk_len, D]`. Use the chunk-length mask derived from `pose_len` to zero out padded frames inside each chunk.

**Chunk Transformer Encoder**
- Concatenate part embeddings along the feature dimension to `[B * N_chunk, chunk_len, part_count * D]`.
- Apply a lightweight transformer encoder that respects the frame mask and emits a fixed number of tokens per chunk: `[B * N_chunk, tokens_per_chunk, E]` where `E` matches the LLM hidden size.
- No separate adapter layer is needed because the transformer already projects into the LLM dimensionality.

**LLM Wrapper**
- Reshape transformer outputs back to `[B, N_chunk, tokens_per_chunk, E]` and drop padded chunks using `pose_len`.
- For sample `i` with `n_i` valid chunks, build a sequence: `[<BOC>, emb_{i,1,1..P}, <EOC>, ... , <BOC>, emb_{i,n_i,1..P}, <EOC>, <BOT>, text_tokens, <EOT>]`.
- Stack batched sequences with right padding; the final sequence shape is `n_i * (tokens_per_chunk + 2) + text_len + 2` embeddings.
- Feed `inputs_embeds = seq[:, :-1, :]`, target indices from `seq[:, 1:, :]`, and compute cross-entropy on every position (special tokens included) so gradients reach the visual stack.
- Training runs without KV cache; generation can add caching later if required.

**Masking Rules**
- Chunk mask: expand `pose_len` to a boolean mask `[B, N_chunk]`, broadcast when reshaping to `[B * N_chunk, 1, 1]` for per-chunk modules, and to `[B * N_chunk, chunk_len]` for frame-level masking.
- Transformer mask: combine the frame mask with any attention masking the encoder expects.
- LLM mask: when packing sequences, build standard causal attention masks plus padding masks so the model ignores padded suffixes.

**Training Notes**
- `training/train_deepspeed.py` must wire dataset -> MultiPart GCN -> chunk transformer -> LLM wrapper, enable DeepSpeed ZeRO, bf16, and checkpoint save/resume.
- Loss: autoregressive cross-entropy on the full packed sequence; reduce over non-padding tokens.
- Align dataloader batch size with DeepSpeed micro-batch settings and respect the chunk-derived masks when accumulating loss.

**Design Notes**
- Keep dataset, encoder, transformer, and LLM wrapper in sync; update this document alongside `dataset/my_dataset.py`, `model/visual_encoder.py`, `model/llm_wrapper.py`, and `training/train_deepspeed.py`.
- Preserve the explicit metadata (`parts`, `part_lens`, `pose_len`) as the single source of truth for downstream reshapes.
- Follow Linus-style patches: small, direct changes with clear intent.