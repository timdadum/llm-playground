# recursion (enhanced v2)

### Key changes (this request)
- **TRM faithful** to K=1 (predict only the next token at the last position).
- **TRMSequentialModel** predicts the next **K** tokens with teacher forcing (flag: `--K`).
- Renamed model type: `shared_weights` → **`recursive_model`**.
- **Recursive model** now explicitly recurses a small 2-block segment **T** times: y = f^{(T)}(x).
- **T** is a global traversal count for any recursive setup (not confined to an attention class only).
- Dataset windows support prediction horizon **K** so we can supervise future tokens for TRMSequential.

### Other features retained
- Dataset: `roneneldan/TinyStories`
- Streaming via `--n_rows` (preferred) or legacy in-memory via `--limit`
- SDPA (flash) attention (`--attn_impl sdpa|eager`)
- Gradient accumulation (`--grad_accum_steps`)
- **wandb** logging with **tokens_seen**
- Exposed hyperparams: `d_model, n_head, n_layer, d_ff, dropout, context_length, lr, epochs`
- Seeding and deterministic flags in `utils.set_seed`

### Example runs
```bash
# TRM (K=1)
python train.py --model_type trm --epochs 1 --n_rows 8000 --context_length 256 --T 1 --seed 42

# TRMSequential (K>1 next tokens)
python train.py --model_type trm_sequential --K 3 --epochs 1 --n_rows 8000 --context_length 260 --seed 7

# Recursive model (segment traversed T times)
python train.py --model_type recursive_model --T 6 --epochs 1 --n_rows 6000 --context_length 256 --seed 1337
```

**Note:** For `trm_sequential`, ensure `context_length >= base_length + (K-1)` at training time, since the teacher-forced prefix grows up to `n + (K-1)`.
