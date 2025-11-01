# -*- coding: utf-8 -*-
"""
Packed DataLoaders for causal LLM training with dense next-K supervision.

- Concatenate documents with EOS separators (no cross-doc leakage of meaning; EOS is a real token).
- Emit non-overlapping input windows of length L_in.
- Also emit a K-token lookahead (peek) from the same stream without consuming it.
- Build dense next-K grid labels of shape (B, L_in, K) via:
      labels[:, t, k] = extended[:, t + (k+1)]
  where extended = concat(input_ids, lookahead).
- No label masking here; EOS and any padding you decide to add are just tokens. Use ignore_index later if needed.
"""

import torch
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer
from datasets import Dataset
from typing import Optional

class PackedDataset(IterableDataset):
    """
    Concatenate all documents with EOS separators. Non-overlapping windows of L_in with K-step lookahead.
    No padding; drop any trailing tokens < L_in + K at the end of the stream.
    """
    def __init__(self, hf_ds, tokenizer, seq_len: int, eos_id: int, lookahead_k: int = 0):
        self.hf_ds = hf_ds
        self.tok = tokenizer
        self.L = int(seq_len)       # L_in
        self.K = int(lookahead_k)   # K (can be 0)
        self.eos = int(eos_id)

    def __iter__(self):
        buffer = []
        for sample in self.hf_ds:
            text = sample.get("text")
            if not isinstance(text, str):
                continue
            ids = self.tok(text, add_special_tokens=False)["input_ids"]
            ids.append(self.eos)  # EOS between docs
            buffer.extend(ids)

            # Emit while we have enough for input + lookahead
            min_needed = self.L + self.K
            while (len(buffer) >= min_needed) if self.K > 0 else (len(buffer) >= self.L):
                chunk = buffer[:self.L]                         # (L_in)
                if self.K > 0:
                    tail = buffer[self.L:self.L + self.K]       # (K)
                    lookahead = torch.tensor(tail, dtype=torch.long)
                else:
                    lookahead = torch.empty(0, dtype=torch.long)

                yield {
                    "input_ids": torch.tensor(chunk, dtype=torch.long),
                    **({"lookahead": lookahead} if self.K > 0 else {}),
                }
                # advance by L (non-overlapping)
                del buffer[:self.L]

        # end: drop any remainder < L_in (+ K), by design (no padding)

def build_nextk_grid_labels(
    input_ids: torch.Tensor,     # (B, L_in)
    lookahead: torch.Tensor,     # (B, K) -- provides K-step peek past the window
    K: int,                      # continuation length
) -> torch.Tensor:
    """
    Build dense next-K grid labels (B, L_in, K) from extended = [input || lookahead],
    with no masking. labels[:, t, k] = extended[:, t + (k+1)] for k in [0..K-1].
    """
    if K <= 0:
        raise ValueError("K must be >= 1 for next-K grid labels.")
    if input_ids.dtype != torch.long:
        input_ids = input_ids.long()
    if lookahead.dtype != torch.long:
        lookahead = lookahead.long()

    B, L_in = input_ids.shape
    assert lookahead.shape == (B, K), f"lookahead must be (B, K); got {lookahead.shape}"

    extended = torch.cat([input_ids, lookahead], dim=1)  # (B, L_in + K)

    labels = input_ids.new_empty((B, L_in, K))
    for k in range(K):
        offset = k + 1
        labels[:, :, k] = extended[:, offset:offset + L_in]
    return labels.long()

def make_collate_tokens(nextk_K: int = 0, eos_id: Optional[int] = None):
    """
    Collate function factory.
      - Always returns 'input_ids', 'attention_mask', and classic 'labels' (next-token LM) for compatibility.
      - If nextk_K > 0, also returns 'nextk_grid_labels' built from [input || lookahead].
    """
    def collate_tokens(batch):
        input_ids = torch.stack([b["input_ids"] for b in batch])  # (B, L_in)
        attention_mask = torch.ones_like(input_ids)
        labels = input_ids.clone()  # classic next-token LM labels

        out = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

        if nextk_K and nextk_K > 0:
            lookahead = torch.stack([b["lookahead"] for b in batch])  # (B, K)
            nextk_grid_labels = build_nextk_grid_labels(
                input_ids=input_ids,
                lookahead=lookahead,
                K=nextk_K,
            )  # (B, L_in, K)
            out["nextk_grid_labels"] = nextk_grid_labels

        return out
    return collate_tokens

def build_packed_dataloaders(
    train: Dataset,
    val: Dataset,
    test: Dataset,
    tokenizer_name: str = "gpt2",
    seq_len: int = 256,
    batch_size: int = 32,
    num_workers: int = 8,
    nextk_K: int = 0,  # 0 = disabled; >0 enables dense next-K grid labels
):
    """
    Returns (train_loader, val_loader, test_loader).
    If `nextk_K > 0`, batches also include:
      - 'lookahead' implicitly (used only to build labels),
      - 'nextk_grid_labels' (B, L_in, K) built from [input || lookahead].
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_id = tokenizer.eos_token_id or tokenizer.pad_token_id

    collate_fn = make_collate_tokens(nextk_K, eos_id=eos_id)

    def make_loader(hf_ds):
        packed = PackedDataset(
            hf_ds,
            tokenizer,
            seq_len,
            eos_id=eos_id,
            lookahead_k=nextk_K,
        )
        return DataLoader(
            packed,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

    return make_loader(train), make_loader(val), make_loader(test)

# ---------------------------
# Debug helper (unchanged idea, renamed prints)
# ---------------------------

def test_two_doc_stream(L_in=16, K=4, eos_id=10_000):
    """
    Simulate two consecutive documents concatenated with EOS.
    Verifies window emission and dense (L_in x K) next-K label construction.
    Uses: doc1 = 56 tokens (0..55), doc2 = 35 tokens (100..134).
    """
    doc1 = list(range(56))             # [0..55]
    doc2 = list(range(100, 135))       # [100..134]
    stream = doc1 + [eos_id] + doc2 + [eos_id]
    total_len = len(stream)

    print(f"Stream length = {total_len} tokens (with EOS separators)\n")

    start = 0
    win_idx = 0
    min_needed = L_in + K

    while start + min_needed <= total_len:
        input_ids = torch.tensor(stream[start:start + L_in], dtype=torch.long)
        lookahead = torch.tensor(stream[start + L_in:start + L_in + K], dtype=torch.long)

        extended = torch.cat([input_ids, lookahead], dim=0)
        labels = torch.empty((L_in, K), dtype=torch.long)
        for k in range(K):
            offset = k + 1
            labels[:, k] = extended[offset:offset + L_in]

        print(f"=== Window {win_idx} ===")
        print(f"start={start}")
        print(f"input_ids ({L_in}):", input_ids.tolist())
        print(f"lookahead ({K}):", lookahead.tolist())
        print(f"extended ({L_in+K}):", extended.tolist())
        print("next-K labels (L_in x K):")
        for row in labels.tolist():
            print(row)
        print()

        start += L_in
        win_idx += 1

    # remainder (optional, padded with EOS)
    if start < total_len:
        tail_start = start + L_in
        pad_len = max(0, L_in - (total_len - start))
        pad = [eos_id] * pad_len

        input_ids = torch.tensor(stream[start:total_len] + pad, dtype=torch.long)
        tail = stream[tail_start:tail_start + K]
        if len(tail) < K:
            tail = tail + [eos_id] * (K - len(tail))
        lookahead = torch.tensor(tail, dtype=torch.long)

        extended = torch.cat([input_ids, lookahead], dim=0)
        labels = torch.empty((L_in, K), dtype=torch.long)
        for k in range(K):
            offset = k + 1
            labels[:, k] = extended[offset:offset + L_in]

        print(f"=== Window {win_idx} (remainder, padded) ===")
        print(f"start={start}")
        print(f"input_ids ({L_in}, padded):", input_ids.tolist())
        print(f"lookahead ({K}):", lookahead.tolist())
        print(f"extended ({L_in+K}):", extended.tolist())
        print("next-K labels (L_in x K):")
        for row in labels.tolist():
            print(row)
        print()

    print(f"Total windows emitted: {win_idx + (1 if start < total_len else 0)}")

# test_two_doc_stream()