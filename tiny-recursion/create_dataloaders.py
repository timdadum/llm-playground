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



# -----------------------------
# Tokenizer helpers
# -----------------------------
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, processors
from transformers import PreTrainedTokenizerFast

def build_tokenizer_from_mode(mode: str, hf_ds=None, cache_dir: str = ".tok_cache", vocab_size: int = 4000, eos_token: str = "<|eos|>"):
    """Return a tokenizer based on mode: 'char' or 'tiny' (BPE trained quickly).
    If mode=='tiny', trains a small BPE on available dataset texts (fast).
    Results are cached to cache_dir.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{mode}_vs{vocab_size}.json"

    if cache_file.exists():
        tok = PreTrainedTokenizerFast(tokenizer_file=str(cache_file))
        tok.add_special_tokens({"eos_token": eos_token})
        return tok

    if mode.lower() == "char":
        # Build a simple character-level tokenizer
        charset = set()
        if hf_ds is not None:
            src_iter = hf_ds["train"]["text"] if isinstance(hf_ds, dict) else hf_ds["text"]
        else:
            src_iter = []
        for txt in list(src_iter)[:50000]:
            charset.update(list(txt))
        # Ensure some basics
        for s in ["\n", " ", ".", ",", "!", "?", "'", '"']:
            charset.add(s)
        # Create vocab
        chars = sorted(charset)
        vocab = {ch: i+4 for i, ch in enumerate(chars)}  # reserve 0..3 for specials
        # Build a PreTrainedTokenizerFast from a Tokenizer model
        model = models.WordLevel(vocab=vocab, unk_token="<|unk|>")
        tok = Tokenizer(model)
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        # Post-processor to append EOS
        tok.post_processor = processors.TemplateProcessing(
            single=f"$A {eos_token}",
            pair=f"$A {eos_token} $B:1 {eos_token}:1",
            special_tokens=[(eos_token, 1)],
        )
        fast = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<|unk|>", eos_token=eos_token)
        fast.save_pretrained(str(cache_dir), legacy_format=False)
        (cache_dir / "tokenizer.json").rename(cache_file)
        return PreTrainedTokenizerFast(tokenizer_file=str(cache_file))

    # Tiny BPE
    wordpiece = Tokenizer(models.BPE(unk_token="<|unk|>"))
    wordpiece.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=2, special_tokens=["<|unk|>", eos_token])
    # Gather a small corpus
    def iter_texts():
        if hf_ds is None:
            return []
        if isinstance(hf_ds, dict):
            for split in ("train", "validation", "val", "test"):
                if split in hf_ds:
                    for t in hf_ds[split]["text"]:
                        yield t
        else:
            for t in hf_ds["text"]:
                yield t
    corpus = list(iter_texts())[:200000]  # cap
    wordpiece.train_from_iterator(corpus, trainer=trainer)
    fast = PreTrainedTokenizerFast(tokenizer_object=wordpiece, unk_token="<|unk|>", eos_token=eos_token)
    fast.save_pretrained(str(cache_dir), legacy_format=False)
    (cache_dir / "tokenizer.json").rename(cache_file)
    return PreTrainedTokenizerFast(tokenizer_file=str(cache_file))

def build_packed_dataloaders_with_tokenizer(train_hfds, val_hfds, test_hfds, tokenizer, seq_len: int, batch_size: int, num_workers: int = 2, nextk_K: int = 0):
    eos_id = tokenizer.eos_token_id
    collate_fn = make_collate_tokens(nextk_K=nextk_K, eos_id=eos_id)

    def make_loader(ds):
        packed = PackedDataset(ds, tokenizer, seq_len, eos_id=eos_id, lookahead_k=nextk_K)
        return DataLoader(packed, batch_size=batch_size, num_workers=num_workers, collate_fn=collate_fn)

    return make_loader(train_hfds), make_loader(val_hfds), make_loader(test_hfds)
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