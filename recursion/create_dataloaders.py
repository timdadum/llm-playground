# -*- coding: utf-8 -*-
"""
Helper to build token-packed DataLoaders for LLM-style training.
Usage:
    from hf_packed_dataloaders import build_packed_dataloaders

    train_loader, val_loader, test_loader = build_packed_dataloaders(
        train, val, test,
        tokenizer_name="gpt2",
        seq_len=256,
        batch_size=32,
    )
"""

import torch
from torch.utils.data import IterableDataset, DataLoader
from transformers import AutoTokenizer
from datasets import Dataset

class PackedDataset(IterableDataset):
    def __init__(self, hf_ds, tokenizer, seq_len, eos_id):
        self.hf_ds = hf_ds
        self.tok = tokenizer
        self.L = seq_len
        self.eos = eos_id

    def __iter__(self):
        buffer = []
        for ex in self.hf_ds:
            text = ex.get("text")
            if not isinstance(text, str):
                continue
            ids = self.tok(text, add_special_tokens=False)["input_ids"]
            if self.eos is not None:
                ids.append(self.eos)
            buffer.extend(ids)
            while len(buffer) >= self.L:
                chunk = buffer[:self.L]
                buffer = buffer[self.L:]
                yield {"input_ids": torch.tensor(chunk, dtype=torch.long)}
        # drop remainder silently

def collate_tokens(batch):
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.ones_like(input_ids)
    labels = input_ids.clone()
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_packed_dataloaders(
    train: Dataset,
    val: Dataset,
    test: Dataset,
    tokenizer_name: str="gpt2",
    seq_len: int=256,
    batch_size: int=32,
    num_workers: int=8
):
    """
    Returns (train_loader, val_loader, test_loader) that yield fixed-length
    token batches ready for causal LM training.

    Args:
        train/val/test: Hugging Face datasets (map- or iterable-style)
        tokenizer_name: any HF tokenizer name or path
        seq_len: fixed sequence length
        batch_size: batch size
        num_workers: DataLoader workers
    """

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    eos_id = tokenizer.eos_token_id or tokenizer.pad_token_id

    def make_loader(hf_ds):
        packed = PackedDataset(hf_ds, tokenizer, seq_len, eos_id)
        return DataLoader(
            packed,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=collate_tokens,
        )

    return make_loader(train), make_loader(val), make_loader(test)