from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import Dataset, IterableDataset
from typing import Optional, Iterable, List

def get_tokenizer(name: str = "gpt2"):
    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

class TextBlocks(Dataset):
    """In-memory contiguous blocks with prediction horizon K."""
    def __init__(self, texts: list, tokenizer, block_size: int, K: int):
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.K = K
        ids: List[int] = []
        for t in texts:
            enc = tokenizer(t, add_special_tokens=False)["input_ids"]
            ids.extend(enc + [tokenizer.eos_token_id])
        self.blocks = []
        step = block_size  # non-overlapping windows
        for i in range(0, max(0, len(ids) - block_size - K), step):
            window = ids[i:i+block_size+K]
            if len(window) == block_size + K:
                x = window[:block_size]
                # standard next-token labels inside the block
                y = window[1:block_size+1]
                future = window[block_size:]
                self.blocks.append((x, y, future))

    def __len__(self): return len(self.blocks)

    def __getitem__(self, idx):
        x, y, future = self.blocks[idx]
        return { "input_ids": x, "labels": y, "future": future }

class TextBlocksStream(IterableDataset):
    """Stream rows and yield blocks for horizon K without loading everything in memory."""
    def __init__(self, ds_iter: Iterable, tokenizer, block_size: int, n_rows: int, K: int):
        self.ds_iter = ds_iter
        self.tok = tokenizer
        self.block = block_size
        self.n_rows = n_rows
        self.K = K

    def __iter__(self):
        buffer: List[int] = []
        seen = 0
        for row in self.ds_iter:
            text = row.get("text", "")
            ids = self.tok(text, add_special_tokens=False)["input_ids"]
            buffer.extend(ids + [self.tok.eos_token_id])
            seen += 1
            # emit as many windows as possible
            while len(buffer) >= self.block + self.K:
                window = buffer[:self.block+self.K]
                x = window[:self.block]
                y = window[1:self.block+1]
                future = window[self.block:]
                yield {"input_ids": x, "labels": y, "future": future}
                buffer = buffer[self.block:]
            if self.n_rows is not None and seen >= self.n_rows:
                break

def load_tinystories(split: str = "train",
                     tokenizer_name: str = "gpt2",
                     block_size: int = 256,
                     K: int = 1,
                     limit: Optional[int] = None,
                     n_rows: Optional[int] = None):
    tok = get_tokenizer(tokenizer_name)
    if n_rows is not None:
        ds = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
        return TextBlocksStream(ds, tok, block_size, n_rows, K), tok
    else:
        ds = load_dataset("roneneldan/TinyStories", split=split)
        if limit is not None:
            ds = ds.select(range(min(limit, len(ds))))
        texts = ds["text"]
        return TextBlocks(texts, tok, block_size, K), tok
