# create_shards.py
from __future__ import annotations
import hashlib, itertools, os, json, gzip
from pathlib import Path
from typing import Tuple, Optional, Dict, List
from datasets import load_dataset, Dataset, load_from_disk, concatenate_datasets

def _bucket_for(key: str, *, train=0.9, val=0.05, seed: bytes=b"fixed-seed") -> str:
    """Deterministic split via keyed hash."""
    h = hashlib.blake2b(key.encode("utf-8"), digest_size=8, key=seed).digest()
    r = int.from_bytes(h, "big") / 2**64
    if r < train: return "train"
    if r < train + val: return "val"
    return "test"

def split_and_shard(
    ds_name: str,
    config: Optional[str],
    hf_split: str,
    out_dir: str,
    *,
    id_key: str = "id",
    text_key: str = "text",
    n_samples: int = 1_000_000,
    chunk_rows: int = 20_000,         # RAM cap per in-memory chunk (~few MB)
    shard_prefix: str = "shard",      # file naming
    compress: bool = False,           # True = JSONL.GZ (slower), False = Arrow shards
) -> None:
    """
    Stream exactly n_samples rows, deterministically split into train/val/test,
    and write many small shards to disk. Peak RAM ~ chunk_rows only.
    """
    out = Path(out_dir)
    for sp in ("train", "val", "test"):
        (out / sp).mkdir(parents=True, exist_ok=True)

    # Stream (no full download). HF only fetches what we read via islice.
    stream = load_dataset(ds_name, config, split=hf_split, streaming=True)

    # One in-RAM chunk per split
    buffers: Dict[str, List[Dict]] = {"train": [], "val": [], "test": []}
    counts  : Dict[str, int]       = {"train": 0,  "val": 0,  "test": 0}
    shard_ix: Dict[str, int]       = {"train": 0,  "val": 0,  "test": 0}

    def flush(split: str):
        if not buffers[split]:
            return
        idx = shard_ix[split]
        dest_dir = out / split
        if compress:
            # JSONL.GZ: universally readable, not memory-mapped
            path = dest_dir / f"{shard_prefix}_{idx:05d}.jsonl.gz"
            with gzip.open(path, "wt", encoding="utf-8") as f:
                for row in buffers[split]:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        else:
            # Arrow: fast + memory-mapped (preferred)
            path = dest_dir / f"{shard_prefix}_{idx:05d}"
            ds_chunk = Dataset.from_list(buffers[split])
            ds_chunk.save_to_disk(str(path))
        buffers[split].clear()
        shard_ix[split] += 1

    for ex in itertools.islice(stream, n_samples):
        text = ex.get(text_key)
        if not isinstance(text, str) or not text:
            continue
        key = str(ex.get(id_key)) if ex.get(id_key) is not None else text[:128]
        split = _bucket_for(key)
        buffers[split].append({id_key: key, text_key: text})
        counts[split] += 1
        if len(buffers[split]) >= chunk_rows:
            flush(split)

    # final flush
    for sp in ("train", "val", "test"):
        flush(sp)

    total = sum(counts.values())
    print(f"[SHARD] Wrote {total:,} rows → {out}")
    for sp in ("train", "val", "test"):
        print(f"  {sp:5s}: {counts[sp]:,} rows in {shard_ix[sp]} shards")

def load_split(
    out_dir: str,
    split: str,
    shuffle_seed: Optional[int] = 42,
):
    """
    Load one dataset split (train/val/test) saved by split_and_shard().
    """
    base = Path(out_dir) / split

    # Prefer Arrow shards (fast, memory-mapped)
    arrow_dirs = sorted(p for p in base.iterdir() if p.is_dir())
    if arrow_dirs:
        parts = [load_from_disk(str(p)) for p in arrow_dirs]
        ds = concatenate_datasets(parts) if len(parts) > 1 else parts[0]
    else:
        import glob
        files = sorted(glob.glob(str(base / "*.jsonl.gz")))
        if not files:
            raise FileNotFoundError(f"No shards found under {base}")
        ds = load_dataset("json", data_files={split: files})[split]

    if shuffle_seed is not None:
        ds = ds.shuffle(seed=shuffle_seed)
    return ds
