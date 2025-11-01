from typing import Optional, Tuple, Dict
import hashlib
import itertools
import shutil
from pathlib import Path
from datasets import load_dataset, Dataset, load_from_disk

def split_and_shard(
    ds_name: Optional[str] = None,
    config: Optional[str] = None,
    hf_split: str = "train",
    ds: Optional[Dataset] = None,
    out_dir: str = "./shards",
    id_key: str = "id",
    text_key: str = "text",
    split_fracs: Tuple[float, float, float] = (0.98, 0.01, 0.01),
    bucket_mod: int = 10_000,
    hash_key: bytes = b"fixed-seed",
    num_proc: int = 8,
    max_shard_size: str = "512MB",
    load_kwargs: Optional[Dict] = None,
    shard_max_examples: int = 50_000
) -> None:
    """
    Stream up to `shard_max_examples` from the HF dataset, deterministically split into
    train/val/test by hashing a stable key, and save each split as Arrow shards.

    Pros: avoids downloading the full corpus.
    """

    assert ds_name or ds, "Provide either ds_name or ds"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 🔥 Always remove any stale dirs first (prevents Windows OSError)
    for split in ["train", "val", "test"]:
        p = out / split
        if p.exists():
            shutil.rmtree(p)

    # === Step 1: load a small streaming subset ===
    if ds is None:
        kw = dict(load_kwargs or {})
        stream = load_dataset(
            ds_name,
            name=config,
            split=hf_split,
            streaming=True,
            **kw,
        )

        # Materialize only N examples
        small_iter = itertools.islice(stream, shard_max_examples)
        small_list = list(small_iter)
        ds = Dataset.from_list(small_list)
        print(f"[SHARD] Streamed and materialized {len(ds)} examples from '{ds_name}' ({hf_split}).")

    # === Step 2: deterministic hash bucketing ===
    def _bucket(example):
        key = example.get(id_key)
        if key is None:
            key = str(example.get(text_key, "")).strip()
        h = hashlib.blake2b(key.encode("utf-8"), digest_size=8, key=hash_key).digest()
        return {"__b": int.from_bytes(h, "big") % bucket_mod}

    ds = ds.map(_bucket, num_proc=num_proc)

    # === Step 3: deterministic split ===
    tr, va, te = split_fracs
    assert abs((tr + va + te) - 1.0) < 1e-6, "split_fracs must sum to 1.0"
    tr_end = int(bucket_mod * tr)
    va_end = tr_end + int(bucket_mod * va)

    train = ds.filter(lambda e: e["__b"] < tr_end, num_proc=num_proc)
    val   = ds.filter(lambda e: tr_end <= e["__b"] < va_end, num_proc=num_proc)
    test  = ds.filter(lambda e: va_end <= e["__b"], num_proc=num_proc)

    # === Step 4: save each split ===
    def _safe_save_split(ds_split, name):
        target = out / name
        target.mkdir(parents=True, exist_ok=True)
        ds_split.save_to_disk(str(target), max_shard_size=max_shard_size)
        print(f"[SHARD] Saved {name} split ({len(ds_split)} samples) to {target}")

    _safe_save_split(train, "train")
    _safe_save_split(val, "val")
    _safe_save_split(test, "test")

def load_split(
    out_dir: str,
    split: str,
    shard_by_rank: bool = True,
    shuffle_seed: int = 42,
):
    """
    Load a split saved by `split_and_shard` and (optionally) shard by DDP rank.
    Returns a map-style Dataset (Arrow, memory-mapped).
    """
    ds = load_from_disk(str(Path(out_dir) / split))

    if shard_by_rank:
        try:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                ds = ds.shard(num_shards=dist.get_world_size(), index=dist.get_rank())
        except Exception:
            pass

    if shuffle_seed is not None:
        ds = ds.shuffle(seed=shuffle_seed)
    return ds
