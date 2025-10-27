from __future__ import annotations
from typing import Optional, Tuple, Dict
import hashlib
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
    max_shard_size: str = "1024MB",
    load_kwargs: Optional[Dict] = None
) -> None:
    """
    Split a corpus deterministically into train/val/test by hashing a stable key,
    then save each split as many Arrow shards capped by `max_shard_size`.

    Provide either `ds_name` (Hugging Face dataset path) or a preloaded `ds` (Dataset).

    Args:
      ds_name: HF repo id.
      config:  HF builder config (aka "name"), e.g. "20231101.simple", "en".
      hf_split: HF split to load from that dataset/config, e.g. "train".
      ds: preloaded map-style Dataset (skips load_dataset).
      load_kwargs: forwarded to datasets.load_dataset (e.g., {'trust_remote_code': True}).
    """
    assert ds_name or ds, "Provide either ds_name or ds"
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if ds is None:
        kw = dict(load_kwargs or {})
        if config is not None:
            ds = load_dataset(ds_name, 
                              name=config, 
                              split=hf_split, 
                              **kw)
        else:
            ds = load_dataset(ds_name, 
                              split=hf_split, 
                              **kw)

    # Stabilize membership by hashing a key per example
    def _bucket(example):
        key = example.get(id_key)
        if key is None:
            key = str(example.get(text_key, "")).strip()  # fallback to text
        h = hashlib.blake2b(key.encode("utf-8"), digest_size=8, key=hash_key).digest()
        return {"__b": int.from_bytes(h, "big") % bucket_mod}

    ds = ds.map(_bucket, num_proc=num_proc)

    # Fractions
    tr, va, te = split_fracs
    assert abs((tr + va + te) - 1.0) < 1e-6, "split_fracs must sum to 1.0"
    tr_end = int(bucket_mod * tr)
    va_end = tr_end + int(bucket_mod * va)

    train = ds.filter(lambda e: e["__b"] < tr_end, num_proc=num_proc)
    val   = ds.filter(lambda e: tr_end <= e["__b"] < va_end, num_proc=num_proc)
    test  = ds.filter(lambda e: va_end <= e["__b"], num_proc=num_proc)

    train.save_to_disk(str(out / "train"), max_shard_size=max_shard_size)
    val.save_to_disk(  str(out / "val"),   max_shard_size=max_shard_size)
    test.save_to_disk( str(out / "test"),  max_shard_size=max_shard_size)


def load_split(
    out_dir: str,
    split: str,
    shard_by_rank: bool = True,
    shuffle_seed: int | None = 42,
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
