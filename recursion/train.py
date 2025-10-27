
# -*- coding: utf-8 -*-
"""
train.py — CLI-driven training script for BasicGPT

Additions in this revision:
- Includes `_extract_loss()` and `evaluate()` helpers you provided (adapted to accept model/device).
- Keeps gradient accumulation (config.grad_accum_steps).
- Makes FlashAttention enabling version-safe across PyTorch releases.

Usage examples:
  python train.py --d_model 512 --max_len 256 --epochs 5 --grad_accum_steps 16
  python train.py --attn_implementation flash --hf_split "train[:5%]"
"""

import argparse
import ast
import math
import os
import time
from dataclasses import fields as dataclass_fields
import wandb

import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from transformers import AutoTokenizer

from create_shards import split_and_shard, load_split
from create_dataloaders import build_packed_dataloaders
from model import BasicGPT, ModelConfig

def _extract_loss(output):
    """
    Robustly extract a scalar loss from various output shapes:
    - CausalLMOutput-like objects (have .loss)
    - dicts with 'loss'
    - tuples/lists with loss in first position
    """
    if hasattr(output, "loss") and output.loss is not None:
        return output.loss
    if isinstance(output, dict) and "loss" in output:
        return output["loss"]
    if isinstance(output, (tuple, list)) and len(output) > 0:
        return output[0]
    raise ValueError("Could not extract loss from model output.")


@torch.no_grad()
def evaluate(model, dataloader, device, max_batches=None):
    model.eval()
    losses, n = 0.0, 0

    for i, batch in enumerate(dataloader):
        if max_batches and i >= max_batches:
            break

        ids = batch.get("input_ids").to(device)
        mask = batch.get("attention_mask")
        labels = batch.get("labels")

        if mask is not None:
            mask = mask.to(device)
        if labels is not None:
            labels = labels.to(device)

        out = model(input_ids=ids, attention_mask=mask, labels=labels)
        loss = _extract_loss(out)
        losses += float(loss.item())
        n += 1

    model.train()
    avg = losses / max(1, n)
    return avg, math.exp(avg)


# ------------------------
# Argparse over full ModelConfig
# ------------------------

def _add_all_config_args(parser: argparse.ArgumentParser):
    from argparse import BooleanOptionalAction
    for f in dataclass_fields(ModelConfig):
        name = f.name
        ann = f.type
        default = f.default
        flag = f"--{name}"
        if ann is bool:
            parser.add_argument(flag, dest=name, action=BooleanOptionalAction, default=None,
                                help=f"(bool) default={default}")
        else:
            parser.add_argument(flag, type=str, default=default, help=f"(type {ann}) default={default}")

def _coerce_value(s: str, target_default):
    if s is None:
        return None
    if target_default is None:
        try:
            return ast.literal_eval(s)
        except Exception:
            return s

    t = type(target_default)
    if isinstance(target_default, tuple):
        inner_t = float if len(target_default) == 0 else type(target_default[0])
        try:
            val = ast.literal_eval(s)
            if isinstance(val, (list, tuple)):
                return tuple(inner_t(x) for x in val)
        except Exception:
            pass
        parts = [p.strip() for p in s.split(",")]
        return tuple(inner_t(p) for p in parts if p != "")
    if t is int:
        return int(s)
    if t is float:
        return float(s)
    if t is bool:
        return s.lower() in ("1","true","t","yes","y")
    if t is str:
        return s
    try:
        return ast.literal_eval(s)
    except Exception:
        return s


def parse_args():
    parser = argparse.ArgumentParser()
    _add_all_config_args(parser)

    # Training/runtime flags
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Micro-batch size per step.")
    parser.add_argument("--num_workers", type=int, default=0, help="DataLoader workers (0 recommended in notebooks).")
    parser.add_argument("--prefetch_factor", type=int, default=2, help="DataLoader prefetch factor.")
    parser.add_argument("--persistent_workers", action=argparse.BooleanOptionalAction, default=False)

    # Dataset/sharding flags
    parser.add_argument("--ds_name", type=str, default="wikimedia/wikipedia")
    parser.add_argument("--ds_config", type=str, default="20231101.en")
    parser.add_argument("--hf_split", type=str, default="train[:10%]")
    # parser.add_argument("--data_out_dir", type=str, default="/workspace/recursion/data/shards")

    return parser.parse_args()


def build_config_from_args(args) -> ModelConfig:
    cfg_kwargs = {}
    defaults = {f.name: f.default for f in dataclass_fields(ModelConfig)}
    for f in dataclass_fields(ModelConfig):
        name = f.name
        default = defaults[name]
        provided = getattr(args, name, None)
        if provided is None:
            continue
        if isinstance(default, bool):
            cfg_kwargs[name] = provided
        else:
            # Only coerce CLI strings; if argparse already gave us a typed value, keep it
            cfg_kwargs[name] = _coerce_value(provided, default) if isinstance(provided, str) else provided


    if "attn_implementation" not in cfg_kwargs:
        cfg_kwargs["attn_implementation"] = defaults["attn_implementation"]
        cfg_kwargs["_attn_implementation_internal"] = defaults["_attn_implementation_internal"]

    return ModelConfig(**cfg_kwargs)


# ------------------------
# Version-safe Flash enabling
# ------------------------

def _enable_flash_if_requested(config):
    if config.attn_implementation != "flash":
        return
    # Try both the "function" API and the "module with enable_*" API
    try:
        from torch.backends.cuda import sdp_kernel
        # If sdp_kernel is callable (old API), call it with kwargs.
        if callable(sdp_kernel):
            sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
            return
        # Otherwise assume it has enable_* attributes (newer API)
        if hasattr(sdp_kernel, "enable_flash"):
            sdp_kernel.enable_flash(True)
        if hasattr(sdp_kernel, "enable_mem_efficient"):
            sdp_kernel.enable_mem_efficient(True)
        if hasattr(sdp_kernel, "enable_math"):
            sdp_kernel.enable_math(False)
    except Exception as e:
        print("[WARN] Could not configure Flash SDPA backend:", e)


# ------------------------
# Main training flow
# ------------------------

def main():
    args = parse_args()
    config = build_config_from_args(args)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = int(1e12)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    DO_TRAIN = False
    if input("Train? (Y/N)").lower() == 'y':
        DO_TRAIN = True

        if input("Load existing model? (Y/N)") == 'y':
            model, checkpoint = BasicGPT.load_weights(f"/workspace/recursion/trained_models/simple_baseline_best.pt", config)
            model.to(device)
            print("Successfully loaded model weights from earlier run. Continuing...")
        else:
            print("Initialized model. Continuing...")
            model = BasicGPT(config).to(device)
            print(model)
            wandb.watch(model, log="gradients", log_freq=200)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Model params: {total_params/1e6:.2f}M")

        _enable_flash_if_requested(config)

        # Shard dataset
        split_and_shard(
            ds_name=args.ds_name,
            config=args.ds_config,
            hf_split=args.hf_split,
            out_dir=args.data_out_dir,
            id_key="id",
            text_key="text",
            split_fracs=(0.98, 0.01, 0.01),
            max_shard_size="1024MB",
            num_proc=8,
        )

        train_hfds = load_split(args.data_out_dir, "train", shard_by_rank=True, shuffle_seed=42)
        eval_hfds  = load_split(args.data_out_dir, "val",   shard_by_rank=True, shuffle_seed=43)
        test_hfds  = load_split(args.data_out_dir, "test",  shard_by_rank=True, shuffle_seed=44)

        train_loader, eval_loader, test_loader = build_packed_dataloaders(
            train_hfds, eval_hfds, test_hfds,
            tokenizer_name="gpt2",
            seq_len=config.max_len,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            persistent_workers=args.persistent_workers,
        )

        # Peek
        it = iter(train_loader)
        first = next(it)
        xb, yb = first["input_ids"], first["labels"]
        print("Sample batch:", xb.shape, yb.shape)
        text = tokenizer.decode(xb[1].tolist(), skip_special_tokens=False)
        print(f"Decoded sample batch: {text[:500]}...")

        optimizer = AdamW(model.parameters(),
                          lr=config.lr,
                          betas=config.betas,
                          weight_decay=config.weight_decay)

        scaler = GradScaler("cuda", enabled=(device == "cuda" and getattr(config, "amp", True)))

        # ---- Training loop with gradient accumulation ----
        global_step = 0
        best_perplexity = float("inf")
        print("[TRAINING]: Starting training...")
        model.to(device).train()
        t0 = time.time()
        running = 0.0

        for epoch in range(args.epochs):
            print(f"[TRAINING]: Starting epoch {epoch+1}/{args.epochs}...")
            for step, batch in enumerate(train_loader, start=1):
                x = batch.get("input_ids").to(device)
                mask = batch.get("attention_mask")
                labels = batch.get("labels")
                if mask is not None:
                    mask = mask.to(device)
                if labels is not None:
                    labels = labels.to(device)

                with autocast("cuda", enabled=(device == "cuda" and config.amp)):
                    out = model(input_ids=x, attention_mask=mask, labels=labels)
                    loss = _extract_loss(out) / config.grad_accum_steps

                scaler.scale(loss).backward()

                # Step every grad_accum_steps
                if (step % config.grad_accum_steps) == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                global_step += 1
                running += float(loss.item()) * config.grad_accum_steps  # undo the division for logging

                # Log every N steps (use config.log_every if present; default 50)
                if global_step % getattr(config, "log_every", 50) == 0:
                    print(f"[TRAINING]: epoch {epoch+1} | step {global_step} | loss {running / getattr(config, 'log_every', 50):.4f}")
                    wandb.log({"train/loss_avg": loss.item()}, step=global_step)
                    running = 0.0

                # Eval every M steps
                if global_step % getattr(config, "eval_every_steps", 500) == 0:
                    eval_loss, eval_ppl = evaluate(model, eval_loader, device)
                    print(f"[TRAINING - EVAL] step {global_step} | val_loss {eval_loss:.4f} | val_ppl {eval_ppl:.2f}")
                    wandb.log({
                        "val/loss": eval_loss,
                        "val/ppl": eval_ppl,
                    }, step=global_step)
                    
                    if eval_ppl < best_perplexity:
                        best_perplexity = eval_ppl
                        os.makedirs(config.out_dir, exist_ok=True)
                        path = os.path.join(config.out_dir, f"{config.project_name}_best.pt")
                        torch.save({
                            "config": vars(config),
                            "model_state": model.state_dict(),
                            "optimizer_state": optimizer.state_dict(),
                            "step": global_step,
                            "val_ppl": eval_ppl,
                        }, path)
                        print("Saved best:", path)

        # Final eval + save
        final_eval_loss, final_eval_ppl = evaluate(model, eval_loader, device)

        wandb.log({
            "final/val_loss": final_eval_loss,
            "final/val_ppl": final_eval_ppl,
            "final/epochs": args.epochs,
        }, step=global_step)
        
        elapsed = time.time() - t0
        print(f"Epoch {args.epochs} in {elapsed:.1f}s | val_loss {final_eval_loss:.4f} | val_ppl {final_eval_ppl:.2f}")
        os.makedirs(config.out_dir, exist_ok=True)
        path = os.path.join(config.out_dir, f"{config.project_name}_e{args.epochs}_step{global_step}.pt")
        torch.save({
            "config": vars(config),
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "step": global_step,
            "val_ppl": final_eval_ppl,
        }, path)
        wandb.finish()
        print("Saved:", path)

    else:
        # Non-training branch: just allow loading a model to inspect/print
        if input("Load existing model? (Y/N)") == 'y':
            model, checkpoint = BasicGPT.load_weights(f"/workspace/recursion/trained_models/{config.project_name}.pt", config)
            print(model)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Model params: {total_params/1e6:.2f}M")
            model.to(device)
            print("Successfully loaded model.")


if __name__ == "__main__":
    project = "llm_playground"
    
    wandb.init(
        project=project
    )
    print('Weights and Biases logging enabled!')
    
    main()
