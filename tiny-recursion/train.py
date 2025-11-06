# -*- coding: utf-8 -*-
"""
train.py — unified trainer for BasicGPT (causal) and TRM (iterative refinement)

Key changes:
- Simple, non-dataclass ModelConfig integration.
- No reflection or dataclass helpers; one source of truth (ModelConfig.__init__ defaults).
- Keeps clean CLI path for both BasicGPT and TRM.
- Logs halting usage metrics to W&B when TRM halting is enabled.
"""

import argparse
import ast
import math
import os
import time
from pathlib import Path
from datasets import load_dataset

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from transformers import AutoTokenizer

# local
from create_shards import split_and_shard, load_split
from create_dataloaders import build_packed_dataloaders, build_packed_dataloaders_with_tokenizer, build_tokenizer_from_mode
from model import BasicGPT, ModelConfig, TRMModel, SequentialTRMModel, RecursiveModel


# ------------------------
# Utils
# ------------------------

def _extract_loss(output):
    """Get a scalar loss from HF-style objects or dicts."""
    if hasattr(output, "loss") and output.loss is not None:
        return output.loss
    if isinstance(output, dict) and "loss" in output and output["loss"] is not None:
        return output["loss"]
    if isinstance(output, (tuple, list)) and len(output) > 0:
        return output[0]
    raise ValueError("Could not extract loss from model output.")


@torch.no_grad()
def evaluate(model, dataloader, device, *, use_trm: bool, n: int, T: int, max_batches=None):
    model.eval()
    total_loss, count = 0.0, 0
    for i, batch in enumerate(dataloader):
        if max_batches and i >= max_batches:
            break

        x = batch.get("input_ids").to(device)
        mask = batch.get("attention_mask")
        labels = batch.get("labels")
        if mask is not None:
            mask = mask.to(device)
        if labels is not None:
            labels = labels.to(device)

        with autocast("cuda", enabled=(device == "cuda" and getattr(model.config, "amp", True))):
            if use_trm:
                out = model(input_ids=x, labels=labels, n=n, T=T)
            else:
                out = model(input_ids=x, attention_mask=mask, labels=labels)

        loss = _extract_loss(out)
        total_loss += float(loss.item())
        count += 1

    model.train()
    avg = total_loss / max(1, count)
    return avg, math.exp(avg)


def _enable_flash_sdpa_if_possible():
    if not torch.cuda.is_available():
        return
    try:
        from torch.backends.cuda import sdp_kernel
        if hasattr(sdp_kernel, "enable_flash"):
            sdp_kernel.enable_flash(True)
        if hasattr(sdp_kernel, "enable_mem_efficient"):
            sdp_kernel.enable_mem_efficient(True)
        if hasattr(sdp_kernel, "enable_math"):
            sdp_kernel.enable_math(False)
        print("SDPA Flash enabled.")
    except Exception as e:
        print("[WARN] Could not configure SDPA Flash backend:", e)


def _build_model(config: ModelConfig, *, use_trm: bool, n: int, T: int, device: str, use_halting: bool = False, next_k: int = 0, model_kind: str = 'auto', recursion_steps: int = 2, next_k=getattr(args, 'next_k', 0), model_kind=getattr(args, 'model_kind', 'auto'), recursion_steps=getattr(args, 'recursion_steps', 2)):
    if use_trm:
        if model_kind == 'recursive':
            model = RecursiveModel(config, recursion_steps=recursion_steps).to(device)
            print(f"[MODEL] RecursiveModel, steps={recursion_steps}.")
        elif next_k and next_k > 0:
            model = SequentialTRMModel(config, n=n, T=T, n_layers=2, use_halting_head=use_halting).to(device)
            print("[MODEL] SequentialTRMModel (next-K). Halting:", use_halting)
        else:
            model = TRMModel(config).to(device)
            print("[MODEL] TRMModel (single-token).")
    else:
        model = BasicGPT(config).to(device)
        print("[MODEL] BasicGPT (causal LM).")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[MODEL] Parameters: {total_params/1e6:.2f}M")
    return model

def _save_checkpoint(model, config, optimizer, 
                     step: int, val_metric: float, 
                     out_dir: str, tag: str, 
                     run_name: str = None, samples_seen: int = 0):
    ckpt_dir = Path(out_dir) / (run_name or f"checkpoint-{tag}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(ckpt_dir, safe_serialization=False)
    config.save_pretrained(ckpt_dir)

    torch.save(
        {
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
            "step": step,
            "val_metric": val_metric,
            "samples_seen": samples_seen
        },
        ckpt_dir / "training_state.pt",
    )

    print(f"[CKPT] Saved checkpoint to {ckpt_dir}")
    return ckpt_dir


# ------------------------
# Config helpers
# ------------------------

def _coerce_value(s: str, default):
    """Coerce string to type of default (for tuple, int, float, bool, etc.)."""
    if s is None:
        return default
    t = type(default)
    if isinstance(default, tuple):
        parts = [p.strip() for p in s.split(",")]
        inner_t = float if not default else type(default[0])
        return tuple(inner_t(p) for p in parts if p != "")
    if t is int:
        return int(s)
    if t is float:
        return float(s)
    if t is bool:
        return s.lower() in ("1", "true", "t", "yes", "y")
    try:
        return ast.literal_eval(s)
    except Exception:
        return s


def build_config_from_args(args) -> ModelConfig:
    """Start from ModelConfig defaults; override with any CLI-provided values."""
    base = ModelConfig()
    # simple subset of fields we may override
    names = [
        "vocab_size","d_model","n_head","n_layer","d_ff","context_length","dropout",
        "attn_implementation","init_range","layer_norm_eps","out_dir","data_out_dir",
        "tie_weights","tie_word_embeddings","amp","use_cache",
        "lr","betas","weight_decay","grad_clip","grad_accum_steps",
        "log_every","eval_every_steps","save_every_epochs",
        "K","n","T",
    ]
    kwargs = {}
    for name in names:
        val = getattr(args, name, None)
        default = getattr(base, name)
        if val is None:
            kwargs[name] = default
        elif name == "betas":
            parts = [p.strip() for p in str(val).split(",")]
            kwargs[name] = tuple(float(p) for p in parts) if len(parts) == 2 else default
        else:
            kwargs[name] = _coerce_value(val, default)
    return ModelConfig(**kwargs)

def _decode(ids, tokenizer, max_chars=120):
    # ids: (L,) LongTensor
    txt = tokenizer.decode(ids.tolist(), skip_special_tokens=True)
    txt = txt.replace("\n", " ")
    return txt[:max_chars]

@torch.no_grad()
def preview_trm_refinement(model, tokenizer, batch, device, *, k=3, max_chars=120):
    model.eval()
    x = batch["input_ids"].to(device)
    labels = batch.get("labels")
    if labels is not None:
        labels = labels.to(device)

    with autocast("cuda", enabled=(device == "cuda" and getattr(model.config, "amp", True))):
        out = model(input_ids=x, labels=labels, return_token_traces=True)

    token_traces = out.__dict__.get("token_traces", None)
    if not token_traces:
        print("[TRM PREVIEW] No token traces available.")
        model.train()
        return

    k = min(k, x.size(0))
    print("--------")
    gt = [_decode(x[i], tokenizer, max_chars) for i in range(k)]
    print(f"Ground truth: {gt}")
    for t, tok_ids in enumerate(token_traces, start=1):
        itxt = [_decode(tok_ids[i], tokenizer, max_chars) for i in range(k)]
        print(f"Iteration {t}: {itxt}")
    print("--------")
    model.train()

def _halting_wandb_payload(halting: dict, T: int) -> dict:
    """
    Compute summary halting metrics for W&B.
    halting: {
      'cum_H': (B,),
      'halt_step': (B,), 1..executed_T (or 0 if never explicitly halted),
      'p_t': list of (B,)
    }
    """
    if not halting or "halt_step" not in halting:
        return {}
    halt_step = halting["halt_step"]  # (B,)
    # samples with explicit halting step > 0
    halted_mask = (halt_step > 0)
    halted_count = int(halted_mask.sum().item())
    B = halt_step.numel()

    # unused outer iterations for halted samples
    unused = torch.clamp(T - halt_step, min=0)
    unused_halted = torch.where(halted_mask, unused, torch.zeros_like(unused))
    total_unused = int(unused_halted.sum().item())
    mean_unused = float(unused_halted.float().mean().item()) if B > 0 else 0.0

    # used fraction: for non-halted treat as 1.0
    used_frac = torch.where(halted_mask, halt_step.float() / float(T), torch.ones_like(halt_step, dtype=torch.float))
    used_frac_mean = float(used_frac.mean().item())

    payload = {
        "halt/unused_outer_total": total_unused,
        "halt/unused_outer_mean": mean_unused,
        "halt/used_frac_mean": used_frac_mean,
    }
    return payload


# ------------------------
# CLI
# ------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    from argparse import BooleanOptionalAction

    # Core config overrides
    parser.add_argument("--vocab_size", type=int, default=None)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--n_head", type=int, default=None)
    parser.add_argument("--n_layer", type=int, default=None)
    parser.add_argument("--d_ff", type=int, default=None)
    parser.add_argument("--context_length", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--attn_implementation", type=str, default=None)
    parser.add_argument("--init_range", type=float, default=None)
    parser.add_argument("--layer_norm_eps", type=float, default=None)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--data_out_dir", type=str, default=None)
    parser.add_argument("--tie_weights", action=BooleanOptionalAction, default=None)
    parser.add_argument("--tie_word_embeddings", action=BooleanOptionalAction, default=None)
    parser.add_argument("--amp", action=BooleanOptionalAction, default=None)
    parser.add_argument("--use_cache", action=BooleanOptionalAction, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--betas", type=str, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--grad_clip", type=float, default=None)
    parser.add_argument("--grad_accum_steps", type=int, default=None)
    parser.add_argument("--log_every", type=int, default=None)
    parser.add_argument("--eval_every_steps", type=int, default=None)
    parser.add_argument("--save_every_epochs", type=int, default=None)
    parser.add_argument("--K", type=int, default=None)
    parser.add_argument("--n", type=int, default=None)
    parser.add_argument("--T", type=int, default=None)

    # runtime/train flags
    parser.add_argument("--do_train", action=BooleanOptionalAction, default=False)
    parser.add_argument("--eval_only", action=BooleanOptionalAction, default=False)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--persistent_workers", action=BooleanOptionalAction, default=False)

    # dataset/sharding
    parser.add_argument("--ds_name", type=str, default="wikimedia/wikipedia")
    parser.add_argument("--ds_config", type=str, default="20231101.en")
    parser.add_argument("--hf_split", type=str, default="train")
    parser.add_argument("--shard_max_examples", type=int, default=50_000)
    parser.add_argument("--max_shard_size", type=str, default="512MB")
    parser.add_argument("--shard_num_proc", type=int, default=4)

    # I/O & logging
    parser.add_argument("--load_path", type=str, default=None)
    parser.add_argument("--save_best", action=BooleanOptionalAction, default=True)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--project", type=str, default="llm_playground")
    parser.add_argument("--wandb", action=BooleanOptionalAction, default=True)
    parser.add_argument("--use_trm", action=BooleanOptionalAction, default=False)

    # NEW: halting toggle (applies only to TRM)
    parser.add_argument("--use_halting", action=BooleanOptionalAction, default=False,
                        help="Enable halting head for TRM (adaptive early-stop across outer iterations).")

    return parser.parse_args()

# ------------------------
# Main
# ------------------------

def main():
    args = parse_args()
    config = build_config_from_args(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[RUNTIME] Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.model_max_length = int(1e12)
    model = _build_model(
        config,
        use_trm=args.use_trm,
        n=args.n or config.n,
        T=args.T or config.T,
        device=device,
        use_halting=bool(getattr(args, "use_halting", False, next_k=getattr(args, 'next_k', 0), model_kind=getattr(args, 'model_kind', 'auto'), recursion_steps=getattr(args, 'recursion_steps', 2))),
    )

    if args.load_path:
        _load_checkpoint_into(model, args.load_path, map_location="cpu")
        model.to(device)
    _enable_flash_sdpa_if_possible()
    
    # Create run name folder
    if args.run_name:
        config.out_dir = os.path.join(config.out_dir, args.run_name)
        os.makedirs(config.out_dir, exist_ok=True)
    
    split_and_shard(
        ds_name=args.ds_name,
        config=args.ds_config,
        hf_split=args.hf_split,
        out_dir=config.data_out_dir,
        n_samples=args.shard_max_examples or 1_000_000,
        chunk_rows=20_000,
        compress=False
    )

    
# Set seeds early
set_global_seed(args.seed)

# Dataset & tokenizer
if args.dataset.lower() in ('tinystories','tiny','ts'):
    hf_train = load_dataset('roneneldan/TinyStories', split='train')
    hf_val = load_dataset('roneneldan/TinyStories', split='validation')
    hf_test = hf_val
    # Build tokenizer quickly
    tokenizer = build_tokenizer_from_mode(args.tokenizer_mode, hf_ds={'train': hf_train, 'validation': hf_val}, cache_dir=str(Path(args.out_dir)/'tok_cache'))
    train_loader, eval_loader, test_loader = build_packed_dataloaders_with_tokenizer(
        hf_train, hf_val, hf_test, tokenizer, seq_len=config.context_length, batch_size=args.batch_size, num_workers=args.num_workers, nextk_K=args.next_k if hasattr(args, 'next_k') else 0
    )
else:
    # Fallback to shards on disk using existing path
    train_hfds = load_split(config.data_out_dir, "train", shuffle_seed=42)
    eval_hfds = load_split(config.data_out_dir, "val", shuffle_seed=43)
    test_hfds = load_split(config.data_out_dir, "test", shuffle_seed=44)
    tokenizer = build_tokenizer_from_mode(args.tokenizer_mode, hf_ds={'train': train_hfds, 'validation': eval_hfds}, cache_dir=str(Path(args.out_dir)/'tok_cache'))
    train_loader, eval_loader, test_loader = build_packed_dataloaders_with_tokenizer(
        train_hfds, eval_hfds, test_hfds, tokenizer, seq_len=config.context_length, batch_size=args.batch_size, num_workers=args.num_workers, nextk_K=args.next_k if hasattr(args, 'next_k') else 0
    )

    eval_hfds = load_split(config.data_out_dir, "val", shuffle_seed=43)
    test_hfds = load_split(config.data_out_dir, "test", shuffle_seed=44)

    train_loader, eval_loader, test_loader = build_packed_dataloaders(
        train_hfds, eval_hfds, test_hfds,
        tokenizer_name="gpt2",
        seq_len=config.context_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

    # Peek
    first = next(iter(train_loader))
    xb, yb = first["input_ids"], first["labels"]
    print("[DATA] Sample batch:", xb.shape, yb.shape)
    print("[DATA] Sample decode:", tokenizer.decode(xb[0].tolist(), skip_special_tokens=False)[:120].replace("\n", " "))

    # W&B
    if args.wandb:
        import wandb
        wandb.init(project=args.project, name=args.run_name)
        wandb.define_metric("samples_seen")          # declare custom step axis
        wandb.define_metric("*", step_metric="samples_seen")  # all metrics use it
        print(f"[W&B] Logging enabled. Run name: {args.run_name or '(auto-generated)'}")
        wandb.watch(model, log="gradients", log_freq=200)

    # Train
    optimizer = AdamW(model.parameters(), lr=config.lr, betas=config.betas, weight_decay=config.weight_decay)
    scaler = GradScaler("cuda", enabled=(device == "cuda" and getattr(config, "amp", True)))

    model.train().to(device)
    global_step = 0
    samples_seen = 0
    best_metric = float("inf")
    running = 0.0

    for epoch in range(args.epochs):
        print(f"[TRAIN] Epoch {epoch+1}/{args.epochs}")
        for step, batch in enumerate(train_loader, start=1):
            x = batch.get("input_ids").to(device)
            mask = batch.get("attention_mask")
            labels = batch.get("labels")
            if mask is not None:
                mask = mask.to(device)
            if labels is not None:
                labels = labels.to(device)

            batch_size_actual = x.size(0)
            samples_seen += batch_size_actual

            with autocast("cuda", enabled=(device == "cuda" and config.amp)):
                if args.use_trm:
                    out = model(input_ids=x, labels=labels, n=args.n or config.n, T=args.T or config.T)
                else:
                    out = model(input_ids=x, attention_mask=mask, labels=labels)
                loss = _extract_loss(out) / config.grad_accum_steps

            # Halting metrics (if enabled)
            if args.use_trm and getattr(args, "use_halting", False) and args.wandb:
                halting = getattr(out, "__dict__", {}).get("halting", None)
                executed_T = getattr(out, "__dict__", {}).get("executed_T", None)
                log_payload = _halting_wandb_payload(halting, T=(args.T or config.T))
                if executed_T is not None:
                    log_payload["halt/executed_T"] = executed_T
                if log_payload:
                    log_payload["samples_seen"] = samples_seen
                    try:
                        import wandb
                        wandb.log(log_payload)
                    except Exception:
                        pass

            scaler.scale(loss).backward()

            if (step % config.grad_accum_steps) == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            global_step += 1
            running += float(loss.item()) * config.grad_accum_steps

            # Logging averaged training loss
            if global_step % getattr(config, "log_every", 50) == 0:
                avg_loss = running / getattr(config, "log_every", 50)
                print(f"[TRAIN] step {global_step} | samples {samples_seen} | loss {avg_loss:.4f}")
                if args.wandb:
                    try:
                        import wandb
                        wandb.log({"train/loss_avg": avg_loss, "samples_seen": samples_seen})
                    except Exception:
                        pass
                running = 0.0

            # Evaluation (use samples_seen as axis)
            if global_step % getattr(config, "eval_every_steps", 500) == 0:
                val_loss, val_ppl = evaluate(model, eval_loader, device,
                                             use_trm=args.use_trm,
                                             n=args.n or config.n,
                                             T=args.T or config.T)
                print(f"[EVAL] step {global_step} | samples {samples_seen} | val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f}")
                if args.wandb:
                    try:
                        import wandb
                        wandb.log({"val/loss": val_loss, "val/ppl": val_ppl, "samples_seen": samples_seen})
                    except Exception:
                        pass

                # Show TRM refinement process over all T iterations
                try:
                    preview_batch = next(iter(eval_loader))
                    print(f"Step {global_step}:")
                    preview_trm_refinement(model, tokenizer, preview_batch, device, k=3, max_chars=120)
                except StopIteration:
                    pass

                # Save model if it has improved
                if args.save_best and val_ppl < best_metric:
                    best_metric = val_ppl
                    _save_checkpoint(
                        model,
                        config,
                        optimizer,
                        global_step,
                        best_metric,
                        config.out_dir,
                        f"{args.run_name}_[best]",
                        run_name=args.run_name,
                        samples_seen=samples_seen
                    )

    # Final eval + save
    val_loss, val_ppl = evaluate(model, eval_loader, device, use_trm=args.use_trm, n=args.n or config.n, T=args.T or config.T)
    print(f"[FINAL] val_loss {val_loss:.4f} | val_ppl {val_ppl:.2f}")
    if args.wandb:
        try:
            import wandb
            wandb.log({
                "final/val_loss": val_loss,
                "final/val_ppl": val_ppl,
                "final/epochs": args.epochs,
                "samples_seen": samples_seen
            })
            wandb.finish()
        except Exception:
            pass

    _save_checkpoint(model, config, optimizer, global_step, val_ppl, config.out_dir, f"e{args.epochs}_step{global_step}", args.run_name, samples_seen=samples_seen)

if __name__ == "__main__":
    main()
