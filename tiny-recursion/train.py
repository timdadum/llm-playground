import argparse, math
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from utils import set_seed, try_import_wandb
from data import load_tinystories
from model import ModelConfig, BasicGPT, TRMModel, TRMSequentialModel, RecursiveModel

MODEL_TYPES = ["gpt", "trm", "trm_sequential", "recursive_model"]

def build_model(model_type, cfg: ModelConfig):
    if model_type == "gpt":
        return BasicGPT(cfg)
    elif model_type == "trm":
        return TRMModel(cfg)
    elif model_type == "trm_sequential":
        return TRMSequentialModel(cfg)
    elif model_type == "recursive_model":
        return RecursiveModel(cfg)
    else:
        raise ValueError(f"Unknown model_type {model_type}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--model_type", type=str, choices=MODEL_TYPES, default="gpt")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum_steps", type=int, default=1)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=8)
    p.add_argument("--n_head", type=int, default=8)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--d_ff", type=int, default=None)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--n_rows", type=int, default=5000)
    p.add_argument("--tokenizer", type=str, default="gpt2")
    p.add_argument("--T", type=int, default=1, help="traversal count in recursive setups")
    p.add_argument("--K", type=int, default=1, help="prediction horizon for TRMSequential")
    p.add_argument("--attn_impl", type=str, choices=["sdpa","eager"], default="sdpa")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="recursion-playground")
    p.add_argument("--wandb_run_name", type=str, default=None)
    args = p.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data with horizon K to support TRMSequential
    if args.n_rows is not None:
        train_ds, tok = load_tinystories(split="train", tokenizer_name=args.tokenizer, block_size=args.context_length, K=max(1,args.K), n_rows=args.n_rows)
        val_ds, _   = load_tinystories(split="validation", tokenizer_name=args.tokenizer, block_size=args.context_length, K=max(1,args.K), n_rows=min(2000, args.n_rows))
        is_iterable = True
    else:
        train_ds, tok = load_tinystories(split="train", tokenizer_name=args.tokenizer, block_size=args.context_length, K=max(1,args.K), limit=args.limit)
        val_ds, _   = load_tinystories(split="validation", tokenizer_name=args.tokenizer, block_size=args.context_length, K=max(1,args.K), limit=2000)
        is_iterable = False

    def collate(batch):
        x = torch.tensor([b["input_ids"] for b in batch], dtype=torch.long)
        y = torch.tensor([b["labels"] for b in batch], dtype=torch.long)
        f = torch.tensor([b["future"] for b in batch], dtype=torch.long)
        return x, y, f

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=not is_iterable, collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,         collate_fn=collate)

    cfg = ModelConfig(
        vocab_size=len(tok), d_model=args.d_model, n_layer=args.n_layer, n_head=args.n_head,
        d_ff=args.d_ff, context_length=args.context_length, dropout=args.dropout,
        T=args.T, K=args.K, attn_impl=args.attn_impl
    )
    model = build_model(args.model_type, cfg).to(device)

    opt = AdamW(model.parameters(), lr=args.lr)
    if not is_iterable:
        steps_per_epoch = (len(train_loader) + args.grad_accum_steps - 1) // args.grad_accum_steps
        total_steps = steps_per_epoch * args.epochs
    else:
        approx_steps_per_epoch = max(1, (args.n_rows // args.batch_size) // max(1, args.grad_accum_steps))
        total_steps = approx_steps_per_epoch * args.epochs
    warmup = min(args.warmup_steps, max(1, total_steps//10))
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup, num_training_steps=total_steps)

    wb = try_import_wandb() if args.wandb else None
    if args.wandb and wb is not None:
        wb.init(project=args.wandb_project, name=args.wandb_run_name, config={
            "model_type": args.model_type, "d_model": args.d_model, "n_head": args.n_head, "n_layer": args.n_layer,
            "d_ff": args.d_ff, "dropout": args.dropout, "context_length": args.context_length, "lr": args.lr,
            "epochs": args.epochs, "batch_size": args.batch_size, "grad_accum_steps": args.grad_accum_steps,
            "tokenizer": args.tokenizer, "T": args.T, "K": args.K, "attn_impl": args.attn_impl, "seed": args.seed,
            "dataset": "roneneldan/TinyStories", "n_rows": args.n_rows, "limit": args.limit
        })

    def evaluate():
        model.eval()
        total, n_batches = 0.0, 0
        with torch.no_grad():
            for xb, yb, fb in val_loader:
                xb, yb, fb = xb.to(device), yb.to(device), fb.to(device)
                if args.model_type == "trm":
                    _, loss = model(xb, future=fb)
                elif args.model_type == "trm_sequential":
                    _, loss = model(xb, future=fb)
                else:
                    _, loss = model(xb, yb)
                total += loss.item(); n_batches += 1
        model.train()
        return total / max(1, n_batches)

    global_step = 0
    tokens_seen = 0
    model.train()
    best_val = float("inf")
    opt.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs+1):
        for micro, (xb, yb, fb) in enumerate(train_loader, 1):
            xb, yb, fb = xb.to(device), yb.to(device), fb.to(device)
            if args.model_type == "trm":
                _, loss = model(xb, future=fb)
            elif args.model_type == "trm_sequential":
                _, loss = model(xb, future=fb)
            else:
                _, loss = model(xb, yb)
            (loss / args.grad_accum_steps).backward()
            tokens_seen += xb.numel()
            if (micro % args.grad_accum_steps) == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                global_step += 1
                if (global_step % 50) == 0:
                    val_loss = evaluate()
                    best_val = min(best_val, val_loss)
                    if wb is not None:
                        wb.log({"train_loss": loss.item(), "val_loss": val_loss, "tokens_seen": tokens_seen}, step=global_step)
                    print(f"epoch {epoch} step {global_step} | train_loss {loss.item():.3f} | val_loss {val_loss:.3f} | tokens_seen {tokens_seen}")

    torch.save({
        "model_type": args.model_type,
        "model_state_dict": model.state_dict(),
        "config": model.cfg if hasattr(model, "cfg") else None,
        "tokenizer": args.tokenizer,
        "context_length": args.context_length
    }, "checkpoint.pt")
    print("Done. Best val loss:", best_val)

if __name__ == "__main__":
    main()
