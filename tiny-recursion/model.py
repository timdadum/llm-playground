import math, torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class ModelConfig:
    vocab_size: int
    d_model: int = 512
    n_layer: int = 8
    n_head: int = 8
    d_ff: int = None
    context_length: int = 256
    dropout: float = 0.0
    T: int = 1                 # traversal count used by recursive setups
    K: int = 1                 # prediction horizon for TRMSequential
    attn_impl: str = "sdpa"    # 'sdpa' or 'eager'

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, context_length, dropout=0.0, impl: str = "sdpa"):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.impl = impl
        self.qkv = nn.Linear(d_model, 3*d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)
        if impl != "sdpa":
            self.register_buffer("mask", torch.tril(torch.ones(context_length, context_length)).view(1,1,context_length,context_length))

    def forward(self, x):
        B, n, d = x.shape
        qkv = self.qkv(x).view(B, n, 3, self.n_head, self.d_head).permute(2,0,3,1,4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if self.impl == "sdpa":
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        else:
            att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
            att = att.masked_fill(self.mask[:,:,:n,:n] == 0, float("-inf"))
            att = torch.softmax(att, dim=-1)
            att = self.attn_drop(att)
            y = att @ v
        y = y.transpose(1,2).contiguous().view(B, n, d)
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):
    def __init__(self, d_model, n_head, context_length, d_ff=None, dropout=0.0, attn_impl="sdpa"):
        super().__init__()
        if d_ff is None: d_ff = 4*d_model
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, context_length, dropout, impl=attn_impl)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class BasicGPT(nn.Module):
    """Standard causal LM with full-position CE."""
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d_model, V = cfg.d_model, cfg.vocab_size
        self.tok_emb = nn.Embedding(V, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.context_length, d_model))
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(d_model, cfg.n_head, cfg.context_length, cfg.d_ff, cfg.dropout, cfg.attn_impl) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, V, bias=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, n = idx.shape
        assert n <= self.cfg.context_length
        x = self.tok_emb(idx) + self.pos_emb[:, :n, :]
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)            # [B,n,V]
        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

class TRMModel(nn.Module):
    """TRM faithful to K=1: predict only the next token after the last input.
    Notation: n = context length, K=1, T is a global traversal symbol (unused here).
    Forward: y_hat = Core(I)[:, -1]; Loss = CE(y_hat, future[:, 0])
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        core_cfg = ModelConfig(
            vocab_size=cfg.vocab_size, d_model=cfg.d_model, n_layer=cfg.n_layer, n_head=cfg.n_head,
            d_ff=cfg.d_ff, context_length=cfg.context_length, dropout=cfg.dropout,
            T=cfg.T, K=1, attn_impl=cfg.attn_impl
        )
        self.core = BasicGPT(core_cfg)

    def forward(self, idx, future=None):
        # future: [B, K] with K>=1; we will use the first column as target
        logits_all, _ = self.core(idx, targets=None)  # [B,n,V]
        last_logits = logits_all[:, -1, :]            # [B,V]
        loss = None
        if future is not None:
            loss = torch.nn.functional.cross_entropy(last_logits, future[:, 0])
        return last_logits, loss

class TRMSequentialModel(nn.Module):
    """TRM but predicting the next K tokens (teacher-forced multi-step).
    For k in {1..K}: run Core on [idx, future[:, :k-1]] and supervise the last position with future[:, k-1].
    Returns logits_k stacked into [B, K, V]; loss is mean of K CE terms.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        core_cfg = ModelConfig(
            vocab_size=cfg.vocab_size, d_model=cfg.d_model, n_layer=cfg.n_layer, n_head=cfg.n_head,
            d_ff=cfg.d_ff, context_length=cfg.context_length, dropout=cfg.dropout,
            T=cfg.T, K=cfg.K, attn_impl=cfg.attn_impl
        )
        self.core = BasicGPT(core_cfg)

    def forward(self, idx, future):
        B, n = idx.shape
        K = self.cfg.K
        logits_list = []
        losses = []
        for k in range(1, K+1):
            # build teacher-forced prefix: idx + future[:,:k-1]
            if k == 1:
                seq = idx
            else:
                prefix = future[:, :k-1]  # [B, k-1]
                seq = torch.cat([idx, prefix], dim=1)
            # respect context_length cap
            if seq.size(1) > self.cfg.context_length:
                seq = seq[:, -self.cfg.context_length:]
            logits_all, _ = self.core(seq, targets=None)    # [B, L, V]
            last = logits_all[:, -1, :]                     # [B, V]
            logits_list.append(last)
            losses.append(torch.nn.functional.cross_entropy(last, future[:, k-1]))
        logits = torch.stack(logits_list, dim=1)            # [B, K, V]
        loss = sum(losses) / len(losses)
        return logits, loss

class RecursiveModel(nn.Module):
    """Recursive model: re-use the same small stack of blocks and traverse it T times.
    Implements y = f^{(T)}(x) = f(f(...f(x))) where f is a 2-block transformer segment.
    """
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d_model, V = cfg.d_model, cfg.vocab_size
        self.tok_emb = nn.Embedding(V, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, cfg.context_length, d_model))
        self.drop = nn.Dropout(cfg.dropout)
        # define a small segment (2 blocks) to be re-used
        self.segment = nn.ModuleList([Block(d_model, cfg.n_head, cfg.context_length, cfg.d_ff, cfg.dropout, cfg.attn_impl) for _ in range(2)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, V, bias=False)

        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Embedding)):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, idx, targets=None):
        B, n = idx.shape
        assert n <= self.cfg.context_length
        x = self.tok_emb(idx) + self.pos_emb[:, :n, :]
        x = self.drop(x)
        # apply the same segment T times
        for _ in range(max(1, self.cfg.T)):
            for blk in self.segment:
                x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss
