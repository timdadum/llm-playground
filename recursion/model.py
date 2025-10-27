import math
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutput
from transformers import PreTrainedModel, PretrainedConfig
from torch.backends.cuda import sdp_kernel

@dataclass
class ModelConfig(PretrainedConfig):
    # model_type: str = "basic_llm"

    # --- Model architecture ---
    vocab_size: int = 50257              # GPT-2 vocab
    d_model: int = 384                   # hidden dimension / embedding size
    n_head: int = 4                      # attention heads
    n_layer: int = 4                     # number of transformer blocks
    d_ff: int = 256                      # feedforward hidden size
    max_len: int = 512                   # maximum sequence length
    dropout: float = 0.1                 # dropout everywhere
    attn_implementation: str = "sdpa"    # ("eager" or "sdpa") -> flash uses SDPA Flash backend
    _attn_implementation_internal: str = "sdpa"

    # --- Positional embeddings ---
    pos_embedding_type: str = "absolute"  # "absolute", "rotary", "alibi"

    # --- Initialization ---
    init_range: float = 0.02             # weight init scale
    tie_weights: bool = True             # share token <-> output embedding

    # --- Training/runtime settings ---
    layer_norm_eps: float = 1e-5         # for numerical stability
    pad_token_id: int = 50256            # often same as EOS for GPT-2
    bos_token_id: int = 50256
    eos_token_id: int = 50256
    gradient_checkpointing: bool = False # save memory on large models
    # device: str = "cuda"                 # runtime device hint

    # --- Logging & evaluation ---
    log_every: int = 50                  # print frequency
    eval_every_steps: int = 500
    save_every_epochs: int = 1
    seed: int = 42

    # --- Optimization (optional defaults) ---
    lr: float = 3e-4
    betas: tuple = (0.9, 0.95)
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    grad_accum_steps: int = 16
    # warmup_steps: int = 100
    # total_steps: int = 10000

    # --- Misc ---
    amp: bool = True                     # mixed precision toggle
    use_cache: bool = False              # skip caching for training
    project_name: str = "basic-llm"
    out_dir: str = "/workspace/recursion/trained_models"
    data_out_dir: str = "/workspace/recursion/data/shards"

def _ensure_flash_enabled_if_requested(attn_impl: str):
    if attn_impl != "sdpa":
        return
    if not torch.cuda.is_available():
        return
    sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)
    print("Flash attention enabled.")

class FlashMultiheadAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, attn_impl: str):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.dropout = dropout
        self.attn_impl = attn_impl

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)

    def _reshape_by_heads(self, x: torch.Tensor, bsz: int, seqlen: int) -> torch.Tensor:
        # (B, T, D) -> (B, nH, T, Dh)
        return x.view(bsz, seqlen, self.n_head, self.d_head).permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q = self._reshape_by_heads(self.q_proj(x), B, T)
        k = self._reshape_by_heads(self.k_proj(x), B, T)
        v = self._reshape_by_heads(self.v_proj(x), B, T)

        if self.attn_impl == "sdpa":
            _ensure_flash_enabled_if_requested(self.attn_impl)
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Eager fallback
            dk = q.size(-1)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dk)
            mask = torch.ones((T, T), dtype=torch.bool, device=x.device).triu(1)
            scores = scores.masked_fill(mask, float("-inf"))
            probs = F.softmax(scores, dim=-1)
            probs = self.attn_drop(probs) if self.training and self.dropout > 0.0 else probs
            out = torch.matmul(probs, v)

        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, D)  # (B, T, D)
        out = self.o_proj(out)
        return out


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.ln2 = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.attn = FlashMultiheadAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.attn_implementation)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class BasicGPT(PreTrainedModel):
    config_class = ModelConfig
    _supports_sdpa = True  # signal SDPA compatibility to HF

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.cfg = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_len, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.lm_head.weight = self.token_emb.weight  # tie weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_range)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def _calculate_loss(self, logits: torch.Tensor, labels: torch.Tensor | None) -> torch.Tensor | None:
        """
        logits: (B, T, V)
        labels: (B, T) with class indices in [0, V) or -100 for positions to ignore
        Returns a scalar Tensor loss (or None if labels is None).
        """
        if labels is None:
            return None
        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]

        vocab = shift_logits.size(-1)
        shift_logits = shift_logits.reshape(-1, vocab).float()
        shift_labels = shift_labels.reshape(-1).long()

        loss = F.cross_entropy(shift_logits, shift_labels, ignore_index=-100)
        return loss

    def forward(self,
                input_ids: torch.LongTensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.LongTensor] = None,
                **kwargs,
    ) -> Dict[str, Any]:
        B, T = input_ids.shape
        device = input_ids.device

        pos = torch.arange(T, device=device)
        h = self.token_emb(input_ids) + self.pos_emb(pos)[None, :, :]
        h = self.drop(h)

        for blk in self.blocks:
            h = blk(h)

        h = self.norm_f(h)
        logits = self.lm_head(h)    # (B, T, V)

        loss = self._calculate_loss(logits, labels)
        return CausalLMOutput(loss=loss, logits=logits)

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        do_sample: bool = True,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,  # unused here; included for API parity
    ):
        """
        Minimal autoregressive generator.
        - Trims context to config.max_len
        - Supports sampling (multinomial) or greedy (argmax)
        - Early-stops if eos_token_id is provided and all batches just produced EOS
        """
        self.eval()
        device = input_ids.device

        for _ in range(max_new_tokens):
            # respect context window
            if input_ids.size(1) > self.cfg.max_len:
                input_ids = input_ids[:, -self.cfg.max_len:]
                if attention_mask is not None:
                    attention_mask = attention_mask[:, -self.cfg.max_len:]

            out = self(input_ids=input_ids, attention_mask=attention_mask)
            logits = out.logits[:, -1, :] / max(temperature, 1e-8)
            probs = torch.softmax(logits, dim=-1)

            if do_sample:
                next_token = torch.multinomial(probs, num_samples=1)             # (B, 1)
            else:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)           # (B, 1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

            if attention_mask is not None:
                attention_mask = torch.cat(
                    [attention_mask, torch.ones_like(next_token, device=device)],
                    dim=1,
                )

            if eos_token_id is not None:
                if (next_token == eos_token_id).all():
                    break

        return input_ids

    @classmethod
    def load_weights(cls,
                     path,
                     config):
        checkpoint = torch.load(path)
        config = ModelConfig(**checkpoint["config"])
        model = cls(config)
        model.load_state_dict(checkpoint["model_state"])
        return model, checkpoint