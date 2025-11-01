from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutput
from transformers import PreTrainedModel, PretrainedConfig
from torch.backends.cuda import sdp_kernel

# =========================
# Config
# =========================
class ModelConfig(PretrainedConfig):
    model_type = "basic_gpt"

    def __init__(
        self,
        
        # architecture
        vocab_size: int = 50257,
        d_model: int = 32,
        n_head: int = 2,
        n_layer: int = 2,
        d_ff: int = 16,
        context_length: int = 64,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        pos_embedding_type: str = "absolute",
        
        # attention mechanism
        attn_implementation: str = "sdpa",
        _attn_implementation_internal: str = "sdpa",
        
        # tied embedding weights
        init_range: float = 0.02,
        tie_weights: bool = True,
        tie_word_embeddings: bool = True,
        
        # special token ids
        pad_token_id: int = 50256,
        bos_token_id: int = 50256,
        eos_token_id: int = 50256,
        
        # training
        log_every: int = 10,
        eval_every_steps: int = 50,
        save_every_epochs: int = 1,
        seed: int = 42,
        lr: float = 1e-4,
        betas: tuple = (0.9, 0.95),
        weight_decay: float = 0.1,
        grad_clip: float = 1.0,
        grad_accum_steps: int = 4,
        amp: bool = True,
        use_cache: bool = False,
        project_name: str = "basic-llm",
        out_dir: str = "checkpoints",
        data_out_dir: str = "data/shards",
        
        # trm defaults
        K: int = 8,
        T: int = 3,
        n: int = 4,
        
        # passthrough
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            use_cache=use_cache,
            **kwargs,
        )
        # assign
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_head = n_head
        self.n_layer = n_layer
        self.d_ff = d_ff
        self.context_length = context_length
        self.dropout = dropout
        self.layer_norm_eps = layer_norm_eps
        self.pos_embedding_type = pos_embedding_type
        self.attn_implementation = attn_implementation
        self._attn_implementation_internal = _attn_implementation_internal
        self.init_range = init_range
        self.tie_weights = tie_weights
        self.tie_word_embeddings = tie_word_embeddings

        self.log_every = log_every
        self.eval_every_steps = eval_every_steps
        self.save_every_epochs = save_every_epochs
        self.seed = seed
        self.lr = lr
        self.betas = betas
        self.weight_decay = weight_decay
        self.grad_clip = grad_clip
        self.grad_accum_steps = grad_accum_steps
        self.amp = amp
        self.project_name = project_name
        self.out_dir = out_dir
        self.data_out_dir = data_out_dir

        self.K = K
        self.T = T
        self.n = n
    
# =========================
# Utilities
# =========================

def _ensure_flash_enabled_if_requested(attn_impl: str):
    if attn_impl != "sdpa":
        return
    if not torch.cuda.is_available():
        return
    sdp_kernel(enable_flash=True, enable_mem_efficient=True, enable_math=False)

# =========================
# Core blocks
# =========================

class FlashMultiheadAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float, attn_impl: str):
        super().__init__()
        assert d_model % n_head == 0
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
                is_causal=True
            )
        else:
            dk = q.size(-1)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dk)
            mask = torch.ones((T, T), dtype=torch.bool, device=x.device).triu(1)
            scores = scores.masked_fill(mask, float("-inf"))
            probs = F.softmax(scores, dim=-1)
            probs = self.attn_drop(probs) if self.training and self.dropout > 0.0 else probs
            out = torch.matmul(probs, v)

        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, D)
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
        self.mlp  = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

# =========================
# Basic GPT
# =========================

class BasicGPT(PreTrainedModel):
    config_class = ModelConfig
    _supports_sdpa = True

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.cfg = config
        self.config.use_cache = False
        self.config.tie_word_embeddings = getattr(config, "tie_word_embeddings", True)

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb   = nn.Embedding(config.context_length, config.d_model)
        self.drop      = nn.Dropout(config.dropout)

        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.norm_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        self.config.tie_word_embeddings = True
        self.config.tie_weights = True

        self.apply(self._init_weights)
        self.tie_weights()

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_range)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "use_cache": False}

    def get_input_embeddings(self):
        return self.token_emb

    def set_input_embeddings(self, new_emb):
        self.token_emb = new_emb

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_out):
        self.lm_head = new_out
    
    def tie_weights(self):
        self.lm_head.weight = self.token_emb.weight

    def _calculate_loss(self, logits: torch.Tensor, labels: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if labels is None:
            return None
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        V = shift_logits.size(-1)
        return F.cross_entropy(shift_logits.reshape(-1, V).float(),
                               shift_labels.reshape(-1).long(),
                               ignore_index=-100)

    def forward(
        self,
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
        logits = self.lm_head(h)
        loss = self._calculate_loss(logits, labels)
        return CausalLMOutput(loss=loss, logits=logits)

# =========================
# TRM
# =========================

class TRMTinyNet(nn.Module):
    def __init__(self, cfg: ModelConfig, n_layers: int = 2):
        super().__init__()
        self.cfg = cfg
        self.ln = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(n_layers)])

    def forward(self, *states: torch.Tensor) -> torch.Tensor:
        """Merges latent inputs y, z and optionally, x, into an updated latent state."""
        h = 0
        for s in states:
            h = h + s
        h = self.ln(h)
        for blk in self.blocks:
            h = blk(h)
        return h

class TRMModel(PreTrainedModel):
    config_class = ModelConfig
    _supports_sdpa = True

    def __init__(
        self,
        config: ModelConfig,
        n: int = 4,
        T: int = 3,
        n_layers: int = 2,
        use_halting_head: bool = False,
        context_length: Optional[int] = None,
        K: Optional[int] = None,
    ):
        super().__init__(config)
        self.cfg = config
        self.config.use_cache = False
        self.config.tie_word_embeddings = True
        self.config.tie_weights = True
        self.context_length = context_length if context_length else config.context_length
        self.K = K if K is not None else config.K

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb   = nn.Embedding(config.context_length, config.d_model)
        self.drop      = nn.Dropout(config.dropout)
        self.net       = TRMTinyNet(config, n_layers=n_layers)
        self.lm_head   = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # optional halting
        self.use_halting = use_halting_head
        if self.use_halting:
            self.halt_head = nn.Linear(config.d_model, 1)

        self.inner_n = int(n)
        self.outer_T = int(T)

        self.apply(self._init_weights)
        self.tie_weights()

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Embedding)):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_range)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    # helpers
    def _embed(self, ids: torch.LongTensor) -> torch.Tensor:
        B, L = ids.shape
        pos = torch.arange(L, device=ids.device)
        h = self.token_emb(ids) + self.pos_emb(pos)[None, :, :]
        return self.drop(h)

    def _reverse_embed(self, y: torch.Tensor) -> torch.Tensor:
        return self.lm_head(y)

    @staticmethod
    def _calculate_K_cross_entropy(logits: torch.Tensor, nextk_grid_labels: torch.Tensor) -> torch.Tensor:
        B, L, V = logits.shape
        _, L2, K = nextk_grid_labels.shape
        assert L == L2
        loss, count = 0.0, 0
        for k in range(K):
            off = k + 1
            if L - off <= 0:
                continue
            yk = nextk_grid_labels[:, :L - off, k]
            lk = logits[:, off:, :]
            loss += F.cross_entropy(lk.reshape(-1, V).float(), yk.reshape(-1).long(), ignore_index=-100)
            count += 1
        return loss / max(1, count)

    # generation helper (no KV cache)
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "use_cache": False}

    # HF embedding hooks
    def get_input_embeddings(self):
        return self.token_emb

    def set_input_embeddings(self, new_emb):
        self.token_emb = new_emb

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_out):
        self.lm_head = new_out

    def tie_weights(self):
        self.lm_head.weight = self.token_emb.weight

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
        nextk_grid_labels: Optional[torch.Tensor] = None,
        answer_init_ids: Optional[torch.LongTensor] = None,
        n: Optional[int] = None,
        T: Optional[int] = None,
        return_all_steps: bool = False,
        **kwargs,
    ):
        device = input_ids.device
        B, L_total = input_ids.shape

        # crop to context window (right side)
        L = min(L_total, self.context_length)
        if L != L_total:
            input_ids = input_ids[:, -L:]
            if labels is not None:
                labels = labels[:, -L:]
            if nextk_grid_labels is not None:
                nextk_grid_labels = nextk_grid_labels[:, -L:, :]

        x = self._embed(input_ids)  # (B, L, D)

        # initial states
        z = torch.zeros_like(x, device=device)
        if answer_init_ids is not None and answer_init_ids.size(1) != L:
            answer_init_ids = answer_init_ids[:, -L:]
        y = self._embed(answer_init_ids) if answer_init_ids is not None else torch.zeros_like(x, device=device)

        inner_n = int(n if n is not None else self.inner_n)
        outer_T = int(T if T is not None else self.outer_T)

        all_logits = []
        total_loss = 0.0 if self.training else None
        loss_steps = 0

        for _t in range(outer_T):
            for _ in range(inner_n):
                dz = self.net(x, y, z)
                z  = z + dz

            dy = self.net(y, z)
            y  = y + dy

            logits_t = self._reverse_embed(y)

            if return_all_steps:
                all_logits.append(logits_t)

            if nextk_grid_labels is not None:
                lbl = nextk_grid_labels[:, :, :self.K] if nextk_grid_labels.size(-1) > self.K else nextk_grid_labels
                step_loss = self._calculate_K_cross_entropy(logits_t, lbl)
            elif labels is not None:
                shift_logits = logits_t[:, :-1, :]
                shift_labels = labels[:, 1:]
                V = shift_logits.size(-1)
                step_loss = F.cross_entropy(shift_logits.reshape(-1, V).float(),
                                            shift_labels.reshape(-1).long(),
                                            ignore_index=-100)
            else:
                step_loss = None

            if step_loss is not None:
                total_loss = step_loss if total_loss is None else (total_loss + step_loss)
                loss_steps += 1

        if loss_steps > 0:
            total_loss = total_loss / loss_steps

        final_logits = logits_t
        if return_all_steps and len(all_logits) > 0:
            all_logits = torch.stack(all_logits, dim=0)
        else:
            all_logits = None

        return CausalLMOutput(loss=total_loss, logits=final_logits)