import torch
import torch.nn as nn
import torch.nn.functional as F

from modules_pt.layers import (
    Attention, BottleneckTextProj, FinalLayer, RMSNorm, SwiGLUFFN,
    TextRotaryEmbeddingFast, TimestepEmbedder,
)


class ELFBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.norm1 = RMSNorm(hidden_size)
        self.attn = Attention(
            hidden_size, num_heads, qkv_bias=True, qk_norm=True,
            attn_drop=attn_drop, proj_drop=proj_drop,
        )
        self.norm2 = RMSNorm(hidden_size)
        self.mlp = SwiGLUFFN(hidden_size, int(hidden_size * mlp_ratio), drop=proj_drop)

    def forward(self, x, rope=None, num_empty_token=0, attention_mask=None):
        x = x + self.attn(self.norm1(x), rope=rope, num_empty_token=num_empty_token,
                          attention_mask=attention_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class ELF(nn.Module):
    def __init__(
        self,
        text_encoder_dim,
        max_length,
        hidden_size=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        attn_drop=0.0,
        proj_drop=0.0,
        bottleneck_dim=128,
        num_time_tokens=4,
        num_self_cond_cfg_tokens=4,
        num_model_mode_tokens=0,
        vocab_size=0,
    ):
        super().__init__()
        self.text_encoder_dim = text_encoder_dim
        self.hidden_size = hidden_size
        self.depth = depth
        self.num_time_tokens = num_time_tokens
        self.num_self_cond_cfg_tokens = num_self_cond_cfg_tokens
        self.num_model_mode_tokens = num_model_mode_tokens
        self.vocab_size = vocab_size

        # Self-conditioning projection (2*text_enc_dim → text_enc_dim)
        self.self_cond_proj = nn.Linear(2 * text_encoder_dim, text_encoder_dim)

        # Text projection with bottleneck
        self.text_proj = BottleneckTextProj(text_encoder_dim, hidden_size, bottleneck_dim)

        # Time conditioning
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.t_emb_tokens = nn.Parameter(torch.zeros(1, num_time_tokens, hidden_size))

        # Self-cond CFG conditioning
        if num_self_cond_cfg_tokens > 0:
            self.self_cond_cfg_embedder = TimestepEmbedder(hidden_size)
            self.self_cond_cfg_tokens = nn.Parameter(
                torch.zeros(1, num_self_cond_cfg_tokens, hidden_size)
            )

        # Model-mode tokens (zero-gated unless decoder_step_active=True)
        if num_model_mode_tokens > 0:
            self.mode_tokens = nn.Parameter(torch.zeros(1, num_model_mode_tokens, hidden_size))

        # RoPE (shared across blocks)
        head_dim = hidden_size // num_heads
        self.rope = TextRotaryEmbeddingFast(head_dim, pt_seq_len=max_length)

        # Transformer blocks
        q1, q3 = depth // 4, depth // 4 * 3
        self.blocks = nn.ModuleList([
            ELFBlock(
                hidden_size, num_heads, mlp_ratio=mlp_ratio,
                attn_drop=attn_drop if q3 > i >= q1 else 0.0,
                proj_drop=proj_drop if q3 > i >= q1 else 0.0,
            )
            for i in range(depth)
        ])

        self.final_layer = FinalLayer(hidden_size, text_encoder_dim)

        # Factored decoder unembedding (raw parameters, matching JAX self.param())
        bn = text_encoder_dim
        self.proj_kernel = nn.Parameter(torch.empty(hidden_size, bn))
        self.proj_bias = nn.Parameter(torch.zeros(bn))
        self.unembed_kernel = nn.Parameter(torch.empty(bn, vocab_size))
        self.unembed_bias = nn.Parameter(torch.zeros(vocab_size))
        nn.init.xavier_uniform_(self.proj_kernel)
        nn.init.xavier_uniform_(self.unembed_kernel)

    def _build_context(self, t, self_cond_cfg_scale=None):
        B = t.shape[0]
        prefix = []

        time_emb = self.t_embedder(t)                             # (B, H)
        t_tokens = self.t_emb_tokens.expand(B, -1, -1) + time_emb[:, None, :]
        prefix.append(t_tokens)

        if self_cond_cfg_scale is not None and self.num_self_cond_cfg_tokens > 0:
            sc_emb = self.self_cond_cfg_embedder(self_cond_cfg_scale)
            sc_tokens = self.self_cond_cfg_tokens.expand(B, -1, -1) + sc_emb[:, None, :]
            prefix.append(sc_tokens)

        return prefix

    def forward(
        self,
        x,
        t,
        attention_mask=None,
        self_cond_cfg_scale=None,
        decoder_step_active=None,
    ):
        """
        x: (B, S, text_enc_dim) or (B, S, 2*text_enc_dim) for self-cond
        t: (B,)
        attention_mask: (B, S) int, 1=valid, 0=pad  [optional]
        self_cond_cfg_scale: (B,)  [optional]
        decoder_step_active: bool or bool tensor  [optional]
        Returns: (output, decoder_logits) matching JAX signature.
        """
        B = x.shape[0]

        # Self-conditioning: concatenated [z, x_pred] → project back to enc_dim
        if x.shape[-1] == 2 * self.text_encoder_dim:
            x = self.self_cond_proj(x)

        # Bottleneck projection to hidden_size
        x = self.text_proj(x)

        # --- Model-mode tokens (zero-gated) ---
        model_mode_offset = 0
        if self.num_model_mode_tokens > 0:
            mode = self.mode_tokens.expand(B, -1, -1)
            if decoder_step_active is None:
                gate = 0.0
            elif isinstance(decoder_step_active, torch.Tensor):
                gate = decoder_step_active.float()
            else:
                gate = float(decoder_step_active)
            mode = mode * gate
            x = torch.cat([mode, x], dim=1)
            model_mode_offset = self.num_model_mode_tokens
            if attention_mask is not None:
                mode_mask = torch.ones(B, self.num_model_mode_tokens,
                                       device=x.device, dtype=attention_mask.dtype)
                attention_mask = torch.cat([mode_mask, attention_mask], dim=1)

        # --- Time / SC-CFG prefix tokens ---
        prefix_parts = self._build_context(t, self_cond_cfg_scale)
        prefix = torch.cat(prefix_parts, dim=1)
        prefix_len = prefix.shape[1]
        x = torch.cat([prefix, x], dim=1)
        if attention_mask is not None:
            pfx_mask = torch.ones(B, prefix_len, device=x.device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([pfx_mask, attention_mask], dim=1)

        num_empty = prefix_len + model_mode_offset

        # --- Transformer ---
        for block in self.blocks:
            x = block(x, rope=self.rope, num_empty_token=num_empty,
                      attention_mask=attention_mask)

        # Strip prefix + mode tokens
        x = x[:, num_empty:]

        # --- Factored decoder ---
        decoder_logits = None
        if decoder_step_active is not None:
            if isinstance(decoder_step_active, torch.Tensor):
                active = bool(decoder_step_active.item())
            else:
                active = bool(decoder_step_active)

            if active:
                h = F.gelu(x @ self.proj_kernel + self.proj_bias)
                decoder_logits = h @ self.unembed_kernel + self.unembed_bias
            else:
                decoder_logits = torch.zeros(
                    *x.shape[:2], self.vocab_size, device=x.device, dtype=x.dtype
                )

        output = self.final_layer(x)
        return output, decoder_logits


def ELF_B(**kwargs):
    return ELF(depth=12, hidden_size=768, num_heads=12, **kwargs)

def ELF_M(**kwargs):
    return ELF(depth=24, hidden_size=1056, num_heads=16, **kwargs)

def ELF_L(**kwargs):
    return ELF(depth=32, hidden_size=1280, num_heads=16, **kwargs)

ELF_models = {'ELF-B': ELF_B, 'ELF-M': ELF_M, 'ELF-L': ELF_L}
