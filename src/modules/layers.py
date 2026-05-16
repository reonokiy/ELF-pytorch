import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


def rotate_half(x):
    x = rearrange(x, '... (d r) -> ... d r', r=2)
    x1, x2 = x.unbind(dim=-1)
    return rearrange(torch.stack((-x2, x1), dim=-1), '... d r -> ... (d r)')


class TextRotaryEmbeddingFast(nn.Module):
    """1D RoPE. Computes cos/sin on the fly; supports a leading block of non-rotated tokens."""
    def __init__(self, dim, pt_seq_len=512, theta=10000.0):
        super().__init__()
        self.dim = dim
        self.pt_seq_len = pt_seq_len
        # pre-computed base frequencies; not a learned param → persistent=False
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('freqs', freqs, persistent=False)

    def forward(self, t, num_empty_token=0):
        """t: (B, H, N, head_dim). Returns same shape."""
        seq_len = t.shape[-2] - num_empty_token
        pos = torch.arange(seq_len, device=t.device, dtype=torch.float32)
        freqs_main = torch.outer(pos, self.freqs)                   # (S, dim//2)
        freqs_main = repeat(freqs_main, 'n d -> n (d r)', r=2)     # (S, dim)

        D = freqs_main.shape[-1]
        if num_empty_token > 0:
            empty_cos = t.new_ones(num_empty_token, D)
            empty_sin = t.new_zeros(num_empty_token, D)
            cos = torch.cat([empty_cos, freqs_main.cos()], dim=0)
            sin = torch.cat([empty_sin, freqs_main.sin()], dim=0)
        else:
            cos = freqs_main.cos()
            sin = freqs_main.sin()

        return t * cos + rotate_half(t) * sin


RMSNorm = nn.RMSNorm


class BottleneckTextProj(nn.Module):
    def __init__(self, text_encoder_dim, hidden_size, bottleneck_dim):
        super().__init__()
        self.proj1 = nn.Linear(text_encoder_dim, bottleneck_dim, bias=False)
        self.proj2 = nn.Linear(bottleneck_dim, hidden_size)

    def forward(self, x):
        return self.proj2(self.proj1(x))


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        # names match JAX checkpoint keys: mlp_0, mlp_2
        self.mlp_0 = nn.Linear(frequency_embedding_size, hidden_size)
        self.act   = nn.SiLU()
        self.mlp_2 = nn.Linear(hidden_size, hidden_size)

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half
        )
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, emb.new_zeros(emb.shape[0], 1)], dim=-1)
        return emb

    def forward(self, t):
        emb = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp_2(self.act(self.mlp_0(emb)))


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_norm=True,
                 attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = attn_drop
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, rope=None, num_empty_token=0, attention_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)   # each (B, H, N, D)

        q = self.q_norm(q)
        k = self.k_norm(k)

        if rope is not None:
            q = rope(q, num_empty_token)
            k = rope(k, num_empty_token)

        # Convert 1=valid/0=pad mask to bool for F.scaled_dot_product_attention
        sdpa_mask = None
        if attention_mask is not None:
            if attention_mask.ndim == 2:
                sdpa_mask = attention_mask[:, None, None, :].bool()   # (B, 1, 1, S)
            elif attention_mask.ndim == 3:
                sdpa_mask = attention_mask[:, None, :, :].bool()      # (B, 1, L, S)
            else:
                sdpa_mask = attention_mask.bool()

        drop_p = self.attn_drop if self.training else 0.0
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=sdpa_mask, dropout_p=drop_p)
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class SwiGLUFFN(nn.Module):
    def __init__(self, dim, hidden_dim, drop=0.0):
        super().__init__()
        hidden = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, 2 * hidden)
        self.w3 = nn.Linear(hidden, dim)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(self.drop(self.act(x1) * x2))


class FinalLayer(nn.Module):
    def __init__(self, hidden_size, out_channels):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.linear(self.norm_final(x))
