import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        var = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (x * self.weight).type_as(x)


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(q, k, freqs_cis):
    q_ = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
    k_ = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.unsqueeze(0).unsqueeze(0)
    q_out = torch.view_as_real(q_ * freqs_cis).flatten(3)
    k_out = torch.view_as_real(k_ * freqs_cis).flatten(3)
    return q_out.type_as(q), k_out.type_as(k)


class GQA(nn.Module):
    """Grouped Query Attention with RoPE."""
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: Optional[int] = None,
                 dropout: float = 0.0, max_seq_len: int = 8192):
        super().__init__()
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        self.head_dim = d_model // n_heads
        self.n_rep = self.n_heads // self.n_kv_heads
        self.dropout = dropout

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

        freqs_cis = precompute_freqs_cis(self.head_dim, max_seq_len)
        self.register_buffer("freqs_cis", freqs_cis)

    def repeat_kv(self, x, n_rep: int):
        if n_rep == 1:
            return x
        B, n, L, d = x.shape
        return x.unsqueeze(2).expand(B, n, n_rep, L, d).reshape(B, n * n_rep, L, d)

    def forward(self, x, mask=None, use_flash=True):
        B, L, _ = x.shape
        q = self.wq(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, L, self.n_kv_heads, self.head_dim).transpose(1, 2)

        freqs_cis = self.freqs_cis[:L]
        q, k = apply_rotary_emb(q, k, freqs_cis)

        k = self.repeat_kv(k, self.n_rep)
        v = self.repeat_kv(v, self.n_rep)

        if use_flash and hasattr(F, 'scaled_dot_product_attention'):
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=mask,
                dropout_p=self.dropout if self.training else 0.0)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores += mask
            attn = F.softmax(scores, dim=-1)
            attn = F.dropout(attn, p=self.dropout, training=self.training)
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.wo(out)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden = int(hidden_mult * d_model * 2 / 3)
        self.w1 = nn.Linear(d_model, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: Optional[int] = None,
                 dropout: float = 0.0, hidden_mult: int = 4, max_seq_len: int = 8192):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.attn = GQA(d_model, n_heads, n_kv_heads, dropout, max_seq_len)
        self.ffn = SwiGLU(d_model, hidden_mult, dropout)

    def forward(self, x, mask=None, use_flash=True):
        x = x + self.attn(self.norm1(x), mask, use_flash)
        x = x + self.ffn(self.norm2(x))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, n_layers: int, n_heads: int,
                 n_kv_heads: Optional[int] = None, dropout: float = 0.0,
                 hidden_mult: int = 4, max_seq_len: int = 8192):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, n_kv_heads,
                                    dropout, hidden_mult, max_seq_len)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(d_model)

    def forward(self, x, mask=None, use_flash=True):
        for layer in self.layers:
            x = layer(x, mask, use_flash)
        return self.norm(x)
