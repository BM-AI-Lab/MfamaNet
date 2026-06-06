import torch
from torch import nn
from torch.nn import functional as F

from .transformer import TransformerEncoder


def multiscale_generator(X, scale_lst):
    """Generate multi-scale sequences."""
    B, T, C = X.shape
    Xs = []
    num_scales = len(scale_lst)

    with torch.no_grad():
        for i in range(num_scales):
            scale_i = scale_lst[i]
            tX = X

            remainder = T % scale_i
            if remainder != 0:
                pad_length = scale_i - remainder
                padding = torch.zeros([B, pad_length, C], device=X.device, dtype=X.dtype)
                tX = torch.cat([tX, padding], dim=1)
                length = T + pad_length
            else:
                length = T

            num_groups = length // scale_i
            tX = tX.reshape(B * num_groups, scale_i, C)
            Xs.append(tX)

    return Xs


class CrossAttention(nn.Module):
    """Pairwise cross-attention between scale CLS tokens."""
    def __init__(self, dim, num_cls, dropout=0.0):
        super().__init__()
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.scale = dim ** -0.5
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('src', torch.triu_indices(num_cls, num_cls, offset=1)[0])
        self.register_buffer('tgt', torch.triu_indices(num_cls, num_cls, offset=1)[1])

    def forward(self, Cls):
        cls = torch.cat(Cls, dim=1)                   # (B, num_cls, H)
        qkv = self.qkv_proj(cls)                      # (B, num_cls, 3H)
        q, k, v = qkv.chunk(3, -1)
        q_pair, k_pair, v_pair = q[:, self.src], k[:, self.tgt], v[:, self.tgt]

        score = torch.einsum('bpc,bpc->bp', q_pair, k_pair) * self.scale
        attn = F.softmax(score, dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bp,bpc->bpc', attn, v_pair)

        cls_update = torch.zeros_like(cls)
        cls_update[:, self.src] += out
        return cls + self.gamma * cls_update


class TRE(nn.Module):
    """Temporal Relation Extractor.

    Multi-scale transformer-based signal encoder with cross-attention
    fusion of learnable CLS tokens.
    """
    def __init__(self, input_channels, num_hidden, scale_lst, dropout, num_layers):
        super().__init__()
        self.H = num_hidden
        self.scale_lst = scale_lst
        self.num_scales = len(scale_lst)

        self.input_proj = nn.Linear(input_channels, num_hidden)
        self.cls_tokens = [nn.Parameter(torch.zeros((1, 1, num_hidden)))
                           for _ in range(self.num_scales)]

        assert num_hidden % 8 == 0, \
            f"num_hidden ({num_hidden}) must be divisible by n_heads (8)"
        self.transformer_layers = nn.ModuleList(
            [TransformerEncoder(num_hidden, num_layers[i], 8, 4, dropout)
             for i in range(self.num_scales)]
        )

        self.cls_proj = nn.AdaptiveAvgPool1d(1)
        self.cross_attn = CrossAttention(num_hidden, self.num_scales, dropout)

    def forward(self, X):
        B, _, _ = X.shape
        Xs = self._multiscale_processing(X)
        for i, x in enumerate(Xs):
            Xs[i] = self.transformer_layers[i](x)

        cls = [x[:, :1, :].reshape(B, -1, self.H) for x in Xs]
        cls = [self.cls_proj(c.permute(0, 2, 1)).permute(0, 2, 1) for c in cls]
        cls = self.cross_attn(cls)    # [B, num_scales, H]

        return cls

    def _multiscale_processing(self, X):
        X = self.input_proj(X)    # [B, T, C] -> [B, T, H]
        Xs = multiscale_generator(X, scale_lst=self.scale_lst)

        for i in range(self.num_scales):
            tX = Xs[i]
            B, _, H = tX.shape
            Cls_i = self.cls_tokens[i].expand(B, 1, H).to(tX.device)
            tX = torch.cat([Cls_i, tX], dim=1)    # [B, T+1, H]
            Xs[i] = tX

        return Xs
