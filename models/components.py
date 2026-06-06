import torch
from torch import nn


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class AdaptiveAggregation(nn.Module):
    """Choquet-inspired fuzzy aggregation.

    https://doi.org/10.1016/j.inffus.2025.103499
    """
    def __init__(self, num_x, num_hiddens):
        super().__init__()
        self.F = nn.Sequential(
            nn.Linear(num_x - 2, num_hiddens),
            nn.GELU(),
            nn.Linear(num_hiddens, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: [B, H, C]
        x_sorted, _ = torch.sort(x, dim=-1)
        out = x_sorted[..., 0]  # x_(1)
        for i in range(1, x.size(-1)):
            delta = x_sorted[..., i] - x_sorted[..., i - 1]
            mask = torch.cat([x_sorted[..., :i - 1], x_sorted[..., i + 1:]], -1)
            out += delta * self.F(mask).squeeze(-1)
        return out    # [B, H]


class ClassificationHead(nn.Module):
    def __init__(self, in_dim, num_classes, dropout=0.2, init_scale=20.0):
        super().__init__()
        self.linear = nn.utils.weight_norm(nn.Linear(in_dim, num_classes))
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()
        self.scale = nn.Parameter(torch.tensor(init_scale))

    def forward(self, x):
        x = torch.nn.functional.normalize(x, p=2, dim=1)
        x = self.dropout(self.act(x))
        logits = self.scale * self.linear(x)
        return logits


class TokenBasedSFE(nn.Module):
    """ViT-style token-based Spatial Feature Extractor.

    Uses S learnable CLS tokens that attend jointly to image patches
    through a shared Transformer encoder, producing [B, S, H].
    """

    def __init__(self, image_chans, num_hiddens, num_scales,
                 patch_size=16, depth=4, num_heads=4, dropout=0.1):
        super().__init__()
        assert num_hiddens % num_heads == 0

        self.num_scales = num_scales
        self.patch_size = patch_size

        self.patch_embed = nn.Conv2d(image_chans, num_hiddens,
                                     kernel_size=patch_size, stride=patch_size)

        self.cls_tokens = nn.Parameter(torch.zeros(1, num_scales, num_hiddens))

        num_patches = (224 // patch_size) ** 2
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + num_scales, num_hiddens))

        self.drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=num_hiddens, nhead=num_heads,
            dim_feedforward=num_hiddens * 4, dropout=dropout,
            activation='gelu', batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(num_hiddens)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_tokens, std=0.02)

    def forward(self, x):
        B = x.shape[0]

        x = self.patch_embed(x)                     # [B, H, P, P]
        x = x.flatten(2).transpose(1, 2)            # [B, N, H]

        cls = self.cls_tokens.expand(B, -1, -1)     # [B, S, H]
        x = torch.cat([cls, x], dim=1)              # [B, S+N, H]
        x = x + self.pos_embed
        x = self.drop(x)

        x = self.transformer(x)                     # [B, S+N, H]
        x = self.norm(x[:, :self.num_scales])       # [B, S, H]
        return x


class Res2NetStyleTRE(nn.Module):
    """Res2Net-style Temporal Relation Extractor.

    S parallel branches of dilated 1D convolutions with different
    receptive fields. Output: [B, num_scales, num_hiddens].
    """

    def __init__(self, signal_chans, num_hiddens, scale_lst, dropout, num_layers):
        super().__init__()
        self.num_scales = len(scale_lst)

        self.input_proj = nn.Sequential(
            nn.Linear(signal_chans, num_hiddens),
            nn.GELU(),
        )

        self.branches = nn.ModuleList()
        for scale, n_layers in zip(scale_lst, num_layers):
            layers = []
            for _ in range(n_layers):
                layers.extend([
                    nn.Conv1d(num_hiddens, num_hiddens, kernel_size=3,
                              dilation=scale, padding=scale),
                    nn.BatchNorm1d(num_hiddens),
                    nn.GELU(),
                ])
            self.branches.append(nn.Sequential(*layers))

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.input_proj(x)                       # [B, T, H]
        x = x.transpose(1, 2)                        # [B, H, T]

        outputs = []
        for branch in self.branches:
            out = branch(x)                           # [B, H, T]
            out = self.pool(out).squeeze(-1)          # [B, H]
            outputs.append(out)

        out = torch.stack(outputs, dim=1)             # [B, S, H]
        return self.dropout(out)
