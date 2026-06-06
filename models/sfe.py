import math
import torch
from torch import nn

from .convnext_blocks import LayerNorm, Block as ConvNeXtBlock


class SpatialAttentionModule(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.avg_pooling = torch.mean
        self.max_pooling = torch.max
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_x = self.avg_pooling(x, dim=1, keepdim=True)
        max_x, _ = self.max_pooling(x, dim=1, keepdim=True)
        v = self.conv(torch.cat((max_x, avg_x), dim=1))
        v = self.sigmoid(v)
        return x * v


class ECALayer(nn.Module):
    """Efficient Channel Attention."""
    def __init__(self, k_size=3):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class CSAM(nn.Module):
    """Channel and Spatial Attention Module."""
    def __init__(self, eca_kernel_size=3, spatial_attention_kernel_size=7):
        super().__init__()
        self.eca_layer = ECALayer(k_size=eca_kernel_size)
        self.sam_layer = SpatialAttentionModule(kernel_size=spatial_attention_kernel_size)

    def forward(self, x):
        x = self.eca_layer(x)
        x = self.sam_layer(x)
        return x


class MHCA(nn.Module):
    """Multi-Head Convolutional Attention."""
    def __init__(self, in_channels, out_channels, head_dim):
        super().__init__()
        self.group_conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                                       padding=1, groups=out_channels // head_dim, bias=False)
        self.norm = nn.BatchNorm2d(out_channels, eps=1e-5)
        self.act = nn.ReLU(inplace=True)
        self.projection = nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.group_conv3x3(x)
        out = self.norm(out)
        out = self.act(out)
        out = self.projection(out)
        return out


class MultiscaleConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, head_dim, num_scales, eca_kernel, sam_kernel):
        super().__init__()
        self.csams = nn.ModuleList()
        for i in range(num_scales):
            self.csams.append(CSAM(eca_kernel, sam_kernel))
        self.mhca = MHCA(num_scales * in_channels, num_scales * out_channels, head_dim)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.num_scales = num_scales

    def forward(self, xs):
        for i in range(self.num_scales):
            if i == 0:
                xs[i] = self.csams[i](xs[i])
            else:
                xs[i] = self.csams[i](xs[i] + xs[i - 1])

        x = torch.stack(xs, dim=1)    # [B, S, C, H, W]
        B, S, C, H, W = x.shape

        x = x.flatten(1, 2)           # [B, S*C, H, W]
        x = self.mhca(x)              # [B, S*C, H, W]
        x = self.gap(x)               # [B, S*C, 1, 1]
        x = x.view(B, S, C)           # [B, S, C]
        return x


class SpatialFeatureExtractor(nn.Module):
    def __init__(self, in_channels, hidden_dims=96, num_layers=[2, 4, 4, 6],
                 scales_kernal=[3, 5, 7, 11], drop_path_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.num_scales = len(num_layers)

        self.conv_stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dims, kernel_size=4, stride=4),
            LayerNorm(hidden_dims, eps=1e-6, data_format="channels_first")
        )

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, max(num_layers))]
        self.convnext_layers = nn.ModuleList()
        for i in range(self.num_scales):
            layer = nn.Sequential(
                nn.Sequential(
                    *[ConvNeXtBlock(dim=hidden_dims, drop_path=dp_rates[j],
                                    layer_scale_init_value=layer_scale_init_value)
                      for j in range(num_layers[i])]
                ),
                LayerNorm(hidden_dims, eps=1e-6, data_format="channels_first"),
                nn.Conv2d(hidden_dims, hidden_dims,
                          kernel_size=scales_kernal[i],
                          padding=scales_kernal[i] // 2, stride=2)
            )
            self.convnext_layers.append(layer)

        assert hidden_dims % 8 == 0
        self.mcb = MultiscaleConvBlock(in_channels=hidden_dims, out_channels=hidden_dims,
                                       head_dim=hidden_dims // 8, num_scales=self.num_scales,
                                       eca_kernel=3, sam_kernel=7)

    def forward(self, x):
        x = self.conv_stem(x)    # [B, C, 224, 224] -> [B, H, 56, 56]
        xs = []
        for i in range(self.num_scales):
            xs.append(self.convnext_layers[i](x))
        x = self.mcb(xs)
        return x
