import torch
from torch import nn
import torch.nn.functional as F

from .tre import TRE
from .sfe import SpatialFeatureExtractor
from .components import AdaptiveAggregation, ClassificationHead, init_weights, TokenBasedSFE, Res2NetStyleTRE


class MfamaNet(nn.Module):
    """
    Multimodal Neural Network with Multiscale Feature Alignment (MFA) loss.

    MFA Loss = MFA_local (per-scale cosine embedding) + MFA_global (cross-modal contrastive).
      - MFA_local:  aligns scale_i(image) <-> scale_i(signal) independently.
      - MFA_global: global cross-modal contrastive on mean-pooled embeddings.

    Ablation flags:
        use_sfe / use_tre : enable modality branches
        sfe_type : 'convnext' | 'token'
        tre_type : 'token' | 'res2net'
        fusion_type : 'choquet' | 'concat' | 'add' | 'attention'
        use_mfa  : toggle MFA loss
    """

    def __init__(self, num_classes, num_hiddens, dropout,
                 signal_chans, scale_lst, num_layers,
                 image_chans, num_blocks, scale_kernels, drop_path_rate,
                 use_sfe=True, use_tre=True,
                 sfe_type='convnext',
                 tre_type='token',
                 fusion_type='choquet',
                 use_mfa=True,
                 mfa_weight=1.0):
        super().__init__()

        assert use_sfe or use_tre, "At least one of use_sfe / use_tre must be True."
        assert len(scale_lst) == len(num_layers) == len(num_blocks) == len(scale_kernels)

        self.num_scales = len(scale_lst)
        self.use_sfe = use_sfe
        self.use_tre = use_tre
        self.sfe_type = sfe_type
        self.fusion_type = fusion_type
        self.use_mfa = use_mfa
        self.mfa_weight = mfa_weight

        # ---- TRE (signal branch) ----
        if use_tre:
            if tre_type == 'token':
                self.tre = TRE(
                    signal_chans, num_hiddens, scale_lst, dropout, num_layers)
            elif tre_type == 'res2net':
                self.tre = Res2NetStyleTRE(
                    signal_chans, num_hiddens, scale_lst, dropout, num_layers)
            else:
                raise ValueError(f"Unsupported tre_type: {tre_type}")
        else:
            self.tre = None

        # ---- SFE (image branch) ----
        if use_sfe:
            if sfe_type == 'convnext':
                self.sfe = SpatialFeatureExtractor(
                    image_chans, num_hiddens, num_blocks, scale_kernels, drop_path_rate)
            elif sfe_type == 'token':
                depth = max(num_blocks)
                self.sfe = TokenBasedSFE(
                    image_chans, num_hiddens, self.num_scales, depth=depth,
                    num_heads=4 if num_hiddens % 4 == 0 else 8, dropout=dropout)
            else:
                raise ValueError(f"Unsupported sfe_type: {sfe_type}")
        else:
            self.sfe = None

        # ---- Fusion ----
        if use_sfe and use_tre:
            if fusion_type == 'choquet':
                self.fusion = AdaptiveAggregation(self.num_scales * 2, 16)
            elif fusion_type == 'concat':
                self.fusion = nn.Sequential(
                    nn.Linear(self.num_scales * 2, num_hiddens),
                    nn.GELU(),
                    nn.Linear(num_hiddens, 1),
                )
            elif fusion_type == 'add':
                self.fusion = None
            elif fusion_type == 'attention':
                self.fusion = nn.MultiheadAttention(
                    embed_dim=num_hiddens, num_heads=4, batch_first=True)
            else:
                raise ValueError(f"Unsupported fusion_type: {fusion_type}")

        self.branch_pool = nn.AdaptiveAvgPool1d(1)

        # ---- Classifier ----
        self.head = ClassificationHead(
            in_dim=num_hiddens, num_classes=num_classes, dropout=dropout)

        # ---- MFA: local (per-scale) + global (cross-modal contrastive) ----
        if use_mfa and use_sfe and use_tre:
            self.mfa_local_img_proj = nn.Sequential(
                nn.Linear(num_hiddens, num_hiddens),
                nn.GELU(),
                nn.Linear(num_hiddens, num_hiddens),
            )
            self.mfa_local_sig_proj = nn.Sequential(
                nn.Linear(num_hiddens, num_hiddens),
                nn.GELU(),
                nn.Linear(num_hiddens, num_hiddens),
            )
            self.mfa_global_img_proj = nn.Sequential(
                nn.Linear(num_hiddens, num_hiddens),
                nn.GELU(),
            )
            self.mfa_global_sig_proj = nn.Sequential(
                nn.Linear(num_hiddens, num_hiddens),
                nn.GELU(),
            )
            self.mfa_global_temp = nn.Parameter(torch.ones([]) * 0.07)

        self.apply(init_weights)

    def forward(self, image, signal):
        # Extract features
        y_image = self.sfe(image) if self.use_sfe else None       # [B, S, H]
        y_signal = self.tre(signal) if self.use_tre else None     # [B, S, H]

        if y_image is not None:
            y_image = y_image.permute(0, 2, 1)                    # [B, H, S]
        if y_signal is not None:
            y_signal = y_signal.permute(0, 2, 1)                  # [B, H, S]

        # Fuse
        if self.use_sfe and self.use_tre:
            fused = self._fuse(y_image, y_signal)                 # [B, H]
        elif self.use_sfe:
            fused = self.branch_pool(y_image).squeeze(-1)
        else:
            fused = self.branch_pool(y_signal).squeeze(-1)

        logits = self.head(fused)

        # MFA loss (local + global)
        loss_total = torch.tensor(0.0, device=logits.device)

        if self.use_mfa and self.use_sfe and self.use_tre:
            feats_img = y_image.permute(0, 2, 1)    # [B, S, H]
            feats_sig = y_signal.permute(0, 2, 1)    # [B, S, H]
            loss_total += self.mfa_weight * (
                self.compute_mfa_local(feats_img, feats_sig) +
                self.compute_mfa_global(feats_img, feats_sig)
            )

        return logits, loss_total

    def forward_features(self, image, signal):
        """Return pooled features for visualization (t-SNE / UMAP)."""
        y_image = self.sfe(image) if self.use_sfe else None       # [B, S, H]
        y_signal = self.tre(signal) if self.use_tre else None     # [B, S, H]

        if y_image is not None:
            y_image = y_image.permute(0, 2, 1)                    # [B, H, S]
        if y_signal is not None:
            y_signal = y_signal.permute(0, 2, 1)                  # [B, H, S]

        if self.use_sfe and self.use_tre:
            image_feat = self.branch_pool(y_image).squeeze(-1)     # [B, H]
            signal_feat = self.branch_pool(y_signal).squeeze(-1)   # [B, H]
            fused = self._fuse(y_image, y_signal)                  # [B, H]
        elif self.use_sfe:
            image_feat = self.branch_pool(y_image).squeeze(-1)
            signal_feat = torch.zeros_like(image_feat)
            fused = image_feat
        else:
            signal_feat = self.branch_pool(y_signal).squeeze(-1)
            image_feat = torch.zeros_like(signal_feat)
            fused = signal_feat

        return image_feat, signal_feat, fused

    def _fuse(self, y_image, y_signal):
        if self.fusion_type == 'choquet':
            x = torch.cat([y_signal, y_image], dim=-1)
            return self.fusion(x)
        elif self.fusion_type == 'concat':
            x = torch.cat([y_signal, y_image], dim=-1)
            return self.fusion(x).squeeze(-1)
        elif self.fusion_type == 'add':
            return self.branch_pool(y_signal + y_image).squeeze(-1)
        elif self.fusion_type == 'attention':
            q = y_image.transpose(1, 2)
            kv = y_signal.transpose(1, 2)
            out, _ = self.fusion(q, kv, kv)
            return self.branch_pool(out.transpose(1, 2)).squeeze(-1)

    def compute_mfa_local(self, spatial_feat, temporal_feat):
        """Per-scale cosine embedding alignment.

        Aligns each scale independently: scale_i(image) <-> scale_i(signal).
        """
        _, S, _ = spatial_feat.shape
        loss = 0.0
        for s in range(S):
            z_img = self.mfa_local_img_proj(spatial_feat[:, s, :])   # [B, H]
            z_sig = self.mfa_local_sig_proj(temporal_feat[:, s, :])  # [B, H]
            z_img = F.normalize(z_img, dim=-1)
            z_sig = F.normalize(z_sig, dim=-1)
            cos_sim = (z_img * z_sig).sum(dim=-1)                     # [B]
            loss += (1.0 - cos_sim).mean()
        return loss / S

    def compute_mfa_global(self, spatial_feat, temporal_feat):
        """Global cross-modal contrastive loss (GLoRIA-style).

        Mean-pool over scales, project, then InfoNCE across modalities.
        """
        with torch.no_grad():
            self.mfa_global_temp.clamp_(0.01, 1.0)

        B = spatial_feat.shape[0]

        img_global = spatial_feat.mean(dim=1)                         # (B, H)
        sig_global = temporal_feat.mean(dim=1)                        # (B, H)

        img_emb = self.mfa_global_img_proj(img_global)                # (B, H)
        sig_emb = self.mfa_global_sig_proj(sig_global)                # (B, H)

        img = F.normalize(img_emb, dim=-1)
        sig = F.normalize(sig_emb, dim=-1)

        labels = torch.arange(B, device=img.device)
        sim = torch.matmul(img, sig.t()) / self.mfa_global_temp

        loss_i2s = F.cross_entropy(sim, labels)
        loss_s2i = F.cross_entropy(sim.t(), labels)
        return loss_i2s + loss_s2i
