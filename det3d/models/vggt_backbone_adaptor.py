"""
VGGTBackboneAdaptor: Converts frozen VGGT intermediate tokens + depth
into MonoDETR-compatible multi-scale feature maps.

Design (refined multi-layer version):
    4 VGGT intermediate layers (idx 4, 11, 17, 23) -> 4 spatial scales.
    Spatial sizes are derived from the input patch grid and downsampled by
    1x, 2x, 4x, and 8x, so both 518x154 and 1280x384 inputs are supported.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionEmbeddingSine2D(nn.Module):
    """Standard 2D sine positional encoding."""

    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W] feature map
        Returns:
            pos: [B, 2*num_pos_feats, H, W]
        """
        B, C, H, W = x.shape
        device = x.device

        y_embed = torch.arange(H, dtype=torch.float32, device=device)
        x_embed = torch.arange(W, dtype=torch.float32, device=device)

        if self.normalize:
            y_embed = (y_embed + 0.5) / (H + 1e-6) * self.scale
            x_embed = (x_embed + 0.5) / (W + 1e-6) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, None] / dim_t  # [W, num_pos_feats]
        pos_y = y_embed[:, None] / dim_t  # [H, num_pos_feats]

        pos_x = torch.stack([pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()], dim=2).flatten(1)  # [W, num_pos_feats]
        pos_y = torch.stack([pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()], dim=2).flatten(1)  # [H, num_pos_feats]

        # Broadcast to [B, H, W, 2*num_pos_feats] then permute
        pos_y = pos_y.unsqueeze(1).expand(H, W, -1)  # [H, W, num_pos_feats]
        pos_x = pos_x.unsqueeze(0).expand(H, W, -1)  # [H, W, num_pos_feats]

        pos = torch.cat([pos_y, pos_x], dim=-1)       # [H, W, 2*num_pos_feats]
        pos = pos.permute(2, 0, 1).unsqueeze(0)        # [1, 2*num_pos_feats, H, W]
        pos = pos.expand(B, -1, -1, -1)                # [B, 2*num_pos_feats, H, W]

        return pos.contiguous()


class VGGTBackboneAdaptor(nn.Module):
    """
    Converts VGGT's intermediate tokens and depth into MonoDETR-compatible inputs.

    Trainable parameters (~2M):
        - 4 × Conv2d(2048 -> 256) projection layers
        - 4 × GroupNorm(32, 256) normalization
        - depth_pos_embed Embedding(81, 256)
        - depth_scale learnable scalar
    """

    LAYER_INDICES = [4, 11, 17, 23]  # cached VGGT intermediate layers

    def __init__(self, hidden_dim=256, depth_max=80.0):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.depth_max = depth_max
        vggt_dim = 2048  # 2 * embed_dim (frame concat global)

        # Per-layer 1x1 conv projection: 2048 -> hidden_dim
        self.proj_layers = nn.ModuleList([
            nn.Conv2d(vggt_dim, hidden_dim, kernel_size=1, bias=False)
            for _ in range(4)
        ])
        self.norm_layers = nn.ModuleList([
            nn.GroupNorm(32, hidden_dim)
            for _ in range(4)
        ])

        # Sine positional encoding (shared across scales, applied per-scale)
        self.pos_encoder = PositionEmbeddingSine2D(num_pos_feats=hidden_dim // 2)

        # Depth position embedding (same as MonoDETR's DepthPredictor)
        # depth_max=80 -> 81 bins (0..80 inclusive)
        self.depth_pos_embed = nn.Embedding(int(depth_max) + 1, hidden_dim)

        # Learnable scale: converts VGGT relative depth to approximate metric depth
        # Initialize at 10.0 as a reasonable starting point (VGGT depth ~ 1/10 of metric)
        self.depth_scale = nn.Parameter(torch.ones(1) * 10.0)

        # Initialize weights
        for proj in self.proj_layers:
            nn.init.xavier_uniform_(proj.weight)

    def interpolate_depth_embed(self, depth):
        """
        Exactly mirrors MonoDETR DepthPredictor.interpolate_depth_embed.
        Args:
            depth: [B, H, W] metric depth in meters (clamped to [0, depth_max])
        Returns:
            [B, hidden_dim, H, W]
        """
        depth = depth.clamp(min=0.0, max=self.depth_max)
        pos = self._interpolate_1d(depth, self.depth_pos_embed)  # [B, H, W, hidden_dim]
        return pos.permute(0, 3, 1, 2)  # [B, hidden_dim, H, W]

    def _interpolate_1d(self, coord, embed):
        """Linearly interpolate between adjacent depth embeddings."""
        floor_coord = coord.floor()
        delta = (coord - floor_coord).unsqueeze(-1)
        floor_coord = floor_coord.long().clamp(0, embed.num_embeddings - 1)
        ceil_coord  = (floor_coord + 1).clamp(0, embed.num_embeddings - 1)
        return embed(floor_coord) * (1.0 - delta) + embed(ceil_coord) * delta

    @staticmethod
    def _get_pyramid_sizes(height, width):
        """Return ceil-downsampled 1x/2x/4x/8x feature sizes."""
        return [
            ((height + scale - 1) // scale, (width + scale - 1) // scale)
            for scale in (1, 2, 4, 8)
        ]

    def forward(self, aggregated_tokens_list, patch_start_idx, images_shape,
                depth, depth_conf):
        """
        Args:
            aggregated_tokens_list: list[24] from VGGT aggregator; non-None at indices 4,11,17,23
            patch_start_idx:        int = 5 (camera token + 4 register tokens)
            images_shape:           tuple (B, S, C, H_img, W_img) of the VGGT input
            depth:                  [B, S, H_img, W_img, 1] VGGT relative depth
            depth_conf:             [B, S, H_img, W_img]  depth confidence

        Returns:
            srcs:               list[4] of [B, hidden_dim, H_i, W_i]
            masks:              list[4] of [B, H_i, W_i] bool (all False)
            pos:                list[4] of [B, hidden_dim, H_i, W_i]
            depth_pos_embed:    [B, hidden_dim, H_1, W_1]   (at scale 1)
            depth_pos_embed_ip: [B, hidden_dim, H_1, W_1]   (same, mirrors MonoDETR)
            weighted_depth:     [B, H_1, W_1]               (metric-ish depth at scale 1)
        """
        B, S, C_img, H_img, W_img = images_shape
        H_p = H_img // 14  # patch grid height
        W_p = W_img // 14  # patch grid width
        pool_sizes = self._get_pyramid_sizes(H_p, W_p)
        device = aggregated_tokens_list[self.LAYER_INDICES[0]].device

        srcs  = []
        masks = []
        pos   = []

        for i, layer_idx in enumerate(self.LAYER_INDICES):
            tokens = aggregated_tokens_list[layer_idx]  # [B, S, P, 2048]
            assert tokens is not None, f"Layer {layer_idx} not cached in VGGT aggregator"

            # Extract patch tokens (drop camera + register tokens)
            patch_tokens = tokens[:, 0, patch_start_idx:, :]  # [B, N_patches, 2048]

            # Reshape to spatial grid
            feat = patch_tokens.reshape(B, H_p, W_p, -1).permute(0, 3, 1, 2)  # [B, 2048, H_p, W_p]

            # Project 2048 -> hidden_dim and normalize
            feat = self.proj_layers[i](feat)
            feat = self.norm_layers[i](feat)
            feat = F.relu(feat)

            # Adaptive pool to target scale
            target_h, target_w = pool_sizes[i]
            feat = F.adaptive_avg_pool2d(feat, (target_h, target_w))  # [B, hidden_dim, H_i, W_i]

            # Sinusoidal position encoding at this scale
            pos_enc = self.pos_encoder(feat)  # [B, hidden_dim, H_i, W_i]

            # All-valid mask (no padding)
            mask = torch.zeros(B, target_h, target_w, dtype=torch.bool, device=device)

            srcs.append(feat)
            masks.append(mask)
            pos.append(pos_enc)

        # Depth position embedding at scale 1.
        # depth: [B, S, H_img, W_img, 1] -> squeeze to [B, H_img, W_img]
        depth_map = depth[:, 0, :, :, 0]  # [B, H_img, W_img]

        # Convert VGGT relative depth to approximate metric depth
        # VGGT depth head uses exp activation, values are log-like scale
        depth_metric = depth_map * self.depth_scale.abs()  # [B, H_img, W_img]

        # Confidence-weighted depth (suppress low-confidence areas)
        conf = depth_conf[:, 0]  # [B, H_img, W_img]
        conf_w = torch.sigmoid(conf)
        depth_metric = depth_metric * conf_w  # soft masking

        # Resize to scale-1 target size
        target_h1, target_w1 = pool_sizes[1]
        weighted_depth = F.interpolate(
            depth_metric.unsqueeze(1),
            size=(target_h1, target_w1),
            mode='bilinear',
            align_corners=True,
        ).squeeze(1)  # [B, 6, 19]

        # Create depth position embedding
        depth_pos_embed    = self.interpolate_depth_embed(weighted_depth)  # [B, hidden_dim, 6, 19]
        depth_pos_embed_ip = depth_pos_embed  # same (MonoDETR uses both; here they're identical)

        return srcs, masks, pos, depth_pos_embed, depth_pos_embed_ip, weighted_depth
