"""
Structure encoder: turns per-residue backbone geometry into one continuous
feature vector per residue, ready to be quantized into discrete tokens.

Deliberately does NOT see the amino acid sequence -- only geometry -- so
that the same fold produces similar features regardless of which amino
acids happen to compose it. This is the actual "understanding shape"
part of the pipeline; VQ quantization (next file) turns these continuous
vectors into discrete tokens.
"""

import torch
import torch.nn as nn

from model.geometry import build_frames, to_local_frame


def pairwise_features(coords: torch.Tensor) -> torch.Tensor:
    """
    Build a feature vector for every (residue_i, residue_j) pair, describing
    residue j's geometry as seen from residue i's own local frame.

    coords: (B, L, 4, 3)
    Returns: (B, L, L, 13) -- for each pair, 3 numbers for relative position,
    9 for relative orientation, 1 for distance.
    """
    rotations, translations = build_frames(coords)  # (B,L,3,3), (B,L,3)
    B, L, _ = translations.shape

    # Relative position: where does residue j's CA sit, measured in
    # residue i's own local axes?
    ca = coords[:, :, 1, :]                       # (B, L, 3) -- CA atoms only
    ca_j = ca.unsqueeze(1).expand(-1, L, -1, -1)   # (B, L, L, 3): j varies along dim 2
    relative_position = to_local_frame(ca_j, rotations, translations)  # (B, L, L, 3)

    # Relative orientation: how is residue j's own frame rotated, compared
    # to residue i's frame? R_i^T @ R_j -- invariant for the same reason
    # relative_position is (a shared whole-protein rotation cancels out).
    rot_i = rotations.unsqueeze(2)                 # (B, L, 1, 3, 3)
    rot_j = rotations.unsqueeze(1)                 # (B, 1, L, 3, 3)
    relative_rotation = rot_i.transpose(-1, -2) @ rot_j  # (B, L, L, 3, 3)
    relative_rotation_flat = relative_rotation.reshape(B, L, L, 9)

    # Distance between residues -- redundant with relative_position (it's
    # that vector's length) but giving it directly speeds up training.
    distance = relative_position.norm(dim=-1, keepdim=True)  # (B, L, L, 1)

    return torch.cat([relative_position, relative_rotation_flat, distance], dim=-1)
    # final shape: (B, L, L, 13)


class GeometricAttentionLayer(nn.Module):
    """
    One layer of attention where the pairwise geometry between residues
    does two separate jobs:
      1. biases WHICH residues a given residue pays attention to
         (pair_to_bias), and
      2. contributes its own content to WHAT gets aggregated
         (pair_to_value).

    Job 2 matters more than it looks: every residue starts from the exact
    same learned vector (see StructureEncoder.initial_embedding below), so
    without job 2, every residue would be averaging together identical
    values -- and averaging identical values always gives back that same
    identical value, no matter how the averaging is weighted. Residues
    would never become distinguishable from one another. pair_to_value
    injects real, pair-specific geometric content directly into the
    aggregation, which is what actually breaks that symmetry.
    """

    def __init__(self, dim: int, num_heads: int, pair_dim: int = 13):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)

        # Turns each pair's 13-number geometry into one attention-logit
        # bias per head -- how much should this pair's real 3D relationship
        # push residue i to attend to residue j.
        self.pair_to_bias = nn.Linear(pair_dim, num_heads)

        # Turns each pair's 13-number geometry into a full per-head content
        # vector -- the "sticker" added onto residue j's value before it's
        # folded into residue i's blend. This is what makes the blended
        # result differ across residues even when every residue's own
        # value vector started out identical.
        self.pair_to_value = nn.Linear(pair_dim, dim)

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, pair_feats: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x:          (B, L, dim)   -- current per-residue features
        pair_feats: (B, L, L, 13) -- from pairwise_features()
        mask:       (B, L)        -- True for real residues, False for padding
        """
        B, L, dim = x.shape
        x_norm = self.norm(x)

        q = self.to_q(x_norm).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, L, hd)
        k = self.to_k(x_norm).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(x_norm).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        attn_logits = q @ k.transpose(-1, -2) / (self.head_dim ** 0.5)  # (B, H, L, L)

        pair_bias = self.pair_to_bias(pair_feats).permute(0, 3, 1, 2)  # (B, H, L, L)
        attn_logits = attn_logits + pair_bias

        pad_mask = mask.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, L)
        attn_logits = attn_logits.masked_fill(~pad_mask, float("-inf"))
        attn_weights = attn_logits.softmax(dim=-1)  # (B, H, L, L)

        # Per-pair "sticker" content: reshape into per-head chunks so it
        # can be added onto v, which is also split per head.
        pair_values = self.pair_to_value(pair_feats)  # (B, L, L, dim)
        pair_values = pair_values.view(B, L, L, self.num_heads, self.head_dim).permute(0, 3, 1, 2, 4)
        # pair_values: (B, H, L, L, hd) -- one sticker-adjusted value per
        # (query residue i, key residue j) pair, per head.

        # v is the same for every query residue i (it only varies by key
        # residue j), so broadcast it across the new "i" dimension before
        # adding the pair-specific stickers, which DO vary by i.
        v_expanded = v.unsqueeze(2).expand(-1, -1, L, -1, -1)  # (B, H, L, L, hd)
        combined_values = v_expanded + pair_values               # (B, H, L, L, hd)

        # Weighted blend: for each query residue i, combine every key
        # residue j's (value + sticker) using the attention weights.
        attn_out = (attn_weights.unsqueeze(-1) * combined_values).sum(dim=3)  # (B, H, L, hd)

        attn_out = attn_out.transpose(1, 2).reshape(B, L, dim)
        out = self.to_out(attn_out)

        # Residual connection: refine the input rather than replace it,
        # standard practice for stable training across stacked layers.
        return x + out


class StructureEncoder(nn.Module):
    """
    Stacks several GeometricAttentionLayers to turn raw backbone geometry
    into a final per-residue feature vector, ready for VQ quantization.
    """

    def __init__(self, dim: int = 128, num_heads: int = 4, num_layers: int = 4):
        super().__init__()

        # Every residue starts from the same learned vector -- there's no
        # sequence information to differentiate them at the input. All
        # differentiation between residues comes from geometry, injected
        # via pair_to_value inside each attention layer.
        self.initial_embedding = nn.Parameter(torch.randn(dim) * 0.02)

        self.layers = nn.ModuleList(
            [GeometricAttentionLayer(dim, num_heads) for _ in range(num_layers)]
        )

    def forward(self, coords: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        coords: (B, L, 4, 3)
        mask:   (B, L)
        Returns: (B, L, dim) -- one feature vector per residue
        """
        B, L = mask.shape
        pair_feats = pairwise_features(coords)  # (B, L, L, 13), computed once, reused every layer

        x = self.initial_embedding.expand(B, L, -1)
        for layer in self.layers:
            x = layer(x, pair_feats, mask)

        return x
