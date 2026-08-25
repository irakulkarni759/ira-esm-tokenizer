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


def knn_pair_mask(coords: torch.Tensor, mask: torch.Tensor, k: int) -> torch.Tensor:
    """
    (B, L, L) boolean: True where residue j is among residue i's k nearest CA
    neighbours. This is the ESM3 receptive field, and passing it into the
    encoder is what makes a residue's description depend on its own local
    surroundings rather than on the whole chain.

    Nearest in SPACE, not along the sequence. Two residues far apart in the
    chain can be neighbours if the fold brings them together, which is the
    whole point of doing it geometrically.

    coords: (B, L, 4, 3)
    mask:   (B, L)  -- True for real residues
    """
    ca = coords[:, :, 1, :]                      # (B, L, 3)
    dist = torch.cdist(ca, ca)                   # (B, L, L)

    # Padded positions sit at the origin, so without this they look like
    # plausible neighbours to every real residue and would get selected.
    dist = dist.masked_fill(~mask.unsqueeze(1), float("inf"))

    k = min(k, dist.shape[-1])
    idx = dist.topk(k, dim=-1, largest=False).indices   # (B, L, k)
    out = torch.zeros_like(dist, dtype=torch.bool).scatter_(-1, idx, True)

    # Force the diagonal on. A row with no allowed keys becomes all -inf,
    # and softmax turns that into NaN which then spreads into the real
    # residues -- the same failure mode as the zero-padding bug in
    # geometry.py. Every residue attending to itself makes that impossible.
    eye = torch.eye(dist.shape[-1], dtype=torch.bool, device=coords.device)
    return out | eye


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

        # Xavier (Glorot) initialization: picks each layer's starting random
        # weights based on both its input and output size, so signal
        # variance stays roughly stable as data passes through many stacked
        # layers. PyTorch's nn.Linear default is Kaiming/He initialization,
        # which is tuned more for ReLU-style networks -- Xavier is the more
        # standard choice for layers without a ReLU in between, which is
        # what we have here (pure attention, no feed-forward block yet).
        # Biases start at zero, standard practice -- Xavier's variance
        # formula is specifically about the weight matrix, not the bias.
        for layer in [self.to_q, self.to_k, self.to_v, self.to_out, self.pair_to_bias, self.pair_to_value]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, pair_feats: torch.Tensor, mask: torch.Tensor,
                pair_mask: torch.Tensor = None) -> torch.Tensor:
        """
        x:          (B, L, dim)   -- current per-residue features
        pair_feats: (B, L, L, 13) -- from pairwise_features()
        mask:       (B, L)        -- True for real residues, False for padding
        pair_mask:  (B, L, L)     -- optional, True where residue i is ALLOWED
                                     to attend to residue j. None means every
                                     residue may look at every other, which is
                                     the default and what the 4096-code run used.
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
        if pair_mask is not None:
            # (B, 1, 1, L) & (B, 1, L, L) broadcasts to (B, 1, L, L): a pair is
            # allowed only if residue j is real AND j is in i's neighbourhood.
            pad_mask = pad_mask & pair_mask.unsqueeze(1)
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

    def __init__(self, dim: int = 128, num_heads: int = 4, num_layers: int = 4,
                 neighbours: int = 0):
        super().__init__()

        # neighbours = 0 means unrestricted, every residue sees the whole
        # chain. That is how the 4096-code run was trained, and it stays the
        # default so old checkpoints keep loading unchanged. Set it to 16 to
        # train with ESM3's receptive field instead.
        self.neighbours = neighbours

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

        # Computed once too, and reused by every layer. The neighbourhood is
        # defined by the input coordinates, which do not change as features
        # flow up the stack, so recomputing it per layer would be waste.
        pair_mask = knn_pair_mask(coords, mask, self.neighbours) if self.neighbours > 0 else None

        x = self.initial_embedding.expand(B, L, -1)
        for layer in self.layers:
            x = layer(x, pair_feats, mask, pair_mask)

        return x
