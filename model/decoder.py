"""
Structure decoder: turns the discrete structure tokens back into 3D backbone
coordinates. This is the half that makes the tokens mean anything.

Without a decoder the codebook is unconstrained -- the encoder and quantizer
could happily settle on 4096 codes that carve up their own feature space
neatly but have nothing to do with actual shape. Forcing the tokens to be
sufficient to REBUILD the backbone is what makes them encode geometry.

Two design consequences follow from how the encoder was built, and they
drive almost every decision in this file:

1. The encoder never sees the amino acid sequence, and it never sees
   absolute positions -- only rotation-invariant pairwise geometry. So the
   tokens genuinely do not contain the protein's orientation in space.
   The decoder therefore CANNOT be trained against raw coordinates; it has
   no way to know which way the protein was pointing. The loss has to be
   orientation-blind too, which is why this file ends in FAPE rather than
   a plain mean-squared error on xyz.

2. The encoder got all of its residue-to-residue structure from geometric
   pair features. The decoder has no geometry to start from (that's the
   thing it's trying to produce), so it uses an ordinary transformer over
   the token sequence instead, plus positional encoding to know chain order.
"""

import math

import torch
import torch.nn as nn

from model.geometry import build_frames, to_local_frame

# Ideal backbone geometry, in Angstroms and degrees. These are the
# textbook average values for a protein backbone, and they barely vary
# between residues in real structures -- bond lengths and bond angles are
# essentially fixed by chemistry. Only the TORSIONS (how the chain rotates
# about its bonds) actually differ from fold to fold, so those are the
# only part worth spending network capacity on predicting.
BOND_N_CA = 1.458
BOND_CA_C = 1.525
BOND_C_O = 1.231
ANGLE_N_CA_C = math.radians(111.2)
ANGLE_CA_C_O = math.radians(120.8)


def ideal_local_backbone() -> dict:
    """
    Where the N, CA, C atoms of a single residue sit inside that residue's
    OWN local frame, assuming ideal chemistry.

    This is fully determined, with no freedom left, because build_frames()
    defines the local frame from those exact three atoms:
      - the origin is CA, so CA is at (0, 0, 0)
      - the first axis points at N, so N lies on the positive x-axis
      - the second axis is the Gram-Schmidt leftover of CA->C, so C lies
        in the xy-plane with no z-component at all

    In other words, once you know a residue's frame you already know where
    its N, CA and C are. The network only has to predict the frame.

    Also returns an orthonormal basis (u, p, q) at the C atom, used for
    placing O -- see place_oxygen() for why O is the odd one out.
    """
    ca = torch.zeros(3)
    n = torch.tensor([BOND_N_CA, 0.0, 0.0])
    c = torch.tensor([
        BOND_CA_C * math.cos(ANGLE_N_CA_C),
        BOND_CA_C * math.sin(ANGLE_N_CA_C),
        0.0,
    ])

    # u: unit vector along the CA->C bond. O sits at a fixed angle away
    # from this axis, and is free to swing around it.
    u = c / c.norm()

    # p, q: two unit vectors perpendicular to u and to each other, giving
    # a "clock face" around the CA->C axis. An angle on this clock face is
    # the one genuinely free number in O's position.
    n_dir = n / n.norm()
    p = n_dir - (n_dir @ u) * u   # strip out the part pointing along u
    p = p / p.norm()
    q = torch.cross(u, p, dim=0)

    return {"n": n, "ca": ca, "c": c, "u": u, "p": p, "q": q}


def rotation_from_6d(v: torch.Tensor) -> torch.Tensor:
    """
    Turn 6 freely-predicted numbers into a valid rotation matrix.

    v: (..., 6)  ->  (..., 3, 3)

    A network can't just output 9 numbers and call it a rotation, because
    almost no 9 numbers form one (they have to be orthonormal with
    determinant +1). Angles are no better: Euler angles and quaternions
    both have discontinuities, points where a tiny change in the true
    rotation demands a huge jump in the predicted numbers, which networks
    learn very badly.

    The fix (Zhou et al. 2019) is to predict two arbitrary 3D vectors and
    Gram-Schmidt them into a frame -- exactly the same construction
    build_frames() uses on N/CA/C. Every possible input maps to a valid
    rotation, and nearby rotations always have nearby inputs.
    """
    a, b = v[..., :3], v[..., 3:]

    e1 = a / a.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    b = b - (b * e1).sum(dim=-1, keepdim=True) * e1
    e2 = b / b.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    e3 = torch.cross(e1, e2, dim=-1)

    # Stacked as COLUMNS, matching build_frames' convention, so the two
    # kinds of frame are interchangeable everywhere downstream.
    return torch.stack([e1, e2, e3], dim=-1)


def sinusoidal_positions(length: int, dim: int, device, dtype) -> torch.Tensor:
    """
    Standard sine/cosine positional encoding, (length, dim).

    The decoder needs this because a plain transformer is order-blind: shuffle
    its inputs and it produces the same outputs shuffled the same way. The
    encoder never needed positional encoding, since residue order was already
    implicit in the 3D geometry it was reading. Here there is no geometry yet,
    so chain order has to be supplied explicitly.

    Sinusoidal rather than learned because proteins vary in length and a
    learned table would cap how long a chain we can decode.
    """
    position = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    freq = torch.exp(
        torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / dim)
    )
    pe = torch.zeros(length, dim, device=device, dtype=dtype)
    pe[:, 0::2] = torch.sin(position * freq)
    pe[:, 1::2] = torch.cos(position * freq)
    return pe


class TransformerLayer(nn.Module):
    """
    One ordinary pre-norm transformer block: self-attention, then a small
    feed-forward network, each wrapped in a residual connection.

    Note the difference from GeometricAttentionLayer in the encoder. That
    one had no feed-forward block, because its pair_to_value path was
    already injecting rich per-pair geometric content and doing the heavy
    lifting. Here there is no such path, so the feed-forward block is
    carrying the per-residue computation instead. Dropping it would leave
    the decoder able only to average token vectors together, which is far
    too weak to reconstruct a fold.
    """

    def __init__(self, dim: int, num_heads: int, ff_mult: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.norm_attn = nn.LayerNorm(dim)
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)
        self.to_out = nn.Linear(dim, dim)

        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Linear(dim * ff_mult, dim),
        )

        # Xavier throughout, same reasoning as the encoder's layers.
        for layer in [self.to_q, self.to_k, self.to_v, self.to_out, self.ff[0], self.ff[2]]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x:    (B, L, dim)
        mask: (B, L) -- True for real residues, False for padding
        """
        B, L, dim = x.shape

        h = self.norm_attn(x)
        q = self.to_q(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v(h).view(B, L, self.num_heads, self.head_dim).transpose(1, 2)

        attn_logits = q @ k.transpose(-1, -2) / (self.head_dim ** 0.5)  # (B, H, L, L)

        # Padding residues must never be attended TO, or real residues would
        # blend in meaningless zero-padding.
        attn_logits = attn_logits.masked_fill(~mask.unsqueeze(1).unsqueeze(1), float("-inf"))
        attn_weights = attn_logits.softmax(dim=-1)

        attn_out = (attn_weights @ v).transpose(1, 2).reshape(B, L, dim)
        x = x + self.to_out(attn_out)

        x = x + self.ff(self.norm_ff(x))
        return x


class StructureDecoder(nn.Module):
    """
    Tokens (as their quantized vectors) in, backbone coordinates out.

    The output head does NOT predict 12 loose xyz numbers per residue. It
    predicts each residue's rigid FRAME (a rotation and a position), then
    drops the atoms into that frame at their ideal chemical offsets. That
    way bond lengths and bond angles are correct by construction and can
    never drift, and the network spends its capacity on the part that
    actually varies between folds -- how each residue is oriented relative
    to its neighbours. This is the same trick AlphaFold's structure module
    uses, and it's the direct inverse of what build_frames() does.
    """

    def __init__(self, code_dim: int = 128, dim: int = 128, num_heads: int = 4, num_layers: int = 4):
        super().__init__()
        self.dim = dim

        self.input_proj = nn.Linear(code_dim, dim)
        self.layers = nn.ModuleList([TransformerLayer(dim, num_heads) for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(dim)

        # Three output heads, one per thing a residue's geometry needs.
        self.to_rotation = nn.Linear(dim, 6)     # -> rotation, via rotation_from_6d
        self.to_translation = nn.Linear(dim, 3)  # -> where this residue's CA sits
        self.to_psi = nn.Linear(dim, 2)          # -> unnormalized (cos, sin) for O's swing

        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)

        # Start every head predicting something neutral and sane: zero
        # weights, and biases set so that before any training every residue
        # comes out as an unrotated frame at the origin. Untrained random
        # rotations would make the first few gradient steps chaotic.
        for head, bias in [
            (self.to_rotation, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),  # -> identity rotation
            (self.to_translation, [0.0, 0.0, 0.0]),
            (self.to_psi, [1.0, 0.0]),                            # -> angle 0
        ]:
            nn.init.zeros_(head.weight)
            with torch.no_grad():
                head.bias.copy_(torch.tensor(bias))

        # Ideal geometry is constant, so register it as buffers: moved to
        # GPU with .to(device) alongside the weights, but never trained.
        for name, tensor in ideal_local_backbone().items():
            self.register_buffer(f"ideal_{name}", tensor)

    def place_oxygen(self, psi: torch.Tensor) -> torch.Tensor:
        """
        Work out O's position in the local frame, given a predicted angle.

        psi: (B, L, 2) -- unnormalized (cos, sin)
        Returns: (B, L, 3) -- O in local-frame coordinates

        O is the one backbone atom the frame does not pin down. N, CA and C
        define the frame, so they're fixed inside it, but O hangs off the C
        atom and is free to swing around the CA->C axis. That swing is the
        psi torsion, and it genuinely varies between residues, so it has to
        be predicted rather than assumed.

        The network outputs a raw (cos, sin) pair which gets normalized to
        unit length, instead of outputting the angle directly. An angle
        would wrap around at 2*pi, and the network would have to make a
        discontinuous jump to cross that seam. A point on a circle has no
        seam. (Whatever constant offset there is between this angle and the
        formal definition of psi just gets absorbed into the learned
        weights, so it doesn't need to be worked out here.)
        """
        psi = psi / psi.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        cos_t, sin_t = psi[..., 0:1], psi[..., 1:2]

        # Direction from C to O: a fixed tilt off the CA->C axis (set by the
        # CA-C-O bond angle), plus a free rotation around it.
        swing = cos_t * self.ideal_p + sin_t * self.ideal_q
        direction = -math.cos(ANGLE_CA_C_O) * self.ideal_u + math.sin(ANGLE_CA_C_O) * swing

        return self.ideal_c + BOND_C_O * direction

    def forward(self, quantized: torch.Tensor, mask: torch.Tensor) -> dict:
        """
        quantized: (B, L, code_dim) -- the quantizer's straight-through output.
                   To decode from raw token ids instead, look them up first:
                   quantizer.codebook(tokens)
        mask:      (B, L) -- True for real residues

        Returns a dict with:
          coords:       (B, L, 4, 3) reconstructed N, CA, C, O positions
          rotations:    (B, L, 3, 3) predicted frames
          translations: (B, L, 3)    predicted CA positions
        """
        B, L, _ = quantized.shape

        x = self.input_proj(quantized)
        x = x + sinusoidal_positions(L, self.dim, x.device, x.dtype)

        for layer in self.layers:
            x = layer(x, mask)
        x = self.final_norm(x)

        rotations = rotation_from_6d(self.to_rotation(x))  # (B, L, 3, 3)
        translations = self.to_translation(x)              # (B, L, 3)

        # Assemble each residue's four atoms in its own local frame: three
        # of them fixed by ideal chemistry, O from the predicted torsion.
        oxygen = self.place_oxygen(self.to_psi(x))                     # (B, L, 3)
        n = self.ideal_n.expand(B, L, 3)
        ca = self.ideal_ca.expand(B, L, 3)
        c = self.ideal_c.expand(B, L, 3)
        local_atoms = torch.stack([n, ca, c, oxygen], dim=2)           # (B, L, 4, 3)

        # Push each residue's local atoms out into shared world space:
        # rotate by that residue's frame, then shift to its CA position.
        # Exactly the inverse of to_local_frame().
        coords = torch.einsum("blij,blaj->blai", rotations, local_atoms) + translations.unsqueeze(2)

        return {"coords": coords, "rotations": rotations, "translations": translations}


def fape_loss(
    pred: dict,
    true_coords: torch.Tensor,
    mask: torch.Tensor,
    clamp: float = 10.0,
    length_scale: float = 10.0,
) -> torch.Tensor:
    """
    Frame Aligned Point Error -- the reconstruction loss, and the only kind
    that can work here.

    A plain MSE between predicted and true coordinates would be unlearnable.
    The encoder's features are rotation-invariant by construction, so the
    tokens carry no information at all about which way the protein was
    pointing in the original PDB file. A coordinate MSE would keep punishing
    the decoder for a global orientation it has no way to know, and the best
    it could do is hedge toward a blurry average.

    FAPE removes that problem by never comparing world positions. Instead,
    for every residue i, it re-expresses every atom in residue i's own local
    frame -- once using the predicted structure's frame and atoms, once
    using the true structure's -- and compares those. Rotate the whole true
    protein and every one of those local views is unchanged, so the loss is
    unchanged. What it's really measuring is "from where residue i is
    standing, does the rest of the protein look right," summed over every
    residue's point of view. Getting that right for all i leaves only one
    possible shape, so it's a strict measure despite ignoring orientation.

    clamp: errors beyond 10A stop growing. Early in training everything is
    wildly wrong, and without the clamp a few far-apart residue pairs would
    produce enormous gradients that drown out the many nearly-correct local
    ones. Capping the penalty keeps the model working on local geometry
    first and global arrangement second.

    length_scale: divides the result so the loss lands in [0, 1] rather than
    in Angstroms, which keeps it on a comparable footing with the
    quantizer's codebook and commitment losses when they're added together.
    """
    B, L = mask.shape

    true_rotations, true_translations = build_frames(true_coords)

    # Flatten the per-residue atom axis: every one of the L*4 atoms will be
    # viewed from every one of the L frames.
    pred_atoms = pred["coords"].reshape(B, L * 4, 3)
    true_atoms = true_coords.reshape(B, L * 4, 3)

    # Broadcast the atom list across the frame axis, then reuse the encoder's
    # own to_local_frame -- the same machinery on both sides guarantees the
    # two views are defined identically.
    pred_local = to_local_frame(
        pred_atoms.unsqueeze(1).expand(-1, L, -1, -1), pred["rotations"], pred["translations"]
    )  # (B, L, L*4, 3)
    true_local = to_local_frame(
        true_atoms.unsqueeze(1).expand(-1, L, -1, -1), true_rotations, true_translations
    )

    # Distance between the two views of each (frame, atom) pair. The epsilon
    # is load-bearing: .norm() of an exactly-zero vector has an undefined
    # (NaN) gradient, and predicted-equals-true happens for real.
    diff = pred_local - true_local
    distance = (diff.pow(2).sum(dim=-1) + 1e-8).sqrt()  # (B, L, L*4)
    distance = distance.clamp(max=clamp)

    # A pair counts only if both the frame residue and the atom's residue
    # are real. repeat_interleave(4) stretches the per-residue mask over
    # that residue's four atoms.
    frame_mask = mask.unsqueeze(2)                             # (B, L, 1)
    atom_mask = mask.repeat_interleave(4, dim=1).unsqueeze(1)  # (B, 1, L*4)
    pair_mask = (frame_mask & atom_mask).float()               # (B, L, L*4)

    return (distance * pair_mask).sum() / pair_mask.sum().clamp(min=1.0) / length_scale
