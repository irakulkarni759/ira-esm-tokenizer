"""
Rigid-body reference frames per residue, built from backbone coordinates.

This is the geometric foundation the encoder is built on (same idea as
AlphaFold's "rigidFrom3Points" and ESM3's structure encoder): each residue
gets its own local coordinate frame (a rotation + a translation) derived
from its own N, CA, C atom positions. Expressing other residues relative
to this frame, instead of in absolute world coordinates, is what makes the
resulting features invariant to how the whole protein is rotated or moved
in space -- the same fold produces the same features regardless of its
orientation.
"""

import torch

# Indices into the 4-atom backbone dimension (N, CA, C, O) used throughout
# the data pipeline -- see BACKBONE_ATOMS in parse_structures.py.
N_IDX, CA_IDX, C_IDX, O_IDX = 0, 1, 2, 3


def build_frames(coords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build one rigid frame per residue from its N, CA, C atoms.

    coords: (B, L, 4, 3) -- batch of structures, backbone atoms per residue.
    Returns (rotations, translations):
      rotations:    (B, L, 3, 3) rotation matrix per residue
      translations: (B, L, 3)    origin (= that residue's CA position)

    The frame is centered at CA, with its axes built from the N-CA and
    C-CA directions via Gram-Schmidt -- this is exactly how AlphaFold
    defines each residue's local frame from three backbone points.
    """
    n = coords[..., N_IDX, :]   # (B, L, 3)
    ca = coords[..., CA_IDX, :]  # (B, L, 3)
    c = coords[..., C_IDX, :]   # (B, L, 3)

    # The frame's origin is the residue's own CA position -- every other
    # residue's position will later be described as "how far from this
    # residue's CA, in this residue's own rotated axes."
    translations = ca

    # First axis: the direction from CA toward N, normalized to unit length.
    # The .clamp(min=1e-8) on every norm below is not cosmetic. collate_fn
    # pads short structures with ZERO coordinates, so a padded position has
    # n == ca == c == 0, giving a zero-length v1 and a 0/0 = NaN axis. NaN
    # then spreads everywhere downstream, including into real residues (it
    # survives being multiplied by a zero attention weight), so masking
    # alone does not contain it. Clamping makes padded frames harmlessly
    # zero instead of NaN, and the masks discard them as intended.
    v1 = n - ca
    e1 = v1 / v1.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    # Second axis: start from the CA->C direction, then Gram-Schmidt it --
    # subtract out whatever part of it points along e1, so e1 and e2 end
    # up exactly perpendicular. Without this step the two axes wouldn't
    # form a valid (orthogonal) coordinate system.
    v2 = c - ca
    v2 = v2 - (v2 * e1).sum(dim=-1, keepdim=True) * e1
    e2 = v2 / v2.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    # Third axis: perpendicular to both e1 and e2, via the cross product --
    # completes a valid right-handed 3D coordinate system.
    e3 = torch.cross(e1, e2, dim=-1)

    # Stack the three axes as columns to form each residue's rotation matrix.
    rotations = torch.stack([e1, e2, e3], dim=-1)  # (B, L, 3, 3)

    return rotations, translations


def to_local_frame(
    point: torch.Tensor, rotations: torch.Tensor, translations: torch.Tensor
) -> torch.Tensor:
    """
    Express world-coordinate point(s) relative to each residue's own frame.

    point:        (B, L, 3) or (B, L, L, 3) -- world-space position(s)
    rotations:    (B, L, 3, 3) from build_frames
    translations: (B, L, 3)    from build_frames

    This answers "where does this point sit, if I measure it using this
    residue's own rotated axes, starting from this residue's own CA?" --
    the actual invariant feature. Rotate or translate the whole protein
    in world space, and this output stays identical.
    """
    # Handle both a single point per residue (B, L, 3) and a full pairwise
    # grid (B, L, L, 3, one entry per residue-pair) by inserting a
    # broadcastable dimension for the pairwise case.
    if point.dim() == translations.dim():
        # (B, L, 3): shift into this frame's origin, then unsqueeze for matmul
        centered = (point - translations).unsqueeze(-1)  # (B, L, 3, 1)
        local = rotations.transpose(-1, -2) @ centered     # (B, L, 3, 1)
        return local.squeeze(-1)                            # (B, L, 3)
    else:
        # (B, L, L, 3): pairwise -- translations/rotations broadcast over
        # the extra L dimension (each row uses its own residue's frame)
        centered = (point - translations.unsqueeze(2)).unsqueeze(-1)  # (B, L, L, 3, 1)
        rot = rotations.unsqueeze(2).transpose(-1, -2)                 # (B, L, 1, 3, 3)
        local = rot @ centered                                          # (B, L, L, 3, 1)
        return local.squeeze(-1)                                        # (B, L, L, 3)
