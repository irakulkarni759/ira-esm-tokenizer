"""
Vector quantizer: the actual "tokenizer" step. Snaps each residue's
continuous 128-number encoder output to the nearest entry in a fixed,
learned codebook, producing a discrete integer per residue -- the
structure token.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """
    num_codes: size of the codebook -- how many distinct structure tokens
    can exist. 4096 matches ESM3's actual structure tokenizer; big enough
    to capture a wide variety of local shapes, small enough to keep the
    vocabulary learnable from ~1000 training structures.

    code_dim: must match the encoder's output size (dim=128), since we're
    comparing encoder outputs directly against codebook entries.
    """

    def __init__(self, num_codes: int = 4096, code_dim: int = 128, commitment_weight: float = 0.25):
        super().__init__()
        self.commitment_weight = commitment_weight

        # The codebook itself: num_codes learned vectors, each code_dim
        # numbers long -- literally the "4096 reference paint chips."
        # These start as random noise and get shaped by training into
        # genuinely useful, distinct local-geometry prototypes.
        self.codebook = nn.Embedding(num_codes, code_dim)

        # Small random initialization, same reasoning as the encoder's
        # initial_embedding -- keeps early training numerically stable.
        nn.init.normal_(self.codebook.weight, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> dict:
        """
        x:    (B, L, code_dim) -- continuous encoder output
        mask: (B, L)           -- True for real residues, False for padding

        Returns a dict with:
          tokens:     (B, L) integer token id per residue -- the actual output
          quantized:  (B, L, code_dim) the snapped-to-codebook vectors,
                      usable as input to the decoder
          loss:       scalar, added to the training loss (see below)
        """
        B, L, D = x.shape
        codebook = self.codebook.weight  # (num_codes, code_dim)

        # Find each residue's nearest codebook entry, measured by squared
        # Euclidean distance -- "which reference chip is the closest color
        # match." Expand x and codebook to compare every residue against
        # every code in one batched operation, rather than looping.
        x_flat = x.reshape(B * L, D)                                # (B*L, D)
        distances = torch.cdist(x_flat, codebook)                    # (B*L, num_codes)
        tokens_flat = distances.argmin(dim=1)                        # (B*L,) -- nearest code index per residue
        tokens = tokens_flat.view(B, L)

        # Look up the actual vector for each chosen code -- this is what
        # gets passed on to the decoder, not the raw continuous x.
        quantized = self.codebook(tokens)  # (B, L, D)

        # --- The straight-through estimator ---
        # quantized.detach() means "treat this as a fixed constant during
        # backpropagation, don't try to compute its gradient." Adding
        # (x - x.detach()), which numerically equals zero, doesn't change
        # the actual value at all -- but it does change what gradients
        # flow backward: gradients end up flowing as if this whole line
        # were just "quantized = x", letting training signal reach the
        # encoder even though argmin itself has no usable gradient.
        quantized_st = x + (quantized - x).detach()

        # --- Two loss terms, both standard VQ-VAE ingredients ---
        # Both computed per-residue, then averaged only over REAL
        # residues, ignoring padding -- same masking principle as
        # everywhere else in this pipeline.
        mask_flat = mask.reshape(-1).float()

        # Codebook loss: pushes each chosen codebook vector to move
        # closer to the encoder outputs that picked it -- literally
        # "adjust the reference chip's color to better match the custom
        # colors that keep getting matched to it."
        per_residue_codebook_loss = F.mse_loss(
            quantized.reshape(-1, D), x.detach().reshape(-1, D), reduction="none"
        ).mean(dim=-1)
        codebook_loss = (per_residue_codebook_loss * mask_flat).sum() / mask_flat.sum()

        # Commitment loss: the reverse pull -- pushes the encoder's
        # output to move closer to whichever codebook vector it already
        # picked, discouraging the encoder from producing wildly varying
        # outputs that hop between different codes for similar inputs.
        # Weighted down (0.25) because this term is a stabilizer, not the
        # main learning signal -- weighting it too heavily can make the
        # encoder collapse to outputting near-identical values everywhere.
        per_residue_commitment_loss = F.mse_loss(
            x.reshape(-1, D), quantized.detach().reshape(-1, D), reduction="none"
        ).mean(dim=-1)
        commitment_loss = (per_residue_commitment_loss * mask_flat).sum() / mask_flat.sum()

        loss = codebook_loss + self.commitment_weight * commitment_loss

        return {
            "tokens": tokens,
            "quantized": quantized_st,
            "loss": loss,
        }
