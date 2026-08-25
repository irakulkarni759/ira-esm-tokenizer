"""
EMA vector quantizer: same tokenizer step as quantizer.py, but the codebook
is updated by exponential moving average instead of by gradient descent.

This is the arm that was missing from the ablation. compare_esm3.py prints
"Dead codes and lopsided usage are the expected cost of gradient updates
over EMA. This is what we paid." -- but nothing in the experiment measured
that, because EMA was never implemented. This file is the counterfactual.

WHAT ACTUALLY CHANGES
---------------------
The gradient version has two loss terms. The codebook loss drags each chosen
code toward the encoder outputs that picked it, one small optimizer step at a
time, and the commitment loss pulls the encoder the other way. Only the second
survives here. EMA replaces the first with a direct, closed-form answer: a
code becomes the running mean of every latent assigned to it. No learning
rate, no optimizer state, no gradient.

Why that is expected to help. A gradient-updated code that stops being chosen
stops receiving gradient, so it stops moving, so it stays unchosen -- dead and
staying dead, which is exactly what revive_dead_codes in train.py exists to
undo. EMA has the same "only chosen codes get updated" property, so it is NOT
immune, but its updates are much larger: a code jumps most of the way to the
mean of its assignments in a few steps rather than crawling there. Codes on
the edge of dying tend to get rescued by their own update rather than needing
an external reset.

WHY THE 'vq' COLUMN IS NOT COMPARABLE ACROSS THE TWO
----------------------------------------------------
train.py logs quantized["loss"]. For the gradient quantizer that is
codebook_loss + 0.25 * commitment_loss. Here there is no codebook loss, so
the same column is 0.25 * commitment_loss alone, and will read lower for
reasons that have nothing to do with quality. Compare 'commitment' between
runs instead -- it is returned separately below and measures the same thing
in both. Reconstruction, perplexity and codes-used are directly comparable.

CHECKPOINT COMPATIBILITY
------------------------
The EMA statistics are registered as NON-PERSISTENT buffers, so state_dict()
contains exactly one entry, codebook.weight, identical to VectorQuantizer.
That means analyze_codebook.py, compare_esm3.py and compare_runs.py all load
an EMA checkpoint unchanged, with no strict=False and no edits. The cost is
that --resume restarts the EMA accumulators from the loaded codebook rather
than continuing them, which is a small momentum discontinuity and nothing
more. Worth it to keep the analysis scripts untouched.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EMAVectorQuantizer(nn.Module):
    """
    Drop-in replacement for VectorQuantizer. Same constructor signature,
    same forward(x, mask) -> {"tokens", "quantized", "loss"} contract, same
    .codebook attribute (a real nn.Embedding), so train.py's
    initialize_codebook and revive_dead_codes both keep working.

    decay: how much of the old codebook survives each update. 0.99 is the
    standard value from the original VQ-VAE paper. Higher is smoother and
    slower; lower tracks the encoder more closely but jitters.

    eps: Laplace smoothing on the cluster counts, so a code chosen zero
    times this step does not divide by zero.

    restart_threshold: optional built-in dead-code revival, off by default.
    Any code whose EMA cluster size falls below this gets reseeded from a
    real latent. Set it above 0 only if you want EMA + revival as its own
    arm; leave it at 0 for the clean EMA-alone measurement.
    """

    def __init__(self, num_codes: int = 4096, code_dim: int = 128,
                 commitment_weight: float = 0.25, decay: float = 0.99,
                 eps: float = 1e-5, restart_threshold: float = 0.0):
        super().__init__()
        self.commitment_weight = commitment_weight
        self.decay = decay
        self.eps = eps
        self.restart_threshold = restart_threshold

        self.codebook = nn.Embedding(num_codes, code_dim)
        nn.init.normal_(self.codebook.weight, mean=0.0, std=0.02)
        # The whole point: this codebook is not trained by the optimizer.
        # AdamW still receives it via quantizer.parameters(), but its .grad
        # stays None so the step is skipped, and clip_grad_norm_ ignores it.
        self.codebook.weight.requires_grad_(False)

        # Running totals the EMA is built from. cluster_size is how many
        # latents each code has been claiming lately; weight_sum is the
        # running sum of those latents. Their ratio is the code's mean,
        # which is what the codebook gets set to.
        self.register_buffer("ema_cluster_size", torch.zeros(num_codes), persistent=False)
        self.register_buffer("ema_weight_sum", torch.zeros(num_codes, code_dim), persistent=False)
        self.register_buffer("initted", torch.zeros((), dtype=torch.bool), persistent=False)

    # -- internals -----------------------------------------------------

    def _implied_codebook(self) -> torch.Tensor:
        """
        The codebook the current EMA statistics imply. Kept in one place so
        the update and the external-write detector below agree exactly.

        The Laplace term spreads a tiny bit of mass onto every code before
        dividing, then rescales so the total count is unchanged. Without it
        a code with zero recent assignments divides by zero.
        """
        counts = self.ema_cluster_size
        n = counts.sum()
        k = counts.numel()
        smoothed = (counts + self.eps) / (n + k * self.eps) * n
        return self.ema_weight_sum / smoothed.unsqueeze(1)

    def _seed_mass(self) -> float:
        """
        How much EMA weight to give a freshly seeded code.

        Setting it to 1.0 always would make revived codes heavier than the
        average live code (with 4096 codes and ~1000 residues a batch, a
        typical steady-state cluster size is well under 1), so they would
        respond more slowly than their neighbours for a long while. Using
        the current live mean instead starts them at a normal update rate.
        """
        live = self.ema_cluster_size[self.ema_cluster_size > 0]
        return float(live.mean()) if live.numel() else 1.0

    @torch.no_grad()
    def _sync_external_writes(self) -> None:
        """
        Pick up any codebook entry that was written from outside this module.

        Two things in train.py do exactly that: initialize_codebook seeds the
        whole codebook before step 0, and revive_dead_codes overwrites dead
        entries mid-run. Neither knows about the EMA accumulators, so without
        this the very next EMA update would compute e = m / N from stale
        statistics and immediately undo their work.

        Detection is a comparison against _implied_codebook(), which is what
        this module last wrote. Rows that no longer match were changed by
        someone else, and get their accumulators reseeded to match.
        """
        if not bool(self.initted):
            self.ema_weight_sum.copy_(self.codebook.weight)
            self.ema_cluster_size.fill_(1.0)
            self.initted.fill_(True)
            return

        implied = self._implied_codebook()
        drift = (self.codebook.weight - implied).abs().amax(dim=1)
        # Relative tolerance, since the encoder's output scale is arbitrary
        # and an absolute threshold would mean different things per run.
        scale = implied.abs().amax(dim=1).clamp(min=1.0)
        changed = drift > 1e-4 * scale
        if bool(changed.any()):
            mass = self._seed_mass()
            self.ema_weight_sum[changed] = self.codebook.weight[changed] * mass
            self.ema_cluster_size[changed] = mass

    @torch.no_grad()
    def _ema_update(self, x_live: torch.Tensor, tokens_live: torch.Tensor) -> None:
        """
        One EMA step. x_live and tokens_live are already stripped of padding.

        counts and sums are gathered with bincount and index_add_ rather than
        a one-hot matmul: one-hot would allocate (residues x num_codes),
        which at 4096 codes is tens of megabytes per step for no reason.
        """
        k = self.ema_cluster_size.numel()

        counts = torch.bincount(tokens_live, minlength=k).to(self.ema_cluster_size.dtype)
        sums = torch.zeros_like(self.ema_weight_sum).index_add_(0, tokens_live, x_live)

        self.ema_cluster_size.mul_(self.decay).add_(counts, alpha=1.0 - self.decay)
        self.ema_weight_sum.mul_(self.decay).add_(sums, alpha=1.0 - self.decay)

        self.codebook.weight.copy_(self._implied_codebook())

        # Optional built-in revival, off unless restart_threshold > 0. Kept
        # here rather than in train.py so an EMA + revival arm needs no edit
        # anywhere else. Note it reseeds the accumulators too, which the
        # train.py version cannot do -- that is what _sync_external_writes
        # is for.
        if self.restart_threshold > 0 and x_live.shape[0] > 0:
            dead = self.ema_cluster_size < self.restart_threshold
            n_dead = int(dead.sum())
            if n_dead:
                picks = torch.randint(0, x_live.shape[0], (n_dead,), device=x_live.device)
                replacement = x_live[picks] + torch.randn_like(x_live[picks]) * 0.01
                mass = self._seed_mass()
                self.codebook.weight[dead] = replacement
                self.ema_weight_sum[dead] = replacement * mass
                self.ema_cluster_size[dead] = mass

    # -- forward -------------------------------------------------------

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> dict:
        """
        x:    (B, L, code_dim) continuous encoder output
        mask: (B, L) True for real residues

        Returns tokens, quantized (straight-through), loss, and two extra
        diagnostics that train.py ignores but a comparison script can use.
        """
        B, L, D = x.shape
        x_flat = x.reshape(B * L, D)
        mask_flat = mask.reshape(-1)

        # Assignment is identical to the gradient version: nearest code by
        # Euclidean distance. Only how the codebook MOVES differs.
        distances = torch.cdist(x_flat, self.codebook.weight)
        tokens_flat = distances.argmin(dim=1)
        tokens = tokens_flat.view(B, L)
        quantized = self.codebook(tokens)

        # Straight-through estimator, same as quantizer.py: numerically a
        # no-op, but it routes the decoder's gradient past the argmin and
        # back into the encoder.
        quantized_st = x + (quantized - x).detach()

        # Commitment loss only. The codebook loss has no job here -- the
        # codebook is not moved by gradients at all -- so this is the single
        # term keeping the encoder from drifting away from its own codes.
        mask_float = mask_flat.float()
        per_residue = F.mse_loss(
            x_flat, quantized.reshape(-1, D).detach(), reduction="none"
        ).mean(dim=-1)
        denom = mask_float.sum().clamp(min=1.0)
        commitment = (per_residue * mask_float).sum() / denom
        loss = self.commitment_weight * commitment

        if self.training:
            self._sync_external_writes()
            live = mask_flat.nonzero(as_tuple=True)[0]
            if live.numel():
                self._ema_update(x_flat.detach()[live], tokens_flat[live])

        # How far the codebook sits from the latents, reported for symmetry
        # with the gradient run's codebook_loss so the two are comparable.
        with torch.no_grad():
            codebook_mse = (per_residue.detach() * mask_float).sum() / denom

        return {
            "tokens": tokens,
            "quantized": quantized_st,
            "loss": loss,
            "commitment": commitment.detach(),
            "codebook_mse": codebook_mse,
        }
