"""
Train with the EMA codebook instead of the gradient codebook.

This is a thin wrapper around train.py, not a copy of it. It swaps the
quantizer class and hands everything else straight through, so the data
loading, batching, schedule, seeding, logging and checkpoint format are
byte-for-byte the same code paths the gradient runs used. That matters:
an ablation is only worth anything if the arm you are comparing against
is genuinely the same experiment with one thing changed.

    python train_ema.py --parsed-dir data/parsed --epochs 200 \
        --num-codes 4096 --revive-every 0

Every train.py flag works. Three are added:

    --ema-decay               how much of the old codebook survives each
                              update. 0.99 is the VQ-VAE paper's value.
    --ema-eps                 Laplace smoothing on the cluster counts.
    --ema-restart-threshold   built-in dead-code revival, off at 0. Use
                              train.py's --revive-every for the external
                              one instead; this exists so an EMA+revival
                              arm needs no edits elsewhere.

THE FOUR ARMS WORTH RUNNING
---------------------------
The question "does EMA beat gradients" and the question "did our revival
mechanism earn its keep" are different, and one run cannot answer both.

    gradient, revival on    python train.py --revive-every 500      (done)
    gradient, revival off   python train.py --revive-every 0
    EMA, revival off        python train_ema.py --revive-every 0
    EMA, revival on         python train_ema.py --revive-every 500

Row 2 is the cheapest and the most informative, because it is the only
one that measures what revive_dead_codes actually bought. Run it first.

READING THE OUTPUT
------------------
The 'vq' column is NOT comparable across the two quantizers. The gradient
version logs codebook_loss + 0.25*commitment; the EMA version has no
codebook loss so it logs 0.25*commitment alone, and reads lower for
reasons unrelated to quality. Compare val reconstruction, codes-used and
perplexity, which mean the same thing in both.

The 'revived' column stays 0 under --ema-restart-threshold, since built-in
restarts happen inside the quantizer and train.py never sees them.
"""

import argparse
import sys

import train
from model.quantizer_ema import EMAVectorQuantizer


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ema-decay", type=float, default=0.99)
    parser.add_argument("--ema-eps", type=float, default=1e-5)
    parser.add_argument("--ema-restart-threshold", type=float, default=0.0)
    # Everything we do not recognise belongs to train.py, so hand it back.
    ema_args, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + remaining

    # train.py builds its quantizer as `VectorQuantizer(args.num_codes, args.dim)`
    # off the module global, so rebinding that name is all the swap takes.
    def build(num_codes, code_dim, *args, **kwargs):
        return EMAVectorQuantizer(
            num_codes, code_dim,
            decay=ema_args.ema_decay,
            eps=ema_args.ema_eps,
            restart_threshold=ema_args.ema_restart_threshold,
        )

    train.VectorQuantizer = build

    # Record the EMA settings on the namespace train.py parses, so they end
    # up in the checkpoint's "args" dict. Without this an EMA checkpoint is
    # indistinguishable from a gradient one on disk, which is a bad thing to
    # discover four runs into an ablation.
    original_parse = argparse.ArgumentParser.parse_args

    def parse_and_tag(self, *a, **kw):
        namespace = original_parse(self, *a, **kw)
        namespace.quantizer = "ema"
        namespace.ema_decay = ema_args.ema_decay
        namespace.ema_eps = ema_args.ema_eps
        namespace.ema_restart_threshold = ema_args.ema_restart_threshold
        return namespace

    argparse.ArgumentParser.parse_args = parse_and_tag

    print(f"EMA codebook: decay={ema_args.ema_decay} eps={ema_args.ema_eps} "
          f"restart_threshold={ema_args.ema_restart_threshold}")
    if "--revive-every" not in remaining:
        print("note: train.py's --revive-every defaults to 500, so external "
              "revival is ON. Pass --revive-every 0 for EMA alone.")

    train.main()


if __name__ == "__main__":
    main()
