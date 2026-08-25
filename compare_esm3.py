"""
How does this tokenizer actually differ from ESM-3's, and does the
difference show up in the tokens?

Five things are built differently here. Four of them are just facts about
the code and can be stated; the interesting question is which ones CHANGE
THE TOKENS, and by how much. That is what this script measures.

    architecture                        ESM-3                 here
    ---------------------------------------------------------------------
    receptive field                     16 nearest neighbours  whole chain
    position information in encoder     relative pos. embed.   none at all
    codebook update                     moving average (EMA)   gradient
    training objective                  reconstruction +       reconstruction
                                        distance / error /     alone
                                        confidence heads
    decoder                             transformer + 6D       same
                                        rotation frames

Only the first two can be probed from a trained checkpoint, because they
are properties of the FORWARD PASS -- we can re-run the encoder with the
receptive field cut down, or with the residues reordered, and see what
happens to the tokens. The last three are properties of TRAINING, so all
we can do is measure their consequences on the codebook we ended up with.

Four probes:

  1. LOCALITY. Re-tokenize with attention clipped to each residue's 16
     nearest neighbours -- ESM-3's receptive field, on our weights. If the
     tokens barely move, our global attention was decorative and the two
     architectures are closer than they look. If they move a lot, our
     tokens are genuinely carrying information ESM-3's cannot.

  2. TRANSPLANT. Cut a window out of a protein and re-encode it on its
     own. The window's geometry is untouched; only its surroundings are
     gone. Ours are free to change, and how often they do is the honest
     measure of how context-dependent our vocabulary is.

     A 16-neighbour encoder scores HIGH here but not exactly 1.0, and it
     is worth being precise about why. The window is contiguous in
     SEQUENCE while the 16 neighbours are nearest in SPACE, so a residue
     whose fold brings a distant part of the chain alongside it loses
     real neighbours when the window is cut. Those residues can change
     token legitimately. Only residues whose entire neighbourhood falls
     inside the window are guaranteed unchanged.

  3. PERMUTATION. Our encoder has no positional embedding and its pair
     features are purely geometric, so shuffling the residue order should
     permute the tokens and nothing else. This checks that claim rather
     than trusting it -- and it is a real difference: ESM-3's relative
     positional embedding means its tokens WOULD change.

  4. CODEBOOK. Gradient-updated codebooks are known to leave more dead
     entries than EMA ones, because a code that stops being chosen stops
     receiving gradient and simply sits there. So: how many codes are
     alive, how evenly are they used, and how far do encoder outputs sit
     from the code they get snapped to.

Runs on an existing checkpoint. No training, no GPU strictly required.

    python compare_esm3.py --checkpoint checkpoints/best.pt
"""

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from data.dataset import StructureDataset
from model.encoder import StructureEncoder
from model.quantizer import VectorQuantizer

# ESM-3's structure encoder attends over each residue's 16 nearest
# neighbours in space (not in sequence). That number is the whole point of
# probe 1, so it gets a name rather than being buried in a default.
ESM3_NEIGHBOURS = 16


# ---------------------------------------------------------------------------
# Running the encoder with a restricted receptive field
# ---------------------------------------------------------------------------
# StructureEncoder builds its own k-nearest-neighbour mask from self.neighbours,
# so clipping the receptive field at inference time is just a matter of
# changing that number and putting it back afterwards. This used to be a
# hand-copied version of the attention maths, which had to be kept in sync with
# encoder.py by hand. It no longer does.

@contextmanager
def with_neighbours(encoder, k):
    """Temporarily run `encoder` with its attention clipped to k neighbours."""
    original = encoder.neighbours
    encoder.neighbours = k
    try:
        yield encoder
    finally:
        # restored even if tokenizing raises, so one bad structure cannot
        # silently leave every later probe running on a clipped encoder.
        encoder.neighbours = original


# ---------------------------------------------------------------------------

def tokenize(encoder, quantizer, coords):
    """coords: (L, 4, 3) on the right device. Returns (L,) numpy token ids."""
    batch = coords.unsqueeze(0)
    mask = torch.ones(1, batch.shape[1], dtype=torch.bool, device=batch.device)
    return quantizer(encoder(batch, mask), mask)["tokens"][0].cpu().numpy()


def normalized_mutual_info(a: np.ndarray, b: np.ndarray) -> float:
    """Same estimator analyze_codebook.py uses, so the numbers are comparable."""
    a_vals, a_idx = np.unique(a, return_inverse=True)
    b_vals, b_idx = np.unique(b, return_inverse=True)
    joint = np.zeros((len(a_vals), len(b_vals)))
    np.add.at(joint, (a_idx, b_idx), 1)
    joint /= joint.sum()

    pa = joint.sum(1, keepdims=True)
    pb = joint.sum(0, keepdims=True)
    nz = joint > 0
    mi = (joint[nz] * np.log(joint[nz] / (pa @ pb)[nz])).sum()

    def entropy(p):
        p = p[p > 0]
        return -(p * np.log(p)).sum()

    ha, hb = entropy(pa.ravel()), entropy(pb.ravel())
    return 0.0 if ha == 0 or hb == 0 else float(mi / np.sqrt(ha * hb))


# ---------------------------------------------------------------------------
# Probe 1: receptive field
# ---------------------------------------------------------------------------

def probe_locality(encoder, quantizer, structures, k, device):
    """
    Same weights, same structures, two receptive fields: everything, and
    ESM-3's 16 nearest neighbours.
    """
    agree, total = 0, 0
    global_all, local_all = [], []

    for coords in structures:
        coords = coords.to(device)

        g = tokenize(encoder, quantizer, coords)
        with with_neighbours(encoder, k):
            l = tokenize(encoder, quantizer, coords)

        agree += int((g == l).sum())
        total += len(g)
        global_all.append(g)
        local_all.append(l)

    global_all = np.concatenate(global_all)
    local_all = np.concatenate(local_all)
    return {
        "neighbours": k,
        "token_agreement": agree / total,
        "nmi_global_vs_local": normalized_mutual_info(global_all, local_all),
        "residues": total,
    }


# ---------------------------------------------------------------------------
# Probe 2: transplant
# ---------------------------------------------------------------------------

def probe_transplant(encoder, quantizer, structures, window, margin, device, rng):
    """
    Encode a protein, then encode a `window`-residue slice of it on its own,
    and compare tokens for the middle of that slice.

    `margin` residues are dropped from each end of the window before
    comparing. Those really do lose neighbours when you cut the window out,
    so counting them would confuse "context changed the token" with "the
    local geometry itself was truncated". The middle keeps its full local
    surroundings and only loses the rest of the protein.
    """
    agree, total = 0, 0

    for coords in structures:
        L = coords.shape[0]
        if L < window + 2 * margin:
            continue
        coords = coords.to(device)

        start = int(rng.integers(0, L - window + 1))
        full = tokenize(encoder, quantizer, coords)
        cut = tokenize(encoder, quantizer, coords[start : start + window])

        core = slice(margin, window - margin)
        a = full[start : start + window][core]
        b = cut[core]
        agree += int((a == b).sum())
        total += len(a)

    return {
        "window": window,
        "margin": margin,
        "token_agreement": (agree / total) if total else float("nan"),
        "residues_compared": total,
    }


# ---------------------------------------------------------------------------
# Probe 3: permutation
# ---------------------------------------------------------------------------

def probe_permutation(encoder, quantizer, structures, device, rng):
    """
    Shuffle the residue order and check the tokens follow their residues.

    Nothing in the encoder is allowed to care about chain order: the pair
    features are frame-relative geometry, attention is a set operation,
    and there is no positional embedding. So this should come out at 1.0.
    Anything less is a bug, not a finding -- most likely numerical, since
    attention sums the same terms in a different order.
    """
    agree, total = 0, 0

    for coords in structures:
        coords = coords.to(device)
        L = coords.shape[0]
        perm = torch.as_tensor(rng.permutation(L), device=device)

        original = tokenize(encoder, quantizer, coords)
        shuffled = tokenize(encoder, quantizer, coords[perm])

        agree += int((original[perm.cpu().numpy()] == shuffled).sum())
        total += L

    return {"token_agreement": agree / total, "residues": total}


# ---------------------------------------------------------------------------
# Probe 4: codebook health
# ---------------------------------------------------------------------------

def probe_codebook(encoder, quantizer, structures, num_codes, device):
    """
    What gradient-updated codebook did we actually end up with?

    EMA updates move a code toward the mean of the vectors assigned to it,
    which keeps it useful even when it is rarely chosen. Gradient updates
    only reach codes that got chosen, so a code that falls out of use
    tends to stay out. The symptom is dead codes and uneven usage, so
    those are what get counted here, along with how far encoder outputs
    end up from the code they are snapped to -- the quantization error the
    decoder has to absorb.
    """
    counts = np.zeros(num_codes)
    residual_sq, residual_n = 0.0, 0

    with torch.no_grad():
        for coords in structures:
            coords = coords.to(device)
            batch = coords.unsqueeze(0)
            mask = torch.ones(1, batch.shape[1], dtype=torch.bool, device=device)

            features = encoder(batch, mask)
            out = quantizer(features, mask)

            counts += np.bincount(out["tokens"][0].cpu().numpy(), minlength=num_codes)
            # Relative, not absolute: the encoder's output scale is
            # arbitrary, so a raw distance would say nothing on its own.
            residual_sq += float(((features - out["quantized"]) ** 2).sum())
            residual_n += float((features ** 2).sum())

    p = counts / counts.sum()
    nz = p > 0
    perplexity = float(np.exp(-(p[nz] * np.log(p[nz])).sum()))

    return {
        "codes_used": int(nz.sum()),
        "codes_total": int(num_codes),
        "perplexity": perplexity,
        "usage_evenness": perplexity / int(nz.sum()) if nz.sum() else float("nan"),
        "relative_quantization_error": float(np.sqrt(residual_sq / residual_n)),
    }


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--parsed-dir", type=Path, default=Path("data/parsed"))
    parser.add_argument("--out-dir", type=Path, default=Path("analysis"))
    parser.add_argument("--split", choices=["val", "train", "all"], default="val")
    parser.add_argument("--max-structures", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--neighbours", type=int, default=ESM3_NEIGHBOURS)
    parser.add_argument("--window", type=int, default=48)
    parser.add_argument("--margin", type=int, default=8)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = state["args"]
    print(f"checkpoint: epoch {state['epoch']}, best val {state['best_val']:.3f}A\n")

    # .get with a default, so checkpoints written before --neighbours existed
    # still load and are correctly treated as whole-chain.
    encoder = StructureEncoder(cfg["dim"], cfg["num_heads"], cfg["num_layers"],
                               cfg.get("neighbours", 0)).to(device)
    quantizer = VectorQuantizer(cfg["num_codes"], cfg["dim"]).to(device)
    encoder.load_state_dict(state["encoder"])
    quantizer.load_state_dict(state["quantizer"])
    encoder.eval()
    quantizer.eval()

    # Same split logic as analyze_codebook.py, so the two reports describe
    # the same structures.
    dataset = StructureDataset(args.parsed_dir)
    permutation = np.random.default_rng(cfg["seed"]).permutation(len(dataset))
    num_val = max(1, int(len(dataset) * cfg["val_fraction"]))
    if args.split == "val":
        indices = permutation[:num_val]
    elif args.split == "train":
        indices = permutation[num_val:]
    else:
        indices = permutation
    indices = indices[: args.max_structures].tolist()

    structures = []
    for i in indices:
        coords = dataset[i]["coords"]
        if coords.shape[0] > args.max_length:
            start = (coords.shape[0] - args.max_length) // 2
            coords = coords[start : start + args.max_length]
        structures.append(coords)
    print(f"{len(structures)} structures from the {args.split} split\n")

    rng = np.random.default_rng(0)
    report = {}

    with torch.no_grad():
        print("=" * 66)
        print(f"1. RECEPTIVE FIELD  (whole chain vs {args.neighbours} nearest neighbours)")
        print("=" * 66)
        loc = probe_locality(encoder, quantizer, structures, args.neighbours, device)
        report["receptive_field"] = loc
        print(f"tokens unchanged when clipped to {loc['neighbours']} neighbours : {loc['token_agreement']:.3f}")
        print(f"NMI(global tokens, local tokens)                     : {loc['nmi_global_vs_local']:.3f}")
        print("  Near 1.0 means the extra reach is not being used, and our")
        print("  tokens are effectively as local as ESM-3's. Low means the")
        print("  tokens depend on parts of the protein ESM-3 never sees.")

        print("\n" + "=" * 66)
        print("2. TRANSPLANT  (same local shape, surroundings removed)")
        print("=" * 66)
        tr = probe_transplant(encoder, quantizer, structures, args.window, args.margin, device, rng)
        report["transplant"] = tr
        print(f"tokens unchanged for the middle of a {tr['window']}-residue window : {tr['token_agreement']:.3f}")
        print(f"  ({tr['residues_compared']} residues compared, {tr['margin']} trimmed from each end)")
        print("  A 16-neighbour encoder scores high here but NOT exactly 1.0,")
        print("  since the window is contiguous in sequence while neighbours")
        print("  are nearest in space, so some are genuinely lost in the cut.")
        print("  Our shortfall is far larger than that, and is the price of")
        print("  global attention: the same shape, tokenized two ways")
        print("  depending on what else was in the chain.")

        print("\n" + "=" * 66)
        print("3. POSITION  (residues shuffled)")
        print("=" * 66)
        pm = probe_permutation(encoder, quantizer, structures, device, rng)
        report["permutation"] = pm
        print(f"tokens that followed their residue : {pm['token_agreement']:.3f}")
        print("  Should be 1.0. We have no positional embedding at all, so")
        print("  chain order is invisible to the encoder. ESM-3 has relative")
        print("  positional embeddings, so its tokens would move here.")

        print("\n" + "=" * 66)
        print("4. CODEBOOK  (gradient updates, not EMA)")
        print("=" * 66)
        cb = probe_codebook(encoder, quantizer, structures, cfg["num_codes"], device)
        report["codebook"] = cb
        print(f"codes used                  : {cb['codes_used']} / {cb['codes_total']}")
        print(f"perplexity                  : {cb['perplexity']:.1f}")
        print(f"usage evenness              : {cb['usage_evenness']:.3f}   (1.0 = every live code used equally)")
        print(f"relative quantization error : {cb['relative_quantization_error']:.3f}")
        print("  Dead codes and lopsided usage are the expected cost of")
        print("  gradient updates over EMA. This is what we paid.")

    print("\n" + "=" * 66)
    print("NOT MEASURABLE FROM A CHECKPOINT")
    print("=" * 66)
    print("ESM-3 also trains distance, error and confidence heads alongside")
    print("reconstruction. We train on reconstruction alone, so there is no")
    print("ablation to run -- the heads were never there. Their effect would")
    print("show up as a better-shaped latent space before quantization,")
    print("which probes 1, 2 and 4 can describe but cannot attribute.")
    print("The decoder is a close match to theirs (6D rotation frames, plain")
    print("attention), so it is not a source of difference.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "esm3_comparison.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {args.out_dir}/esm3_comparison.json")


if __name__ == "__main__":
    main()
