"""
Any number of checkpoints, same structures, one ablation table.

Written for one question. Of the four ways this tokenizer differs from ESM-3,
which ones actually matter, and what does each one cost?

You cannot answer that by reading. Each difference is a training choice, so
the only way to score it is to train a run that changes exactly that one thing
and compare. This prints the table.

    receptive field    --neighbours 16     (vs whole chain)
    codebook size      --num-codes 512     (vs 4096)
    positional info    not yet implemented
    EMA codebook       not yet implemented
    auxiliary heads    not yet implemented

TWO AXES, NOT ONE. Reconstruction error on its own will pick the wrong run
every time, because giving each residue a near-unique code makes the decoder's
job EASIER. That is exactly how the 4096-code run reached 4.1 residues per
code. A tokenizer can score well on reconstruction precisely because its
tokens are useless as a vocabulary.

So every run gets scored on both:

  val reconstruction (A)   can the decoder rebuild the backbone
  transplant agreement     does a shape keep its token when you cut the
                           surrounding protein away
  codes per local shape    how many codes one shape gets. Ideal 1.
  NMI(code, protein)       how much a token leaks about which protein it is
                           in. Ideal 0.

A change that improves the bottom three while barely moving the top one is a
real gain. One that improves them because reconstruction collapsed is not.

Every run must share --seed and --val-fraction, or they were not scored on the
same structures. That is checked, not assumed.

    python compare_runs.py \\
        --run 4096=checkpoints/best.pt \\
        --run 512=checkpoints-512/best.pt \\
        --run knn16=checkpoints-knn16/best.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data.dataset import StructureDataset
from model.encoder import StructureEncoder
from model.quantizer import VectorQuantizer

# Reuse the metric code rather than reimplementing it. If the definition of
# "codes per local shape" ever changes, it must change in one place or these
# numbers stop being comparable to the single-run reports.
from analyze_codebook import annotate_sse, local_geometry, normalized_mutual_info
from compare_esm3 import probe_locality, probe_transplant, tokenize


def load_run(checkpoint: Path, device):
    """Rebuild encoder and quantizer from a checkpoint, in eval mode."""
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    cfg = state["args"]

    # .get with a default, so checkpoints written before --neighbours
    # existed still load and are correctly treated as whole-chain.
    encoder = StructureEncoder(cfg["dim"], cfg["num_heads"], cfg["num_layers"],
                               cfg.get("neighbours", 0)).to(device)
    quantizer = VectorQuantizer(cfg["num_codes"], cfg["dim"]).to(device)
    encoder.load_state_dict(state["encoder"])
    quantizer.load_state_dict(state["quantizer"])
    encoder.eval()
    quantizer.eval()

    return encoder, quantizer, cfg, state


def evaluate(encoder, quantizer, cfg, state, structures, names, args, device):
    """Every number we compare, for one checkpoint."""
    codes, sse, angle, torsion, protein = [], [], [], [], []

    with torch.no_grad():
        for coords, name in zip(structures, names):
            tokens = tokenize(encoder, quantizer, coords.to(device))
            ca = coords[:, 1].numpy()
            a, t = local_geometry(ca)

            codes.append(tokens)
            sse.append(annotate_sse(ca))
            angle.append(a)
            torsion.append(t)
            protein.append(np.full(len(tokens), name))

    code    = np.concatenate(codes)
    sse     = np.concatenate(sse)
    angle   = np.concatenate(angle)
    torsion = np.concatenate(torsion)
    protein = np.concatenate(protein)

    # --- usage ---
    counts = np.bincount(code, minlength=cfg["num_codes"]).astype(float)
    p = counts / counts.sum()
    nz = p > 0
    perplexity = float(np.exp(-(p[nz] * np.log(p[nz])).sum()))
    used = int(nz.sum())

    # --- geometry buckets, identical binning to analyze_codebook.py ---
    valid = ~(np.isnan(angle) | np.isnan(torsion))
    angle_bin = np.digitize(angle[valid], np.linspace(60, 150, 13))
    torsion_bin = np.digitize(torsion[valid], np.linspace(-180, 180, 13))
    geom_bucket = angle_bin * 100 + torsion_bin

    # --- how many codes does one local shape get? ---
    codes_v = code[valid]
    spreads, sizes = [], []
    for bucket in np.unique(geom_bucket):
        sel = geom_bucket == bucket
        if sel.sum() < 50:
            continue
        _, cnts = np.unique(codes_v[sel], return_counts=True)
        q = cnts / cnts.sum()
        spreads.append(float(np.exp(-(q * np.log(q)).sum())))
        sizes.append(int(sel.sum()))
    codes_per_shape = float(np.average(spreads, weights=sizes)) if spreads else float("nan")

    with torch.no_grad():
        loc = probe_locality(encoder, quantizer, structures, args.neighbours, device)
        # Fresh rng per checkpoint, so both see the SAME random windows.
        tr = probe_transplant(encoder, quantizer, structures, args.window, args.margin,
                              device, np.random.default_rng(args.window_seed))

    return {
        "num_codes": int(cfg["num_codes"]),
        # 0 means whole chain. Recorded per run so the table shows
        # which receptive field each one was actually TRAINED with.
        "trained_neighbours": int(cfg.get("neighbours", 0)),
        "epoch": int(state["epoch"]),
        "val_angstrom": float(state["best_val"]),
        "residues": int(counts.sum()),
        "codes_used": used,
        "perplexity": perplexity,
        "residues_per_code": float(counts.sum() / used) if used else float("nan"),
        "codes_per_local_shape": codes_per_shape,
        "nmi_sse": normalized_mutual_info(code, sse),
        "nmi_geometry": normalized_mutual_info(codes_v, geom_bucket),
        "nmi_protein": normalized_mutual_info(code, protein),
        "locality_agreement": loc["token_agreement"],
        "transplant_agreement": tr["token_agreement"],
    }


# Each row: printed label, key, how many decimals, and which direction is
# better. "up" means higher is better, "down" lower, None means it is context
# rather than a score.
ROWS = [
    ("codebook size",           "num_codes",            0, None),
    ("receptive field",         "trained_neighbours",   0, None),
    ("epoch",                   "epoch",                0, None),
    ("val reconstruction (A)",  "val_angstrom",         3, "down"),
    ("codes used",              "codes_used",           0, None),
    ("perplexity",              "perplexity",           1, None),
    ("residues per code",       "residues_per_code",    1, "up"),
    ("codes per local shape",   "codes_per_local_shape",1, "down"),
    ("NMI(code, SSE)",          "nmi_sse",              3, "up"),
    ("NMI(code, geometry)",     "nmi_geometry",         3, "up"),
    ("NMI(code, protein)",      "nmi_protein",          3, "down"),
    ("locality agreement",      "locality_agreement",   3, "up"),
    ("transplant agreement",    "transplant_agreement", 3, "up"),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", action="append", required=True, metavar="LABEL=PATH",
        help="A checkpoint to include, e.g. --run 512=checkpoints-512/best.pt. "
             "Repeat for each run. The first one is treated as the baseline.",
    )
    parser.add_argument("--parsed-dir", type=Path, default=Path("data/parsed"))
    parser.add_argument("--out", type=Path, default=Path("analysis/ablation.json"))
    parser.add_argument("--split", choices=["val", "train", "all"], default="val")
    parser.add_argument("--max-structures", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--neighbours", type=int, default=16)
    parser.add_argument("--window", type=int, default=48)
    parser.add_argument("--margin", type=int, default=8)
    parser.add_argument("--window-seed", type=int, default=0,
                        help="Fixed so every run gets the same transplant windows.")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    runs = []
    for spec in args.run:
        if "=" not in spec:
            raise SystemExit(f"--run needs LABEL=PATH, got {spec!r}")
        label, path = spec.split("=", 1)
        runs.append((label, Path(path)))

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    loaded = [(label,) + load_run(path, device) for label, path in runs]

    # A comparison across different splits is not a comparison. Fail here
    # rather than printing a table that looks fine and means nothing.
    base_label, _, _, base_cfg, _ = loaded[0]
    for label, _, _, cfg, _ in loaded[1:]:
        for key in ["seed", "val_fraction"]:
            if cfg[key] != base_cfg[key]:
                raise SystemExit(
                    f"{key} differs between {base_label} ({base_cfg[key]}) and "
                    f"{label} ({cfg[key]}), so they were validated on different "
                    "structures and cannot be compared"
                )

    dataset = StructureDataset(args.parsed_dir)
    permutation = np.random.default_rng(base_cfg["seed"]).permutation(len(dataset))
    num_val = max(1, int(len(dataset) * base_cfg["val_fraction"]))
    if args.split == "val":
        indices = permutation[:num_val]
    elif args.split == "train":
        indices = permutation[num_val:]
    else:
        indices = permutation
    indices = indices[: args.max_structures].tolist()

    structures, names = [], []
    for i in indices:
        coords = dataset[i]["coords"]
        if coords.shape[0] > args.max_length:
            start = (coords.shape[0] - args.max_length) // 2
            coords = coords[start : start + args.max_length]
        structures.append(coords)
        names.append(dataset.files[i].stem)

    print(f"{len(structures)} structures from the {args.split} split, "
          f"cropped at {args.max_length}\n")

    results = {}
    for label, encoder, quantizer, cfg, state in loaded:
        print(f"  scoring {label} ...", flush=True)
        results[label] = evaluate(encoder, quantizer, cfg, state,
                                  structures, names, args, device)
    print()

    labels = [label for label, *_ in loaded]
    print_table(results, labels, base_label)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


def print_table(results, labels, base_label):
    """One column per run, with the best value in each scored row marked."""
    label_w = max(len(r[0]) for r in ROWS) + 2
    col_w = max(max(len(x) for x in labels) + 2, 11)

    header = f"{'':<{label_w}}" + "".join(f"{x:>{col_w}}" for x in labels)
    print("=" * len(header))
    print(header)
    print("=" * len(header))

    for label, key, dp, direction in ROWS:
        values = [results[x][key] for x in labels]
        shown = [f"{v:.{dp}f}" for v in values]

        # Mark on the PRINTED value, not the underlying float. Crowning a
        # winner on a difference too small to display reads as a real result
        # when it is rounding noise.
        if direction is not None:
            finite = [(s_, v) for s_, v in zip(shown, values) if np.isfinite(v)]
            if finite:
                best = (max if direction == "up" else min)(v for _, v in finite)
                best_shown = f"{best:.{dp}f}"
                if sum(1 for s_ in shown if s_ == best_shown) < len(shown):
                    shown = [s_ + "*" if s_ == best_shown else s_ + " " for s_ in shown]

        print(f"{label:<{label_w}}" + "".join(f"{s_:>{col_w}}" for s_ in shown))

    print("=" * len(header))
    print(f"\n* = best in that row. Baseline is {base_label}.")
    print("\nRead the bottom rows together with 'val reconstruction'. A run that")
    print("wins on consistency while barely moving reconstruction is a real gain.")
    print("One that wins because reconstruction collapsed is not.")


if __name__ == "__main__":
    main()
