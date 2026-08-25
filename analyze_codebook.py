"""
What does each of the 4096 structure codes actually MEAN?

Nothing in training ever told the model what a token should stand for. The
only pressure was "the decoder has to rebuild the backbone from these." So
whether the codes ended up carrying real geometry is a question we have to
go and check, not something we can assume.

This script checks it three ways:

  1. Do codes line up with secondary structure? If code 3012 is picked
     almost only on helices, it has clearly learned "helix-ish".
  2. Do codes line up with local backbone geometry? Secondary structure is
     a coarse 3-way label; local geometry is the continuous thing the
     encoder actually sees, so this is the finer-grained version.
  3. Is the mapping CONSISTENT? This is the important one. Our encoder sees
     the whole protein at once (unlike real ESM-3, which only lets each
     residue see its 16 nearest neighbours). So the same local shape in two
     different proteins is ALLOWED to get two different codes here. If that
     happens a lot, our tokens are partly describing "which protein am I
     in" rather than "what shape am I", which would make them much less
     useful as a vocabulary.

Nothing here needs training. It runs on an existing checkpoint.

    python analyze_codebook.py --checkpoint checkpoints/best.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from data.dataset import StructureDataset
from model.encoder import StructureEncoder
from model.quantizer import VectorQuantizer


# ---------------------------------------------------------------------------
# Local backbone geometry
# ---------------------------------------------------------------------------
# Two classic CA-only descriptors. Between them they separate helix from
# sheet cleanly, which is why they're worth computing:
#
#   pseudo-angle   : the bend at residue i, using CA(i-1), CA(i), CA(i+1).
#                    ~89 degrees in a helix, ~120 in a strand.
#   pseudo-torsion : the twist across CA(i-1..i+2). ~+50 degrees in a
#                    helix, ~-170 (nearly flat) in a strand.
#
# We use these rather than phi/psi because they need only CA positions and
# because they're exactly the kind of local shape the encoder can see.

def local_geometry(ca: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    ca: (L, 3) CA coordinates.
    Returns (angle_deg, torsion_deg), both (L,), NaN where undefined at
    the chain ends.
    """
    L = len(ca)
    angle = np.full(L, np.nan)
    torsion = np.full(L, np.nan)

    if L < 3:
        return angle, torsion

    # --- bend angle at each interior residue ---
    v1 = ca[:-2] - ca[1:-1]   # CA(i) -> CA(i-1)
    v2 = ca[2:] - ca[1:-1]    # CA(i) -> CA(i+1)
    cos = (v1 * v2).sum(-1) / (
        np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-8
    )
    angle[1:-1] = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))

    if L < 4:
        return angle, torsion

    # --- dihedral across four consecutive CAs, standard formula ---
    # b0 points BACKWARDS along the chain (p0 - p1), not forwards. Getting
    # this the wrong way round flips every torsion by exactly 180 degrees,
    # which silently turns helices into strands and vice versa.
    b0 = ca[:-3] - ca[1:-2]
    b1 = ca[2:-1] - ca[1:-2]
    b2 = ca[3:] - ca[2:-1]

    b1n = b1 / (np.linalg.norm(b1, axis=-1, keepdims=True) + 1e-8)
    # Project b0 and b2 onto the plane perpendicular to b1, then measure
    # the angle between those projections. That angle is the twist.
    v = b0 - (b0 * b1n).sum(-1, keepdims=True) * b1n
    w = b2 - (b2 * b1n).sum(-1, keepdims=True) * b1n
    x = (v * w).sum(-1)
    y = (np.cross(b1n, v) * w).sum(-1)
    # Assigned to residue i, the second of the four atoms.
    torsion[1:-2] = np.degrees(np.arctan2(y, x))

    return angle, torsion


def secondary_structure(ca: np.ndarray) -> np.ndarray:
    """
    Crude 3-state secondary structure from CA geometry alone: 'H' helix,
    'E' strand, 'C' coil.

    This is the P-SEA idea reduced to its essentials. It is NOT as good as
    DSSP, which uses backbone hydrogen bonds. It's here so the script runs
    with no external binary and no extra install. If biotite is available
    we use its proper implementation instead (see annotate_sse below).
    """
    angle, torsion = local_geometry(ca)
    sse = np.full(len(ca), "C", dtype="<U1")

    helix = (np.abs(angle - 89.0) < 12.0) & (np.abs(torsion - 50.0) < 40.0)
    strand = (np.abs(angle - 124.0) < 14.0) & (
        (np.abs(torsion) > 140.0) | (np.abs(torsion - 180.0) < 40.0)
    )
    sse[helix] = "H"
    sse[strand & ~helix] = "E"
    return sse


def annotate_sse(ca: np.ndarray) -> np.ndarray:
    """Prefer biotite's tested P-SEA if it's installed, else fall back."""
    try:
        import biotite.structure as struc

        array = struc.AtomArray(len(ca))
        array.coord = ca.astype(np.float32)
        array.atom_name = np.full(len(ca), "CA")
        array.res_name = np.full(len(ca), "GLY")
        array.res_id = np.arange(1, len(ca) + 1)
        array.chain_id = np.full(len(ca), "A")
        array.element = np.full(len(ca), "C")
        labels = struc.annotate_sse(array)
        # biotite returns lowercase a/b/c; map to our H/E/C.
        mapping = {"a": "H", "b": "E", "c": "C"}
        out = np.array([mapping.get(str(x), "C") for x in labels], dtype="<U1")
        # Older biotite versions return one label per residue, which for a
        # CA-only array is the same length. Guard anyway.
        if len(out) == len(ca):
            return out
    except Exception:
        pass
    return secondary_structure(ca)


# ---------------------------------------------------------------------------
# Information-theoretic agreement, written out rather than pulled from
# sklearn so the script has no extra dependencies.
# ---------------------------------------------------------------------------

def normalized_mutual_info(a: np.ndarray, b: np.ndarray) -> float:
    """
    How much does knowing a tell you about b, on a 0-1 scale?
    0 means the two labellings are unrelated. 1 means one determines the
    other. Symmetric.
    """
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
    if ha == 0 or hb == 0:
        return 0.0
    return float(mi / np.sqrt(ha * hb))


# ---------------------------------------------------------------------------
# Tokenizing
# ---------------------------------------------------------------------------

def tokenize_all(encoder, quantizer, dataset, indices, device, max_length):
    """
    Run every chosen structure through encoder + quantizer, and collect one
    row per residue: which code it got, its secondary structure, its local
    geometry, and which protein it came from.
    """
    rows = {"code": [], "sse": [], "angle": [], "torsion": [], "protein": []}

    encoder.eval()
    quantizer.eval()
    with torch.no_grad():
        for n, i in enumerate(indices):
            item = dataset[i]
            coords = item["coords"]

            # Crop long chains from the centre. Attention is quadratic in
            # length, and we're only characterising codes, not scoring the
            # model, so a representative window is enough.
            if coords.shape[0] > max_length:
                start = (coords.shape[0] - max_length) // 2
                coords = coords[start : start + max_length]

            batch = coords.unsqueeze(0).to(device)
            mask = torch.ones(1, batch.shape[1], dtype=torch.bool, device=device)

            tokens = quantizer(encoder(batch, mask), mask)["tokens"][0].cpu().numpy()

            ca = coords[:, 1].numpy()  # index 1 of (N, CA, C, O)
            angle, torsion = local_geometry(ca)
            sse = annotate_sse(ca)

            rows["code"].append(tokens)
            rows["sse"].append(sse)
            rows["angle"].append(angle)
            rows["torsion"].append(torsion)
            rows["protein"].append(np.full(len(tokens), dataset.files[i].stem))

            if (n + 1) % 25 == 0:
                print(f"  tokenized {n + 1}/{len(indices)}")

    return {k: np.concatenate(v) for k, v in rows.items()}


# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--parsed-dir", type=Path, default=Path("data/parsed"))
    parser.add_argument("--out-dir", type=Path, default=Path("analysis"))
    parser.add_argument(
        "--split",
        choices=["val", "train", "all"],
        default="val",
        help="Which structures to analyse. val = the ones the model never trained on.",
    )
    parser.add_argument("--max-structures", type=int, default=200)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--plots", action="store_true", help="Also save figures.")
    args = parser.parse_args()

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )

    # --- rebuild the model exactly as it was trained ---
    state = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = state["args"]
    print(f"checkpoint: epoch {state['epoch']}, best val {state['best_val']:.3f}A")

    # .get with a default, so checkpoints written before --neighbours
    # existed still load and are correctly treated as whole-chain.
    encoder = StructureEncoder(cfg["dim"], cfg["num_heads"], cfg["num_layers"],
                               cfg.get("neighbours", 0)).to(device)
    quantizer = VectorQuantizer(cfg["num_codes"], cfg["dim"]).to(device)
    encoder.load_state_dict(state["encoder"])
    quantizer.load_state_dict(state["quantizer"])

    # --- reproduce the exact same train/val split train.py used ---
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
    print(f"analysing {len(indices)} structures from the {args.split} split\n")

    data = tokenize_all(encoder, quantizer, dataset, indices, device, args.max_length)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    report = {}

    # -----------------------------------------------------------------
    # 1. Is the codebook actually being used?
    # -----------------------------------------------------------------
    counts = np.bincount(data["code"], minlength=cfg["num_codes"]).astype(float)
    p = counts / counts.sum()
    perplexity = float(np.exp(-(p[p > 0] * np.log(p[p > 0])).sum()))
    report["codes_used"] = int((counts > 0).sum())
    report["codes_total"] = cfg["num_codes"]
    report["perplexity"] = perplexity
    report["residues"] = int(counts.sum())

    print("=" * 62)
    print("CODEBOOK USAGE")
    print("=" * 62)
    print(f"residues tokenized : {report['residues']}")
    print(f"codes used         : {report['codes_used']} / {report['codes_total']}")
    print(f"perplexity         : {perplexity:.1f}")
    print("  (perplexity is the effective number of codes in play. Equal to")
    print("   codes-used if they're used evenly, much lower if a few dominate.)")

    # -----------------------------------------------------------------
    # 2. Do codes correspond to secondary structure?
    # -----------------------------------------------------------------
    sse = data["sse"]
    code = data["code"]

    nmi_sse = normalized_mutual_info(code, sse)

    # Purity: for each code, what fraction of its residues share the single
    # most common secondary structure? Averaged over residues.
    purity_num = 0.0
    per_code = {}
    for c in np.unique(code):
        sel = sse[code == c]
        vals, cnts = np.unique(sel, return_counts=True)
        purity_num += cnts.max()
        per_code[int(c)] = {
            "n": int(len(sel)),
            "dominant_sse": str(vals[cnts.argmax()]),
            "sse_purity": float(cnts.max() / len(sel)),
        }
    purity = purity_num / len(code)

    # The number to beat: always guessing the globally most common label.
    _, marginal = np.unique(sse, return_counts=True)
    baseline = float(marginal.max() / marginal.sum())

    report["sse_nmi"] = nmi_sse
    report["sse_purity"] = float(purity)
    report["sse_purity_baseline"] = baseline

    print("\n" + "=" * 62)
    print("DO CODES MEAN SECONDARY STRUCTURE?")
    print("=" * 62)
    print(f"purity          : {purity:.3f}   (baseline {baseline:.3f})")
    print(f"NMI(code, SSE)  : {nmi_sse:.3f}")
    print("  purity = how often a code's residues agree on H/E/C.")
    print("  baseline = what you'd get by always guessing the commonest label.")
    print("  Beating the baseline by a lot means codes carry real structure.")

    # -----------------------------------------------------------------
    # 3. Do codes correspond to fine-grained local geometry?
    # -----------------------------------------------------------------
    # Bucket the continuous (bend, twist) pair into a grid, then ask how
    # well code predicts bucket. Finer-grained than the 3-way SSE label.
    valid = ~(np.isnan(data["angle"]) | np.isnan(data["torsion"]))
    angle_bin = np.digitize(data["angle"][valid], np.linspace(60, 150, 13))
    torsion_bin = np.digitize(data["torsion"][valid], np.linspace(-180, 180, 13))
    geom_bucket = angle_bin * 100 + torsion_bin

    nmi_geom = normalized_mutual_info(code[valid], geom_bucket)
    report["geometry_nmi"] = nmi_geom

    print("\n" + "=" * 62)
    print("DO CODES MEAN LOCAL GEOMETRY?")
    print("=" * 62)
    print(f"NMI(code, local geometry bucket) : {nmi_geom:.3f}")

    # -----------------------------------------------------------------
    # 4. THE ESM-3 QUESTION: is the mapping consistent across proteins?
    # -----------------------------------------------------------------
    # Take one geometry bucket at a time. Within it, every residue has
    # essentially the same local shape. If our tokens were purely local
    # (as ESM-3's are, by construction) they'd almost all get the same
    # code. Spread here means whole-protein context is leaking into the
    # token.
    print("\n" + "=" * 62)
    print("ARE CODES CONSISTENT ACROSS PROTEINS?")
    print("=" * 62)

    codes_v, prot_v = code[valid], data["protein"][valid]
    spreads, sizes = [], []
    for bucket in np.unique(geom_bucket):
        sel = geom_bucket == bucket
        if sel.sum() < 50:
            continue
        b_codes = codes_v[sel]
        vals, cnts = np.unique(b_codes, return_counts=True)
        q = cnts / cnts.sum()
        # Effective number of distinct codes used for this one local shape.
        spreads.append(float(np.exp(-(q * np.log(q)).sum())))
        sizes.append(int(sel.sum()))

    codes_per_shape = float(np.average(spreads, weights=sizes)) if spreads else float("nan")

    # And the direct leak test: how much does knowing the code tell you
    # about WHICH PROTEIN the residue came from? For a purely local
    # tokenizer this should be near zero.
    nmi_protein = normalized_mutual_info(code, data["protein"])

    report["codes_per_local_shape"] = codes_per_shape
    report["protein_identity_nmi"] = nmi_protein

    print(f"codes per distinct local shape : {codes_per_shape:.1f}")
    print("  1.0 would mean each local shape always gets the same code.")
    print("  Large values mean the same shape is being tokenized")
    print("  inconsistently, which is what global attention allows.")
    print(f"\nNMI(code, protein identity)    : {nmi_protein:.3f}")
    print("  How much a token reveals about WHICH protein it came from.")
    print("  ESM-3's local encoder cannot leak this. Ours can. Near 0 is good.")

    # -----------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------
    (args.out_dir / "summary.json").write_text(json.dumps(report, indent=2))

    # Per-code table, most-used first, with mean local geometry attached.
    lines = ["code,n,dominant_sse,sse_purity,mean_angle,mean_torsion"]
    order = np.argsort(-counts)
    for c in order:
        if counts[c] == 0:
            continue
        info = per_code[int(c)]
        sel = (code == c) & valid
        mean_angle = float(np.mean(data["angle"][sel])) if sel.sum() else float("nan")
        # Torsion is circular, so average it as unit vectors, not as numbers.
        if sel.sum():
            t = np.radians(data["torsion"][sel])
            mean_torsion = float(np.degrees(np.arctan2(np.sin(t).mean(), np.cos(t).mean())))
        else:
            mean_torsion = float("nan")
        lines.append(
            f"{int(c)},{info['n']},{info['dominant_sse']},"
            f"{info['sse_purity']:.3f},{mean_angle:.1f},{mean_torsion:.1f}"
        )
    (args.out_dir / "per_code.csv").write_text("\n".join(lines))

    print(f"\nwrote {args.out_dir}/summary.json and per_code.csv")

    if args.plots:
        make_plots(data, counts, per_code, valid, args.out_dir)
        print(f"wrote figures to {args.out_dir}/")


def make_plots(data, counts, per_code, valid, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    code = data["code"]

    # Figure 1: where each code sits in local-geometry space, coloured by
    # the secondary structure it mostly represents. If the codebook learned
    # geometry, this should show clear helix and strand clusters.
    fig, ax = plt.subplots(figsize=(7, 6))
    colours = {"H": "#d62728", "E": "#1f77b4", "C": "#999999"}
    xs, ys, cs, ss = [], [], [], []
    for c, info in per_code.items():
        if info["n"] < 20:
            continue
        sel = (code == c) & valid
        if not sel.sum():
            continue
        t = np.radians(data["torsion"][sel])
        xs.append(np.mean(data["angle"][sel]))
        ys.append(np.degrees(np.arctan2(np.sin(t).mean(), np.cos(t).mean())))
        cs.append(colours[info["dominant_sse"]])
        ss.append(min(info["n"] / 5, 120))
    ax.scatter(xs, ys, c=cs, s=ss, alpha=0.6, edgecolors="none")
    ax.set_xlabel("mean bend angle (degrees)")
    ax.set_ylabel("mean twist / torsion (degrees)")
    ax.set_title("Each dot is one code, placed at its average local geometry\n"
                 "red = mostly helix, blue = mostly strand, grey = coil")
    fig.tight_layout()
    fig.savefig(out_dir / "code_geometry_map.png", dpi=150)
    plt.close(fig)

    # Figure 2: how evenly the codebook is used.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.sort(counts)[::-1])
    ax.set_yscale("log")
    ax.set_xlabel("code, most used to least")
    ax.set_ylabel("times used (log)")
    ax.set_title("Codebook usage. A cliff means most codes are doing nothing.")
    fig.tight_layout()
    fig.savefig(out_dir / "codebook_usage.png", dpi=150)
    plt.close(fig)

    # Figure 3: purity distribution.
    fig, ax = plt.subplots(figsize=(7, 4))
    pur = [i["sse_purity"] for i in per_code.values() if i["n"] >= 20]
    ax.hist(pur, bins=30, color="#4c72b0")
    ax.set_xlabel("fraction of a code's residues sharing one secondary structure")
    ax.set_ylabel("number of codes")
    ax.set_title("Codes to the right are specific. Codes to the left are vague.")
    fig.tight_layout()
    fig.savefig(out_dir / "code_purity.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
