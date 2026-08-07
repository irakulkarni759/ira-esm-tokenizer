"""
Parse raw .pdb files into backbone coordinate arrays for structure
tokenizer training.

For each structure, extracts the (N, CA, C, O) backbone atom coordinates
per residue and the amino acid sequence, then saves them together as a
single .npz file. This is step 3 of the pipeline: raw .pdb files (from
download_pdbs.py) -> clean per-residue coordinate arrays (this script)
-> PyTorch Dataset (next step).
"""

import argparse
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
from Bio.SeqUtils import seq1

# The four backbone atoms every standard amino acid residue has. Together
# they define the protein's fold; side-chain atoms vary per residue type
# and aren't needed for a backbone-level structure tokenizer.
BACKBONE_ATOMS = ["N", "CA", "C", "O"]


def extract_backbone(pdb_path: Path, min_length: int) -> tuple[np.ndarray, str] | None:
    """
    Parse one .pdb file and return (coords, sequence), or None if the
    structure doesn't yield a usable single chain.

    coords has shape (L, 4, 3): L residues x 4 backbone atoms x xyz.
    sequence is the one-letter amino acid code string, length L.
    """
    # QUIET=True suppresses BioPython's warnings about minor format
    # quirks (common in real PDB files) that we don't need to see per-file.
    parser = PDBParser(QUIET=True)

    try:
        structure = parser.get_structure(pdb_path.stem, pdb_path)
    except Exception:
        # Malformed file (truncated download, unparseable header, etc.)
        return None

    # Some PDB entries (e.g. NMR structures) contain multiple models of
    # the same molecule. We only want one static structure per entry, so
    # take the first model and ignore the rest.
    model = structure[0]

    # We filtered for single-chain entries when building the ID list, but
    # some entries still have a water/ligand "chain" alongside the real
    # one. Grab the first chain that actually contains amino acids.
    chain = None
    for candidate in model:
        if any(is_aa(res, standard=True) for res in candidate):
            chain = candidate
            break
    if chain is None:
        return None

    coords = []
    sequence = []
    for residue in chain:
        # Skip anything that isn't a standard amino acid: waters, metal
        # ions, ligands, and modified/non-standard residues all show up
        # as separate "residues" in BioPython's model but aren't part of
        # the sequence we want to tokenize.
        if not is_aa(residue, standard=True):
            continue

        # Crystal structures often have missing density for some atoms
        # (flexible loops, disordered regions). If any backbone atom is
        # absent for this residue, we can't get a coordinate for it, so
        # we drop the whole residue rather than leaving a gap or faking
        # a value.
        if not all(atom in residue for atom in BACKBONE_ATOMS):
            continue

        residue_coords = [residue[atom].get_coord() for atom in BACKBONE_ATOMS]
        coords.append(residue_coords)
        sequence.append(seq1(residue.get_resname()))

    # After dropping incomplete residues, the chain may have shrunk below
    # our length floor (e.g. a 55-residue entry losing 10 disordered
    # residues). Re-check here rather than trusting the original RCSB filter.
    if len(coords) < min_length:
        return None

    coords_array = np.array(coords, dtype=np.float32)  # shape (L, 4, 3)
    sequence_str = "".join(sequence)
    return coords_array, sequence_str


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pdb-dir",
        type=Path,
        default=Path("data/raw_pdb"),
        help="Directory of raw .pdb files (output of download_pdbs.py).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/parsed"),
        help="Directory to save parsed .npz files into.",
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=50,
        help="Drop structures shorter than this after removing incomplete residues.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pdb_files = sorted(args.pdb_dir.glob("*.pdb"))
    print(f"Parsing {len(pdb_files)} structures from {args.pdb_dir}/")

    failed = []
    succeeded = 0
    for i, pdb_path in enumerate(pdb_files, start=1):
        out_path = args.out_dir / f"{pdb_path.stem}.npz"

        # Same resumability pattern as download_pdbs.py: skip work already done.
        if out_path.exists():
            succeeded += 1
            continue

        result = extract_backbone(pdb_path, args.min_length)
        if result is None:
            failed.append(pdb_path.stem)
            continue

        coords, sequence = result
        # savez (not savez_compressed) — these arrays are small (a few KB
        # each), so compression overhead isn't worth the slower read speed
        # during training, when this file gets loaded repeatedly.
        np.savez(out_path, coords=coords, sequence=sequence)
        succeeded += 1

        if i % 100 == 0:
            print(f"  {i}/{len(pdb_files)} processed...")

    print(f"Done. {succeeded} succeeded, {len(failed)} failed/skipped.")
    if failed:
        failed_path = args.out_dir / "_failed_ids.txt"
        failed_path.write_text("\n".join(failed))
        print(f"Failed IDs written to {failed_path}")


if __name__ == "__main__":
    main()
