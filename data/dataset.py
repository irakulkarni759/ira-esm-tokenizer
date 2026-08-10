"""
PyTorch Dataset over parsed protein structures (.npz files produced by
parse_structures.py). Step 4 of the pipeline: parsed coordinate arrays
-> batches ready to feed into a model.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# The 20 standard amino acids, in a fixed order. Every sequence letter
# gets mapped to its index in this string (e.g. "A" -> 0, "C" -> 1, ...).
# The model works with these integer indices, not raw letters.
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

# Reserved index for padding -- positions we add just to make a batch
# rectangular, not real residues. Placed after the 20 real amino acids.
PAD_INDEX = len(AMINO_ACIDS)


class StructureDataset(Dataset):
    """
    One item = one parsed protein structure: backbone coordinates plus
    its amino acid sequence, both loaded from a single .npz file.
    """

    def __init__(self, parsed_dir: Path):
        self.parsed_dir = Path(parsed_dir)
        # Build the file list once, at startup, rather than re-scanning
        # the directory on every __getitem__ call.
        self.files = sorted(self.parsed_dir.glob("*.npz"))

    def __len__(self):
        # PyTorch's DataLoader calls this to know how many items exist,
        # e.g. to decide how many batches make up one epoch.
        return len(self.files)

    def __getitem__(self, idx):
        # PyTorch calls this with an integer index whenever it needs one
        # specific example -- this is where the actual file gets read.
        data = np.load(self.files[idx])
        coords = torch.from_numpy(data["coords"])  # (L, 4, 3) float32

        # sequence was saved as a numpy string; convert each letter to
        # its integer index using the lookup table above.
        sequence_str = str(data["sequence"])
        sequence = torch.tensor(
            [AA_TO_INDEX[aa] for aa in sequence_str], dtype=torch.long
        )

        return {"coords": coords, "sequence": sequence}


def collate_fn(batch: list[dict]) -> dict:
    """
    Combine a list of variable-length examples into one padded batch.

    Structures have different numbers of residues (L varies per protein),
    but a batch tensor needs one fixed shape. We pad every example up to
    the longest one in this batch, and return a mask marking which
    positions are real residues vs. padding.
    """
    lengths = [item["coords"].shape[0] for item in batch]
    max_len = max(lengths)
    batch_size = len(batch)

    # Pre-allocate zero-filled tensors of the final padded shape, then
    # fill in each example's real data. Padded positions stay zero.
    coords_batch = torch.zeros(batch_size, max_len, 4, 3)
    sequence_batch = torch.full((batch_size, max_len), PAD_INDEX, dtype=torch.long)
    # mask is True for real residues, False for padding -- models use this
    # to ignore padded positions in attention/loss computations.
    mask = torch.zeros(batch_size, max_len, dtype=torch.bool)

    for i, item in enumerate(batch):
        length = lengths[i]
        coords_batch[i, :length] = item["coords"]
        sequence_batch[i, :length] = item["sequence"]
        mask[i, :length] = True

    return {"coords": coords_batch, "sequence": sequence_batch, "mask": mask}
