# ira-esm-tokenizer

A protein structure tokenizer built from scratch, trained with randomly initialised weights, and
compared probe by probe against ESM-3's structure tokenizer.

The model takes backbone coordinates and turns each residue into one discrete integer from a learned
codebook, then rebuilds the 3D backbone from those integers alone. Nothing about ESM-3 is reused.
No pretrained weights, no ESM code paths, no sequence information anywhere in the encoder.

Everything here runs on a single Kaggle GPU session.

---

## The pipeline

```
RCSB search  ->  raw .pdb  ->  parsed .npz  ->  Dataset  ->  Encoder  ->  Quantizer  ->  Decoder
fetch_pdb_ids   download_pdbs   parse_structures   dataset     encoder     quantizer     decoder
                                                                             |              |
                                                                       structure       backbone
                                                                         tokens        coordinates
```

1. `data/fetch_pdb_ids.py` queries RCSB for single-chain X-ray structures of 50 to 300 residues.
2. `data/download_pdbs.py` pulls the raw coordinate files, throttled.
3. `data/parse_structures.py` extracts per-residue (N, CA, C, O) backbone atoms plus the sequence,
   saving one `.npz` per structure.
4. `data/dataset.py` serves those to PyTorch, with length-bucketed batching under a
   `count x longest^2` token budget so padding waste stays bounded across variable-length chains.
5. `model/` holds the three learned pieces, described below.
6. `train.py` runs the loop. `analyze_codebook.py`, `compare_esm3.py` and `compare_runs.py`
   ask whether the tokens mean anything.

---

## Architecture

### Frames, `model/geometry.py`

Every residue gets its own local coordinate frame, a rotation and a translation built from its own
N, CA, C atoms. Same construction as AlphaFold's `rigidFrom3Points`. Expressing one residue relative
to another's frame is what makes the whole encoder invariant to rotating or translating the input
structure.

### Encoder, `model/encoder.py`

Geometry only, no sequence, 4 layers, 4 heads, 128 dimensions per residue, Xavier initialised.

Every residue starts as the *same* vector, because there is no sequence to distinguish them. That
creates a real problem. If geometry only enters as an attention bias, then q, k and v are identical
for every residue, the softmax weights still sum to 1, and every output collapses to the same value
no matter what the protein looks like.

So geometry enters twice. For every ordered pair (i, j) the encoder builds a **13-dimensional
pairwise feature**, which is 3 numbers of relative position, 9 numbers of flattened relative
rotation, and 1 distance, all measured in residue i's own local frame. Those 13 numbers then feed
two separate learned projections.

| Projection | Output | Role |
|---|---|---|
| `pair_to_bias` | 1 number per head | Added to the attention logit before softmax, so geometry decides *how much* i attends to j |
| `pair_to_value` | a full 128-dim vector | Added to j's value vector before the weighted sum, so geometry changes *what* actually gets averaged |

The second one is the fix for the collapse. Without it the thing being averaged is identical for
every pair and only the weights differ, which as shown above is mathematically guaranteed to
produce the same output for every residue.

### Quantizer, `model/quantizer.py` and `model/quantizer_ema.py`

A 4096 x 128 codebook. Each residue's continuous vector snaps to the nearest code by L2 distance,
and that code's index is the structure token.

Snapping is not differentiable, so training uses a straight-through estimator. The decoder's
gradient is routed past the snap and delivered to the encoder output directly, as though the
quantization step had not happened. Codebook loss plus a 0.25 commitment weight keeps the two sides
from drifting apart.

Two details that mattered in practice. The codebook is **seeded from real encoder outputs** before
step 1, because an encoder output with norm around 227 against a codebook initialised near 0.23
leaves the optimiser dragging codes across empty space for thousands of steps. And **dead codes are
revived** every 500 steps by moving unused entries onto real encoder outputs, since a code that is
never selected never receives a gradient and would otherwise stay dead permanently.

`quantizer_ema.py` is the alternative arm, updating each code toward the running mean of the vectors
assigned to it instead of by gradient. This is the mechanism ESM-3 uses.

### Decoder, `model/decoder.py`

Takes tokens back to coordinates. Predicts a rigid frame per residue, using the 6D Gram-Schmidt
rotation representation, and trains with FAPE clamped at 10 Angstroms. The clamp keeps a single
badly placed domain from dominating the loss for an otherwise correct structure.

The decoder is a deliberately close match to ESM-3's, so that it is not a source of difference when
the two tokenizers are compared.

### Training, `train.py`

AdamW, learning rate 1e-4 with 500 warmup steps then cosine decay, weight decay 0.01, gradient
clipping at 1.0. Chains longer than 256 residues are cropped, with a random window during training
and a fixed centre window during validation, so validation numbers are comparable across epochs.
Checkpoints are written every epoch, best and last kept separately.

---

## Where this deliberately differs from ESM-3

| | ESM-3 | This | Measured effect |
|---|---|---|---|
| Receptive field | 16 nearest neighbours per residue | Whole chain, full L x L attention | Only 0.001 of tokens survive clipping to 16 neighbours, NMI 0.590 |
| Positional info | Relative positional embedding in the encoder | None at all | Shuffling residues moves 1.000 of tokens with their residue, confirming the encoder is blind to chain order |
| Codebook updates | EMA | Gradient plus 0.25 commitment | 3458 of 4096 codes live, evenness 0.776, relative quantization error 0.061 |
| Auxiliary heads | Distance, error and confidence trained alongside | Reconstruction only | Not measurable from a checkpoint, the heads were never there |
| Decoder | 6D rotation frames, plain attention | Same | Not a source of difference by design |

The global receptive field is the substantive one. It makes reconstruction easier during training
while making each individual token mean less, because a token can absorb information from residues
200 positions away and is therefore not guaranteed to transfer between proteins the way a purely
local token is.

---

## Results

Best run, 829 structures, 200 epochs, roughly 9 seconds per epoch.

| Metric | Value |
|---|---|
| Train reconstruction | 6.654 A |
| Validation reconstruction | 7.034 A |
| VQ loss | 0.145 |
| Codes in use | 4096 / 4096 |
| Codebook perplexity | 3077 |

Token quality, 92 validation structures, 14,457 residues.

| Question | Number | Reading |
|---|---|---|
| Do codes mean secondary structure? | NMI 0.167 | Weakly. Purity reads 0.722 against a 0.433 baseline but should not be quoted, see below |
| Do codes mean local geometry? | NMI 0.456 | Yes, more strongly than secondary structure |
| Are codes consistent across proteins? | 490.9 codes per distinct local shape | No, the same shape gets tokenized many different ways |
| Do tokens leak which protein they came from? | NMI 0.479 | Yes, and this tracks codebook size more than receptive field |

**Three honest caveats**, all of which should stay attached to these numbers.

At 14,457 residues over 3,489 live codes, each code holds about 4.1 residues. Secondary-structure
purity of 0.722 is therefore inflated, since with four residues per code purity is mostly measuring
how few residues a code has rather than whether the code means anything. NMI is the honest number
here. The deeper reading is that at 4096 codes the tokens behave more like serial numbers than like
a vocabulary, because nothing in training rewards reuse and codes are never scarce.

The protein-identity leak is not purely a global-attention story. NMI is 0.479 for global-4096 and
0.459 for knn16, nearly identical, while the 512-code run sits at 0.248. That tracks codebook
capacity, not receptive field. A 4096-code book over 14,457 residues has enough room to memorise
chains.

The transplant probe is not currently working as a discriminator. knn16 scores 0.025, worse than
global-4096's 0.015, when a 16-neighbour encoder should score near 1.0 by construction. The window
is built from a contiguous sequence stretch rather than from spatial neighbours, which likely
accounts for more of the shortfall than the write-up allows. Either rebuild the window spatially or
report probe 1 as the receptive-field evidence and drop probe 2.

---

## Running it

Build the dataset.

```bash
python data/fetch_pdb_ids.py --out data/pdb_ids.txt --max-results 2000
python data/download_pdbs.py --id-list data/pdb_ids.txt --out-dir data/raw_pdb
python data/parse_structures.py --pdb-dir data/raw_pdb --out-dir data/parsed
```

Train the main run.

```bash
python train.py --epochs 200 --num-codes 4096 --checkpoint-dir checkpoints
```

Train the ablation arms, each changing exactly one thing.

```bash
python train.py --epochs 200 --num-codes 512  --checkpoint-dir checkpoints-512
python train.py --epochs 200 --neighbours 16  --checkpoint-dir checkpoints-knn16
python train_ema.py --epochs 200 --checkpoint-dir checkpoints-ema
```

Score them.

```bash
python analyze_codebook.py --checkpoint checkpoints/best.pt
python compare_esm3.py     --checkpoint checkpoints/best.pt
python compare_runs.py --run global4096=checkpoints/best.pt \
                       --run codes512=checkpoints-512/best.pt \
                       --run knn16=checkpoints-knn16/best.pt \
                       --run ema=checkpoints-ema/best.pt
```

Useful flags. `--budget` is the batch size limit expressed as `count x longest^2`, halve it on
out-of-memory. `--neighbours 16` reproduces ESM-3's receptive field. `--revive-every 0` disables
dead-code revival. `--no-data-init` skips codebook seeding, which is not recommended and exists
mainly to demonstrate why the seeding is there.

---

## Repo map

```
data/
  fetch_pdb_ids.py      RCSB search, single chain, 50-300 residues, X-ray
  download_pdbs.py      throttled coordinate download
  parse_structures.py   .pdb -> per-residue backbone .npz
  dataset.py            PyTorch Dataset, length-bucketed batching
model/
  geometry.py           per-residue rigid frames from N, CA, C
  encoder.py            geometry-only attention, 13-dim pair features, bias + value injection
  quantizer.py          4096-code VQ, straight-through, seeding, dead-code revival
  quantizer_ema.py      same, with EMA codebook updates
  decoder.py            tokens -> frames -> coordinates, 6D rotations, clamped FAPE
train.py                the training loop
train_ema.py            thin wrapper swapping in the EMA quantizer
analyze_codebook.py     what do the 4096 codes actually mean
compare_esm3.py         four probes against ESM-3's design choices
compare_runs.py         any number of checkpoints, one ablation table
notebooks/
  ira-esm-tokenizer-full.ipynb            build, train and analyse on Kaggle
  ira-esm-ablation.ipynb                  unattended multi-arm ablation run
  kaggle_esm_tokenizer_selfcontained.ipynb  older, writes every source file inline
```

---

## Open

- EMA against gradient codebook, arm built, run not yet scored.
- Bigger model, `--dim 256 --num-layers 6`, GPU headroom exists.
- More structures, past 1000, best return per unit effort.
- Unclamped FAPE on a fraction of batches, to get signal on global arrangement.
- IPA against the simplified 13-feature geometry, an explicit comparison.
- Sticker ablation, with and without the `pair_to_value` injection.
- An MLP block in the encoder, not yet added.
- A positional embedding, the one gap that is a real gap rather than a simplification.
- Rebuild or retire the transplant probe.
