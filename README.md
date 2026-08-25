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

The global receptive field is the substantive one, and the ablation below settles it. Restricting
the encoder to ESM-3's 16 neighbours costs 0.05 A of reconstruction and improves every measure of
what the tokens mean. Global attention makes reconstruction marginally easier while making each
individual token mean less, because a token can absorb information from residues 200 positions away
and therefore does not transfer between proteins the way a purely local token does.

---

## Results

Three arms of one ablation, 200 epochs each, identical data, decoder, optimizer and seed, with
exactly one knob changed per arm. 829 train / 92 val structures, 14,457 val residues, roughly
9 seconds per epoch.

- **4096**, the reference build. Global attention, 4096-code codebook.
- **512**, changes only vocabulary size.
- **knn16**, changes only receptive field, to ESM-3's 16 nearest spatial neighbours.

### Reconstruction is flat

| | 4096 | 512 | knn16 |
|---|---|---|---|
| Validation reconstruction | 7.034 A | 7.051 A | 7.087 A |
| Train reconstruction | 6.65 A | 6.96 A | 6.70 A |
| Train/val gap | 0.38 | 0.09 | 0.38 |

An 8x codebook and a whole-chain receptive field together buy 0.05 A, which is noise. All three
curves are flat from epoch 175 to 200, so this is a real tie rather than one arm being cut short.

The read is not that codebook size does not matter. It is that **at 7 A nothing downstream of the
encoder is the binding constraint**. The encoder is 128 dimensions, 4 layers, 4 heads, with no
feed-forward block, and every residue starts from one shared vector with all differentiation
injected through `pair_to_value`. That is the ceiling all three arms hit from different directions.
The train/val gaps say where the extra capacity went, since the two big codebooks memorised the
extra 0.3 A rather than generalising it.

### Token quality, where the arms actually separate

| | 4096 | 512 | knn16 | Better |
|---|---|---|---|---|
| VQ loss | 0.145 | 0.091 | 0.050 | lower |
| Codes live (val) | 3458 / 4096 | 508 / 512 | 3139 / 4096 | higher |
| Usage evenness | 0.776 | 0.931 | 0.758 | higher |
| Relative quantization error | 0.061 | 0.039 | 0.069 | lower |
| SSE purity | 0.722 | 0.559 | 0.813 | higher |
| NMI(code, SSE) | 0.168 | 0.053 | 0.229 | higher |
| NMI(code, local geometry) | 0.458 | 0.255 | 0.466 | higher |
| Codes per distinct local shape | 479.3 | n/a | 353.8 | lower |
| NMI(code, protein identity) | 0.480 | 0.251 | 0.463 | lower |
| Locality agreement | 0.001 | n/a | 1.000 | n/a |

### The headline

**knn16 wins.** It ties on reconstruction and wins on every semantic measure, against the same
4096-code vocabulary, so the comparison is fair.

Clipping the global model to 16 neighbours leaves only 1 token in 1000 unchanged. Global attention
is not being used occasionally for long-range contacts, it is being used constantly, for every
residue, and it makes the tokens **worse** by every measure of what a token means. That is the
central result of the project.

The mechanism is visible in the VQ loss. knn16 sits at 0.050 against 4096's 0.145, a 3x lower
quantization cost on an identical codebook, and it starts near 0.6 before a single training step
while the global arms start at 10 to 15. A locally-restricted encoder produces latents that are
inherently more clusterable. Whole-chain attention aggregates over hundreds of residues, so every
residue's vector gets contaminated by its particular protein and the clusters smear.

### Four caveats that must stay attached to these numbers

**Purity only compares fairly at similar code counts.** It inflates automatically as codes are
added. 4096 against knn16 is fair, 3458 codes against 3139. Any comparison against 512 is not, so
512's 0.559 should be read as its own row and not as a loss.

**The identity leak tracks codebook size, not receptive field.** 0.480 for global-4096 against
0.463 for knn16 is nearly no change, while 512 sits at 0.251. A 4096-code book over 14,457 residues
simply has the capacity to memorise chains. Attributing the leak to global attention would
contradict these numbers.

**The 512 run nearly died and the revival mechanism saved it.** Perplexity crashed to about 3 codes
in use around epoch 4 to 6, which is textbook codebook collapse and normally unrecoverable. It came
back to roughly 480 by epoch 50 because of `initialize_codebook` and `revive_dead_codes`. This
reframes the gradient-versus-EMA point. The dead-code cost was not paid, it was engineered around.

**The transplant probe is not working as a discriminator.** knn16 scores 0.025, worse than
global-4096's 0.015, when a 16-neighbour encoder should score near 1.0 by construction. The window
is a contiguous sequence stretch rather than a spatial neighbourhood, which likely accounts for most
of the shortfall. Use probe 1 for the receptive-field argument and either rebuild probe 2 spatially
or retire it.

### One more thing the curves show

Codebook perplexity goes flat at about epoch 75 in all three arms, while reconstruction keeps
improving until about epoch 150. For the second half of training the vocabulary was frozen and all
the gains came from the encoder and decoder learning to use it better. That is the same finding from
another angle, since the vocabulary settles early and then stops being what matters.

A note on precision. Numbers from the standalone `analyze_codebook.py` pass and the `compare_runs.py`
pass differ in the third digit (3489 against 3458 live codes, 490.9 against 479.3 codes per shape)
because they subsample structures differently. The table above uses the `compare_runs.py` values so
that every column comes from one pass.

---

## Running it

Build the dataset.

```bash
python data/fetch_pdb_ids.py --out-file data/pdb_ids.txt --min-length 50 --max-length 300 --max-results 2000
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

- **knn16 at 512 codes, highest priority.** Both knobs have been varied alone and never together.
  This is the missing corner of the 2x2, and it is where the best configuration probably sits, with
  local semantics plus a healthy codebook plus no identity leak.
- A codebook-size sweep at 1024 and 2048, since two points make a line and three make a trend.
- EMA against gradient codebook, arm built, run not yet scored.
- A final LayerNorm on the encoder output. Evidence in hand but not applied, since encoder outputs
  have norm around 227 while the codebook initialises near 0.23. A test on synthetic data gave
  val 6.66 A against 7.50 A and VQ loss 0.043 against 4.4 over 120 epochs. This would change the
  encoder, so it is a deliberate call rather than a cleanup.
- Bigger model, `--dim 256 --num-layers 6`, GPU headroom exists.
- More structures, past 1000, best return per unit effort.
- Unclamped FAPE on a fraction of batches, to get signal on global arrangement.
- IPA against the simplified 13-feature geometry, an explicit comparison.
- Sticker ablation, with and without the `pair_to_value` injection.
- An MLP block in the encoder, not yet added.
- A positional embedding, the one gap that is a real gap rather than a simplification.
- Rebuild or retire the transplant probe.
