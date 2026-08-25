"""
Training loop: the file that actually turns the encoder, quantizer and
decoder into a working structure tokenizer.

The whole thing is one loop over protein structures, doing:

    backbone coords -> encoder -> continuous vectors
                    -> quantizer -> discrete tokens (+ VQ loss)
                    -> decoder -> rebuilt backbone coords
                    -> FAPE loss vs. the original

and then nudging all three parts to make that round trip better. Nothing
supervises the tokens directly; they only have to be good enough that the
decoder can rebuild the shape from them. That reconstruction pressure is
the entire training signal.

Usage:
    python train.py --parsed-dir data/parsed --epochs 100

Resume an interrupted run:
    python train.py --resume checkpoints/last.pt
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler, Subset

from data.dataset import StructureDataset, collate_fn
from model.decoder import StructureDecoder, fape_loss
from model.encoder import StructureEncoder
from model.quantizer import VectorQuantizer


def set_seed(seed: int) -> None:
    """Make a run repeatable -- same data order, same weight init."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(requested: str) -> torch.device:
    """cuda if there's a GPU, else Apple's mps, else cpu."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def structure_lengths(files: list[Path]) -> list[int]:
    """
    Read just the residue count of every structure, once, at startup.

    Needed up front so batches can be assembled by length (see below).
    Cheap enough to brute-force for a few thousand files, and the result
    is cached to disk so restarts don't repeat it.
    """
    lengths = []
    for path in files:
        with np.load(path) as data:
            lengths.append(int(data["coords"].shape[0]))
    return lengths


class QuadraticBudgetSampler(Sampler):
    """
    Decides which structures go together in a batch.

    A fixed batch size does not work for this model. The encoder builds a
    feature for every PAIR of residues, so its memory grows with L squared,
    not with L. A batch of 8 proteins of 100 residues is comfortable; a
    batch of 8 proteins of 400 residues is 16x the memory and will run out.
    Picking a batch size small enough for the worst case would waste most
    of the GPU on the many short structures.

    So instead of a fixed count, batches are filled up to a fixed budget of
    (batch size) x (longest structure in the batch) squared. Short proteins
    come many at a time, long ones a few at a time, and peak memory stays
    roughly constant either way.

    Structures are also sorted by length before batching, so each batch
    holds proteins of similar size and very little of it is wasted padding.
    A little random noise is added to the sort key, and the finished
    batches are shuffled, so the model doesn't see the exact same groupings
    in the same order every epoch.
    """

    def __init__(self, lengths: list[int], budget: int, max_length: int, shuffle: bool = True, seed: int = 0):
        # Structures longer than max_length get cropped before they reach
        # the model, so their memory cost is capped at the crop size.
        self.effective = [min(length, max_length) for length in lengths]
        self.budget = budget
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self._batches = self._build(seed)

    def _build(self, seed: int) -> list[list[int]]:
        rng = np.random.default_rng(seed)
        indices = list(range(len(self.effective)))

        if self.shuffle:
            # Jitter the sort key by up to +/-10% so batch groupings vary
            # between epochs while staying length-homogeneous.
            noise = rng.uniform(0.9, 1.1, size=len(indices))
            indices.sort(key=lambda i: self.effective[i] * noise[i])
        else:
            indices.sort(key=lambda i: self.effective[i])

        batches, current, longest = [], [], 0
        for i in indices:
            candidate_longest = max(longest, self.effective[i])
            # Would adding this structure push the batch over budget?
            if current and (len(current) + 1) * candidate_longest ** 2 > self.budget:
                batches.append(current)
                current, longest = [i], self.effective[i]
            else:
                current.append(i)
                longest = candidate_longest
        if current:
            batches.append(current)

        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def set_epoch(self, epoch: int) -> None:
        """Re-roll the groupings each epoch, reproducibly."""
        self.epoch = epoch
        self._batches = self._build(self.seed + epoch)

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


def make_collate(max_length: int, train: bool):
    """
    Wrap the dataset's collate_fn with cropping.

    Very long chains are cut down to max_length before batching. This is
    standard for structure models and costs less than it sounds: local fold
    geometry is what the tokens are describing, and a 256-residue window
    contains plenty of it. During training the crop starts at a random
    offset, so over many epochs the model still sees every part of every
    long protein. During validation it's a fixed centre crop, so the
    validation number means the same thing every time it's computed.
    """

    def _collate(batch: list[dict]) -> dict:
        cropped = []
        for item in batch:
            length = item["coords"].shape[0]
            if length > max_length:
                if train:
                    start = random.randint(0, length - max_length)
                else:
                    start = (length - max_length) // 2
                item = {
                    "coords": item["coords"][start : start + max_length],
                    "sequence": item["sequence"][start : start + max_length],
                }
            cropped.append(item)
        return collate_fn(cropped)

    return _collate


@torch.no_grad()
def initialize_codebook(models, loader, device, num_codes) -> int:
    """
    Seed the codebook from real encoder outputs before training starts.

    The quantizer initializes its codebook to small random noise (norm about
    0.2), which is the textbook default, but the encoder's untrained outputs
    come out roughly a thousand times larger (norm about 227 -- it has no
    output normalization, and four residual layers keep adding to the signal).

    So on step 0 every single residue in the dataset is nearer to the same
    handful of codes than to any other, the VQ loss starts around 500 while
    the reconstruction loss is under 1, and the optimizer spends its first
    long stretch doing nothing but dragging 4096 codes across empty space.
    Reconstruction, the thing we actually care about, barely moves meanwhile.

    Dropping the codes onto actual encoder outputs instead starts them where
    the data already is. Same idea as the dead-code revival below, just done
    once for the whole codebook before the first step.
    """
    encoder, quantizer, _ = models
    encoder.eval()

    # A few batches is plenty to sample from; no need to sweep the dataset.
    pool = []
    collected = 0
    for batch in loader:
        latents = encoder(batch["coords"].to(device), batch["mask"].to(device))
        pool.append(latents[batch["mask"].to(device)])
        collected += pool[-1].shape[0]
        if collected >= num_codes * 4:
            break

    pool = torch.cat(pool)
    picks = torch.randint(0, pool.shape[0], (num_codes,), device=device)
    chosen = pool[picks]
    quantizer.codebook.weight.copy_(chosen + torch.randn_like(chosen) * 0.01)
    return collected


@torch.no_grad()
def revive_dead_codes(quantizer, encoder_output, mask, usage_counts) -> int:
    """
    Reset codebook entries that nothing is using.

    This is the characteristic failure mode of VQ-VAEs, and with 4096 codes
    and only ~1000 training structures it is close to guaranteed. A code
    that starts out far from every encoder output never gets chosen, so it
    never receives a gradient, so it never moves closer -- it's dead, and
    it stays dead. Left alone, a run can end up genuinely using a few
    hundred of its 4096 codes, and the tokenizer is far coarser than the
    codebook size suggests.

    The standard fix is blunt and works: every so often, find the codes
    that went unused and move them on top of actual encoder outputs from
    the current batch, where they'll be near real data and stand a chance
    of being picked. A little noise is added so that two codes revived from
    the same batch don't land in exactly the same place.
    """
    dead = (usage_counts == 0).nonzero().squeeze(-1)
    if dead.numel() == 0:
        return 0

    pool = encoder_output[mask]  # (num_real_residues, code_dim)
    if pool.shape[0] == 0:
        return 0

    picks = torch.randint(0, pool.shape[0], (dead.numel(),), device=pool.device)
    replacement = pool[picks]
    quantizer.codebook.weight[dead] = replacement + torch.randn_like(replacement) * 0.01
    return int(dead.numel())


def codebook_report(counts: torch.Tensor) -> dict:
    """
    Two numbers describing how much of the codebook is really in use.

    'used' is the blunt count of codes that appeared at least once.
    'perplexity' is the subtler one -- roughly "how many codes are in
    meaningful rotation." If usage is spread evenly over 500 codes,
    perplexity is about 500. If 490 of those are used once each and 10
    codes absorb everything else, 'used' still says 500 but perplexity
    drops to near 10, which is the honest answer.
    """
    total = counts.sum()
    if total == 0:
        return {"used": 0, "perplexity": 0.0}
    probs = counts.float() / total
    nonzero = probs[probs > 0]
    entropy = -(nonzero * nonzero.log()).sum()
    return {"used": int((counts > 0).sum()), "perplexity": float(entropy.exp())}


def run_epoch(models, loader, device, optimizer=None, scheduler=None, args=None,
              usage_since_revive=None, global_step=0):
    """
    One pass over the data. Training when an optimizer is given, evaluating
    when it isn't -- the forward pass is identical either way, which is the
    point of keeping them in one function.
    """
    encoder, quantizer, decoder = models
    training = optimizer is not None
    for module in models:
        module.train(training)

    totals = {"recon": 0.0, "vq": 0.0, "residues": 0}
    epoch_counts = torch.zeros(quantizer.codebook.num_embeddings, dtype=torch.long, device=device)
    revived = 0

    for batch in loader:
        global_step += 1
        coords = batch["coords"].to(device)
        mask = batch["mask"].to(device)

        with torch.set_grad_enabled(training):
            latents = encoder(coords, mask)
            quantized = quantizer(latents, mask)
            rebuilt = decoder(quantized["quantized"], mask)

            reconstruction_loss = fape_loss(rebuilt, coords, mask)
            loss = reconstruction_loss + quantized["loss"]

        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            # Clip gradients before stepping. Early in training a badly
            # wrong structure can produce a huge gradient that knocks the
            # weights somewhere they never recover from; clipping caps the
            # step size without changing its direction.
            torch.nn.utils.clip_grad_norm_(
                [p for m in models for p in m.parameters()], args.grad_clip
            )
            optimizer.step()
            scheduler.step()

        # Tally which codes got used, for the dead-code report and revival.
        used = quantized["tokens"][mask].detach()
        batch_counts = torch.bincount(used, minlength=epoch_counts.numel())
        epoch_counts += batch_counts
        if usage_since_revive is not None:
            usage_since_revive += batch_counts

        if training and args.revive_every > 0 and global_step % args.revive_every == 0:
            revived += revive_dead_codes(quantizer, latents.detach(), mask, usage_since_revive)
            usage_since_revive.zero_()

        # Weight each batch's loss by how many real residues it held, so
        # the epoch average isn't skewed by batches of different sizes.
        residues = int(mask.sum())
        totals["recon"] += reconstruction_loss.detach().item() * residues
        totals["vq"] += quantized["loss"].detach().item() * residues
        totals["residues"] += residues

    n = max(totals["residues"], 1)
    report = codebook_report(epoch_counts)
    return {
        "recon": totals["recon"] / n,
        # The same reconstruction error in Angstroms rather than in
        # normalized loss units, which is the number worth watching. It is
        # a clamped average, so it can never exceed 10 no matter how bad
        # the prediction is.
        "recon_angstrom": (totals["recon"] / n) * 10.0,
        "vq": totals["vq"] / n,
        "codes_used": report["used"],
        "perplexity": report["perplexity"],
        "revived": revived,
        "global_step": global_step,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parsed-dir", type=Path, default=Path("data/parsed"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=256, help="Crop structures longer than this.")
    parser.add_argument(
        "--budget",
        type=int,
        default=4 * 256 ** 2,
        help="Batch size limit as (count x longest^2). Halve it if you hit out-of-memory.",
    )
    parser.add_argument("--num-codes", type=int, default=4096)
    parser.add_argument("--dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument(
        "--neighbours", type=int, default=0,
        help="Restrict each residue's attention to its k nearest CA neighbours. "
             "0 = whole chain (default). 16 reproduces ESM3's receptive field.",
    )
    parser.add_argument(
        "--no-data-init",
        action="store_true",
        help="Skip seeding the codebook from real encoder outputs (not recommended).",
    )
    parser.add_argument("--revive-every", type=int, default=500, help="Steps between dead-code resets; 0 disables.")
    parser.add_argument("--num-workers", type=int, default=2, help="Use 0 if dataloader workers misbehave on macOS.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset = StructureDataset(args.parsed_dir)
    if len(dataset) == 0:
        raise SystemExit(
            f"No .npz files in {args.parsed_dir}. Run data/parse_structures.py first."
        )

    # Cache the length list -- reading every file's header takes a moment
    # and the answer never changes.
    cache_path = args.parsed_dir / "_lengths.json"
    names = [p.name for p in dataset.files]
    cached = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    if cached.get("names") != names:
        cached = {"names": names, "lengths": structure_lengths(dataset.files)}
        cache_path.write_text(json.dumps(cached))
    lengths = cached["lengths"]

    # Split by a seeded shuffle so the same structures are held out on
    # every run, including after a resume.
    permutation = np.random.default_rng(args.seed).permutation(len(dataset))
    num_val = max(1, int(len(dataset) * args.val_fraction))
    val_indices, train_indices = permutation[:num_val], permutation[num_val:]

    train_sampler = QuadraticBudgetSampler(
        [lengths[i] for i in train_indices], args.budget, args.max_length, shuffle=True, seed=args.seed
    )
    val_sampler = QuadraticBudgetSampler(
        [lengths[i] for i in val_indices], args.budget, args.max_length, shuffle=False
    )

    train_loader = DataLoader(
        Subset(dataset, train_indices.tolist()),
        batch_sampler=train_sampler,
        collate_fn=make_collate(args.max_length, train=True),
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices.tolist()),
        batch_sampler=val_sampler,
        collate_fn=make_collate(args.max_length, train=False),
        num_workers=args.num_workers,
    )

    encoder = StructureEncoder(args.dim, args.num_heads, args.num_layers,
                               args.neighbours).to(device)
    quantizer = VectorQuantizer(args.num_codes, args.dim).to(device)
    decoder = StructureDecoder(args.dim, args.dim, args.num_heads, args.num_layers).to(device)
    models = (encoder, quantizer, decoder)

    parameters = [p for m in models for p in m.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)

    # Linear warmup, then a slow cosine decay. The warmup matters more than
    # usual here: at step 0 the codebook is random noise, so the tokens are
    # meaningless and the decoder's gradients are pure nonsense. Easing in
    # stops those first few steps from doing lasting damage.
    total_steps = max(1, args.epochs * len(train_sampler))

    def lr_at(step: int) -> float:
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(progress, 1.0)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_at)

    start_epoch, best_val = 0, float("inf")
    if args.resume and args.resume.exists():
        state = torch.load(args.resume, map_location=device)
        encoder.load_state_dict(state["encoder"])
        quantizer.load_state_dict(state["quantizer"])
        decoder.load_state_dict(state["decoder"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best_val = state["epoch"] + 1, state["best_val"]
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    if not args.no_data_init and not (args.resume and args.resume.exists()):
        sampled = initialize_codebook(models, train_loader, device, args.num_codes)
        print(f"seeded codebook from {sampled} encoder outputs")

    num_params = sum(p.numel() for p in parameters)
    print(f"device: {device}   parameters: {num_params/1e6:.2f}M")
    print(f"structures: {len(train_indices)} train / {len(val_indices)} val")
    print(f"batches per epoch: {len(train_sampler)}")

    usage_since_revive = torch.zeros(args.num_codes, dtype=torch.long, device=device)
    # Counted across the whole run, not per epoch -- otherwise --revive-every
    # would silently never trigger on datasets with few batches per epoch.
    global_step = start_epoch * len(train_sampler)

    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        started = time.time()

        train_stats = run_epoch(
            models, train_loader, device, optimizer, scheduler, args,
            usage_since_revive, global_step,
        )
        global_step = train_stats["global_step"]
        val_stats = run_epoch(models, val_loader, device)

        print(
            f"epoch {epoch:3d}  "
            f"train {train_stats['recon_angstrom']:.3f}A  "
            f"val {val_stats['recon_angstrom']:.3f}A  "
            f"vq {train_stats['vq']:.4f}  "
            f"codes {train_stats['codes_used']}/{args.num_codes}  "
            f"ppl {train_stats['perplexity']:.0f}  "
            f"revived {train_stats['revived']}  "
            f"{time.time() - started:.0f}s"
        )

        state = {
            "encoder": encoder.state_dict(),
            "quantizer": quantizer.state_dict(),
            "decoder": decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "best_val": best_val,
            # Paths stringified so the checkpoint stays loadable under
            # torch.load's weights_only=True default (PyTorch 2.6+), which
            # refuses to unpickle a PosixPath.
            "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        }
        # Always overwrite last.pt so an interrupted run can resume; keep
        # best.pt separately so a late overfitting slide can't erase the
        # best model we actually found.
        torch.save(state, args.checkpoint_dir / "last.pt")
        if val_stats["recon_angstrom"] < best_val:
            best_val = state["best_val"] = val_stats["recon_angstrom"]
            torch.save(state, args.checkpoint_dir / "best.pt")

    print(f"Done. Best validation reconstruction: {best_val:.3f}A")


if __name__ == "__main__":
    main()
