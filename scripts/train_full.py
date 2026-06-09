"""Full-config GenTac trajectory training.

Paper config (M=4, d=256, 100 diffusion steps, batch=200, 60 epochs).
Designed to run on a single A100/H100 — see docs/cloud_training.md for the
~$25/run Lambda Labs flow.

Usage:
    python scripts/train_full.py                          # default: GPU if available
    python scripts/train_full.py --epochs 30              # short run
    python scripts/train_full.py --precision bf16-mixed   # H100 sweet spot
    python scripts/train_full.py --resume path/to.ckpt

Checkpoints land in checkpoints/full/ — both the last epoch and the best val/loss
are kept. Re-loadable by the inference server and scripts/sample.py because the
cfg dict is now persisted via `save_hyperparameters` (no smoke flag guessing).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.config import GenTacConfig
from src.model.lightning_module import GenTacDataModule, GenTacTrajectoryModule


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=None,
                   help="path to data/processed (defaults to repo data/processed)")
    p.add_argument("--ckpt-dir", type=Path, default=None,
                   help="where to write checkpoints (defaults to repo checkpoints/full)")
    p.add_argument("--epochs", type=int, default=None, help="override cfg.epochs")
    p.add_argument("--batch-size", type=int, default=None, help="override cfg.batch_size")
    p.add_argument("--lr", type=float, default=None,
                   help="override cfg.lr_traj. Paper uses 1e-3 at batch 200; scale down for "
                        "smaller batches (linear rule: ~3e-4 at batch 64) or training diverges.")
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--precision", default="16-mixed",
                   choices=["32-true", "16-mixed", "bf16-mixed"])
    p.add_argument("--resume", type=Path, default=None, help="resume from checkpoint")
    p.add_argument("--devices", type=int, default=1)
    p.add_argument("--seed", type=int, default=42,
                   help="Global seed: torch + Lightning + Python random. Same seed reproduces the run.")
    p.add_argument("--dry-run", action="store_true",
                   help="Run one batch end-to-end (data → fwd → backward → checkpoint write) and exit. "
                        "Use locally to catch path/data/device bugs before burning GPU $.")
    p.add_argument("--ema-decay", type=float, default=None,
                   help="Override cfg.ema_decay. Set 0 to disable EMA.")
    return p.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]

    # Deterministic where possible. EMA buffers + dropout + CUDA matmul randomness
    # still leave some non-determinism; this gets you within ~1e-4 of bit-equal.
    pl.seed_everything(args.seed, workers=True)

    cfg = GenTacConfig()  # paper defaults — NOT smoke
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.lr is not None:
        cfg.lr_traj = args.lr
    if args.ema_decay is not None:
        cfg.ema_decay = args.ema_decay

    data_dir = args.data_dir or (root / "data" / "processed")
    ckpt_dir = args.ckpt_dir or (root / "checkpoints" / "full")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        # Dry-run mode: don't require GPU. Run one tiny batch to exercise every code
        # path (data load, model fwd, loss, backward, checkpoint write). Catches
        # cloud-specific bugs (mounted-volume paths, mixed-precision dtype mismatches,
        # missing deps) without burning GPU time.
        cfg.epochs = 1
        cfg.batch_size = min(cfg.batch_size, 2)
        accelerator = "gpu" if torch.cuda.is_available() else "cpu"
        precision = args.precision if torch.cuda.is_available() else "32-true"
        print(f"[full] DRY RUN  data_dir={data_dir}  ckpt_dir={ckpt_dir}  "
              f"accelerator={accelerator}  precision={precision}")
        dm = GenTacDataModule(data_dir, cfg, num_workers=0)
        mod = GenTacTrajectoryModule(cfg)
        trainer = pl.Trainer(
            max_epochs=1, accelerator=accelerator, devices=1, precision=precision,
            default_root_dir=str(ckpt_dir), limit_train_batches=2, limit_val_batches=1,
            log_every_n_steps=1, enable_checkpointing=True,
        )
        trainer.fit(mod, dm)
        out = ckpt_dir / "dryrun.ckpt"
        trainer.save_checkpoint(str(out))
        loaded_cfg = GenTacTrajectoryModule.cfg_from_checkpoint(out)
        assert loaded_cfg.schema_version == cfg.schema_version
        print(f"[full] dry-run OK → {out} (schema_version={loaded_cfg.schema_version})")
        return

    if not torch.cuda.is_available():
        raise RuntimeError(
            "train_full.py requires CUDA. For Mac smoke runs use scripts/smoke_train.py; "
            "for a code-path validation pass use --dry-run (works on CPU)."
        )

    print(f"[full] cfg: d={cfg.d_model} layers={cfg.n_layers} steps={cfg.n_diffusion_steps} "
          f"batch={cfg.batch_size} epochs={cfg.epochs} ema_decay={cfg.ema_decay}")
    print(f"[full] data_dir={data_dir}  ckpt_dir={ckpt_dir}  precision={args.precision}  seed={args.seed}")

    dm = GenTacDataModule(data_dir, cfg, num_workers=args.num_workers)
    mod = GenTacTrajectoryModule(cfg)
    print(f"[full] params: {sum(p.numel() for p in mod.model.parameters()):,}")

    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir, filename="best-{epoch:02d}-{val/loss:.4f}",
            monitor="val/loss", mode="min", save_top_k=3, save_last=True,
            auto_insert_metric_name=False,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        accelerator="gpu",
        devices=args.devices,
        precision=args.precision,
        default_root_dir=str(ckpt_dir),
        callbacks=callbacks,
        gradient_clip_val=1.0,
        log_every_n_steps=10,
        deterministic="warn",                       # Lightning will warn on non-determ ops, not crash
    )
    trainer.fit(mod, dm, ckpt_path=str(args.resume) if args.resume else None)
    print(f"[full] training complete. best ckpts in {ckpt_dir}")


if __name__ == "__main__":
    main()
