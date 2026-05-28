"""Smoke-train GenTac trajectory diffusion on Mac MPS.

Tiny config, 1 epoch, batch 4 — proves the whole pipeline runs end-to-end and the
loss curve goes the right direction. NOT a real training run.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytorch_lightning as pl
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.config import GenTacConfig
from src.model.lightning_module import GenTacDataModule, GenTacTrajectoryModule


def main():
    # Deterministic smoke runs — same seed reproduces the loss curve bit-for-bit
    # (modulo MPS non-determinism in fp16 reductions).
    pl.seed_everything(0, workers=True)

    cfg = GenTacConfig(smoke=True)
    # nudge epochs up a bit so we can actually see learning
    cfg.epochs = 3
    cfg.batch_size = 4

    print(f"smoke config: d={cfg.d_model}  layers={cfg.n_layers}  steps={cfg.n_diffusion_steps}  "
          f"batch={cfg.batch_size}  epochs={cfg.epochs}")

    root = Path(__file__).resolve().parents[1]
    dm = GenTacDataModule(root / "data" / "processed", cfg, num_workers=0)
    mod = GenTacTrajectoryModule(cfg)

    n_params = sum(p.numel() for p in mod.model.parameters())
    print(f"model params: {n_params:,}")

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"device: {device}")

    ckpt_dir = root / "checkpoints" / "smoke"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    trainer = pl.Trainer(
        max_epochs=cfg.epochs,
        accelerator=device,
        devices=1,
        precision="32-true",
        log_every_n_steps=1,
        enable_progress_bar=True,
        enable_checkpointing=False,        # we save manually at end
        default_root_dir=str(ckpt_dir),
    )
    trainer.fit(mod, dm)
    ckpt_path = ckpt_dir / "smoke_final.ckpt"
    trainer.save_checkpoint(str(ckpt_path))
    print(f"smoke train completed without errors. checkpoint → {ckpt_path}")


if __name__ == "__main__":
    main()
