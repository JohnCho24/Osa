"""PyTorch Lightning wrappers for GenTac trajectory training.

`GenTacTrajectoryModule` randomly picks one of the pretraining modes
{unconditioned, opp_conditioned_team0, opp_conditioned_team1} per batch and computes
the DDPM noise-prediction loss (paper §6.3.1).

`GenTacDataModule` discovers all converted match JSONs and splits matches between
train and val (match-level holdout, not window-level — windows from the same match
are highly correlated, so within-match splits would leak).
"""
from __future__ import annotations

import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import List

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from .config import GenTacConfig
from .dataset import TrajectoryDataset
from .diffusion import GenTacDiffusion, LinearBetaSchedule, build_target_mask, compute_diffusion_loss


PRETRAIN_MODES = ("unconditioned", "opp_conditioned_team0", "opp_conditioned_team1")


class GenTacTrajectoryModule(pl.LightningModule):
    def __init__(self, cfg: GenTacConfig):
        super().__init__()
        self.cfg = cfg
        self.model = GenTacDiffusion(cfg)
        self.schedule = LinearBetaSchedule(cfg.n_diffusion_steps, cfg.beta_start, cfg.beta_end)
        # Persist the post-init config dict so any checkpoint can be reloaded standalone
        # (server, sample.py, etc.) without the caller having to know smoke vs paper config.
        self.save_hyperparameters({"cfg_dict": asdict(cfg)})

        # EMA shadow weights. Standard diffusion-training trick: the EMA-averaged
        # weights generate smoother, higher-fidelity samples than the live weights
        # at the same step count. We register them as buffers so they checkpoint
        # alongside the live model and Lightning syncs them across devices in DDP.
        self.ema_decay: float = getattr(cfg, "ema_decay", 0.0)         # 0 disables EMA
        if self.ema_decay > 0:
            self._ema_buffers: dict[str, torch.Tensor] = {}
            for name, p in self.model.named_parameters():
                if p.requires_grad:
                    buf_name = f"ema_{name.replace('.', '__')}"
                    self.register_buffer(buf_name, p.detach().clone(), persistent=True)
                    self._ema_buffers[name] = buf_name

    def on_train_batch_end(self, *args, **kwargs):
        # Update EMA shadow after each optimizer step. decay closer to 1 → slower update.
        if self.ema_decay <= 0:
            return
        with torch.no_grad():
            for name, p in self.model.named_parameters():
                if not p.requires_grad:
                    continue
                buf = getattr(self, self._ema_buffers[name])
                buf.mul_(self.ema_decay).add_(p.detach(), alpha=1 - self.ema_decay)

    def ema_state_dict(self) -> dict:
        """Returns a model state_dict whose parameters are the EMA shadows."""
        if self.ema_decay <= 0:
            return self.model.state_dict()
        sd = self.model.state_dict()
        for name in self._ema_buffers:
            sd[name] = getattr(self, self._ema_buffers[name]).clone()
        return sd

    @classmethod
    def cfg_from_checkpoint(cls, ckpt_path: str | Path) -> GenTacConfig:
        """Reconstruct the GenTacConfig that was used to train this checkpoint.

        Validates the persisted `schema_version` matches the runtime `GenTacConfig`.
        A schema mismatch means the checkpoint's architecture is incompatible with
        the current code path (e.g. trained before the learned-waypoint extension
        landed) and must be retrained rather than silently misloaded.
        """
        data = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        cfg_dict = data.get("hyper_parameters", {}).get("cfg_dict")
        if cfg_dict is None:
            raise ValueError(f"checkpoint {ckpt_path} has no cfg_dict; retrain after the config-persistence fix")
        ckpt_version = cfg_dict.get("schema_version", 1)
        runtime_version = GenTacConfig().schema_version
        if ckpt_version != runtime_version:
            raise ValueError(
                f"checkpoint schema_version={ckpt_version} but runtime expects {runtime_version}. "
                f"Retrain with the current code (scripts/smoke_train.py or scripts/train_full.py)."
            )
        return GenTacConfig(**cfg_dict)

    def setup(self, stage: str | None = None):
        self.schedule.to(self.device)

    def _step(self, batch: dict, stage: str) -> torch.Tensor:
        history, future, valid = batch["history"], batch["future"], batch["mask"]
        # randomly pick a mode for this batch (paper pretrains on both)
        mode = random.choice(PRETRAIN_MODES)
        target_mask = build_target_mask(valid, mode, self.cfg)
        loss, info = compute_diffusion_loss(self.model, self.schedule, history, future, valid, target_mask)
        self.log(f"{stage}/loss", loss, prog_bar=(stage == "train"), on_step=(stage == "train"), on_epoch=True)
        self.log(f"{stage}/n_targets", float(info["n_targets"]), on_step=False, on_epoch=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def configure_optimizers(self):
        optim = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr_traj, weight_decay=self.cfg.weight_decay)
        # cosine + 2% linear warmup (paper §6.3.3)
        warmup_steps = max(1, int(self.trainer.estimated_stepping_batches * 0.02))
        total_steps = max(1, self.trainer.estimated_stepping_batches)
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(1, warmup_steps)
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
        return {
            "optimizer": optim,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


class GenTacDataModule(pl.LightningDataModule):
    def __init__(self, processed_dir: Path, cfg: GenTacConfig, num_workers: int = 0):
        super().__init__()
        self.processed_dir = processed_dir
        self.cfg = cfg
        self.num_workers = num_workers
        self.train_ds: TrajectoryDataset | None = None
        self.val_ds: TrajectoryDataset | None = None

    def _discover(self) -> List[Path]:
        # Exclude `_clip` (dev slices) and `samples.json` (inference output — different schema).
        return sorted(
            p for p in self.processed_dir.glob("*.json")
            if "_clip" not in p.name and p.name != "samples.json"
        )

    def setup(self, stage: str | None = None):
        matches = self._discover()
        if len(matches) < 2:
            # one-match fallback: train and val on the same data (smoke only)
            self.train_ds = TrajectoryDataset(matches, self.cfg)
            self.val_ds = TrajectoryDataset(matches, self.cfg, stride=self.cfg.train_stride_frames * 4)
        else:
            # hold out the last match as val (match-level, not window-level)
            train_paths = matches[:-1]
            val_paths = matches[-1:]
            self.train_ds = TrajectoryDataset(train_paths, self.cfg)
            self.val_ds = TrajectoryDataset(val_paths, self.cfg, stride=self.cfg.train_stride_frames * 4)
        print(f"[DataModule] train={len(self.train_ds)} windows  val={len(self.val_ds)} windows")

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, batch_size=self.cfg.batch_size, shuffle=True,
            num_workers=self.num_workers, drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, batch_size=self.cfg.batch_size, shuffle=False,
            num_workers=self.num_workers, drop_last=False,
        )
