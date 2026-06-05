"""Schema-versioning regressions: catch silent misload before it happens.

This is the test we wish we'd had before the schema_version field existed.
"""
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from src.model.config import GenTacConfig
from src.model.lightning_module import GenTacTrajectoryModule


def _save_fake_ckpt(cfg_dict: dict, tmp: Path) -> Path:
    """Save a minimal checkpoint with the given cfg_dict in hparams."""
    ckpt = {
        "hyper_parameters": {"cfg_dict": cfg_dict},
        "state_dict": {},
        "pytorch-lightning_version": "2.4.0",
    }
    p = tmp / "fake.ckpt"
    torch.save(ckpt, p)
    return p


def test_cfg_roundtrips_via_dict():
    """asdict(cfg) → GenTacConfig(**dict) is the identity (modulo post_init)."""
    cfg = GenTacConfig(smoke=True)
    rebuilt = GenTacConfig(**asdict(cfg))
    assert asdict(rebuilt) == asdict(cfg)


def test_schema_mismatch_refuses_load(tmp_path):
    """A checkpoint with an older schema_version must refuse to load with a clear
    error, not silently misload into a different architecture."""
    bad_cfg = asdict(GenTacConfig(smoke=True))
    bad_cfg["schema_version"] = 1  # pretend this is an old checkpoint
    ckpt = _save_fake_ckpt(bad_cfg, tmp_path)
    with pytest.raises(ValueError, match="schema_version"):
        GenTacTrajectoryModule.cfg_from_checkpoint(ckpt)


def test_no_cfg_dict_refuses_load(tmp_path):
    """Pre-cfg-persistence checkpoints (no cfg_dict in hparams) must fail loudly."""
    ckpt_data = {"hyper_parameters": {}, "state_dict": {}}
    p = tmp_path / "fake.ckpt"
    torch.save(ckpt_data, p)
    with pytest.raises(ValueError, match="cfg_dict"):
        GenTacTrajectoryModule.cfg_from_checkpoint(p)


def test_smoke_overrides_apply():
    """smoke=True must actually shrink the architecture."""
    cfg = GenTacConfig(smoke=True)
    assert cfg.d_model == 128
    assert cfg.n_layers == 2
    assert cfg.ema_decay == 0.0           # smoke disables EMA


def test_paper_config_has_ema():
    """Full config must enable EMA by default (canonical diffusion training)."""
    cfg = GenTacConfig()
    assert cfg.d_model == 256
    assert cfg.n_layers == 4
    assert cfg.ema_decay > 0
