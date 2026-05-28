# Cloud training

Mac MPS is enough to verify the code runs but not enough to train paper-quality
GenTac. This document covers running the real training on a single A100 / H100 via
Modal, Lambda Labs, or any rented GPU box. Pick whichever pricing/access is best.

## Estimating cost

Paper config: M=4 layers, d=256, n_heads=8, S=100, batch=200, 60 epochs.
Single A100 should finish in **≤24 hours** based on the paper's own setup
(§6.3.3: "All experiments in this work are conducted on a single NVIDIA A100").

Approx pricing (May 2026):
- Lambda Labs A100 40GB on-demand: ~$1.10/hr → **~$25/run**
- Modal A100 80GB: ~$3.50/hr → **~$80/run**
- RunPod / vast.ai A100 community: ~$0.80–1.00/hr → **~$20/run**

Budget ~$30 for the first real pretraining run + a couple of fine-tunes.

## Option A — Modal (easiest, but priciest)

`pip install modal` locally, `modal token new`, then:

```python
# scripts/modal_train.py (sketch)
import modal

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "pytorch-lightning", "einops", "numpy")
)
app = modal.App("gentac")
volume = modal.Volume.from_name("gentac-data", create_if_missing=True)

@app.function(gpu="A100-40GB", image=image, volumes={"/data": volume}, timeout=86400)
def train():
    import subprocess, sys
    sys.path.insert(0, "/data/code")
    from src.model.config import GenTacConfig
    from src.model.lightning_module import GenTacDataModule, GenTacTrajectoryModule
    import pytorch_lightning as pl
    cfg = GenTacConfig()        # full paper config
    dm = GenTacDataModule("/data/processed", cfg, num_workers=4)
    mod = GenTacTrajectoryModule(cfg)
    trainer = pl.Trainer(
        max_epochs=cfg.epochs, accelerator="gpu", devices=1, precision="16-mixed",
        default_root_dir="/data/checkpoints",
    )
    trainer.fit(mod, dm)
```

Before running: upload `data/processed/*.json` to the volume and sync `src/`.

```bash
modal volume put gentac-data data/processed /processed
modal volume put gentac-data src             /code/src
modal run scripts/modal_train.py
```

## Option B — Lambda Labs (cheapest serious option)

1. Provision an A100 40GB instance from `lambdalabs.com/service/gpu-cloud`.
2. SSH in. `git clone` your project, then:

```bash
pip install torch pytorch-lightning einops numpy
# upload data: scp -r data/processed lambda:~/gentac/data/
python scripts/train_full.py   # paper config (see below)
```

`scripts/train_full.py` is just `smoke_train.py` with `GenTacConfig()` (no `smoke=True`).

## Option C — RunPod / vast.ai

Same idea as Lambda but the OS image and storage details differ. Use the same
`train_full.py` script. Don't forget to back up checkpoints to S3 / a volume before
shutting down the instance — community boxes can disappear without notice.

## Things to verify before paying for GPU time

1. **Loss decreases on the smoke run.** See `scripts/smoke_train.py`; we logged
   1.38 → 0.378 → 0.144 over 3 epochs on MPS. If it doesn't move, fix bugs first.
2. **Data is enough.** Paper used 2,838 trajectory segments; we currently have ~1,000
   from Metrica. Pull SkillCorner (M2.2) before the real run or expect overfitting.
3. **Sampling output looks sane on the smoke checkpoint.** Run `scripts/sample.py`
   and check that the inter-sample stddev is > 0 (it was 3.42m for us, healthy)
   and that coords don't fly off the pitch by more than ~5m.

## After training

```bash
# Copy the checkpoint back
scp lambda:~/gentac/checkpoints/last.ckpt checkpoints/
# Sample with the real model
python scripts/sample.py --ckpt checkpoints/last.ckpt --mode opp_conditioned_team0
```

Then plug the resulting `data/processed/samples.json` into the renderer (M3).

---

## Pre-flight: validate locally before paying

The single most expensive failure mode is "spent $25, discovered at minute 9 that
my checkpoint cfg dict didn't round-trip / numpy version drifted / data volume
didn't mount". Both training scripts have a cheap pre-flight that exercises the
exact code path the cloud run will take, but on CPU/MPS and a 2-batch budget:

```bash
# Direct
python scripts/train_full.py --dry-run

# Via the Modal entrypoint (works WITHOUT `modal` installed — useful before any
# cloud setup at all)
python scripts/modal_train.py --validate-local
```

Either path: runs data-load → model fwd → backward → checkpoint write → checkpoint
reload + cfg roundtrip. ~30 s end-to-end. If this fails, the cloud run will fail.
If it passes, the cloud run is overwhelmingly likely to make it past epoch 0.

---

## Troubleshooting

### Modal

| Symptom | Likely cause | Fix |
|---|---|---|
| `RuntimeError: src not populated` at start of `train` | Forgot the `modal volume put` step after `sync_repo`. The volume exists but is empty. | `modal volume put gentac-data ./src /code/src` and `modal volume put gentac-data ./data/processed /data/processed` before `modal run scripts/modal_train.py::train`. |
| `CUDA out of memory` near step ~50 | Default batch=200 + paper d=256 + 16-mixed exceeds A100-40GB headroom under some sequence lengths. | Bump to A100-80GB (`gpu="A100-80GB"` in `modal_train.py`) or pass `--batch-size 128` on `train`. |
| Cold-start timeout > 10 min before training begins | Modal image rebuild — usually after a `pip_install` version bump in `modal_train.py`. | First run rebuilds; subsequent runs are cached. Wait it out once. |
| Random `Killed` / OOM kill on CPU during data load | `num_workers > 4` on a small Modal container — workers OOM the host before the GPU sees data. | Lower `num_workers` (the `train_full.py` default of 4 is safe). |
| `download_checkpoints` returns 0 bytes | Training never wrote to `CKPT_DIR` — usually a permissions or path bug. | Inspect: `modal volume ls gentac-data /checkpoints/full`. If empty, the training job died silently — check its logs. |
| `modal.exception.AuthError` | `modal token new` was never run or the token expired. | Re-auth. |

### Lambda Labs (or any SSH-to-VM)

| Symptom | Likely cause | Fix |
|---|---|---|
| `ImportError: torch ... GLIBC_X.Y not found` | The instance image has a system glibc older than your torch wheel expects. | Use the Lambda "PyTorch" base image rather than "Base"; or pin `torch` to a version matching the system. |
| `Killed` during validation | Default `num_workers=4` × `batch_size=200` blows the 30 GB Lambda VM RAM. | Lower `num_workers` to 2 OR `--batch-size 128`. |
| SCP of `last.ckpt` is 0 bytes | Checkpoint was still being written when you `scp`'d. | Wait for "training complete" in the trainer log, then SCP. |
| Instance disappears mid-run | Lambda preempted the spot instance or you hit the 24h limit. | Use on-demand pricing, OR `--resume` from the most recent `best-XX-...ckpt` on next run. The repo's `ModelCheckpoint` writes every epoch. |

### Mixed-precision (16-mixed / bf16-mixed)

| Symptom | Fix |
|---|---|
| Loss = NaN within first ~100 steps | Drop to `--precision bf16-mixed` if hardware supports it; otherwise `--precision 32-true` for one epoch then mix in fp16 once loss is stable. |
| Loss jumps wildly between steps then stabilizes | Normal warmup behavior under fp16; check `grad_norm` is not exploding. The trainer applies `gradient_clip_val=1.0`. |
| H100 not noticeably faster than A100 under `16-mixed` | The default schedule is bandwidth-bound, not compute-bound. Try `bf16-mixed` on H100 — usually 1.3–1.6× speedup. |

### Checkpoint reload errors (server / sample.py)

| Symptom | Likely cause | Fix |
|---|---|---|
| `checkpoint has no cfg_dict; retrain after the config-persistence fix` | Checkpoint was produced before the cfg-persistence fix (anything before 2026-05-28). | Retrain. The fix lives at [`src/model/lightning_module.py`](../src/model/lightning_module.py). |
| `checkpoint schema_version=2 but runtime expects 3` | Architecture changed (e.g. EMA buffers added). | Retrain — schema bumps are explicit and refuse to silently misload. |
| Missing/extra keys in `load_state_dict` | Custom changes to model layers between checkpoint and code. | Re-derive the cfg dict, retrain. There's no automatic migration. |

### Sanity checks before paying

(Repeated for prominence — these are the four things to verify pre-cloud.)

1. **`python scripts/smoke_train.py` runs and loss decreases** — proves the model + data are wired.
2. **`python scripts/train_full.py --dry-run` exits 0** — proves the FULL training script (paper config, real Lightning Trainer) runs end-to-end on CPU.
3. **`python scripts/eval.py --ckpt checkpoints/smoke/smoke_final.ckpt --n-frames 4 --k 3`** — proves the inference + physics + eval harness all work.
4. **`python scripts/modal_train.py --validate-local` exits 0** — proves the Modal entrypoint can launch the training subprocess. (No Modal account needed.)

If all four pass, the only thing the cloud run can fail on is something genuinely cloud-specific: dep version mismatch in the Modal image, volume mount, or actual training pathology that needs more data. The first two surface in the first 60 s of the cloud run; the third surfaces in the loss curve.

