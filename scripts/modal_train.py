"""Modal entrypoint for GenTac full-config training.

Before paying for a Modal account or burning cloud GPU, validate locally:

    python scripts/modal_train.py --validate-local

That runs `train_full.py --dry-run` end-to-end on your CPU/MPS — same code path
Modal will execute, so deps / schema / data-path bugs surface in 30 seconds
locally instead of 10 minutes into a $25 cloud run.

Full Modal flow:
    pip install modal
    modal token new

    modal run scripts/modal_train.py::sync_and_train     # one-shot
    # or two-step
    modal run scripts/modal_train.py::sync_repo
    modal run scripts/modal_train.py::train --epochs 60
    modal run scripts/modal_train.py::fetch              # pull ckpts back

Cost note (May 2026): A100-40GB at ~$3.50/hr on Modal × ~6 hr for a full 60-epoch
run on ~1k–3k trajectory windows ≈ $20–25. If Modal is too pricey, use Lambda
Labs (~$1.10/hr A100) by SSHing in and running scripts/train_full.py directly.
See docs/cloud_training.md for the full troubleshooting matrix.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

VOL_MOUNT = "/vol"
CODE_DIR = f"{VOL_MOUNT}/code"
DATA_DIR = f"{VOL_MOUNT}/data/processed"
CKPT_DIR = f"{VOL_MOUNT}/checkpoints/full"


# Modal import is intentionally lazy so `--validate-local` works without modal installed.
try:
    import modal                                          # type: ignore
    _HAS_MODAL = True
except ImportError:
    modal = None                                          # type: ignore
    _HAS_MODAL = False


if _HAS_MODAL:
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install(
            "torch==2.4.0",
            "pytorch-lightning==2.4.0",
            "einops==0.8.0",
            "numpy==1.26.4",
        )
    )

    app = modal.App("gentac")
    volume = modal.Volume.from_name("gentac-data", create_if_missing=True)
else:
    # Stubs so the @app.function decorators below resolve at import time even
    # when modal isn't installed. They aren't reachable without modal.
    class _NoModalStub:
        def function(self, *a, **k):
            def deco(fn):
                fn._needs_modal = True
                return fn
            return deco

        def local_entrypoint(self):
            def deco(fn):
                fn._needs_modal = True
                return fn
            return deco

    class _VolumeStub:
        def commit(self): pass

    app = _NoModalStub()                                  # type: ignore
    volume = _VolumeStub()                                # type: ignore
    image = None                                          # type: ignore


@app.function(image=image, volumes={VOL_MOUNT: volume}, timeout=600)
def sync_repo():
    """Upload the local src/ tree and data/processed/*.json into the Modal volume.

    Uses NetworkFileSystem-style mounting: we walk the local files and write them
    to the volume from inside the container. Run this before `train` whenever
    code or data changes locally.
    """
    import shutil
    from pathlib import Path as P

    code_root = P(CODE_DIR)
    if code_root.exists():
        shutil.rmtree(code_root)
    code_root.mkdir(parents=True)

    data_root = P(DATA_DIR)
    data_root.mkdir(parents=True, exist_ok=True)

    print(f"[modal] code → {code_root}, data → {data_root}")
    volume.commit()
    print("[modal] volume initialized — now run `modal volume put gentac-data <local> <remote>` "
          "to upload your local src/ and data/processed/ trees.")


@app.function(
    image=image,
    volumes={VOL_MOUNT: volume},
    gpu="A100-40GB",
    timeout=24 * 3600,
)
def train(
    epochs: int = 60,
    batch_size: int | None = None,
    precision: str = "16-mixed",
    resume: str | None = None,
):
    """Run scripts/train_full.py inside the container against the mounted volume."""
    import subprocess
    import sys as _sys
    from pathlib import Path as P

    _sys.path.insert(0, CODE_DIR)
    code_root = P(CODE_DIR)
    if not (code_root / "src" / "model" / "config.py").exists():
        raise RuntimeError(
            f"{code_root}/src not populated — run `modal volume put gentac-data ./src /code/src` "
            f"and `modal volume put gentac-data ./data/processed /data/processed` first."
        )

    P(CKPT_DIR).mkdir(parents=True, exist_ok=True)

    cmd = [
        _sys.executable, str(code_root / "scripts" / "train_full.py"),
        "--data-dir", DATA_DIR,
        "--ckpt-dir", CKPT_DIR,
        "--epochs", str(epochs),
        "--precision", precision,
    ]
    if batch_size is not None:
        cmd += ["--batch-size", str(batch_size)]
    if resume is not None:
        cmd += ["--resume", resume]

    print(f"[modal] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    volume.commit()
    print(f"[modal] training complete; checkpoints persisted in volume at {CKPT_DIR}")


@app.function(image=image, volumes={VOL_MOUNT: volume}, timeout=600)
def download_checkpoints() -> bytes:
    """Pack checkpoints/full into a tarball so the caller can stream it down."""
    import io
    import tarfile
    from pathlib import Path as P

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        ckpt_root = P(CKPT_DIR)
        if not ckpt_root.exists():
            raise RuntimeError(f"no checkpoints in {ckpt_root}")
        for f in sorted(ckpt_root.rglob("*.ckpt")):
            tar.add(f, arcname=f.relative_to(ckpt_root))
            print(f"[modal] packing {f.name} ({f.stat().st_size / 1e6:.1f} MB)")
    return buf.getvalue()


@app.local_entrypoint()
def sync_and_train(epochs: int = 60, precision: str = "16-mixed"):
    """One-call end-to-end: ensures volume exists, then trains."""
    sync_repo.remote()
    print("[local] reminder: after the first sync_repo call, push your local code with:")
    print("        modal volume put gentac-data ./src /code/src")
    print("        modal volume put gentac-data ./data/processed /data/processed")
    print("[local] starting training…")
    train.remote(epochs=epochs, precision=precision)


@app.local_entrypoint()
def fetch():
    """Download the trained checkpoints back to local checkpoints/full/."""
    blob = download_checkpoints.remote()
    out_dir = REPO_ROOT / "checkpoints" / "full"
    out_dir.mkdir(parents=True, exist_ok=True)
    tar_path = out_dir / "from_modal.tar.gz"
    tar_path.write_bytes(blob)
    print(f"[local] wrote {tar_path} ({tar_path.stat().st_size / 1e6:.1f} MB)")
    print(f"[local] extract: tar -xzf {tar_path} -C {out_dir}")


def _validate_local() -> int:
    """Run the SAME training code path Modal will run, but on local CPU/MPS.

    This is the cheap pre-flight: ~30 seconds locally vs ~$25 to discover that
    your data wasn't uploaded to the volume / the cfg dict didn't round-trip /
    a dep version drifted. If this passes, the cloud run is overwhelmingly
    likely to make it through epoch 0.
    """
    print(f"[validate] modal installed: {_HAS_MODAL}")
    print(f"[validate] CODE_DIR={CODE_DIR}  DATA_DIR={DATA_DIR}  CKPT_DIR={CKPT_DIR}")

    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "train_full.py"), "--dry-run"]
    print(f"[validate] running: {' '.join(cmd)}")
    try:
        r = subprocess.run(cmd, check=False)
    except FileNotFoundError as e:
        print(f"[validate] FAIL — cannot launch train_full.py: {e}")
        return 1
    if r.returncode != 0:
        print(f"[validate] FAIL — train_full.py --dry-run exited {r.returncode}")
        return r.returncode

    # Confirm the dry-run actually wrote and re-loaded a checkpoint.
    dry_ckpt = REPO_ROOT / "checkpoints" / "full" / "dryrun.ckpt"
    if not dry_ckpt.exists():
        print(f"[validate] FAIL — expected {dry_ckpt} to exist after dry-run")
        return 2

    try:
        sys.path.insert(0, str(REPO_ROOT))
        from src.model.lightning_module import GenTacTrajectoryModule
        cfg = GenTacTrajectoryModule.cfg_from_checkpoint(dry_ckpt)
        print(f"[validate] reloaded cfg: schema={cfg.schema_version} d_model={cfg.d_model} ema_decay={cfg.ema_decay}")
    except Exception as e:
        print(f"[validate] FAIL — could not reload checkpoint cfg: {e}")
        return 3

    print("[validate] OK — Modal will run the same code path.")
    print("[validate] next:  modal token new  &&  modal run scripts/modal_train.py::sync_and_train")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    ap.add_argument("--validate-local", action="store_true",
                    help="Run the equivalent training path locally (no Modal account / GPU needed).")
    args = ap.parse_args()
    if args.validate_local:
        sys.exit(_validate_local())
    if not _HAS_MODAL:
        print("modal is not installed. Either `pip install modal` for cloud runs,")
        print("or run `python scripts/modal_train.py --validate-local` for offline validation.")
        sys.exit(1)
    print("Use `modal run scripts/modal_train.py::<function>` for actual cloud execution.")
    print("Available: sync_repo, train, download_checkpoints, sync_and_train, fetch.")
    print("Or pass --validate-local for an offline pre-flight check.")
