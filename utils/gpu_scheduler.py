"""Cross-process GPU scheduler for single-process training entry points.

The goal: when a user launches several ``python train.py`` (or
``run_improvement_experiments.py``) commands in different panes, they should
automatically spread across the free GPUs instead of all piling onto GPU 0.
If every GPU is busy, later commands should block and wait in a queue until
a GPU frees up.

The coordination primitive is a per-GPU advisory lock file under
``/tmp/angiography_vqa_gpu_locks/``. Acquiring the lock is race-free
(``fcntl.flock(LOCK_EX | LOCK_NB)``); the OS releases the lock automatically
when the holding process exits, so there is no stale-file cleanup to worry
about even on crashes.

This module is intentionally stdlib-only so it is cheap to import and has
no chance of pulling in torch before the caller wants it.

Usage
-----

    from utils.gpu_scheduler import acquire_free_gpu
    gpu_id = acquire_free_gpu()   # int, e.g. 1. Blocks until a GPU is free.

The caller passes ``gpu_id`` straight to Ultralytics / torch as ``device``.

The scheduler is a no-op when ``CUDA_VISIBLE_DEVICES`` is already set — that
means the caller is inside a ``run_stenosis_strategies.py`` worker (or any
other parent that has pre-masked the GPU view), and we must not fight the
parent's allocation. In that case we return ``0`` (the only GPU visible to
the process after masking) without touching any lockfile.
"""

from __future__ import annotations

import atexit
import fcntl
import logging
import os
import subprocess
import time
from pathlib import Path
from typing import List, Optional

LOG = logging.getLogger(__name__)

DEFAULT_LOCK_DIR = Path("/tmp/angiography_vqa_gpu_locks")
DEFAULT_MIN_FREE_MB = 2000
DEFAULT_POLL_INTERVAL_S = 10

# Module-level registry so fds acquired via flock are not garbage-collected
# (which would release the lock) while the process is still alive.
_HELD_LOCKS: list = []


def list_gpu_ids() -> List[int]:
    """Return the list of visible GPU indices via ``nvidia-smi -L``.

    Returns an empty list if nvidia-smi is unavailable or returns nothing,
    so callers can treat that as "no GPUs" and fall back to CPU without
    crashing.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "-L"], text=True, stderr=subprocess.DEVNULL
        )
    except Exception:
        return []
    ids: list[int] = []
    for line in out.strip().split("\n"):
        line = line.strip()
        if not line.startswith("GPU "):
            continue
        # Format: "GPU 0: NVIDIA GeForce RTX 5090 (UUID: GPU-...)"
        try:
            after = line[len("GPU "):]
            idx = int(after.split(":", 1)[0])
            ids.append(idx)
        except (ValueError, IndexError):
            continue
    return ids


def gpu_free_mb(gpu_id: int) -> int:
    """Return free VRAM in MB on ``gpu_id`` via nvidia-smi.

    Returns -1 if nvidia-smi is unavailable or parsing fails, so callers
    can treat an unknown result as "don't block".
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
                "-i", str(gpu_id),
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return int(out.strip().split("\n")[0])
    except Exception:
        return -1


def gpu_compute_pids(
    gpu_id: int, exclude_pids: Optional[set] = None
) -> List[int]:
    """Return compute-mode PIDs currently running on ``gpu_id``."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader,nounits",
                "-i", str(gpu_id),
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    pids: list[int] = []
    for line in out.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            pid = int(line)
        except ValueError:
            continue
        if exclude_pids is None or pid not in exclude_pids:
            pids.append(pid)
    return pids


def _try_lock(lock_dir: Path, gpu_id: int) -> Optional[int]:
    """Try to acquire the per-GPU lock file. Returns the fd on success,
    ``None`` if the lock is already held by another process.
    """
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_path = lock_dir / f"gpu_{gpu_id}.lock"
    # Open with O_CREAT so the first caller makes the file, then flock.
    fd = os.open(str(lock_path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(fd)
        return None
    # Write our pid into the lockfile for debuggability. Best-effort.
    try:
        os.ftruncate(fd, 0)
        os.write(fd, f"{os.getpid()}\n".encode())
    except OSError:
        pass
    return fd


def acquire_free_gpu(
    min_free_mb: int = DEFAULT_MIN_FREE_MB,
    poll_interval_s: int = DEFAULT_POLL_INTERVAL_S,
    lock_dir: Path = DEFAULT_LOCK_DIR,
) -> int:
    """Acquire an exclusive reservation on a free GPU.

    Algorithm:

    1. If ``CUDA_VISIBLE_DEVICES`` is already set in the environment, honor
       the caller's pre-masking and return 0 without touching any lockfile.
       The caller is inside a parent that has already assigned a GPU.
    2. Otherwise enumerate GPUs via ``nvidia-smi -L``.
    3. For each GPU whose free VRAM is at least ``min_free_mb``, try to
       flock ``gpu_<id>.lock``. First success wins, and the fd is stashed
       in a module-level registry so the lock survives for the lifetime of
       the process.
    4. If no GPU is claimable right now, log a waiting message and sleep
       ``poll_interval_s`` seconds, then try again. Blocks indefinitely
       (queue semantics) unless the environment variable
       ``ANGIO_GPU_ACQUIRE_TIMEOUT_S`` is set, in which case
       ``RuntimeError`` is raised after that many seconds.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None and cvd.strip() != "":
        # Parent masked the GPU view; never contend with them on lockfiles.
        LOG.debug(
            "CUDA_VISIBLE_DEVICES=%r already set; skipping lock-based "
            "acquire and returning 0",
            cvd,
        )
        return 0

    gpu_ids = list_gpu_ids()
    if not gpu_ids:
        raise RuntimeError(
            "[gpu_scheduler] nvidia-smi reported no GPUs; cannot auto-acquire. "
            "Set device explicitly in config.yaml or export CUDA_VISIBLE_DEVICES."
        )

    timeout_env = os.environ.get("ANGIO_GPU_ACQUIRE_TIMEOUT_S")
    deadline: Optional[float] = None
    if timeout_env:
        try:
            deadline = time.monotonic() + float(timeout_env)
        except ValueError:
            deadline = None

    waited = False
    while True:
        for gpu_id in gpu_ids:
            free = gpu_free_mb(gpu_id)
            if 0 <= free < min_free_mb:
                # nvidia-smi says this card is busy; don't bother locking.
                continue
            fd = _try_lock(lock_dir, gpu_id)
            if fd is None:
                continue
            _HELD_LOCKS.append(fd)
            print(
                f"[gpu_scheduler] acquired GPU {gpu_id} "
                f"(free={free} MB, lock={lock_dir}/gpu_{gpu_id}.lock, "
                f"pid={os.getpid()})",
                flush=True,
            )
            return gpu_id

        if deadline is not None and time.monotonic() >= deadline:
            raise RuntimeError(
                f"[gpu_scheduler] timed out waiting for a free GPU after "
                f"{timeout_env}s (pool={gpu_ids})"
            )

        if not waited:
            print(
                f"[gpu_scheduler] all GPUs busy (pool={gpu_ids}, "
                f"min_free_mb={min_free_mb}); waiting for a free slot…",
                flush=True,
            )
            waited = True
        time.sleep(poll_interval_s)


def resolve_device(value, **kwargs) -> object:
    """Idempotently resolve a ``device`` config value.

    Returns ``value`` unchanged unless it is the string ``"auto"`` (case
    insensitive), in which case it calls :func:`acquire_free_gpu` and
    returns the acquired integer. Lists, other strings and ints pass
    through untouched.
    """
    if isinstance(value, str) and value.strip().lower() == "auto":
        return acquire_free_gpu(**kwargs)
    return value


def _release_all() -> None:
    """Close all held lock fds. Registered via ``atexit`` as belt-and-
    suspenders; the kernel would release the locks anyway when the process
    exits.
    """
    while _HELD_LOCKS:
        fd = _HELD_LOCKS.pop()
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass
        try:
            os.close(fd)
        except OSError:
            pass


atexit.register(_release_all)
