from __future__ import annotations

import contextlib
import os
import sys


@contextlib.contextmanager
def suppress_stderr():
    """Suppress noisy stderr output from third-party audio backends."""
    try:
        stderr_fd = sys.stderr.fileno()
    except Exception:
        yield
        return

    original = os.dup(stderr_fd)
    try:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, stderr_fd)
        os.close(devnull)
        yield
    finally:
        os.dup2(original, stderr_fd)
        os.close(original)


__all__ = ['suppress_stderr']
