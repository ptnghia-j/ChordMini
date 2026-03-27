from __future__ import annotations

from pathlib import Path
import sys


def ensure_src_on_path(start: str | Path | None = None) -> Path:
    start_path = Path(start).resolve() if start is not None else Path(__file__).resolve()
    src_dir = start_path.parents[1] if start_path.is_file() else start_path
    src_dir_str = str(src_dir)
    if src_dir_str not in sys.path:
        sys.path.insert(0, src_dir_str)
    return src_dir


def bootstrap_cli(start: str | Path | None = None):
    ensure_src_on_path(start)
    from .paths import bootstrap_project_root

    return bootstrap_project_root(start)


__all__ = ['bootstrap_cli', 'ensure_src_on_path']
