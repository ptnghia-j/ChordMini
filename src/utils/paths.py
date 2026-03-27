from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys


def discover_src_dir(start: str | Path | None = None) -> Path:
    start_path = Path(start).resolve() if start is not None else Path(__file__).resolve()
    current = start_path if start_path.is_dir() else start_path.parent

    for candidate in (current, *current.parents):
        if candidate.name == 'src':
            return candidate
        nested_src = candidate / 'src'
        if nested_src.is_dir():
            return nested_src

    raise RuntimeError(f'Could not locate src directory from {start_path}')


def _looks_like_project_root(path: Path) -> bool:
    return (path / 'src').is_dir() and (
        (path / 'requirements.txt').exists() or (path / 'README.md').exists()
    )


@lru_cache(maxsize=None)
def discover_project_root(start: str | Path | None = None) -> Path:
    start_path = Path(start).resolve() if start is not None else Path(__file__).resolve()
    current = start_path if start_path.is_dir() else start_path.parent

    for candidate in (current, *current.parents):
        if _looks_like_project_root(candidate):
            return candidate

    raise RuntimeError(f'Could not locate project root from {start_path}')


def ensure_project_root_on_path(start: str | Path | None = None) -> Path:
    root = discover_project_root(start)
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return root


def bootstrap_project_root(start: str | Path | None = None) -> Path:
    src_dir = discover_src_dir(start)
    src_dir_str = str(src_dir)
    if src_dir_str not in sys.path:
        sys.path.insert(0, src_dir_str)
    return ensure_project_root_on_path(start)


def project_path(*parts: str, start: str | Path | None = None) -> Path:
    return discover_project_root(start).joinpath(*parts)


__all__ = [
    'bootstrap_project_root',
    'discover_project_root',
    'discover_src_dir',
    'ensure_project_root_on_path',
    'project_path',
]
