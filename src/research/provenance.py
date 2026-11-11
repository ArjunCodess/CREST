"""Content-addressed provenance for research runs."""
from __future__ import annotations

import hashlib
from pathlib import Path


def research_code_content_hash(repo_root: Path | None = None) -> str:
    root = repo_root or Path(__file__).resolve().parents[2]
    h = hashlib.sha256()
    extra_roots = [
        root / 'pyproject.toml',
        root / 'requirements.txt',
        root / 'src' / 'research' / 'bundled_configs',
    ]
    patterns = [root / 'src' / 'research', root / 'configs' / 'research']
    files: list[Path] = []
    for base in extra_roots:
        if base.is_file():
            files.append(base)
        elif base.is_dir():
            files.extend(sorted(p for p in base.rglob('*') if p.is_file()))
    lock = root / 'requirements-lock.txt'
    if lock.is_file():
        files.append(lock)
    for base in patterns:
        if not base.exists():
            continue
        if base.is_file():
            files.append(base)
        else:
            files.extend(sorted(p for p in base.rglob('*') if p.is_file() and p.suffix in {'.py', '.json'}))
    seen: set[Path] = set()
    deduped: list[Path] = []
    for p in files:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            deduped.append(p)
    files = deduped
    for path in files:
        h.update(path.relative_to(root).as_posix().encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:16]
